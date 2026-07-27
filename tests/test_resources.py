import json
import os
import subprocess
import sys
from pathlib import Path

import pytest

from g2p_mix import resources

CASE_FILE = Path(__file__).parent / "cases" / "resources.json"
CASE_GROUPS = json.loads(CASE_FILE.read_text(encoding="utf-8"))


def cases(group):
    return [pytest.param(case, id=case["id"]) for case in CASE_GROUPS[group]]


@pytest.mark.parametrize("case", cases("english_force_spellings"))
def test_english_force_spelling_resource_controls_cmudict_filter(case, monkeypatch):
    from nltk.corpus import cmudict

    spellings = resources.load_english_force_spellings()
    assert spellings == resources.load_lines(case["resource"])
    assert len(spellings) == len(set(spellings))
    assert all(spelling == spelling.upper() for spelling in spellings)

    resources.ensure_bundled_nltk_data()
    unfiltered = cmudict.dict()

    assert all(spelling.lower() in unfiltered for spelling in spellings)
    resources.load_cmudict.cache_clear()
    filtered = resources.load_cmudict()
    assert all(spelling.lower() not in filtered for spelling in spellings)

    synthetic_spelling = case["synthetic_spelling"]
    assert synthetic_spelling not in spellings
    monkeypatch.setattr(resources, "load_english_force_spellings", lambda: (synthetic_spelling,))
    resources.load_cmudict.cache_clear()
    try:
        synthetic_filter = resources.load_cmudict()
        assert synthetic_spelling.lower() not in synthetic_filter
        assert all(spelling.lower() in synthetic_filter for spelling in spellings)
    finally:
        resources.load_cmudict.cache_clear()


@pytest.mark.parametrize("case", cases("jieba_phrase_installation"))
def test_jieba_phrase_installation_concurrent_first_call(case):
    code = """
import json
import sys
import threading
import time
from concurrent.futures import ThreadPoolExecutor

import jieba

from g2p_mix import resources

case = json.loads(sys.argv[1])
workers = case["workers"]
barrier = threading.Barrier(workers)
calls = []
calls_lock = threading.Lock()

def record_add_word(word):
    time.sleep(case["delay_seconds"])
    with calls_lock:
        calls.append(word)

def install_after_barrier():
    barrier.wait()
    return resources.install_jieba_phrases()

original_add_word = jieba.add_word
jieba.add_word = record_add_word
try:
    expected_words = tuple(word for word, _ in resources.load_pinyin_phrases())
    expected = frozenset(expected_words)
    with ThreadPoolExecutor(max_workers=workers) as executor:
        results = tuple(executor.map(lambda _: install_after_barrier(), range(workers)))

    calls_after_race = tuple(calls)
    later_results = tuple(
        resources.install_jieba_phrases() for _ in range(case["subsequent_calls"])
    )
    print(json.dumps({
        "expected_word_count": len(expected),
        "installed_in_resource_order": calls_after_race == expected_words,
        "call_counts": [calls_after_race.count(word) for word in expected_words],
        "all_race_results_match": all(result == expected for result in results),
        "all_later_results_match": all(result == expected for result in later_results),
        "later_add_call_count": len(calls) - len(calls_after_race),
    }))
finally:
    jieba.add_word = original_add_word
"""
    completed = subprocess.run(
        [sys.executable, "-c", code, json.dumps(case)],
        check=True,
        capture_output=True,
        text=True,
        env=os.environ.copy(),
        timeout=case["subprocess_timeout_seconds"],
    )
    payload = json.loads(completed.stdout.splitlines()[-1])

    assert payload["expected_word_count"] > 0
    assert payload == {
        "expected_word_count": payload["expected_word_count"],
        "installed_in_resource_order": True,
        "call_counts": [1] * payload["expected_word_count"],
        "all_race_results_match": True,
        "all_later_results_match": True,
        "later_add_call_count": 0,
    }


@pytest.mark.parametrize("case", cases("jieba_phrase_installation"))
def test_jieba_phrase_installation_failure_resumes_without_replay(case):
    code = """
import json
import sys
import threading
import time
from concurrent.futures import ThreadPoolExecutor

import jieba

from g2p_mix import resources

case = json.loads(sys.argv[1])
workers = case["workers"]
barrier = threading.Barrier(workers)
expected_words = tuple(word for word, _ in resources.load_pinyin_phrases())
expected = frozenset(expected_words)
failure_word = expected_words[case["failure_index"]]
attempts = {}
successes = {}
attempt_threads = {}
successful_order = []
calls_lock = threading.Lock()

class InjectedFailure(RuntimeError):
    pass

def record_add_word(word):
    time.sleep(case["delay_seconds"])
    thread_id = threading.get_ident()
    with calls_lock:
        attempt = attempts.get(word, 0) + 1
        attempts[word] = attempt
        attempt_threads.setdefault(word, []).append(thread_id)
    if word == failure_word and attempt == 1:
        raise InjectedFailure("injected add_word failure")
    with calls_lock:
        successes[word] = successes.get(word, 0) + 1
        successful_order.append(word)

def install_after_barrier():
    barrier.wait()
    try:
        result = resources.install_jieba_phrases()
    except InjectedFailure:
        return {"installed": False, "thread": threading.get_ident()}
    return {"installed": result == expected, "thread": threading.get_ident()}

original_add_word = jieba.add_word
jieba.add_word = record_add_word
try:
    with ThreadPoolExecutor(max_workers=workers) as executor:
        results = tuple(executor.map(lambda _: install_after_barrier(), range(workers)))

    attempts_after_race = dict(attempts)
    successful_order_after_race = tuple(successful_order)
    later_results = tuple(
        resources.install_jieba_phrases() for _ in range(case["subsequent_calls"])
    )
    failure_threads = attempt_threads[failure_word]
    print(json.dumps({
        "expected_word_count": len(expected_words),
        "failure_index": case["failure_index"],
        "attempt_counts": [attempts.get(word, 0) for word in expected_words],
        "success_counts": [successes.get(word, 0) for word in expected_words],
        "successful_order_matches": successful_order_after_race == expected_words,
        "injected_failure_count": sum(not result["installed"] for result in results),
        "successful_result_count": sum(result["installed"] for result in results),
        "takeover_thread_changed": (
            len(failure_threads) == 2 and failure_threads[0] != failure_threads[1]
        ),
        "all_later_results_match": all(result == expected for result in later_results),
        "later_add_call_count": sum(attempts.values()) - sum(attempts_after_race.values()),
    }))
finally:
    jieba.add_word = original_add_word
"""
    completed = subprocess.run(
        [sys.executable, "-c", code, json.dumps(case)],
        check=True,
        capture_output=True,
        text=True,
        env=os.environ.copy(),
        timeout=case["subprocess_timeout_seconds"],
    )
    payload = json.loads(completed.stdout.splitlines()[-1])
    expected_attempts = [1] * payload["expected_word_count"]
    expected_attempts[payload["failure_index"]] = 2

    assert payload == {
        "expected_word_count": payload["expected_word_count"],
        "failure_index": case["failure_index"],
        "attempt_counts": expected_attempts,
        "success_counts": [1] * payload["expected_word_count"],
        "successful_order_matches": True,
        "injected_failure_count": 1,
        "successful_result_count": case["workers"] - 1,
        "takeover_thread_changed": True,
        "all_later_results_match": True,
        "later_add_call_count": 0,
    }


@pytest.mark.parametrize("case", cases("jieba_phrase_publication_failure"))
def test_jieba_phrase_publication_failure_allows_waiter_takeover(case):
    code = """
import json
import sys
import threading
from concurrent.futures import ThreadPoolExecutor

import jieba

from g2p_mix import resources

case = json.loads(sys.argv[1])
workers = case["workers"]
barrier = threading.Barrier(workers)
expected_words = tuple(word for word, _ in resources.load_pinyin_phrases())
expected = frozenset(expected_words)
add_calls = []
frozenset_threads = []
waiter_threads = set()
calls_lock = threading.Lock()
failure_lock = threading.Lock()
all_waiters_ready = threading.Event()
failure_remaining = case["failure_trigger"] == "frozenset"
condition = resources._JIEBA_PHRASE_INSTALL_CONDITION
original_wait = condition.wait

class InjectedFailure(BaseException):
    pass

def record_wait(timeout=None):
    with calls_lock:
        waiter_threads.add(threading.get_ident())
        if len(waiter_threads) == workers - 1:
            all_waiters_ready.set()
    return original_wait(timeout)

def record_add_word(word):
    with calls_lock:
        add_calls.append(word)

def injected_frozenset(values):
    global failure_remaining
    thread_id = threading.get_ident()
    with calls_lock:
        frozenset_threads.append(thread_id)
    with failure_lock:
        should_fail = failure_remaining
        failure_remaining = False
    if should_fail:
        if not all_waiters_ready.wait(case["waiter_ready_timeout_seconds"]):
            raise AssertionError("other installers did not reach the wait state")
        raise InjectedFailure("injected frozenset failure")
    return frozenset(values)

def install_after_barrier():
    barrier.wait()
    try:
        result = resources.install_jieba_phrases()
    except InjectedFailure as error:
        return {
            "failed": str(error) == "injected frozenset failure",
            "installed": False,
            "thread": threading.get_ident(),
        }
    return {
        "failed": False,
        "installed": result == expected,
        "thread": threading.get_ident(),
    }

original_add_word = jieba.add_word
jieba.add_word = record_add_word
condition.wait = record_wait
resources.frozenset = injected_frozenset
try:
    with ThreadPoolExecutor(max_workers=workers) as executor:
        results = tuple(executor.map(lambda _: install_after_barrier(), range(workers)))

    add_calls_after_race = tuple(add_calls)
    later_results = tuple(
        resources.install_jieba_phrases() for _ in range(case["subsequent_calls"])
    )
    failed_results = tuple(result for result in results if result["failed"])
    successful_results = tuple(result for result in results if result["installed"])
    print(json.dumps({
        "expected_word_count": len(expected_words),
        "failure_trigger": case["failure_trigger"],
        "installed_in_resource_order": add_calls_after_race == expected_words,
        "add_call_counts": [
            add_calls_after_race.count(word) for word in expected_words
        ],
        "injected_failure_count": len(failed_results),
        "successful_result_count": len(successful_results),
        "takeover_thread_changed": (
            len(frozenset_threads) == 2
            and len(failed_results) == 1
            and frozenset_threads[0] == failed_results[0]["thread"]
            and frozenset_threads[1] != frozenset_threads[0]
        ),
        "takeover_was_waiting": (
            len(frozenset_threads) == 2
            and frozenset_threads[1] in waiter_threads
        ),
        "all_later_results_match": all(
            result == expected for result in later_results
        ),
        "later_add_call_count": len(add_calls) - len(add_calls_after_race),
        "later_frozenset_call_count": len(frozenset_threads) - 2,
    }))
finally:
    del resources.frozenset
    condition.wait = original_wait
    jieba.add_word = original_add_word
"""
    completed = subprocess.run(
        [sys.executable, "-c", code, json.dumps(case)],
        check=True,
        capture_output=True,
        text=True,
        env=os.environ.copy(),
        timeout=case["subprocess_timeout_seconds"],
    )
    payload = json.loads(completed.stdout.splitlines()[-1])

    assert payload == {
        "expected_word_count": payload["expected_word_count"],
        "failure_trigger": "frozenset",
        "installed_in_resource_order": True,
        "add_call_counts": [1] * payload["expected_word_count"],
        "injected_failure_count": 1,
        "successful_result_count": case["workers"] - 1,
        "takeover_thread_changed": True,
        "takeover_was_waiting": True,
        "all_later_results_match": True,
        "later_add_call_count": 0,
        "later_frozenset_call_count": 0,
    }


@pytest.mark.parametrize("case", cases("jieba_phrase_claim_failure"))
def test_jieba_phrase_claim_faults_preserve_state_and_allow_retry(case):
    code = """
import json
import linecache
import sys
import threading
import time
from concurrent.futures import ThreadPoolExecutor

import jieba

from g2p_mix import resources

case = json.loads(sys.argv[1])
expected_words = tuple(word for word, _ in resources.load_pinyin_phrases())
expected = frozenset(expected_words)
condition = resources._JIEBA_PHRASE_INSTALL_CONDITION
original_wait = condition.wait
original_add_word = jieba.add_word
trace_hits = 0
trace_state = None
trace_thread = None
invariant_samples = []
calls = []
calls_lock = threading.Lock()
waiter_threads = set()
waiters_ready = threading.Event()

class InjectedTraceFailure(BaseException):
    pass

def state_is_valid(state):
    shape_is_valid = (
        (
            state.status == resources._JIEBA_INSTALL_PENDING
            and state.owner is None
            and state.installed is None
        )
        or (
            state.status == resources._JIEBA_INSTALL_INSTALLING
            and state.owner is not None
            and state.installed is None
        )
        or (
            state.status == resources._JIEBA_INSTALL_INSTALLED
            and state.owner is None
            and state.installed is not None
        )
    )
    return shape_is_valid and state.next_index >= 0

def sample_invariant():
    state = resources._jieba_phrase_install_state
    invariant_samples.append(state_is_valid(state))
    return state

def inject_trace(frame, event, arg):
    global trace_hits, trace_state, trace_thread
    if (
        frame.f_code is resources.install_jieba_phrases.__code__
        and event == "line"
        and case["fault_line_fragment"]
        in linecache.getline(frame.f_code.co_filename, frame.f_lineno)
    ):
        trace_hits += 1
        trace_state = sample_invariant()
        trace_thread = threading.get_ident()
        if trace_hits == case["fault_occurrence"]:
            raise InjectedTraceFailure(case["phase"])
    return inject_trace

def record_wait(timeout=None):
    with calls_lock:
        waiter_threads.add(threading.get_ident())
        if len(waiter_threads) >= case["workers"] - 1:
            waiters_ready.set()
    sample_invariant()
    return original_wait(timeout)

def record_add_word_for_retry(word):
    with calls_lock:
        calls.append(word)
        first_call = len(calls) == 1
    if first_call and not waiters_ready.wait(case["waiter_ready_timeout_seconds"]):
        raise AssertionError("retry installers did not reach the wait state")
    time.sleep(case["delay_seconds"])
    sample_invariant()

fault_caught = False
active_owner_preserved = True
owner_was_active_after_fault = False
fault_thread_was_owner = False

try:
    if case["fault_role"] == "claimant":
        sys.settrace(inject_trace)
        try:
            resources.install_jieba_phrases()
        except InjectedTraceFailure:
            fault_caught = True
        finally:
            sys.settrace(None)

        post_fault_state = sample_invariant()
        condition.wait = record_wait
        jieba.add_word = record_add_word_for_retry
        barrier = threading.Barrier(case["workers"])

        def retry_after_barrier():
            barrier.wait()
            return resources.install_jieba_phrases()

        with ThreadPoolExecutor(max_workers=case["workers"]) as executor:
            futures = [
                executor.submit(retry_after_barrier) for _ in range(case["workers"])
            ]
            results = tuple(
                future.result(timeout=case["future_timeout_seconds"])
                for future in futures
            )
        owner_was_active_after_fault = post_fault_state.owner is not None
    else:
        owner_started = threading.Event()
        release_owner = threading.Event()
        first_add_started = False
        owner_thread = None

        def blocking_add_word(word):
            global first_add_started, owner_thread
            with calls_lock:
                calls.append(word)
                is_first = not first_add_started
                if is_first:
                    first_add_started = True
                    owner_thread = threading.get_ident()
            if is_first:
                owner_started.set()
                if not release_owner.wait(case["future_timeout_seconds"]):
                    raise AssertionError("owner was not released")
            time.sleep(case["delay_seconds"])
            sample_invariant()

        def faulting_waiter():
            sys.settrace(inject_trace)
            try:
                resources.install_jieba_phrases()
            except InjectedTraceFailure:
                return True
            finally:
                sys.settrace(None)
            return False

        condition.wait = record_wait
        jieba.add_word = blocking_add_word
        with ThreadPoolExecutor(max_workers=case["workers"]) as executor:
            owner_future = executor.submit(resources.install_jieba_phrases)
            if not owner_started.wait(case["future_timeout_seconds"]):
                raise AssertionError("owner did not begin installation")
            owner_state = sample_invariant()
            waiter_future = executor.submit(faulting_waiter)
            if not waiters_ready.wait(case["waiter_ready_timeout_seconds"]):
                raise AssertionError("non-owner did not enter Condition.wait")
            with condition:
                condition.notify_all()
            fault_caught = waiter_future.result(
                timeout=case["future_timeout_seconds"]
            )
            post_fault_state = sample_invariant()
            active_owner_preserved = (
                post_fault_state == owner_state
                and post_fault_state.owner == owner_thread
                and not owner_future.done()
            )
            owner_was_active_after_fault = (
                post_fault_state.status == resources._JIEBA_INSTALL_INSTALLING
            )
            release_owner.set()
            try:
                owner_result = owner_future.result(
                    timeout=case["future_timeout_seconds"]
                )
            finally:
                release_owner.set()
        results = (owner_result,)

    calls_after_install = tuple(calls)
    later_results = tuple(
        resources.install_jieba_phrases() for _ in range(case["subsequent_calls"])
    )
    final_state = sample_invariant()
    if trace_state is not None:
        fault_thread_was_owner = trace_state.owner == trace_thread
    print(json.dumps({
        "phase": case["phase"],
        "fault_caught": fault_caught,
        "trace_hit_count": trace_hits,
        "trace_state_status": None if trace_state is None else trace_state.status,
        "trace_state_owner_present": (
            False if trace_state is None else trace_state.owner is not None
        ),
        "fault_thread_was_owner": fault_thread_was_owner,
        "post_fault_state_status": post_fault_state.status,
        "owner_was_active_after_fault": owner_was_active_after_fault,
        "active_owner_preserved": active_owner_preserved,
        "all_invariant_samples_valid": all(invariant_samples),
        "actual_waiter_count": len(waiter_threads),
        "installed_in_resource_order": calls_after_install == expected_words,
        "add_call_counts": [
            calls_after_install.count(word) for word in expected_words
        ],
        "all_retry_results_match": all(result == expected for result in results),
        "all_later_results_match": all(
            result == expected for result in later_results
        ),
        "later_add_call_count": len(calls) - len(calls_after_install),
        "final_state_status": final_state.status,
        "final_owner_is_none": final_state.owner is None,
        "final_installed_matches": final_state.installed == expected,
    }))
finally:
    condition.wait = original_wait
    jieba.add_word = original_add_word
"""
    for _ in range(case["trace_repetitions"]):
        completed = subprocess.run(
            [sys.executable, "-c", code, json.dumps(case)],
            check=True,
            capture_output=True,
            text=True,
            env=os.environ.copy(),
            timeout=case["subprocess_timeout_seconds"],
        )
        payload = json.loads(completed.stdout.splitlines()[-1])

        assert payload["phase"] == case["phase"]
        assert payload["fault_caught"] is True
        assert payload["trace_hit_count"] == case["fault_occurrence"]
        assert payload["trace_state_status"] == case["expected_trace_status"]
        assert payload["trace_state_owner_present"] is case["expected_trace_owner"]
        assert payload["all_invariant_samples_valid"] is True
        assert payload["active_owner_preserved"] is True
        assert payload["actual_waiter_count"] == case["workers"] - 1
        assert payload["installed_in_resource_order"] is True
        assert payload["add_call_counts"] == [1] * len(payload["add_call_counts"])
        assert payload["all_retry_results_match"] is True
        assert payload["all_later_results_match"] is True
        assert payload["later_add_call_count"] == 0
        assert payload["final_state_status"] == "installed"
        assert payload["final_owner_is_none"] is True
        assert payload["final_installed_matches"] is True

        if case["fault_role"] == "claimant":
            assert payload["post_fault_state_status"] == "pending"
            assert payload["owner_was_active_after_fault"] is False
            assert payload["fault_thread_was_owner"] is case["expected_trace_owner"]
        else:
            assert payload["post_fault_state_status"] == "installing"
            assert payload["owner_was_active_after_fault"] is True
            assert payload["fault_thread_was_owner"] is False


@pytest.mark.parametrize("case", cases("jieba_phrase_installation"))
def test_jieba_phrase_installation_same_thread_reentry_fails_fast(case):
    code = """
import json
import sys

import jieba

from g2p_mix import resources

case = json.loads(sys.argv[1])
expected_words = tuple(word for word, _ in resources.load_pinyin_phrases())
expected = frozenset(expected_words)
reentrant_word = expected_words[case["reentrant_index"]]
calls = []
reentry_errors = []
reentry_attempted = False

def record_add_word(word):
    global reentry_attempted
    calls.append(word)
    if word == reentrant_word and not reentry_attempted:
        reentry_attempted = True
        try:
            resources.install_jieba_phrases()
        except RuntimeError as error:
            reentry_errors.append(str(error))

original_add_word = jieba.add_word
jieba.add_word = record_add_word
try:
    result = resources.install_jieba_phrases()
    calls_after_install = tuple(calls)
    later_results = tuple(
        resources.install_jieba_phrases() for _ in range(case["subsequent_calls"])
    )
    print(json.dumps({
        "expected_word_count": len(expected_words),
        "result_matches": result == expected,
        "installed_in_resource_order": calls_after_install == expected_words,
        "call_counts": [calls_after_install.count(word) for word in expected_words],
        "reentry_error_count": len(reentry_errors),
        "reentry_error_is_clear": (
            len(reentry_errors) == 1
            and "re-entered" in reentry_errors[0]
            and "installing thread" in reentry_errors[0]
        ),
        "all_later_results_match": all(result == expected for result in later_results),
        "later_add_call_count": len(calls) - len(calls_after_install),
    }))
finally:
    jieba.add_word = original_add_word
"""
    completed = subprocess.run(
        [sys.executable, "-c", code, json.dumps(case)],
        check=True,
        capture_output=True,
        text=True,
        env=os.environ.copy(),
        timeout=case["subprocess_timeout_seconds"],
    )
    payload = json.loads(completed.stdout.splitlines()[-1])

    assert payload == {
        "expected_word_count": payload["expected_word_count"],
        "result_matches": True,
        "installed_in_resource_order": True,
        "call_counts": [1] * payload["expected_word_count"],
        "reentry_error_count": 1,
        "reentry_error_is_clear": True,
        "all_later_results_match": True,
        "later_add_call_count": 0,
    }
