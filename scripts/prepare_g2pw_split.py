#!/usr/bin/env python
"""Split the monolithic g2pw.onnx into a shared encoder and a light head.

The upstream G2PW inference path runs the full BERT encoder once per
polyphonic position on a +/-16 character window. The split boundary is the
encoder's final hidden states (tensor ``1663``, ``[B, L, 768]``, right
before the classifier flattens the sequence), so:
  * g2pw_encoder.onnx — BERT pass, run once per sentence
  * g2pw_head.onnx    — classifier head, scores every polyphonic position
                        of the sentence from the shared encoding

The script verifies round-trip parity against the monolithic model on a
handful of built-in sentences, and can additionally emit dynamic INT8
quantizations of the monolithic graph and the encoder.

Requires the optional model stack plus onnx:

    pip install "g2p-mix[g2pw]" onnx

Usage:

    python scripts/prepare_g2pw_split.py [--quantize]
"""

from __future__ import annotations

import argparse
import copy
import sys

PARITY_SENTENCES = [
    "重庆银行的重要性",
    "他在银行工作了很多年，头发都白了。",
    "音乐会使人快乐，我乐意听轻音乐。",
    "这个地道很长，走进去黑漆漆的。",
    "王昭君出塞和亲，是整个汉朝的一件大事。",
    "他还了欠款，还没走出法院就被记者围住了。",
    "辅导员重视学生的心理健康，常找他们谈心。",
    "这条裙子很漂亮，着实让人羡慕。",
]

_BOUNDARY = "1663"  # bert-hidden [B, L, 768] right before Flatten
_FEED_NAMES = ("input_ids", "token_type_ids", "attention_mask", "phoneme_mask", "char_ids", "position_ids")


def split_model(src_dir: str) -> None:
    import onnx

    onnx_path = f"{src_dir}/g2pw.onnx"
    model = onnx.shape_inference.infer_shapes(onnx.load(onnx_path))
    graph = model.graph

    value_info = next((v for v in graph.value_info if v.name == _BOUNDARY), None)
    if value_info is None:
        raise SystemExit(f"boundary tensor {_BOUNDARY!r} has no inferred shape")
    fixed = onnx.helper.make_tensor_value_info(_BOUNDARY, onnx.TensorProto.FLOAT, ("B", "L", 768))
    value_info.CopyFrom(fixed)

    # The head must not depend on BERT-side mask/segment inputs; it rebuilds
    # flat gather indices from input_ids' shape and the query signals alone.
    after_boundary = False
    for node in graph.node:
        if _BOUNDARY in node.output:
            after_boundary = True
            continue
        if after_boundary and any(name in {"token_type_ids", "attention_mask"} for name in node.input):
            raise SystemExit(f"head node {node.op_type} depends on BERT-side inputs: {node.input}")

    original_inputs = {v.name: v for v in graph.input}
    original_outputs = {v.name: v for v in graph.output}
    initializer_names = {init.name for init in graph.initializer}

    def boundary_value_info():
        return onnx.helper.make_tensor_value_info(_BOUNDARY, onnx.TensorProto.FLOAT, ("B", "L", 768))

    def build_subgraph(name, input_names, output_names):
        nodes = list(graph.node)
        producer = {}
        for index, node in enumerate(nodes):
            for output in node.output:
                producer[output] = index
        keep = set()
        stack = list(output_names)
        seen = set()
        while stack:
            tensor = stack.pop()
            if tensor in seen:
                continue
            seen.add(tensor)
            if tensor in input_names:
                continue  # declared input: do not chase upstream
            index = producer.get(tensor)
            if index is None:
                if tensor not in initializer_names:
                    raise SystemExit(f"unresolved tensor {tensor!r} while building {name}")
                continue
            keep.add(index)
            stack.extend(nodes[index].input)
        sub_model = copy.deepcopy(model)
        subgraph = sub_model.graph
        del subgraph.node[:]
        subgraph.node.extend(nodes[i] for i in sorted(keep))
        del subgraph.input[:]
        for input_name in input_names:
            if input_name == _BOUNDARY:
                subgraph.input.extend([boundary_value_info()])
            else:
                subgraph.input.extend([copy.deepcopy(original_inputs[input_name])])
        del subgraph.output[:]
        for output_name in output_names:
            if output_name == _BOUNDARY:
                subgraph.output.extend([boundary_value_info()])
            else:
                subgraph.output.extend([copy.deepcopy(original_outputs[output_name])])
        del subgraph.value_info[:]
        used = {name for node in subgraph.node for name in node.input if name in initializer_names}
        kept = [init for init in subgraph.initializer if init.name in used]
        del subgraph.initializer[:]
        subgraph.initializer.extend(kept)
        subgraph.name = name
        subgraph.doc_string = ""
        onnx.checker.check_model(sub_model)
        print(f"{name}: {len(keep)} nodes, {len(kept)}/{len(initializer_names)} initializers kept")
        return sub_model

    encoder = build_subgraph("g2pw_encoder", ("input_ids", "token_type_ids", "attention_mask"), (_BOUNDARY,))
    onnx.save(encoder, f"{src_dir}/g2pw_encoder.onnx")

    head_inputs = (_BOUNDARY, "input_ids", "position_ids", "char_ids", "phoneme_mask")
    head = build_subgraph("g2pw_head", head_inputs, ("probs",))
    onnx.save(head, f"{src_dir}/g2pw_head.onnx")
    print("saved into", src_dir)


def check_parity(src_dir: str, model_source: str) -> None:
    import numpy as np
    import onnxruntime as ort

    options = ort.SessionOptions()
    options.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_ALL
    mono = ort.InferenceSession(f"{src_dir}/g2pw.onnx", sess_options=options, providers=["CPUExecutionProvider"])
    encoder = ort.InferenceSession(
        f"{src_dir}/g2pw_encoder.onnx", sess_options=options, providers=["CPUExecutionProvider"]
    )
    head = ort.InferenceSession(f"{src_dir}/g2pw_head.onnx", sess_options=options, providers=["CPUExecutionProvider"])

    from g2pw import G2PWConverter
    from g2pw.dataset import TextDataset

    converter = G2PWConverter(
        model_dir=src_dir,
        style="pinyin",
        model_source=model_source,
        num_workers=0,
        enable_non_tradional_chinese=True,
    )
    texts, query_ids, _sent_ids, _partial = converter._prepare_data(PARITY_SENTENCES)
    dataset = TextDataset(
        converter.tokenizer,
        converter.labels,
        converter.char2phonemes,
        converter.chars,
        texts,
        query_ids,
        use_mask=converter.config.use_mask,
        use_char_phoneme=converter.config.use_char_phoneme,
        window_size=converter.config.window_size,
        for_train=False,
    )
    data = dataset.create_mini_batch([dataset[i] for i in range(len(dataset))])
    feeds = {name: data[name].numpy() for name in _FEED_NAMES}

    probs_mono = mono.run([], feeds)[0]
    hidden = encoder.run([], {k: feeds[k] for k in ("input_ids", "token_type_ids", "attention_mask")})[0]
    probs_split = head.run(
        [],
        {
            _BOUNDARY: hidden,
            "input_ids": feeds["input_ids"],
            "position_ids": feeds["position_ids"],
            "char_ids": feeds["char_ids"],
            "phoneme_mask": feeds["phoneme_mask"],
        },
    )[0]

    mismatches = int((probs_mono.argmax(-1) != probs_split.argmax(-1)).sum())
    max_diff = float(np.abs(probs_mono - probs_split).max())
    print(f"parity: samples={len(probs_mono)} argmax mismatches={mismatches} max|diff|={max_diff:.3e}")
    if mismatches or max_diff > 1e-4:
        raise SystemExit("parity check FAILED: split graphs diverge from the monolithic model")


def quantize_models(src_dir: str) -> None:
    import onnx
    from onnxruntime.quantization import QuantType, quantize_dynamic

    for source, output in (
        ("g2pw.onnx", "g2pw_int8.onnx"),
        ("g2pw_encoder.onnx", "g2pw_encoder_int8.onnx"),
        ("g2pw_head.onnx", "g2pw_head_int8.onnx"),
    ):
        output_path = f"{src_dir}/{output}"
        quantize_dynamic(
            model_input=f"{src_dir}/{source}",
            model_output=output_path,
            weight_type=QuantType.QInt8,
        )
        onnx.checker.check_model(output_path)
        print("saved", output_path)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0] if __doc__ else None)
    parser.add_argument("--quantize", action="store_true", help="also emit a dynamic INT8 encoder")
    arguments = parser.parse_args()

    try:
        from modelscope import snapshot_download
    except ImportError as error:
        raise SystemExit("install the optional model stack first: pip install 'g2p-mix[g2pw]' onnx") from error

    # Only the monolithic graph plus converter metadata is needed; skip the
    # derived graphs and the unused 393 MB tokenizer checkpoint.
    repo = snapshot_download(
        "pengzhendong/g2pw",
        allow_patterns=[
            "G2PWModel/version",
            "G2PWModel/config.py",
            "G2PWModel/POLYPHONIC_CHARS.txt",
            "G2PWModel/MONOPHONIC_CHARS.txt",
            "G2PWModel/g2pw.onnx",
            "bert-base-chinese/vocab.txt",
            "bert-base-chinese/tokenizer.json",
        ],
    )
    src_dir = f"{repo}/G2PWModel"

    split_model(src_dir)
    check_parity(src_dir, f"{repo}/bert-base-chinese")
    if arguments.quantize:
        quantize_models(src_dir)
    print("done")
    sys.exit(0)


if __name__ == "__main__":
    main()
