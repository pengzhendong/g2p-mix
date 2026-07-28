from __future__ import annotations

import math
import unicodedata
from dataclasses import dataclass
from enum import Enum
from functools import lru_cache
from typing import Protocol, Sequence

from .errors import SimilarityError
from .models import G2PResult, Pronunciation, PronunciationUnit
from .transcription import IpaTranscriber


@lru_cache(maxsize=1)
def _load_panphon_distance():
    try:
        import panphon.distance

        return panphon.distance.Distance()
    except (ImportError, OSError) as error:
        raise SimilarityError(
            'PanPhon similarity is unavailable; install it with pip install "g2p-mix[similarity]"'
        ) from error


class EditOperation(str, Enum):
    MATCH = "match"
    SUBSTITUTE = "substitute"
    INSERT = "insert"
    DELETE = "delete"


@dataclass(frozen=True)
class AlignmentStep:
    operation: EditOperation
    left: str | None
    right: str | None
    cost: float


@dataclass(frozen=True)
class DistanceResult:
    distance: float
    maximum_distance: float
    left: tuple[str, ...]
    right: tuple[str, ...]
    alignment: tuple[AlignmentStep, ...]


@dataclass(frozen=True)
class SimilarityResult:
    score: float
    distance: float
    maximum_distance: float
    left: tuple[str, ...]
    right: tuple[str, ...]
    alignment: tuple[AlignmentStep, ...]
    backend: str


class PhoneticDistanceBackend(Protocol):
    name: str

    def compare(self, left: Sequence[str], right: Sequence[str]) -> DistanceResult:
        pass


class PanPhonDistanceBackend:
    name = "panphon"
    _PANPHON_ALIASES = {
        "k̚": "k",
        "p̚": "p",
        "t̚": "t",
        "ŋ̍": "ŋ̩",
        "ɚ": "ə˞",
        "ɝ": "ɜ˞",
    }

    def __init__(self, distance=None) -> None:
        self._distance = distance
        self._vector_cache: dict[str, list[int]] = {}

    @property
    def distance(self):
        if self._distance is None:
            self._distance = _load_panphon_distance()
        return self._distance

    def compare(self, left: Sequence[str], right: Sequence[str]) -> DistanceResult:
        left_segments = self._normalize(left)
        right_segments = self._normalize(right)
        if not left_segments or not right_segments:
            raise SimilarityError("Phonetic similarity requires phones on both sides")

        left_vectors = tuple(self._vector(phone) for phone in left_segments)
        right_vectors = tuple(self._vector(phone) for phone in right_segments)
        costs, operations = self._dynamic_programming(left_segments, right_segments, left_vectors, right_vectors)
        maximum = max(
            sum(self.distance.weighted_deletion_cost(vector) for vector in left_vectors),
            sum(self.distance.weighted_insertion_cost(vector) for vector in right_vectors),
        )
        value = costs[-1][-1]
        if not math.isfinite(value) or value < 0 or not math.isfinite(maximum) or maximum <= 0:
            raise SimilarityError("PanPhon returned an invalid phonetic distance")

        return DistanceResult(
            distance=value,
            maximum_distance=maximum,
            left=left_segments,
            right=right_segments,
            alignment=self._backtrace(
                left_segments,
                right_segments,
                costs,
                operations,
            ),
        )

    def _normalize(self, phones: Sequence[str]) -> tuple[str, ...]:
        result = []
        for phone in phones:
            if not isinstance(phone, str) or not phone:
                raise SimilarityError(f"Invalid IPA phone: {phone!r}")
            panphon_phone = self._PANPHON_ALIASES.get(phone, phone)
            segments = tuple(self.distance.fm.ipa_segs(panphon_phone))
            if len(segments) != 1 or unicodedata.normalize("NFD", segments[0]) != unicodedata.normalize(
                "NFD", panphon_phone
            ):
                raise SimilarityError(f"PanPhon does not recognize IPA phone {phone!r}")
            result.append(phone)
        return tuple(result)

    def _vector(self, phone: str) -> list[int]:
        cached = self._vector_cache.get(phone)
        if cached is not None:
            return cached
        panphon_phone = self._PANPHON_ALIASES.get(phone, phone)
        vectors = self.distance.fm.word_to_vector_list(panphon_phone, numeric=True)
        if len(vectors) != 1:
            raise SimilarityError(f"PanPhon did not parse {phone!r} as one IPA segment")
        vector = list(vectors[0])
        self._vector_cache[phone] = vector
        return vector

    def _dynamic_programming(self, left, right, left_vectors, right_vectors):
        rows = len(left) + 1
        columns = len(right) + 1
        costs = [[0.0] * columns for _ in range(rows)]
        operations: list[list[EditOperation | None]] = [[None] * columns for _ in range(rows)]

        for row in range(1, rows):
            costs[row][0] = costs[row - 1][0] + self.distance.weighted_deletion_cost(left_vectors[row - 1])
            operations[row][0] = EditOperation.DELETE
        for column in range(1, columns):
            costs[0][column] = costs[0][column - 1] + self.distance.weighted_insertion_cost(right_vectors[column - 1])
            operations[0][column] = EditOperation.INSERT

        for row in range(1, rows):
            for column in range(1, columns):
                if left[row - 1] == right[column - 1]:
                    diagonal_operation = EditOperation.MATCH
                    substitution = 0.0
                else:
                    diagonal_operation = EditOperation.SUBSTITUTE
                    substitution = self.distance.weighted_substitution_cost(
                        left_vectors[row - 1],
                        right_vectors[column - 1],
                    )

                candidates = (
                    (costs[row - 1][column - 1] + substitution, diagonal_operation),
                    (
                        costs[row - 1][column] + self.distance.weighted_deletion_cost(left_vectors[row - 1]),
                        EditOperation.DELETE,
                    ),
                    (
                        costs[row][column - 1] + self.distance.weighted_insertion_cost(right_vectors[column - 1]),
                        EditOperation.INSERT,
                    ),
                )
                costs[row][column], operations[row][column] = min(candidates, key=lambda candidate: candidate[0])

        return costs, operations

    @staticmethod
    def _backtrace(left, right, costs, operations) -> tuple[AlignmentStep, ...]:
        result = []
        row, column = len(left), len(right)
        while row or column:
            operation = operations[row][column]
            if operation in {EditOperation.MATCH, EditOperation.SUBSTITUTE}:
                previous_row, previous_column = row - 1, column - 1
                left_phone, right_phone = left[row - 1], right[column - 1]
            elif operation is EditOperation.DELETE:
                previous_row, previous_column = row - 1, column
                left_phone, right_phone = left[row - 1], None
            elif operation is EditOperation.INSERT:
                previous_row, previous_column = row, column - 1
                left_phone, right_phone = None, right[column - 1]
            else:
                raise SimilarityError("Could not reconstruct phonetic alignment")

            result.append(
                AlignmentStep(
                    operation=operation,
                    left=left_phone,
                    right=right_phone,
                    cost=costs[row][column] - costs[previous_row][previous_column],
                )
            )
            row, column = previous_row, previous_column

        result.reverse()
        return tuple(result)


PronunciationInput = G2PResult | Pronunciation | PronunciationUnit | Sequence[PronunciationUnit]


class PhoneticMatcher:
    def __init__(self, backend: PhoneticDistanceBackend | None = None, transcriber=None) -> None:
        self.backend = backend or PanPhonDistanceBackend()
        self._transcriber = transcriber or IpaTranscriber()

    def compare(self, left: PronunciationInput, right: PronunciationInput) -> SimilarityResult:
        measured = self.backend.compare(self._phones(left), self._phones(right))
        normalized_distance = measured.distance / measured.maximum_distance
        return SimilarityResult(
            score=max(0.0, min(1.0, 1.0 - normalized_distance)),
            distance=measured.distance,
            maximum_distance=measured.maximum_distance,
            left=measured.left,
            right=measured.right,
            alignment=measured.alignment,
            backend=self.backend.name,
        )

    def _phones(self, value: PronunciationInput) -> tuple[str, ...]:
        units = self._units(value)
        return tuple(phone for unit in units for phone in self._transcriber.transcribe_unit(unit).phones)

    @staticmethod
    def _units(value: PronunciationInput) -> tuple[PronunciationUnit, ...]:
        if isinstance(value, G2PResult):
            return value.units
        if isinstance(value, Pronunciation):
            return value.units
        if isinstance(value, PronunciationUnit):
            return (value,)
        if isinstance(value, Sequence) and not isinstance(value, (str, bytes)):
            units = tuple(value)
            if all(isinstance(unit, PronunciationUnit) for unit in units):
                return units
        raise TypeError("pronunciation input must contain PronunciationUnit values")
