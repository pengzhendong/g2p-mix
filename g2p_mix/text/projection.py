from __future__ import annotations

from typing import List, Sequence, Tuple

from ..models import (
    Language,
    LanguageProjection,
    ProjectionKind,
    ProjectionToken,
    TextToken,
)

PLACEHOLDERS = {
    Language.CHINESE: "<ZH>",
    Language.ENGLISH: "<EN>",
}


class ProjectionBuilder:
    def build(
        self,
        tokens: Sequence[TextToken],
        target: Language,
    ) -> LanguageProjection:
        if target not in {Language.CHINESE, Language.ENGLISH}:
            raise ValueError(f"Unsupported projection target: {target!r}")

        foreign = Language.ENGLISH if target is Language.CHINESE else Language.CHINESE
        projected: List[ProjectionToken] = []
        index = 0

        while index < len(tokens):
            token = tokens[index]
            if token.language is target:
                projected.append(
                    ProjectionToken(
                        text=token.text,
                        source_ids=(token.id,),
                        kind=ProjectionKind.TARGET,
                        boundary_before=token.boundary_before,
                    )
                )
                index += 1
                continue

            if token.language is foreign:
                source_ids, index = self._consume_language_island(
                    tokens,
                    index,
                    foreign,
                )
                projected.append(
                    ProjectionToken(
                        text=PLACEHOLDERS[foreign],
                        source_ids=source_ids,
                        kind=ProjectionKind.PLACEHOLDER,
                        boundary_before=token.boundary_before,
                    )
                )
                continue

            projected.append(
                ProjectionToken(
                    text=token.text,
                    source_ids=(token.id,),
                    kind=ProjectionKind.CONTEXT,
                    boundary_before=token.boundary_before,
                )
            )
            index += 1

        return LanguageProjection(target=target, tokens=tuple(projected))

    @staticmethod
    def _consume_language_island(
        tokens: Sequence[TextToken],
        start: int,
        language: Language,
    ) -> Tuple[Tuple[int, ...], int]:
        source_ids = [tokens[start].id]
        index = start + 1

        while index < len(tokens):
            token = tokens[index]
            if token.language is language:
                source_ids.append(token.id)
                index += 1
                continue
            if token.language is Language.SPACE and index + 1 < len(tokens) and tokens[index + 1].language is language:
                source_ids.append(token.id)
                source_ids.append(tokens[index + 1].id)
                index += 2
                continue
            break

        return tuple(source_ids), index
