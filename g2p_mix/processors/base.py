from typing import Mapping, Protocol, Sequence

from ..models import Pronunciation, TextToken


class PronunciationProcessor(Protocol):
    def process(
        self,
        tokens: Sequence[TextToken],
        pronunciations: Mapping[int, Pronunciation],
    ) -> Mapping[int, Pronunciation]:
        pass
