from typing import Mapping, Protocol, Sequence

from ..models import Pronunciation, TextToken


class PronunciationProcessor(Protocol):
    """Post-process pronunciation attributes without changing alignment.

    Implementations must preserve token IDs, phone alphabets, unit text and source-span
    coverage. They may only replace pronunciation attributes such as phones, native
    forms, tone, stress, or confidence while keeping those alignment invariants intact.
    """

    def process(
        self,
        tokens: Sequence[TextToken],
        pronunciations: Mapping[int, Pronunciation],
    ) -> Mapping[int, Pronunciation]:
        pass
