import importlib.abc
import sys

import pytest

from g2p_mix import PanPhonDistanceBackend, SimilarityError
from g2p_mix.similarity import _load_panphon_distance


def test_missing_panphon_has_an_optional_install_hint():
    class BlockPanPhon(importlib.abc.MetaPathFinder):
        def find_spec(self, fullname, path, target=None):
            if fullname == "panphon" or fullname.startswith("panphon."):
                raise ModuleNotFoundError("blocked panphon", name=fullname)
            return None

    original_modules = {
        name: module for name, module in tuple(sys.modules.items()) if name == "panphon" or name.startswith("panphon.")
    }
    _load_panphon_distance.cache_clear()
    for name in original_modules:
        sys.modules.pop(name, None)

    blocker = BlockPanPhon()
    sys.meta_path.insert(0, blocker)
    try:
        with pytest.raises(SimilarityError, match=r"g2p-mix\[similarity\]"):
            PanPhonDistanceBackend().compare(("i",), ("i",))
    finally:
        sys.meta_path.remove(blocker)
        sys.modules.update(original_modules)
        _load_panphon_distance.cache_clear()
