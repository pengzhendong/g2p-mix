from importlib.metadata import PackageNotFoundError, version

from .api import G2P
from .errors import G2PError
from .models import G2PResult

try:
    __version__ = version("g2p-mix")
except (PackageNotFoundError, TypeError):
    __version__ = "0+unknown"

__all__ = [
    "__version__",
    "G2P",
    "G2PResult",
    "G2PError",
]
