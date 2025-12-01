try:  # Prefer stdlib importlib.metadata
	from importlib.metadata import version, PackageNotFoundError
except ImportError:  # Python <3.8 fallback (if backport installed)
	try:
		from importlib_metadata import version, PackageNotFoundError  # type: ignore
	except ImportError:
		version = None  # type: ignore
		PackageNotFoundError = Exception  # type: ignore

try:
	__version__ = version("ldpc") if version is not None else "0+local"
except PackageNotFoundError:  # package not installed (e.g. running from source tree)
	__version__ = "0+local"

from ldpc.bp_decoder import BpDecoder
from ldpc.bp_decoder import SoftInfoBpDecoder

try:
	from ldpc.bposd_decoder import BpOsdDecoder
except ImportError:  # optional module not built
	BpOsdDecoder = None

try:
	from ldpc.belief_find_decoder import BeliefFindDecoder
except ImportError:
	BeliefFindDecoder = None

try:
	from ldpc.sinter_decoders import SinterBpOsdDecoder
except ImportError:
	SinterBpOsdDecoder = None

# Legacy syntax
from ldpc.bp_decoder import bp_decoder
try:
	from ldpc.bposd_decoder import bposd_decoder
except ImportError:
	bposd_decoder = None
