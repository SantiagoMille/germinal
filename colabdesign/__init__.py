import os as _os
from pkgutil import extend_path as _extend_path

# Aggregate inner implementation directory for imports
__path__ = _extend_path(__path__, __name__)  # type: ignore[name-defined]
_inner = _os.path.join(_os.path.dirname(__file__), 'colabdesign')
if _inner not in __path__:
    __path__.append(_inner)

# Export stable API
from colabdesign.shared.utils import clear_mem  # noqa: F401
from colabdesign.af.model import mk_af_model  # noqa: F401
from colabdesign.mpnn.model import mk_mpnn_model  # noqa: F401
from colabdesign.tr.model import mk_tr_model  # noqa: F401

# Backwards compatibility
mk_design_model = mk_afdesign_model = mk_af_model
mk_trdesign_model = mk_tr_model


