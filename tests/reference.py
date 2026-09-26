"""Resolve the real `aicsimageio` for parity testing, or fall back to NumPy.

`aicsimageio` is not installed in the shared test venv, and its top-level
`__init__` pulls in a long tail of optional reader dependencies that the port
does not need. This module therefore resolves the *transforms* module only:

1. an already importable `aicsimageio`;
2. an extraction cached under `tests/.refcache/`;
3. a fresh wheel download from PyPI, extracted into the same cache.

The top-level `__init__.py` of the extracted copy is emptied, because
`aicsimageio.transforms` needs only `dask`, `numpy`, `xarray` and the
package's own `types` / `exceptions` / `dimensions` / `image_container` modules.

If every route fails -- offline machine, no wheel -- `REFERENCE` is None and
the tests compare against the analytic NumPy reference in
`test_transforms.py` instead. `reference_kind()` says which is in use.
"""

import io
import json
import os
import pathlib
import sys
import urllib.request
import zipfile

_ROOT = pathlib.Path(__file__).resolve().parents[1]
_CACHE = pathlib.Path(__file__).resolve().parent / ".refcache"


def _import_transforms():
    try:
        from aicsimageio import transforms  # noqa: PLC0415
    except Exception:
        return None
    return transforms


def _download_and_extract():
    _CACHE.mkdir(parents=True, exist_ok=True)
    with urllib.request.urlopen(
        "https://pypi.org/pypi/aicsimageio/json", timeout=30
    ) as resp:
        meta = json.load(resp)
    url = next(
        u["url"]
        for u in meta["urls"]
        if u["packagetype"] == "bdist_wheel"
    )
    with urllib.request.urlopen(url, timeout=120) as resp:
        payload = resp.read()
    target = _CACHE / meta["info"]["version"]
    target.mkdir(parents=True, exist_ok=True)
    with zipfile.ZipFile(io.BytesIO(payload)) as zf:
        zf.extractall(target)
    # Only the pure-python dimension engine is needed; the package's own
    # top-level __init__ imports every optional reader in the ecosystem.
    (target / "aicsimageio" / "__init__.py").write_text("")
    return target


def _load_from_cache():
    if not _CACHE.is_dir():
        return None
    for candidate in sorted(_CACHE.iterdir(), reverse=True):
        if (candidate / "aicsimageio" / "transforms.py").exists():
            return candidate
    return None


def _resolve():
    override = os.environ.get("MOJO_AICSIMAGEIO_REF")
    if override:
        sys.path.insert(0, override)
        found = _import_transforms()
        if found is not None:
            return found, "installed"
        return None, None
    found = _import_transforms()
    if found is not None:
        return found, "installed"
    base = _load_from_cache()
    if base is None and not _CACHE.exists():
        try:
            base = _download_and_extract()
        except Exception:
            base = None
    if base is not None:
        sys.path.insert(0, str(base))
        found = _import_transforms()
        if found is not None:
            return found, "wheel"
        for mod in [m for m in sys.modules if m.startswith("aicsimageio")]:
            del sys.modules[mod]
    return None, None


REFERENCE, REFERENCE_SOURCE = _resolve()


def reference_kind():
    """'upstream' when parity is measured against the real package."""
    return REFERENCE_SOURCE or "analytic-numpy"


def upstream_reshape(data, given_dims, return_dims, **kwargs):
    if REFERENCE is None:
        raise RuntimeError("upstream aicsimageio is unavailable")
    return REFERENCE.reshape_data(data, given_dims, return_dims, **kwargs)


def upstream_reduce_to_slice(values):
    if REFERENCE is None:
        raise RuntimeError("upstream aicsimageio is unavailable")
    return REFERENCE.reduce_to_slice(values)
