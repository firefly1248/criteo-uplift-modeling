"""Drop-in replacement for ``sklift.datasets.fetch_criteo``.

scikit-uplift downloads Criteo from an S3 bucket that Criteo has since disabled
(the request now returns HTTP 403 / ``AllAccessDisabled``), so the stock loader
no longer works. This module reads the same dataset from the official Hugging
Face mirror (``criteo/criteo-uplift``) and returns the same ``Bunch`` the stock
loader did, so the notebooks keep using the familiar API.

The mirrored file is byte-identical to the one scikit-uplift used to ship
(same MD5), so results are unchanged. It is cached under ``~/scikit-uplift-data``
and downloads only once (~297 MB).

Usage mirrors the original::

    from criteo_data import fetch_criteo
    ds = fetch_criteo(target_col='conversion')
    X, y, t = ds.data, ds.target, ds.treatment
"""
from pathlib import Path
import urllib.request

import pandas as pd
from sklearn.utils import Bunch

_URL = (
    "https://huggingface.co/datasets/criteo/criteo-uplift/"
    "resolve/main/criteo-research-uplift-v2.1.csv.gz"
)
_CACHE = Path.home() / "scikit-uplift-data" / "criteo-research-uplift-v2.1.csv.gz"

_TREATMENT_COLS = ["exposure", "treatment"]
_TARGET_COLS = ["visit", "conversion"]


def _cached_path():
    _CACHE.parent.mkdir(parents=True, exist_ok=True)
    if not _CACHE.exists():
        print(f"Downloading Criteo dataset (~297 MB) to {_CACHE} ...")
        urllib.request.urlretrieve(_URL, _CACHE)
    return _CACHE


def fetch_criteo(target_col="conversion", treatment_col="treatment", return_X_y_t=False):
    """Load the Criteo Uplift dataset.

    Args:
        target_col: 'visit', 'conversion', or 'all' (both label columns).
        treatment_col: 'treatment', 'exposure', or 'all' (both columns).
        return_X_y_t: if True, return (data, target, treatment) instead of a Bunch.
    """
    if treatment_col == "all":
        treatment_col = _TREATMENT_COLS
    if target_col == "all":
        target_col = _TARGET_COLS

    dtypes = {c: "Int8" for c in _TREATMENT_COLS + _TARGET_COLS}
    df = pd.read_csv(_cached_path(), dtype=dtypes)

    treatment, target = df[treatment_col], df[target_col]
    data = df.drop(_TARGET_COLS + _TREATMENT_COLS, axis=1)

    if return_X_y_t:
        return data, target, treatment
    return Bunch(
        data=data,
        target=target,
        treatment=treatment,
        feature_names=list(data.columns),
        target_name=target_col,
        treatment_name=treatment_col,
    )
