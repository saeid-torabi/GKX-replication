"""Locations of data files that live outside this repository.

The 94 firm characteristics come from Dacheng Xiu's posted ``datashare.csv``,
which is far too large to version-control and therefore sits outside the
project. Hard-coding one machine's absolute path makes the notebooks break
whenever the project is moved or opened on another computer, so the path is
resolved at run time instead.

Resolution order:
  1. the DATASHARE_PATH environment variable, if set;
  2. a few conventional locations relative to this repository.

To pin it explicitly, either export the variable

    export DATASHARE_PATH=~/Documents/Codes/EquityCharacteristics-master/datashare/datashare.csv

or place the EquityCharacteristics folder beside this repository.
"""
import os
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent

_DATASHARE_CANDIDATES = (
    "EquityCharacteristics-master/datashare/datashare.csv",
    "../EquityCharacteristics-master/datashare/datashare.csv",
    "../../EquityCharacteristics-master/datashare/datashare.csv",
    "datashare/datashare.csv",
    "data_csv/datashare.csv",
)


def datashare_path():
    """Absolute path to datashare.csv, or a FileNotFoundError naming what to fix."""
    override = os.environ.get("DATASHARE_PATH")
    if override:
        path = Path(override).expanduser().resolve()
        if path.exists():
            return path
        raise FileNotFoundError(
            f"DATASHARE_PATH is set to {path}, which does not exist. "
            "Correct the variable or unset it to fall back to the default search."
        )

    searched = []
    for relative in _DATASHARE_CANDIDATES:
        candidate = (REPO_ROOT / relative).resolve()
        searched.append(candidate)
        if candidate.exists():
            return candidate

    locations = "\n  ".join(str(p) for p in searched)
    raise FileNotFoundError(
        "Could not find datashare.csv (the 94 firm characteristics).\n"
        f"Searched:\n  {locations}\n"
        "Set DATASHARE_PATH to its location, or move the EquityCharacteristics "
        "folder beside this repository."
    )
