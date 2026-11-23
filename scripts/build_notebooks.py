"""Write notebooks 08c, 10 and 11 into notebooks/.

Usage (from the project root):
    python scripts/build_notebooks.py

The notebooks are generated from this file so that their code is exactly the code that was tested.
Existing files with the same names are overwritten (the empty placeholders 10_ablations.ipynb and
11_paper_figures_tables.ipynb are replaced).
"""
from __future__ import annotations

from pathlib import Path

import nbformat as nbf

ROOT = Path(__file__).resolve().parents[1]
md, code = nbf.v4.new_markdown_cell, nbf.v4.new_code_cell


def save(cells, name: str) -> None:
    nb = nbf.v4.new_notebook()
    nb["cells"] = cells
    nb["metadata"] = {"kernelspec": {"display_name": "Python 3", "language": "python", "name": "python3"}}
    path = ROOT / "notebooks" / name
    path.parent.mkdir(parents=True, exist_ok=True)
    nbf.write(nb, path)
    print("wrote", path)
