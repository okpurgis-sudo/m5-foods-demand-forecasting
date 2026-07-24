"""Notebookに記録した分析・学習処理を一括実行するランナー。"""

from __future__ import annotations

from pathlib import Path
import subprocess
import sys

PROJECT_ROOT = Path(__file__).resolve().parent.parent
NOTEBOOK = PROJECT_ROOT / "notebooks" / "m5_foods_demand_forecasting.ipynb"


def main() -> None:
    if not NOTEBOOK.is_file():
        raise FileNotFoundError(f"Notebook was not found: {NOTEBOOK}")

    command = [
        sys.executable,
        "-m",
        "jupyter",
        "nbconvert",
        "--to",
        "notebook",
        "--execute",
        "--inplace",
        "--ExecutePreprocessor.timeout=-1",
        str(NOTEBOOK),
    ]
    print(f"Executing notebook: {NOTEBOOK}")
    subprocess.run(command, cwd=PROJECT_ROOT, check=True)
    print("Notebook execution completed successfully.")


if __name__ == "__main__":
    main()
