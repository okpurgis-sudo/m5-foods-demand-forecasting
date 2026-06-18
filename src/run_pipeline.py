import argparse
import os
import subprocess
import sys
from pathlib import Path


def run_command(command: list[str], cwd: Path) -> None:
    print("Running command:")
    print(" ".join(command))
    print("Working directory:")
    print(cwd)

    process = subprocess.Popen(
        command,
        cwd=str(cwd),  # ここで実行場所をプロジェクト直下に固定
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        text=True,
        bufsize=1
    )

    assert process.stdout is not None

    for line in process.stdout:
        print(line, end="")

    return_code = process.wait()

    if return_code != 0:
        raise RuntimeError(f"Command failed with return code {return_code}")


def main():
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--notebook",
        default="notebooks/m5_foods_demand_forecasting.ipynb",
        help="Path to the notebook to execute."
    )

    parser.add_argument(
        "--output-notebook",
        default="notebooks/executed_m5_foods_demand_forecasting.ipynb",
        help="Path to save executed notebook."
    )

    parser.add_argument(
        "--timeout",
        type=int,
        default=7200,
        help="Notebook execution timeout in seconds."
    )

    args = parser.parse_args()

    # このファイル src/run_pipeline.py から見て1つ上がプロジェクト直下
    project_root = Path(__file__).resolve().parents[1]

    # 念のため、Pythonプロセス自体のカレントディレクトリも変更
    os.chdir(project_root)

    notebook_path = project_root / args.notebook
    output_path = project_root / args.output_notebook

    if not notebook_path.exists():
        raise FileNotFoundError(f"Notebook not found: {notebook_path}")

    (project_root / "results").mkdir(exist_ok=True)
    (project_root / "models").mkdir(exist_ok=True)
    (project_root / "images").mkdir(exist_ok=True)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    command = [
        sys.executable,
        "-m",
        "jupyter",
        "nbconvert",
        "--to",
        "notebook",
        "--execute",
        str(notebook_path),
        "--output",
        output_path.name,
        "--output-dir",
        str(output_path.parent),
        f"--ExecutePreprocessor.timeout={args.timeout}"
    ]

    run_command(command, cwd=project_root)

    print("Pipeline completed successfully.")
    print(f"Executed notebook saved to: {output_path}")


if __name__ == "__main__":
    main()