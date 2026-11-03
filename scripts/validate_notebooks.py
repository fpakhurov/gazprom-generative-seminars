"""Execute the seminar notebooks and save portable output and validation evidence."""

import argparse
import base64
import hashlib
import importlib.metadata
import json
import os
from pathlib import Path
import platform
import sys
import time

import nbformat
from nbclient import NotebookClient
from PIL import Image, ImageDraw


def validate(path, destination):
    notebook = nbformat.read(path, as_version=4)
    nbformat.validate(notebook)
    started = time.perf_counter()
    client = NotebookClient(
        notebook,
        timeout=900,
        kernel_name="python3",
        resources={"metadata": {"path": str(path.parent)}},
        allow_errors=False,
    )
    client.execute()
    seconds = time.perf_counter() - started
    nbformat.write(notebook, path)
    screenshots = []
    text_outputs = []
    for cell_index, cell in enumerate(notebook.cells):
        for output_index, output in enumerate(cell.get("outputs", [])):
            if output.output_type == "error":
                raise RuntimeError(f"Unexpected error in cell {cell_index}")
            if "text" in output:
                text_outputs.append({"cell": cell_index, "text": output["text"]})
            png = output.get("data", {}).get("image/png")
            if png:
                target = destination / f"{path.stem}_cell_{cell_index:02d}_output_{output_index:02d}.png"
                target.write_bytes(base64.b64decode(png))
                screenshots.append(target)
    # A compact sheet makes missing plots or unreadable panels easy to spot.
    width, height = 600, 330
    rows = (len(screenshots) + 1) // 2
    sheet = Image.new("RGB", (width * 2, height * max(rows, 1)), "white")
    drawer = ImageDraw.Draw(sheet)
    for index, picture_path in enumerate(screenshots):
        picture = Image.open(picture_path).convert("RGB")
        picture.thumbnail((width - 20, height - 35))
        left = (index % 2) * width
        top = (index // 2) * height
        sheet.paste(picture, (left + (width - picture.width) // 2, top + 25))
        drawer.text((left + 10, top + 5), picture_path.stem, fill="black")
    sheet.save(destination / f"{path.stem}_contact_sheet.png")
    report = {
        "notebook": path.name,
        "execution": "full default local CPU run",
        "elapsed_seconds": round(seconds, 2),
        "python": sys.version.split()[0],
        "platform": platform.platform(),
        "packages": {
            name: importlib.metadata.version(name)
            for name in ["torch", "numpy", "matplotlib", "scikit-learn", "nbformat", "nbclient"]
        },
        "code_cells": sum(cell.cell_type == "code" for cell in notebook.cells),
        "executed_code_cells": sum(
            cell.cell_type == "code" and cell.get("execution_count") is not None
            for cell in notebook.cells
        ),
        "plot_count": len(screenshots),
        "error_count": 0,
        "text_outputs": text_outputs,
        "notebook_sha256": hashlib.sha256(path.read_bytes()).hexdigest(),
        "colab_hosted_runtime_tested": False,
    }
    (destination / f"{path.stem}_execution.json").write_text(
        json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    print(json.dumps({key: report[key] for key in ["notebook", "elapsed_seconds", "code_cells", "executed_code_cells", "plot_count", "error_count"]}))


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("notebooks", nargs="+")
    arguments = parser.parse_args()
    os.environ["MPLBACKEND"] = "module://matplotlib_inline.backend_inline"
    os.environ["JUPYTER_PATH"] = str(Path(sys.prefix) / "share" / "jupyter")
    root = Path(__file__).resolve().parents[1]
    destination = root / "validation"
    destination.mkdir(exist_ok=True)
    for name in arguments.notebooks:
        validate(root / name, destination)
