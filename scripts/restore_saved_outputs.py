"""Preserve verified outputs after an editorial-only notebook rebuild."""

import argparse
import copy
import hashlib
import json
from pathlib import Path

import nbformat


parser = argparse.ArgumentParser()
parser.add_argument("--previous-dir", required=True)
parser.add_argument("--revision-date", required=True)
arguments = parser.parse_args()
root = Path(__file__).resolve().parents[1]
previous = Path(arguments.previous_dir)
figures = json.loads((root / "assets" / "architecture_links.json").read_text())

for name in ["01_vae_gan", "02_flows_diffusion"]:
    path = root / f"{name}.ipynb"
    old_path = previous / path.name
    old = nbformat.read(old_path, as_version=4)
    current = nbformat.read(path, as_version=4)
    old_code = [cell for cell in old.cells if cell.cell_type == "code"]
    current_code = [cell for cell in current.cells if cell.cell_type == "code"]
    old_sources = [cell.source for cell in old_code]
    current_sources = [cell.source for cell in current_code]
    if old_sources != current_sources:
        raise ValueError(f"Code changed in {name}; execute the notebook instead of restoring outputs")
    for target, source in zip(current_code, old_code):
        target.outputs = copy.deepcopy(source.outputs)
        target.execution_count = source.execution_count
        target.metadata = {**copy.deepcopy(source.metadata), **target.metadata}
    current.metadata["language_info"] = copy.deepcopy(old.metadata["language_info"])
    current.metadata["editorial_revision"] = {
        "date": arguments.revision_date,
        "code_unchanged": True,
        "saved_outputs_preserved": True,
    }
    nbformat.validate(current)
    nbformat.write(current, path)
    report_path = root / "validation" / f"{name}_execution.json"
    report = json.loads(report_path.read_text())
    old_words = sum(len(cell.source.split()) for cell in old.cells if cell.cell_type == "markdown")
    new_words = sum(len(cell.source.split()) for cell in current.cells if cell.cell_type == "markdown")
    report["editorial_revision"] = {
        "date": arguments.revision_date,
        "code_cells_identical": True,
        "new_training_run": False,
        "outputs_preserved_from_sha256": hashlib.sha256(old_path.read_bytes()).hexdigest(),
        "code_sources_sha256": hashlib.sha256(json.dumps(current_sources, ensure_ascii=False).encode()).hexdigest(),
        "markdown_words_before": old_words,
        "markdown_words_after": new_words,
        "remote_figure_count": sum(figure["notebook"] == name for figure in figures),
    }
    report["notebook_sha256"] = hashlib.sha256(path.read_bytes()).hexdigest()
    report_path.write_text(json.dumps(report, ensure_ascii=False, indent=2))
    print(f"{name}: {len(current_code)} unchanged code cells; Markdown {old_words} -> {new_words} words")
