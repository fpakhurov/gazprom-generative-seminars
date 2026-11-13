"""Check saved execution evidence and create a verified portable ZIP package."""

import ast
import argparse
import hashlib
import json
from pathlib import Path
import re
import runpy
import zipfile

from bs4 import BeautifulSoup
import nbformat


parser = argparse.ArgumentParser()
parser.add_argument("--source-only", action="store_true", help="Package a clearly marked partial source revision without claiming a full execution")
arguments = parser.parse_args()
root = Path(__file__).resolve().parents[1]
figures = json.loads((root / "assets" / "architecture_links.json").read_text())
figure_urls = {figure["url"] for figure in figures}
for script in (root / "scripts").glob("*.py"):
    ast.parse(script.read_text())
for name in ["01_vae_gan", "02_flows_diffusion"]:
    path = root / f"{name}.ipynb"
    notebook = nbformat.read(path, as_version=4)
    nbformat.validate(notebook)
    settings = notebook.metadata.get("gazprom_seminar", notebook.metadata.get("seminar", {}))
    assert settings["duration_minutes"] == 90
    assert settings["image_size"] == 28
    builder = ast.parse((root / "scripts" / f"build_{name[:2]}.py").read_text())
    expected = [(node.value.func.id, ast.literal_eval(node.value.args[0]).strip())
                for node in builder.body
                if isinstance(node, ast.Expr) and isinstance(node.value, ast.Call)
                and isinstance(node.value.func, ast.Name) and node.value.func.id in {"md", "code"}]
    actual = [("md" if cell.cell_type == "markdown" else "code", cell.source)
              for cell in notebook.cells
              if not {"photo_extension", "pet_extension"}.intersection(cell.metadata.get("tags", []))]
    assert actual == expected, f"Builder/notebook content mismatch: {name}"
    code_cells = [cell for cell in notebook.cells if cell.cell_type == "code"]
    for cell in code_cells:
        ast.parse(cell.source)
        if not arguments.source_only:
            assert cell.execution_count is not None
        assert all(output.output_type != "error" for output in cell.get("outputs", []))
    report = json.loads((root / "validation" / f"{name}_execution.json").read_text())
    assert report["notebook_sha256"] == hashlib.sha256(path.read_bytes()).hexdigest()
    if arguments.source_only:
        assert report.get("validation_scope") == "partial"
        assert report.get("full_notebook_executed") is False
    else:
        assert report.get("full_notebook_executed", True)
        assert report["executed_code_cells"] == len(code_cells)
    if report.get("colab_hosted_runtime_tested"):
        colab_report = json.loads((root / "validation" / report["colab_verification_report"]).read_text())
        assert colab_report["completed_code_cells"] == len(code_cells)
        assert colab_report["execution_errors_observed"] == 0
        assert colab_report["code_sha256"] == [hashlib.sha256(cell.source.strip().encode()).hexdigest() for cell in code_cells]
    pacing = json.loads((root / "validation" / "pacing_audit.json").read_text())
    plan = next(plan for plan in pacing["plans"] if plan["notebook"] == path.name)
    actual_ranges = [[int(a), int(b)] for a, b in re.findall(r"\| (\d+)–(\d+) \|", expected[0][1])]
    assert plan["ranges_minutes"] == actual_ranges
    assert plan["core_minutes"] == 75 and plan["reserve_minutes"] == 15
    soup = BeautifulSoup((root / "validation" / f"{name}.html").read_text(), "html.parser")
    assert not soup.find_all("script")
    assert not soup.select(".katex-error")
    assert soup.select(".katex")
    assert all(image.get("src", "").startswith("data:") or image.get("src") in figure_urls
               for image in soup.find_all("img"))
    expected_figures = {figure["url"] for figure in figures if figure["notebook"] == name}
    actual_figures = {image.get("src") for image in soup.find_all("img") if not image.get("src", "").startswith("data:")}
    assert actual_figures == expected_figures
    print(name, len(code_cells), "source code cells;" if arguments.source_only else "executed code cells;", len(soup.select(".katex")), "rendered formulas")

homework_path = root / "hw-4-celeba.ipynb"
homework = nbformat.read(homework_path, as_version=4)
nbformat.validate(homework)
homework_builder = runpy.run_path(str(root / "scripts" / "build_homework.py"))
assert homework == homework_builder["build_notebook"]()
assert homework.metadata.homework.points == 10
for cell in homework.cells:
    if cell.cell_type == "code":
        ast.parse(cell.source)
        assert not cell.outputs
homework_report = json.loads((root / "validation" / "hw-4-celeba.json").read_text())
assert homework_report["notebook_sha256"] == hashlib.sha256(homework_path.read_bytes()).hexdigest()
print("hw-4-celeba: student template validated")

for text in root.glob("*.md"):
    for target in re.findall(r"\]\(([^)]+)\)", text.read_text()):
        if not re.match(r"[a-z]+://", target) and not target.startswith("#"):
            assert (text.parent / target.split("#")[0]).exists(), (text.name, target)

excluded_parts = {".git", ".venv", "venv", ".cache", ".ipynb_checkpoints",
                  "__pycache__", "cyclegan_cache", "node_modules", "data", "checkpoints"}
files = [path for path in root.rglob("*") if path.is_file()
         and not excluded_parts.intersection(path.relative_to(root).parts)
         and path.name not in {".DS_Store", "MANIFEST.sha256", ".env"}
         and path.suffix.lower() != ".zip"
         and not path.name.startswith(".env.")]
manifest = "\n".join(f"{hashlib.sha256(path.read_bytes()).hexdigest()}  {path.relative_to(root)}"
                     for path in sorted(files)) + "\n"
(root / "MANIFEST.sha256").write_text(manifest)
files.append(root / "MANIFEST.sha256")
archive = root.with_suffix(".zip")
with zipfile.ZipFile(archive, "w", compression=zipfile.ZIP_DEFLATED, compresslevel=9) as bundle:
    for path in sorted(files):
        bundle.write(path, arcname=str(Path(root.name) / path.relative_to(root)))
with zipfile.ZipFile(archive) as bundle:
    assert bundle.testzip() is None
    for line in manifest.splitlines():
        expected, relative = line.split("  ", 1)
        assert hashlib.sha256(bundle.read(f"{root.name}/{relative}")).hexdigest() == expected
print("Verified archive:", archive.name, "; files:", len(files), "; MB:", round(archive.stat().st_size / 1e6, 2))
