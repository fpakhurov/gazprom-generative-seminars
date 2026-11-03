"""Export notebook previews with embedded math and attributed remote figures."""

import argparse
import base64
import json
from pathlib import Path
import re
import subprocess

from bs4 import BeautifulSoup
import nbformat
from nbconvert import HTMLExporter
from traitlets.config import Config


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--katex-dir", required=True)
    arguments = parser.parse_args()
    katex = Path(arguments.katex_dir).resolve()
    root = Path(__file__).resolve().parents[1]
    css = (katex / "dist" / "katex.min.css").read_text()

    def font_url(match):
        path = katex / "dist" / match.group(1)
        media = {".woff2": "font/woff2", ".woff": "font/woff", ".ttf": "font/ttf"}[path.suffix]
        encoded = base64.b64encode(path.read_bytes()).decode()
        return f'url(data:{media};base64,{encoded})'

    css = re.sub(r"url\((fonts/[^)]+)\)", font_url, css)
    pattern = re.compile(r"\$\$(.*?)\$\$|(?<!\$)\$(?!\$)([^$\n]+?)\$(?!\$)", re.S)
    for name in ["01_vae_gan", "02_flows_diffusion"]:
        notebook = nbformat.read(root / f"{name}.ipynb", as_version=4)
        for cell in notebook.cells:
            if cell.metadata.get("jupyter", {}).get("source_hidden"):
                cell.metadata.setdefault("tags", []).append("remove_input")
        config = Config({"TagRemovePreprocessor": {"enabled": True, "remove_input_tags": ["remove_input"]}})
        html, _ = HTMLExporter(template_name="lab", config=config).from_notebook_node(notebook)
        soup = BeautifulSoup(html, "html.parser")
        for script in soup.find_all("script"):
            script.decompose()
        fragments = []
        formulas = []
        for text_node in list(soup.find_all(string=True)):
            if any(parent.name in {"code", "pre", "style", "script", "title"} for parent in text_node.parents):
                continue
            matches = list(pattern.finditer(str(text_node)))
            if matches:
                fragments.append((text_node, matches))
                formulas.extend({"tex": match.group(1) if match.group(1) is not None else match.group(2),
                                 "display": match.group(1) is not None} for match in matches)
        javascript = """
const fs = require('fs');
const katex = require(process.argv[1]);
const formulas = JSON.parse(fs.readFileSync(0, 'utf8'));
process.stdout.write(JSON.stringify(formulas.map(f => katex.renderToString(f.tex, {
  displayMode: f.display, throwOnError: true, strict: 'ignore', output: 'htmlAndMathml'
}))));
"""
        rendered = iter(json.loads(subprocess.run(
            ["node", "-e", javascript, str(katex)],
            input=json.dumps(formulas), text=True, capture_output=True, check=True,
        ).stdout))
        for text_node, matches in fragments:
            original = str(text_node)
            replacements = []
            position = 0
            for match in matches:
                replacements.append(original[position:match.start()])
                replacements.extend(list(BeautifulSoup(next(rendered), "html.parser").contents))
                position = match.end()
            replacements.append(original[position:])
            text_node.replace_with(*replacements)
        style = soup.new_tag("style")
        style.string = css + "\n.katex-display{overflow-x:auto;overflow-y:hidden;padding:0.35em 0}\n.jp-RenderedHTMLCommon img{height:auto;max-width:100%}"
        soup.head.append(style)
        output = root / "validation" / f"{name}.html"
        output.write_text(str(soup))
        print(f"Rendered {name}: {len(formulas)} math expressions, no external scripts")
