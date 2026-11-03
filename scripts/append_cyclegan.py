"""Embed the verified photo demo and opt-in inference in seminar 1."""

import base64
import json
from pathlib import Path

import nbformat as nbf


ROOT = Path(__file__).resolve().parents[1]
path = ROOT / "01_vae_gan.ipynb"
notebook = nbf.read(path, as_version=4)
notebook.cells = [cell for cell in notebook.cells if "photo_extension" not in cell.metadata.get("tags", [])]
location = next(index for index, cell in enumerate(notebook.cells) if "<!-- CYCLEGAN_EXTENSION -->" in cell.source)
report = json.loads((ROOT / "assets" / "cyclegan_execution.json").read_text())
preview = base64.b64encode((ROOT / "assets" / "cyclegan_photo.png").read_bytes()).decode()
input_png = base64.b64encode((ROOT / "assets" / "horse_input.png").read_bytes()).decode()

introduction = nbf.v4.new_markdown_cell(r"""
### CycleGAN: применение предобученной модели

- **Модель:** horse→zebra из [официального репозитория](https://github.com/junyanz/pytorch-CycleGAN-and-pix2pix); обучение не выполняется.
- **Вход:** левая половина первого кадра авторской иллюстрации. Результат получен отдельным проверенным CPU-запуском.
- **Анализ:** сохранение геометрии и фона; изменения вне целевого объекта.
- **Новый inference:** `RUN_CYCLEGAN_DEMO=True`; первоначальная загрузка весов — около 46 MB. Сохранённый результат доступен без сети.

**Теория:** [Cycle consistency](https://fpakhurov.com/ru/notes/generative-models/cyclegan/#cycle-consistency), [ограничения](https://fpakhurov.com/ru/notes/generative-models/cyclegan/#limitations).
""".strip())

preview_cell = nbf.v4.new_code_cell(
    '#@title Verified saved photo experiment\n'
    'import base64\nfrom IPython.display import Image as DisplayImage, display\n'
    f'PHOTO_INPUT_BASE64 = "{input_png}"\n'
    f'PHOTO_PREVIEW_BASE64 = "{preview}"\n'
    'display(DisplayImage(data=base64.b64decode(PHOTO_PREVIEW_BASE64)))\n'
    f'print("Cached pretrained inference; CPU forward time: {report["forward_seconds_cpu"]:.3f} s. No training performed.")'
)
preview_cell.metadata.update({"cellView": "form", "jupyter": {"source_hidden": True}})

module = (ROOT / "scripts" / "cyclegan_photo.py").read_text().split('if __name__ == "__main__":')[0]
module = module.replace('weights_path = download(MODEL_URL, cache / "horse2zebra.pth")',
                        f'weights_path = download(MODEL_URL, cache / "horse2zebra.pth", "{report["weights_sha256"]}")')
live_cell = nbf.v4.new_code_cell(
    '#@title Optional live photo inference\n'
    'RUN_CYCLEGAN_DEMO = False #@param {type:"boolean"}\n\n'
    + module
    + '\nif RUN_CYCLEGAN_DEMO:\n'
      '    from io import BytesIO\n'
      '    photo_input = Image.open(BytesIO(base64.b64decode(PHOTO_INPUT_BASE64))).convert("RGB")\n'
      '    photo_device = "cuda" if torch.cuda.is_available() else "cpu"\n'
      '    photo_prediction, photo_seconds = translate(photo_input, Path.cwd() / "cyclegan_cache", device=photo_device)\n'
      '    display(photo_prediction)\n'
      '    print(f"New pretrained forward pass: {photo_seconds:.3f} s on {photo_device}")\n'
      'else:\n'
      '    print("Live photo inference disabled. The verified cached result is shown above.")\n'
)
live_cell.metadata.update({"cellView": "form", "jupyter": {"source_hidden": True}})

for cell in [introduction, preview_cell, live_cell]:
    cell.metadata["tags"] = ["photo_extension"]
notebook.cells[location + 1:location + 1] = [introduction, preview_cell, live_cell]
nbf.validate(notebook)
nbf.write(notebook, path)
print(f"Embedded verified CycleGAN example in {path.name}")
