"""Add self-contained real cat/dog image experiments to the two seminars."""

import argparse
from pathlib import Path
import nbformat as nbf

root = Path(__file__).resolve().parents[1]
module = (root / "scripts" / "pet_demo.py").read_text().split('if __name__ == "__main__":')[0]
parser = argparse.ArgumentParser()
parser.add_argument("notebooks", nargs="*", default=["01_vae_gan.ipynb", "02_flows_diffusion.ipynb"])
arguments = parser.parse_args()
for filename in arguments.notebooks:
    path = root / filename
    notebook = nbf.read(path, as_version=4)
    notebook.cells = [cell for cell in notebook.cells if "pet_extension" not in cell.metadata.get("tags", [])]
    explanation = nbf.v4.new_markdown_cell(r"""
## Фотографии кошки и собаки: латентное представление

- **Данные:** две реальные фотографии, центрированный crop 256×256; атрибуция — [Wikimedia Commons](https://commons.wikimedia.org/).
- **Модель:** [TAESD](https://github.com/madebyollin/taesd), предобученный компактный автоэнкодер. Первоначальная загрузка весов — около 10 MB.
- **Эксперимент:** восстановление $D(E(x))$ и возмущение кода $D(E(x)+\sigma\varepsilon)$; параметры доступны ниже.
- TAESD детерминирован: гауссовский posterior и генеративный prior здесь не обучаются. Случайный код не является корректным источником новых фотографий.
- Линейное смешение кодов кошки и собаки может создавать гибридные признаки. Для заданного класса нужно обученное условие $y$; это показано на MNIST в cVAE, cGAN и DDPM.
- **Слайдер PET_NOISE:** масштаб возмущения латентного кода, а не уровень диффузионной цепочки. Показываются σ=0, PET_NOISE и 3×PET_NOISE. Входные фотографии, веса и направление случайного возмущения фиксированы; нового обучения нет. RUN_PET_DEMO только включает/выключает этот опыт.
- **Задание:** сопоставить сохранение формы и мелких деталей при разных $\sigma$. Различить случайность внутри класса и смену класса.

**Теория:** [Латентная диффузия: роль автоэнкодера](https://fpakhurov.com/ru/notes/generative-models/latent-diffusion/), [условный перевод доменов](https://fpakhurov.com/ru/notes/generative-models/stargan/).
""".strip())
    if filename.startswith("02_"):
        explanation.source = explanation.source.replace("## Фотографии", "## Дополнительно: фотографии", 1)
        explanation.source += "\n\nЭтот опыт разобран в первом семинаре; повторный показ во втором в основной тайминг не входит."
    demo = nbf.v4.new_code_cell(
        '#@title Real cat and dog: reconstruction and latent perturbation\n'
        'RUN_PET_DEMO = True #@param {type:"boolean"}\n'
        'PET_NOISE = 0.2 #@param {type:"slider", min:0.05, max:0.6, step:0.05}\n\n'
        + module
        + '\nif RUN_PET_DEMO:\n'
          '    from IPython.display import display\n'
          '    import matplotlib.pyplot as plt\n'
          '    pet_figures = run_pet_demo(noise_levels=(0.0, PET_NOISE, 3 * PET_NOISE))\n'
          '    display(pet_figures[0]); display(pet_figures[1])\n'
          '    plt.close(pet_figures[0]); plt.close(pet_figures[1])\n'
          '    print("Pet demo:", pet_figures[2]["input_shape"], "->", pet_figures[2]["latent_shape"], "; reconstruction MSE:", pet_figures[2]["reconstruction_mse"])\n'
          'else:\n'
          '    print("Optional pet experiment disabled.")\n'
    )
    for cell in [explanation, demo]:
        cell.metadata["tags"] = ["pet_extension"]
    demo.metadata.update({"cellView": "form", "jupyter": {"source_hidden": True}})
    notebook.cells[-1:-1] = [explanation, demo]
    nbf.write(notebook, path)
    print("Added cat/dog experiment:", path.name)
