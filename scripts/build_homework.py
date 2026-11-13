"""Build a self-contained Colab port of the HSE-LAMBDA CelebA homework."""

from pathlib import Path

import nbformat


ROOT = Path(__file__).resolve().parents[1]
UPSTREAM_URL = "https://github.com/HSE-LAMBDA/DeepGenerativeModels/blob/42dd64b535f5bedf1d2b3db28f2f21ce96c9f09f/homeworks/03_hw_vapnev.ipynb"

# The original task text and cells are embedded below for offline rebuilding.
UPSTREAM_CELLS = [{'cell_type': 'markdown',
  'source': '# Домашнее задание 3. VAE + NF + VAPNEV\n'
            '\n'
            'В этом задании вам предстоит поработать с набором данных CelebA. Решение домашней '
            'работы должно быть представлено в виде .zip архива, содержащего:\n'
            '- Jupyter Notebook(и) и/или модули с кодом,\n'
            '- PDF-отчет, в котором необходимо описать проведённые эксперименты, а также '
            'обосновать выбор используемых методов (например, дополнительные лоссы для '
            'стабилизации, виды потоков, особенности реализации VAE и пр.).'},
 {'cell_type': 'markdown', 'source': '## Настройка окружения'},
 {'cell_type': 'code',
  'source': 'import sys\n'
            "sys.path.append('../seminars/utils/')  # Добавляем путь к CelebADataset\n"
            '\n'
            'import torch\n'
            'from torchvision import transforms\n'
            '\n'
            'import matplotlib.pyplot as plt\n'
            'import numpy as np\n'
            '\n'
            'from datasets.celeba import CelebADataset\n'
            '\n'
            '# Set device\n'
            'cuda = torch.cuda.is_available()\n'
            'device = torch.device("cuda" if cuda else ("mps" if torch.backends.mps.is_available() '
            'else "cpu"))\n'
            'torch.backends.cudnn.benchmark = cuda\n'
            '\n'
            'print(f"Using device: {device}")'},
 {'cell_type': 'markdown', 'source': '## Загрузка данных'},
 {'cell_type': 'code',
  'source': '# Определение трансформаций\n'
            '\n'
            't_normalize = lambda x: x * 2 - 1\n'
            't_invnormalize = lambda x: (x + 1) / 2\n'
            '\n'
            'transform = transforms.Compose([\n'
            '    transforms.Resize(64),\n'
            '    transforms.CenterCrop(64),\n'
            '    transforms.ToTensor(),\n'
            '    t_normalize,\n'
            '])'},
 {'cell_type': 'code',
  'source': '# Кастомный датасет CelebA, возвращающий только изображения\n'
            '\n'
            'class CelebaCustomDataset(CelebADataset):\n'
            '    def __getitem__(self, idx):\n'
            '        image, _ = super().__getitem__(idx)\n'
            '        return image\n'
            '\n'
            'dataset = CelebaCustomDataset(\n'
            '    transform=transform,\n'
            "    root_dir='../seminars/data/celeba',\n"
            ')\n'
            '\n'
            '# Пример получения изображения\n'
            'img = dataset[0]\n'
            'print("Размер изображения:", img.shape)'},
 {'cell_type': 'markdown', 'source': '### Визуализация датасета'},
 {'cell_type': 'code',
  'source': 'def plot_img(img, transform):\n'
            '    plt.imshow(transform(img).permute(1, 2, 0))\n'
            "    plt.axis('off')"},
 {'cell_type': 'code',
  'source': 'plt.figure(figsize=(15, 15))\n'
            'for i in range(25):\n'
            '    random_idx = np.random.randint(len(dataset))\n'
            '    img = dataset[random_idx]\n'
            '    plt.subplot(5, 5, i + 1)\n'
            '    plot_img(img, t_invnormalize)\n'
            'plt.show()'},
 {'cell_type': 'markdown',
  'source': '### Задача 1 (4/13 баллов): Построение и обучение нормального потока (Normalizing '
            'Flow) на CelebA\n'
            '\n'
            'Необходимо обучить нормальный поток до качества, удовлетворяющего базовым порогам '
            '(точные пороги будут предоставлены позже). \n'
            '\n'
            'Измерьте FID и Negative Log Likelihood для оценки модели.\n'
            '\n'
            'Внутри потока можно использовать любые компоненты: Coupling/Autoregressive/Linear '
            'слои, произвольные трансформации и т.д. Главное — реализовать их самостоятельно без '
            'копипаста.\n'
            '\n'
            'Можно использовать как сверточные потоки (обычно дают лучшие результаты, но сложнее '
            'реализуются), так и линейные потоки (будут менее качественными), при этом можно '
            'предварительно свести изображение к вектору.\n'
            '\n'
            'Можно использовать готовый код из репозитория курса, но рекомендуется написать свою '
            'реализацию.'},
 {'cell_type': 'code', 'source': '# TODO: Реализуйте нормальный поток для CelebA.'},
 {'cell_type': 'markdown',
  'source': '### Задача 2 (2/13 баллов): Построение и обучение VAE на CelebA\n'
            '\n'
            'Обучите вариационный автоэнкодер (VAE) до достижения приемлемого качества генерации. '
            'Измерьте FID и сохраните результаты для дальнейшего сравнения.\n'
            '\n'
            'Можно использовать готовые решения из репозитория курса, но предпочтительно написать '
            'свою реализацию.'},
 {'cell_type': 'code', 'source': '# TODO: Реализуйте VAE для CelebA.'},
 {'cell_type': 'markdown',
  'source': '### Задача 3 (6/13 баллов)\n'
            '\n'
            '#### Задача 3.1 (4 балла): Построение и обучение VAPNEV на CelebA\n'
            '\n'
            'Изучите статью [VAPNEV](https://arxiv.org/pdf/1611.05209.pdf), реализуйте и обучите '
            'модель до качества, удовлетворяющего базовым порогам (точные пороги будут '
            'предоставлены позже). Измерьте FID и сохраните результаты.\n'
            '\n'
            '#### Задача 3.2 (2 балла): Построение и обучение conditional VAPNEV на CelebA\n'
            '\n'
            'Реализуйте условный (conditional) вариант VAPNEV. Подробности можно найти в описании '
            'задачи и в статье.'},
 {'cell_type': 'code', 'source': '# TODO: Реализуйте VAPNEV и conditional VAPNEV для CelebA.'},
 {'cell_type': 'markdown',
  'source': '### Задача 4 (1/13 баллов): Анализ\n'
            '\n'
            'Проведите анализ латентного пространства и оцените качество генерации:\n'
            '1. Определите, у какой модели лучше интерполируется латентное пространство.\n'
            '2. Попробуйте взять внешнее изображение (не из CelebA), отобразить его в латентное '
            'пространство и найдите ближайшие по значению векторы.\n'
            '3. Продемонстрируйте самый интересный (или забавный) результат генерации, который у '
            'вас получался.'},
 {'cell_type': 'code',
  'source': '# TODO: Добавьте код для анализа латентного пространства и демонстрации результатов.'},
 {'cell_type': 'markdown',
  'source': '### Задача 0: Отчет\n'
            '\n'
            'Для получения полного балла по каждому пункту задания необходимо включить в отчет '
            '(предоставляемый отдельно в формате PDF) примеры генераций моделей. \n'
            '\n'
            'Сравните результаты различных моделей, постарайтесь объяснить, почему одни подходы '
            'работают лучше других.'}]


def build_notebook():
    cells = []

    def md(source):
        cells.append(nbformat.v4.new_markdown_cell(source.strip()))

    def code(source):
        cells.append(nbformat.v4.new_code_cell(source.strip()))

    md(r"""
# Домашняя работа. VAE + NF + VAPNEV

Colab-версия задания HSE-LAMBDA. Сохранены задания, разрешение **64×64** и требования к сдаче. Шкала переведена в **10 баллов**.

| Задание | Баллы |
|---|---:|
| Нормализующий поток | 3 |
| VAE | 2 |
| VAPNEV | 3 |
| Conditional VAPNEV | 1 |
| Анализ | 1 |
| **Всего** | **10** |

1. Сохраните копию ноутбука в Google Drive. Выберите GPU в настройках среды выполнения.
2. Выполните ячейки подготовки. Ноутбук автоматически скачает `celeba.zip` по открытой ссылке Google Drive, проверит контрольную сумму и распакует данные.
3. Изображения и `annotations/list_attr_celeba.txt` входят в один архив. Настройки загрузки можно оставить по умолчанию.
4. После визуализации данных заполните исходные `TODO` и проведите эксперименты.

Архив занимает около 1,49 ГБ. Загрузка выполняется в файловую систему Colab. Данные в `/content` хранятся до завершения сессии. [Архив CelebA](https://drive.google.com/file/d/1vQZtj1uLUa4pbBddsIYYk3MkjyHqoEMH/view)

Загрузчик `CelebADataset` из курса включён в ноутбук с адаптацией путей и загрузки локальных архивов. Подготовка работает без клонирования репозитория.

Исходник — [HSE-LAMBDA, spring-2025-long](https://github.com/HSE-LAMBDA/DeepGenerativeModels/blob/42dd64b535f5bedf1d2b3db28f2f21ce96c9f09f/homeworks/03_hw_vapnev.ipynb). Атрибуция и лицензия приведены в конце.
""")
    for index, original in enumerate(UPSTREAM_CELLS):
        if index == 2:
            code(r'''
import importlib.util
import importlib.metadata
import subprocess
import sys

packages = {"torch": "torch", "PIL": "pillow", "numpy": "numpy", "matplotlib": "matplotlib"}
missing = [package for module, package in packages.items() if importlib.util.find_spec(module) is None]
if missing:
    subprocess.check_call([sys.executable, "-m", "pip", "install", "-q", *missing])
if not importlib.util.find_spec("gdown") or importlib.metadata.version("gdown") != "6.5.1":
    subprocess.check_call([sys.executable, "-m", "pip", "install", "-q", "gdown==6.5.1"])

from pathlib import Path
import random
import torch
from PIL import Image
import matplotlib.pyplot as plt
import numpy as np

SEED = 42
random.seed(SEED)
np.random.seed(SEED)
torch.manual_seed(SEED)
cuda = torch.cuda.is_available()
device = torch.device("cuda" if cuda else ("mps" if torch.backends.mps.is_available() else "cpu"))
torch.backends.cudnn.benchmark = cuda
print("Python:", sys.version.split()[0], "PyTorch:", torch.__version__)
print("Using device:", device)
''')
            code((ROOT / "scripts/celeba_dataset.py").read_text())
            cells[-1].metadata["jupyter"] = {"source_hidden": True}
            md(r"""
## Архив изображений и атрибуты

Исходный загрузчик курса ожидает следующую структуру

```text
celeba/
├── img_align_celeba/
│   ├── 000001.jpg
│   └── ...
└── annotations/
    └── list_attr_celeba.txt
```

По умолчанию `DOWNLOAD_FROM_DRIVE=True` скачивает открытый архив в `IMAGE_ZIP`. Подключение личного Google Drive для этого не требуется. Корректный локальный архив используется повторно. Прерванное скачивание можно продолжить повторным запуском ячейки.

`ATTR_ZIP` и `ATTR_PATH` оставьте пустыми. Эти поля нужны для отдельного архива атрибутов или уже распакованного `list_attr_celeba.txt` при использовании собственного набора файлов.

Проверенный архив занимает около 1,49 ГБ и содержит 202 599 изображений и 40 атрибутов. Достаточно указать один `IMAGE_ZIP`. Служебные файлы macOS в `__MACOSX` загрузчик пропускает.

`DATA_DIR` задаёт каталог распаковки. Дополнительная папка `celeba` внутри архива поддерживается. Для уже распакованных данных выключите `DOWNLOAD_FROM_DRIVE` и `EXTRACT_ARCHIVES` и укажите их каталог в `DATA_DIR`. При нескольких наборах данных сузьте каталог поиска, при нескольких файлах атрибутов задайте `ATTR_PATH` явно.

Если Google Drive временно ограничит скачивания, можно использовать свою копию архива. Выключите `DOWNLOAD_FROM_DRIVE`, включите `USE_GOOGLE_DRIVE` и укажите путь вида `/content/drive/MyDrive/celeba.zip`. Для загруженного локального файла выключите `DOWNLOAD_FROM_DRIVE` и укажите `IMAGE_ZIP`. При повторном запуске подготовки можно выключить `EXTRACT_ARCHIVES`.
""")
            code(r'''
#@title Download and prepare CelebA
DOWNLOAD_FROM_DRIVE = True #@param {type:"boolean"}
USE_GOOGLE_DRIVE = False #@param {type:"boolean"}
IMAGE_ZIP = "/content/celeba.zip" #@param {type:"string"}
ATTR_ZIP = "" #@param {type:"string"}
ATTR_PATH = "" #@param {type:"string"}
DATA_DIR = "/content/DeepGenerativeModels/seminars/data/celeba" #@param {type:"string"}
EXTRACT_ARCHIVES = True #@param {type:"boolean"}

if USE_GOOGLE_DRIVE:
    from google.colab import drive
    drive.mount("/content/drive")

if DOWNLOAD_FROM_DRIVE:
    if not IMAGE_ZIP.strip():
        raise ValueError("Set IMAGE_ZIP to the local destination for the download.")
    download_celeba(
        file_id="1vQZtj1uLUa4pbBddsIYYk3MkjyHqoEMH",
        archive_path=IMAGE_ZIP,
        expected_size=1491502698,
        expected_sha256="ec9b0e280f8a3ea83c08ae9813c2a2f714a229eb80699868698a91974089e014",
    )

if EXTRACT_ARCHIVES:
    for archive_path in (IMAGE_ZIP, ATTR_ZIP):
        if archive_path.strip():
            if not Path(archive_path).expanduser().is_file():
                raise FileNotFoundError(f"Upload the archive and update its path: {archive_path}")
            print("Extracting:", archive_path)
            extract_celeba_zip(archive_path, DATA_DIR)
    # The upstream dataset may also be supplied as an inner image ZIP.
    for image_archive in Path(DATA_DIR).expanduser().rglob("img_align_celeba.zip"):
        extract_celeba_zip(image_archive, image_archive.parent)

dataset_root, attribute_file = find_celeba_files(DATA_DIR, ATTR_PATH or None)
print("Dataset root:", dataset_root)
print("Attributes:", attribute_file)
''')
        elif index == 4:
            code(r'''
# Resize the shorter side to 64, center-crop, then normalize to [-1, 1].
# Equivalent preprocessing stages to the original torchvision pipeline.
t_normalize = lambda x: x * 2 - 1
t_invnormalize = lambda x: (x + 1) / 2

def transform(image):
    image = image.convert("RGB")
    width, height = image.size
    if width <= height:
        resized = (64, int(64 * height / width))
    else:
        resized = (int(64 * width / height), 64)
    image = image.resize(resized, resample=Image.Resampling.BILINEAR)
    left = int(round((image.width - 64) / 2))
    top = int(round((image.height - 64) / 2))
    image = image.crop((left, top, left + 64, top + 64))
    pixels = np.array(image, dtype=np.float32) / 255.0
    tensor = torch.from_numpy(pixels).permute(2, 0, 1).contiguous()
    return t_normalize(tensor)
''')
        elif index == 5:
            code(r'''
# Preserve the original image-only interface for unconditional models.
class CelebaCustomDataset(CelebADataset):
    def __getitem__(self, idx):
        image, _ = super().__getitem__(idx)
        return image

dataset = CelebaCustomDataset(
    transform=transform,
    root_dir=dataset_root,
    attributes_path=attribute_file,
)
img = dataset[0]
assert img.shape == (3, 64, 64)
assert torch.isfinite(img).all() and img.min() >= -1 and img.max() <= 1
print("Images:", len(dataset), "Image shape:", tuple(img.shape))
print("Attribute names:", dataset.header)
''')
            md(r"""
`CelebaCustomDataset` возвращает только изображение, как в исходном задании. Для conditional VAPNEV используйте `CelebADataset` с теми же аргументами. Он возвращает `(image, info)`. Вектор `info["attributes"]` хранит значения −1/+1 в порядке `dataset.header`. Для бинарного условия можно преобразовать выбранное значение по формуле `(value + 1) // 2`.
""")
        else:
            constructor = md if original["cell_type"] == "markdown" else code
            source = original["source"]
            for old_score, new_score in [("4/13 баллов", "3/10 баллов"), ("2/13 баллов", "2/10 баллов"), ("6/13 баллов", "4/10 баллов"), ("1/13 баллов", "1/10 баллов"), ("(4 балла)", "(3 балла)"), ("(2 балла)", "(1 балл)")]:
                source = source.replace(old_score, new_score)
            constructor(source)
            cells[-1].metadata["upstream_cell_index"] = index

    md(r"""
## Источники и атрибуция

- [Исходная домашняя работа](https://github.com/HSE-LAMBDA/DeepGenerativeModels/blob/42dd64b535f5bedf1d2b3db28f2f21ce96c9f09f/homeworks/03_hw_vapnev.ipynb).
- [Исходный загрузчик CelebADataset](https://github.com/HSE-LAMBDA/DeepGenerativeModels/blob/42dd64b535f5bedf1d2b3db28f2f21ce96c9f09f/seminars/utils/datasets/celeba.py). Адаптированы пути и чтение выданных архивов, сохранён интерфейс данных и атрибутов.
- [CelebA](https://mmlab.ie.cuhk.edu.hk/projects/CelebA.html).
- [Подробная теория VAE](https://fpakhurov.com/ru/notes/generative-models/vae/) · [нормализующие потоки](https://fpakhurov.com/ru/notes/generative-models/normalizing-flows/) · [архитектуры потоков](https://fpakhurov.com/ru/notes/generative-models/flow-architectures/).

Исходные материалы HSE-LAMBDA распространяются по MIT License.
""")
    md((ROOT / "assets/HSE_LAMBDA_LICENSE.txt").read_text())
    for index, cell in enumerate(cells):
        cell.id = f"celeba-hw-{index:02d}"
    notebook = nbformat.v4.new_notebook(cells=cells)
    notebook.metadata.update({
        "kernelspec": {"display_name": "Python 3", "language": "python", "name": "python3"},
        "language_info": {"name": "python"},
        "colab": {"name": "03_homework_celeba.ipynb", "provenance": []},
        "homework": {"upstream": UPSTREAM_URL, "image_size": 64, "points": 10},
    })
    return notebook


if __name__ == "__main__":
    notebook = build_notebook()
    nbformat.validate(notebook)
    nbformat.write(notebook, ROOT / "03_homework_celeba.ipynb")
    print("Built Colab homework:", len(notebook.cells), "cells")
