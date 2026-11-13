# Практические семинары по генеративным моделям

Материалы двух занятий по 90 минут, подготовленных для Газпрома. Для работы нужны Python и основы нейросетей. В каждом блокноте есть краткая теория, реализация на PyTorch и эксперименты с параметрами моделей.

## Программа

| Семинар | Темы | Блокнот |
| --- | --- | --- |
| **1. VAE и GAN** | Автоэнкодер как базовая модель. Реконструкция, генерация и интерполяция в VAE. Обучение GAN и генерация цифр заданного класса. | [Открыть в Colab](https://colab.research.google.com/github/fpakhurov/gazprom-generative-seminars/blob/main/01_vae_gan.ipynb) |
| **2. Нормализующие потоки и диффузия** | Оценка плотности с RealNVP. Зашумление и обратный процесс DDPM. Условная генерация и редактирование изображений методом SDEdit. | [Открыть в Colab](https://colab.research.google.com/github/fpakhurov/gazprom-generative-seminars/blob/main/02_flows_diffusion.ipynb) |

Основные эксперименты используют двумерные распределения и изображения MNIST размером 28 × 28 пикселей. Семейства моделей сравниваются по целевой функции, процедуре генерации и вычислительной стоимости. Дополнительные демонстрации включают преобразование изображений с CycleGAN и реконструкцию фотографий кошки и собаки.

## Запуск в Google Colab

1. Войдите в аккаунт Google и откройте блокнот из таблицы. Для обучения свёрточных моделей выберите GPU T4 в настройках среды выполнения. Запуск на CPU также поддерживается.
2. Выберите **Среда выполнения → Выполнить все**. Для основных экспериментов оставьте `QUICK_MODE = False`. Режим `True` подходит для быстрой проверки запуска.
3. После обучения меняйте параметры в интерактивных ячейках и повторно выполняйте ячейки генерации или редактирования.

## Материалы

- Сохранённые результаты для просмотра без запуска Python. Скачайте HTML-файл и откройте его в браузере. [Семинар 1](validation/01_vae_gan.html) · [Семинар 2](validation/02_flows_diffusion.html)
- Примеры изображений. [MNIST](assets/mnist_training_preview.png) · [Реконструкция фотографий](assets/pets_reconstruction.jpg)
- Теория курса на fpakhurov.com, первоисточники и схемы архитектур. [Список материалов](SOURCES.md)
- План занятий и пояснения к экспериментам. [Методические указания](INSTRUCTOR_GUIDE.md) · [Заметки к семинару 1](LECTURER_NOTES_01.md) · [Заметки к семинару 2](LECTURER_NOTES_02.md)
- [Архив семинаров](https://github.com/fpakhurov/gazprom-generative-seminars/releases/latest) · [Сведения о среде и проверках](VALIDATION.md)

## Домашняя работа

[Открыть в Colab](https://colab.research.google.com/github/fpakhurov/gazprom-generative-seminars/blob/main/03_homework_celeba.ipynb) · [Скачать ноутбук](03_homework_celeba.ipynb)

VAE, нормализующий поток, VAPNEV и conditional VAPNEV на CelebA 64×64. Задания адаптированы для запуска в Colab из курса HSE-LAMBDA. Суммарная оценка — 10 баллов. Ноутбук автоматически скачивает архив изображений и атрибутов из Google Drive и проверяет SHA-256. Инструкция и класс датасета включены в ноутбук.

<details>
<summary>Локальный запуск</summary>

Выполните команды из корня репозитория в активированном виртуальном окружении Python 3.11 или новее.

```bash
python -m pip install -r requirements.txt
python -m ipykernel install --prefix "$VIRTUAL_ENV" --name python3
python scripts/validate_notebooks.py 01_vae_gan.ipynb 02_flows_diffusion.ipynb
```

</details>
