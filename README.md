# Генеративные модели: практические семинары

**Газпром · 2×90 минут · Python и основы нейросетей.**

| Занятие | Содержание |
|---|---|
| 1 | AE → VAE → GAN; преобразование фотографий предобученным CycleGAN |
| 2 | RealNVP → DDPM; SDEdit-подобное редактирование и дополнение по маске |

| Блокнот | Google Colab |
|---|---|
| 1. VAE и GAN | [Открыть в Colab](https://colab.research.google.com/github/fpakhurov/gazprom-generative-seminars/blob/main/01_vae_gan.ipynb) |
| 2. Потоки и диффузия | [Открыть в Colab](https://colab.research.google.com/github/fpakhurov/gazprom-generative-seminars/blob/main/02_flows_diffusion.ipynb) |

- Формат: тезисы, схемы, формулы, готовый код и управляемые эксперименты.
- Основные данные: встроенные цифры 8×8 и двумерные точки. GPU необязателен.
- LLM не рассматриваются. Дополнительные разделы предназначены для самостоятельной работы.

## Материалы

- [Семинар 1: блокнот](01_vae_gan.ipynb) · [HTML-файл](validation/01_vae_gan.html).
- [Семинар 2: блокнот](02_flows_diffusion.ipynb) · [HTML-файл](validation/02_flows_diffusion.html).
- [Методические указания и тайминг](INSTRUCTOR_GUIDE.md).
- [Теория по темам, источники и схемы](SOURCES.md).
- [Проверки и параметры среды](VALIDATION.md).
- [ZIP для локального просмотра и скачивания](https://github.com/fpakhurov/gazprom-generative-seminars/releases/latest).

## Запуск

1. Открыть блокнот по ссылке Colab в таблице выше.
2. Оставить Python 3 и CPU. При необходимости сохранить копию в свой Google Drive.
3. Выбрать **Среда выполнения → Выполнить все**.
4. Для основного 90-минутного маршрута следовать пометкам блокнота; расширения разбирать после занятия.

- Google Drive и API-ключи не требуются.
- CycleGAN по умолчанию использует сохранённую иллюстрацию. Новый inference: RUN_CYCLEGAN_DEMO=True; нужны интернет и загрузка около 46 MB весов.
- После изменения архитектуры или расписания шума повторить создание модели и оптимизатора.

Локальное воспроизведение в отдельном окружении Python 3.11+:

    python -m pip install -r requirements.txt
    python -m ipykernel install --prefix "$VIRTUAL_ENV" --name python3
    python scripts/validate_notebooks.py 01_vae_gan.ipynb 02_flows_diffusion.ipynb

Исходники сборки находятся в scripts/. HTML доступен без выполнения Python; для внешних архитектурных схем требуется интернет; сохранённые результаты позволяют проводить разбор без повторного обучения.

Проверено локальное выполнение; hosted Colab отдельно не тестировался. Демонстрации на малых данных не подтверждают эффективность на производственных задачах.
