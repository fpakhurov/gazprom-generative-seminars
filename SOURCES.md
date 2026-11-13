# Теория и источники

Проверка: **3 ноября 2026 года**. Код семинаров — самостоятельные малые реализации; исходные курсы используются для структуры и теоретических объяснений.

## Домашняя работа CelebA

Задания и интерфейс датасета взяты из HSE-LAMBDA, revision `42dd64b535f5bedf1d2b3db28f2f21ce96c9f09f`. Для Colab добавлены загрузка выданных ZIP, настройка путей и встроенный датасет. Преобразование изображений сохраняет Resize → CenterCrop → нормировку [-1,1]. Баллы пересчитаны до 10. [Исходное задание](https://github.com/HSE-LAMBDA/DeepGenerativeModels/blob/42dd64b535f5bedf1d2b3db28f2f21ce96c9f09f/homeworks/03_hw_vapnev.ipynb) · [Загрузчик](https://github.com/HSE-LAMBDA/DeepGenerativeModels/blob/42dd64b535f5bedf1d2b3db28f2f21ce96c9f09f/seminars/utils/datasets/celeba.py) · [MIT License](assets/HSE_LAMBDA_LICENSE.txt)

## fpakhurov.com: разбор по темам

Указанные страницы проверены на основном сайте: HTTP 200, соответствующие заголовки и canonical URL. Идентификаторы разделов в ссылках проверены в HTML.

| Тема | Углублённое объяснение |
|---|---|
| AE | [Реконструкция](https://fpakhurov.com/ru/notes/generative-models/autoencoders/#reconstruction-loss); [почему случайные коды не гарантируют генерацию](https://fpakhurov.com/ru/notes/generative-models/autoencoders/#why-the-codes-do-not-generate) |
| VAE: posterior и ELBO | [Латентная модель и posterior](https://fpakhurov.com/ru/notes/generative-models/vae/#the-latent-variable-model-and-its-posterior); [вариационное приближение](https://fpakhurov.com/ru/notes/generative-models/vae/#variational-approximation-and-the-elbo) |
| VAE: градиенты | [Репараметризация](https://fpakhurov.com/ru/notes/generative-models/vae/#the-reparameterization-trick) |
| GAN | [Чередование шагов](https://fpakhurov.com/ru/notes/generative-models/gans/#how-to-optimize-alternate-the-players); [практическая цель генератора](https://fpakhurov.com/ru/notes/generative-models/gans/#which-generator-loss-is-used-in-practice) |
| Image-to-image | [Парная постановка pix2pix](https://fpakhurov.com/ru/notes/generative-models/cyclegan/#paired-translation-pix2pix); [cycle consistency](https://fpakhurov.com/ru/notes/generative-models/cyclegan/#cycle-consistency); [ограничения](https://fpakhurov.com/ru/notes/generative-models/cyclegan/#limitations) |
| Нормализующие потоки | [Замена переменных](https://fpakhurov.com/ru/notes/generative-models/normalizing-flows/#change-of-variables); [coupling NICE/RealNVP](https://fpakhurov.com/ru/notes/generative-models/flow-architectures/#a-coupling-layers-nice-realnvp) |
| DDPM: прямой процесс | [Переход к произвольному шагу](https://fpakhurov.com/ru/notes/generative-models/ddpm-forward-process/#the-jump-from-x-to-any-step) |
| DDPM: обучение и генерация | [Истинный posterior](https://fpakhurov.com/ru/notes/generative-models/ddpm-training-objective/#the-true-posterior); [предсказание шума](https://fpakhurov.com/ru/notes/generative-models/ddpm-training-objective/#from-means-to-noise); [сэмплирование](https://fpakhurov.com/ru/notes/generative-models/ddpm-training-objective/#sampling-running-the-chain-backwards) |
| Связь с SDEdit | [Редактирование тем же генератором](https://fpakhurov.com/ru/notes/generative-models/latent-diffusion/#editing-with-the-same-model); точная постановка — в статье SDEdit ниже |

## HSE-LAMBDA: практические материалы

[DeepGenerativeModels](https://github.com/HSE-LAMBDA/DeepGenerativeModels/tree/spring-2025-long), ветка spring-2025-long. Проверены дерево и [MIT license](https://github.com/HSE-LAMBDA/DeepGenerativeModels/blob/42dd64b535f5bedf1d2b3db28f2f21ce96c9f09f/LICENSE); commit **42dd64b535f5bedf1d2b3db28f2f21ce96c9f09f**.

| Раздел | Блокнот курса |
|---|---|
| AE | [Autoencoders](https://github.com/HSE-LAMBDA/DeepGenerativeModels/blob/42dd64b535f5bedf1d2b3db28f2f21ce96c9f09f/seminars/00_Autoencoders/autoencoders.ipynb) |
| VAE | [Variational Autoencoders](https://github.com/HSE-LAMBDA/DeepGenerativeModels/blob/42dd64b535f5bedf1d2b3db28f2f21ce96c9f09f/seminars/06_Variational-Autoencoders/VAE.ipynb) |
| GAN | [GANs](https://github.com/HSE-LAMBDA/DeepGenerativeModels/blob/42dd64b535f5bedf1d2b3db28f2f21ce96c9f09f/seminars/02_GAN/GANs.ipynb) |
| Image-to-image | [CycleGAN](https://github.com/HSE-LAMBDA/DeepGenerativeModels/blob/42dd64b535f5bedf1d2b3db28f2f21ce96c9f09f/seminars/04_Image-to-Image/CycleGAN.ipynb) |
| NF | [Normalizing flows](https://github.com/HSE-LAMBDA/DeepGenerativeModels/blob/42dd64b535f5bedf1d2b3db28f2f21ce96c9f09f/seminars/09_Normalizing-Flows/nf.ipynb) |
| DDPM | [Diffusion models](https://github.com/HSE-LAMBDA/DeepGenerativeModels/blob/42dd64b535f5bedf1d2b3db28f2f21ce96c9f09f/seminars/10_Diffusion-Models/02_DPM_Models.ipynb) |

Выполнение исходных блокнотов HSE в текущем Colab не проверялось.

## BIASED: структура проверки понимания

[BIASED School](https://biased.school).

- Темы: vae-elbo, vae-reparameterization, gans, cyclegan, normalizing-flows, flow-architectures, ddpm-forward-process, ddpm-training-objective.
- Используемый принцип: тезис → короткая проверка → эксперимент → интерпретация.

## Первоисточники

| Метод | Источник |
|---|---|
| VAE | Kingma, Welling. [Auto-Encoding Variational Bayes](https://arxiv.org/abs/1312.6114) |
| GAN | Goodfellow et al. [Generative Adversarial Networks](https://arxiv.org/abs/1406.2661) |
| RealNVP | Dinh et al. [Density estimation using Real NVP](https://arxiv.org/abs/1605.08803) |
| DDPM | Ho et al. [Denoising Diffusion Probabilistic Models](https://arxiv.org/abs/2006.11239) |
| pix2pix | Isola et al. [Image-to-Image Translation with Conditional Adversarial Networks](https://arxiv.org/abs/1611.07004) |
| CycleGAN | Zhu et al. [Unpaired Image-to-Image Translation](https://junyanz.github.io/CycleGAN/) |
| SDEdit | Meng et al. [Guided Image Synthesis and Editing with SDEs](https://arxiv.org/abs/2108.01073) |
| Данные | [MNIST](https://docs.pytorch.org/vision/stable/generated/torchvision.datasets.MNIST.html): 70 000 объектов, 10 классов, 28×28 |

## Фотографический пример

- Официальный [pytorch-CycleGAN-and-pix2pix](https://github.com/junyanz/pytorch-CycleGAN-and-pix2pix), архитектура на commit **2a7afba2895d52556dd5dfe07e8555ef657ced6f**.
- Веса horse→zebra: официальный HTTPS-сервер Berkeley; демонстрируется inference, обучение не воспроизводится.
- В 90-минутном маршруте использовать подготовленный кеш или сохранённое изображение.
- Статус выполнения и ограничения: [VALIDATION.md](VALIDATION.md). Полный запуск v1.2.0 в Colab на Tesla T4 проверен; текущая редакция и границы её проверки описаны в VALIDATION.md.

## Архитектурные схемы

| Иллюстрация | Изображение | Источник |
|---|---|---|
| VAE: вычислительная схема | [PNG](https://arxiv.org/html/1906.02691v3/vae_cartoon.png) | [Страница авторов / публикация](https://arxiv.org/html/1906.02691v3) |
| DCGAN: генератор | [PNG](https://arxiv.org/html/1511.06434v2/lsun_bedrooms_generator.png) | [Страница авторов / публикация](https://arxiv.org/html/1511.06434v2) |
| CycleGAN: обучающие данные | [PNG](https://arxiv.org/html/1703.10593v7/paired_unpaired2.png) | [Страница авторов / публикация](https://arxiv.org/html/1703.10593v7) |
| Поток: композиция преобразований | [PNG](https://lilianweng.github.io/posts/2018-10-13-flow-models/normalizing-flow.png) | [Страница авторов / публикация](https://lilianweng.github.io/posts/2018-10-13-flow-models/) |
| DDPM: цепочки переходов | [PNG](https://hojonathanho.github.io/diffusion/assets/img/pgm_diagram_xarrow.png) | [Страница авторов / публикация](https://hojonathanho.github.io/diffusion/) |
| U-Net: исходная архитектура | [PNG](https://lmb.informatik.uni-freiburg.de/people/ronneber/u-net/u-net-architecture.png) | [Страница авторов / публикация](https://lmb.informatik.uni-freiburg.de/people/ronneber/u-net/) |

- Проверены HTTP 200 и тип `image/png` для всех шести URL.
- Изображения загружаются из внешних источников; для отображения требуется интернет.
- Авторство указано в подписях. В практикуме используются компактный свёрточный генератор 7→14→28 и условная U-Net с кодированием времени; размеры и состав блоков отличаются от исходных схем.

## Условная диффузия

- [Ho, Salimans — Classifier-Free Diffusion Guidance](https://arxiv.org/abs/2207.12598): совместное обучение условной и безусловной ветвей, компромисс качества и разнообразия.

## Кошка и собака: предобученный автоэнкодер

- [TAESD](https://github.com/madebyollin/taesd), commit `fdbae2df989db95761f482efb8977e1182fb68d2`, MIT. Детерминированная дистиллированная модель; гауссовский posterior в этом опыте не вычисляется.
- Фотографии и их производные: [атрибуция и лицензия](assets/PETS_LICENSE.md).
- Результаты и SHA-256 архитектуры, весов и входов: `assets/pets_execution.json`.
- Условная генерация по метке демонстрируется отдельно на cVAE, cGAN, условном RealNVP и DDPM с classifier-free guidance.
