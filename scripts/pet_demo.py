"""Reconstruct real cat/dog photos using a small pretrained image autoencoder."""

from pathlib import Path
import hashlib
import importlib.util
import json
import time
import urllib.request

import numpy as np
from PIL import Image, ImageOps
import torch

PET_MODEL_COMMIT = "fdbae2df989db95761f482efb8977e1182fb68d2"
PET_HASHES = {'taesd.py': '5943493eb291f768e0a87dd9b54c60461d3be0a9141003db7f62bd6b8e9efa6f', 'taesd_encoder.pth': '15bc6128f0ac51c673d3427216082ebfa62402ffc89763af74edfe259b32d49d', 'taesd_decoder.pth': '02873377c3f4659cd9f9adb2f718dcb434ba4c7bba3af3ee7bb95cbdabe2d3cf', 'cat.jpg': '74d09a7c6463a844572c58634a798d4fa1b646aeff8be81a42ab68915e54e049', 'dog.jpg': '1e71abf29876f248088a11211a350ed3d26a0c68aa34f6e45738771eb2d45a22'}
PET_URLS = {
    "taesd.py": f"https://raw.githubusercontent.com/madebyollin/taesd/{PET_MODEL_COMMIT}/taesd.py",
    "taesd_encoder.pth": f"https://raw.githubusercontent.com/madebyollin/taesd/{PET_MODEL_COMMIT}/taesd_encoder.pth",
    "taesd_decoder.pth": f"https://raw.githubusercontent.com/madebyollin/taesd/{PET_MODEL_COMMIT}/taesd_decoder.pth",
    "cat.jpg": "https://upload.wikimedia.org/wikipedia/commons/b/b6/Felis_catus-cat_on_snow.jpg",
    "dog.jpg": "https://upload.wikimedia.org/wikipedia/commons/6/6e/Golde33443.jpg",
}


def pet_download(name, cache):
    path = Path(cache) / name
    path.parent.mkdir(parents=True, exist_ok=True)
    if not path.exists():
        request = urllib.request.Request(PET_URLS[name], headers={"User-Agent": "EducationalSeminar/1.0"})
        with urllib.request.urlopen(request, timeout=90) as response:
            path.write_bytes(response.read())
    digest = hashlib.sha256(path.read_bytes()).hexdigest()
    expected = PET_HASHES.get(name)
    if expected and digest != expected:
        raise ValueError(f"Pet demo checksum mismatch: {name}")
    return path


def run_pet_demo(cache="data/pets", noise_levels=(0.0, 0.2, 0.6), seed=42):
    cache = Path(cache)
    files = {name: pet_download(name, cache) for name in PET_URLS}
    spec = importlib.util.spec_from_file_location("official_taesd", files["taesd.py"])
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    model = module.TAESD(encoder_path=str(files["taesd_encoder.pth"]), decoder_path=str(files["taesd_decoder.pth"]), latent_channels=4, image_channels=3).eval()
    originals = [ImageOps.fit(Image.open(files[name]).convert("RGB"), (256, 256), method=Image.Resampling.LANCZOS) for name in ["cat.jpg", "dog.jpg"]]
    x = torch.tensor(np.stack([np.asarray(im, dtype="float32") / 255 for im in originals])).permute(0, 3, 1, 2)
    generator = torch.Generator().manual_seed(seed)
    start = time.perf_counter()
    with torch.inference_mode():
        z = model.encoder(x)
        noise = torch.randn(z.shape, generator=generator)
        decoded = [model.decoder(z + float(level) * noise).clamp(0, 1) for level in noise_levels]
        weights = torch.linspace(0, 1, 5).view(-1, 1, 1, 1)
        latent_path = (1 - weights) * z[:1] + weights * z[1:2]
        interpolated = model.decoder(latent_path).clamp(0, 1)
    report = {
        "method": "TAESD deterministic distilled image autoencoder",
        "model_commit": PET_MODEL_COMMIT,
        "input_shape": list(x.shape),
        "latent_shape": list(z.shape),
        "noise_levels": list(noise_levels),
        "reconstruction_mse": [float((decoded[0][i] - x[i]).square().mean()) for i in range(2)],
        "seconds_cpu": round(time.perf_counter() - start, 3),
        "training_performed": False,
        "gaussian_posterior": False,
        "files": {name: {"url": PET_URLS[name], "sha256": hashlib.sha256(path.read_bytes()).hexdigest()} for name, path in files.items()},
    }
    import matplotlib.pyplot as plt

    fig, axes = plt.subplots(2, 1 + len(decoded), figsize=(12, 6), dpi=140)
    for row, label in enumerate(["Cat", "Dog"]):
        axes[row, 0].imshow(originals[row])
        axes[row, 0].set_title(label + ": original 256 x 256")
        for col, (level, result) in enumerate(zip(noise_levels, decoded), start=1):
            axes[row, col].imshow(result[row].permute(1, 2, 0).numpy())
            axes[row, col].set_title(f"D(E(x) + {level:.1f} eps)")
    for axis in axes.flat:
        axis.axis("off")
    fig.tight_layout()
    fig2, axes2 = plt.subplots(1, 5, figsize=(12, 2.7), dpi=140)
    for index, axis in enumerate(axes2):
        axis.imshow(interpolated[index].permute(1, 2, 0).numpy())
        axis.set_title(f"Latent mix: {index / 4:.2f}")
        axis.axis("off")
    fig2.suptitle("Latent interpolation is not class-conditional translation")
    fig2.tight_layout()
    return fig, fig2, report


if __name__ == "__main__":
    torch.set_num_threads(2)
    figures = run_pet_demo()
    output = Path(__file__).resolve().parents[1] / "assets"
    figures[0].savefig(output / "pets_reconstruction.jpg", dpi=140, pil_kwargs={"quality": 92})
    figures[1].savefig(output / "pets_interpolation.jpg", dpi=140, pil_kwargs={"quality": 92})
    (output / "pets_execution.json").write_text(json.dumps(figures[2], indent=2))
    print(json.dumps(figures[2]))
