"""Run the optional official pretrained CycleGAN photo experiment.

Architecture code is downloaded from an immutable upstream commit; it is not
included in this package. The source input is the left panel of upstream's
horse2zebra demonstration, first frame. No original training is performed.
"""

import argparse
import hashlib
import importlib.util
import json
from pathlib import Path
import time
import urllib.request

import numpy as np
from PIL import Image
import torch


REPO_SHA = "2a7afba2895d52556dd5dfe07e8555ef657ced6f"
SOURCE_HASH = "2bfcb7adb19f7b7ff6fd8dd76e313200acf000e96784b920658a62e1a11fc04f"
SOURCE_URL = f"https://raw.githubusercontent.com/junyanz/pytorch-CycleGAN-and-pix2pix/{REPO_SHA}/models/networks.py"
MODEL_URL = "https://efrosgans.eecs.berkeley.edu/cyclegan/pretrained_models/horse2zebra.pth"
MODEL_HASH = "5bfc34b154035be1cd592cd273fdf328810bee4756044c9cf7deb8d4b6d2fadb"
IMAGE_URL = f"https://raw.githubusercontent.com/junyanz/pytorch-CycleGAN-and-pix2pix/{REPO_SHA}/imgs/horse2zebra.gif"


def file_hash(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def download(url, target, expected_hash=None):
    target = Path(target)
    if not target.exists():
        temporary = target.with_suffix(target.suffix + ".download")
        with urllib.request.urlopen(url, timeout=90) as response, temporary.open("wb") as output:
            while chunk := response.read(1024 * 1024):
                output.write(chunk)
        temporary.replace(target)
    if expected_hash is not None and file_hash(target) != expected_hash:
        raise ValueError(f"Checksum mismatch: {target.name}")
    return target


def translate(image, cache, device="cpu"):
    cache = Path(cache)
    cache.mkdir(parents=True, exist_ok=True)
    source = download(SOURCE_URL, cache / "networks.py", SOURCE_HASH)
    weights_path = download(MODEL_URL, cache / "horse2zebra.pth", MODEL_HASH)
    spec = importlib.util.spec_from_file_location("official_cyclegan_networks", source)
    networks = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(networks)
    network = networks.ResnetGenerator(
        3, 3, ngf=64,
        norm_layer=torch.nn.InstanceNorm2d,
        use_dropout=False, n_blocks=9,
    )
    weights = torch.load(weights_path, map_location="cpu", weights_only=True)
    # Older InstanceNorm checkpoints contain buffers that current PyTorch omits.
    weights = {
        key: value for key, value in weights.items()
        if not key.endswith(("running_mean", "running_var", "num_batches_tracked"))
    }
    network.load_state_dict(weights, strict=True)
    network = network.to(device).eval()
    pixels = np.array(image.convert("RGB"), dtype=np.float32) / 127.5 - 1
    tensor = torch.from_numpy(pixels).permute(2, 0, 1).unsqueeze(0).to(device)
    started = time.perf_counter()
    with torch.inference_mode():
        result = network(tensor).clamp(-1, 1)
    elapsed = time.perf_counter() - started
    result = ((result[0].permute(1, 2, 0).cpu().numpy() + 1) * 127.5).round().astype("uint8")
    return Image.fromarray(result), elapsed


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--cache", required=True)
    parser.add_argument("--output", required=True)
    arguments = parser.parse_args()
    torch.set_num_threads(2)
    cache = Path(arguments.cache)
    cache.mkdir(parents=True, exist_ok=True)
    gif_path = download(IMAGE_URL, cache / "horse2zebra.gif")
    with Image.open(gif_path) as animation:
        frame = animation.convert("RGB")
        image = frame.crop((0, 0, frame.width // 2, frame.height)).resize((448, 256))
    prediction, seconds = translate(image, cache)
    output = Path(arguments.output)
    output.mkdir(parents=True, exist_ok=True)
    image.save(output / "horse_input.png")
    prediction.save(output / "zebra_prediction.png")
    import matplotlib.pyplot as plt

    fig, axes = plt.subplots(1, 2, figsize=(10, 3.4))
    for axis, picture, title in zip(axes, [image, prediction], ["Input photograph", "Our pretrained CycleGAN inference"]):
        axis.imshow(picture)
        axis.set_title(title)
        axis.axis("off")
    fig.tight_layout()
    fig.savefig(output / "cyclegan_photo.png", dpi=150, bbox_inches="tight")
    report = {
        "architecture_commit": REPO_SHA,
        "architecture_sha256": file_hash(cache / "networks.py"),
        "weights_url": MODEL_URL,
        "weights_sha256": file_hash(cache / "horse2zebra.pth"),
        "input_source": IMAGE_URL,
        "input_transformation": "first frame, left half only, resized to 448x256",
        "input_shape": [1, 3, 256, 448],
        "forward_seconds_cpu": round(seconds, 3),
        "training_performed": False,
        "pytorch": torch.__version__,
    }
    (output / "cyclegan_execution.json").write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(json.dumps(report))
