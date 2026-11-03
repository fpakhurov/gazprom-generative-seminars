"""Load native-resolution MNIST with a small verified download cache."""

from pathlib import Path
import gzip
import hashlib
import struct
import urllib.request

import numpy as np
from sklearn.model_selection import train_test_split

MNIST_FILES = {
    "train-images-idx3-ubyte.gz": "f68b3c2dcbeaaa9fbdd348bbdeb94873",
    "train-labels-idx1-ubyte.gz": "d53e105ee54ea40749a09fcbcd1e9432",
    "t10k-images-idx3-ubyte.gz": "9fb629c4189551a2d022fa330f9573f3",
    "t10k-labels-idx1-ubyte.gz": "ec29112dd5afa0611ce80d1b7f02629c",
}


def load_mnist(cache="data/mnist", train_limit=12000, test_limit=2000, seed=42):
    cache = Path(cache)
    cache.mkdir(parents=True, exist_ok=True)
    arrays = {}
    for filename, expected in MNIST_FILES.items():
        path = cache / filename
        if not path.exists():
            url = "https://storage.googleapis.com/cvdf-datasets/mnist/" + filename
            with urllib.request.urlopen(url, timeout=90) as response:
                path.write_bytes(response.read())
        if hashlib.md5(path.read_bytes()).hexdigest() != expected:
            raise ValueError(f"MNIST checksum mismatch: {filename}; remove the cached file and rerun")
        payload = gzip.decompress(path.read_bytes())
        magic, count = struct.unpack(">II", payload[:8])
        if magic == 2051:
            rows, cols = struct.unpack(">II", payload[8:16])
            assert (rows, cols) == (28, 28)
            values = np.frombuffer(payload, dtype=np.uint8, offset=16).reshape(count, rows, cols)
        elif magic == 2049:
            values = np.frombuffer(payload, dtype=np.uint8, offset=8)
            assert len(values) == count
        else:
            raise ValueError(f"Unexpected MNIST magic: {magic}")
        arrays[filename] = values

    def subset(images, labels, limit):
        assert len(images) == len(labels)
        indices, _ = train_test_split(np.arange(len(labels)), train_size=limit,
                                      stratify=labels, random_state=seed)
        return images[indices].astype("float32").reshape(-1, 784) / 255.0, labels[indices].astype("int64")

    train_x, train_y = subset(arrays["train-images-idx3-ubyte.gz"], arrays["train-labels-idx1-ubyte.gz"], train_limit)
    test_x, test_y = subset(arrays["t10k-images-idx3-ubyte.gz"], arrays["t10k-labels-idx1-ubyte.gz"], test_limit)
    return train_x, test_x, train_y, test_y
