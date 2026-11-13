"""Colab adaptation of HSE-LAMBDA's CelebADataset.

Source: DeepGenerativeModels/seminars/utils/datasets/celeba.py
Revision: 42dd64b535f5bedf1d2b3db28f2f21ce96c9f09f
Copyright (c) 2024 Laboratory of methods for Big Data Analysis at HSE.
MIT License; see assets/HSE_LAMBDA_LICENSE.txt.

The original dataset interface is preserved. Data is supplied explicitly by the
instructor instead of downloaded from the upstream Google Drive links.
"""

from pathlib import Path, PurePosixPath
import shutil
import stat
import tempfile
import zipfile

from PIL import Image
import torch
from torch.utils.data import Dataset


def extract_celeba_zip(archive_path, destination):
    """Extract provided files atomically and reject paths outside destination."""
    destination = Path(destination).expanduser().resolve()
    destination.mkdir(parents=True, exist_ok=True)
    with zipfile.ZipFile(Path(archive_path).expanduser()) as archive:
        for member in archive.infolist():
            relative = PurePosixPath(member.filename.replace("\\", "/"))
            if relative.is_absolute() or ".." in relative.parts:
                raise ValueError(f"Invalid archive path: {member.filename}")
            if any(part.startswith(".") or part == "__MACOSX" for part in relative.parts):
                continue
            target = destination.joinpath(*relative.parts).resolve()
            if not target.is_relative_to(destination) or stat.S_ISLNK(member.external_attr >> 16):
                raise ValueError(f"Invalid archive member: {member.filename}")
            if member.is_dir():
                target.mkdir(parents=True, exist_ok=True)
                continue
            target.parent.mkdir(parents=True, exist_ok=True)
            with tempfile.NamedTemporaryFile(dir=target.parent, delete=False) as output:
                temporary = Path(output.name)
                try:
                    with archive.open(member) as source:
                        shutil.copyfileobj(source, output)
                except BaseException:
                    temporary.unlink(missing_ok=True)
                    raise
            temporary.replace(target)


def find_celeba_files(search_root, attributes_path=None):
    """Locate the upstream image directory and its TXT attributes."""
    search_root = Path(search_root).expanduser()
    image_directories = sorted(
        folder for folder in search_root.rglob("img_align_celeba")
        if folder.is_dir() and any(folder.glob("*.jpg"))
    )
    if search_root.name == "img_align_celeba" and any(search_root.glob("*.jpg")):
        image_directories.insert(0, search_root)
    if len(image_directories) != 1:
        raise FileNotFoundError(
            f"Expected one img_align_celeba directory with JPG images under {search_root}; "
            f"found {len(image_directories)}. Set DATA_DIR to the extracted CelebA directory."
        )
    if attributes_path:
        attribute_file = Path(attributes_path).expanduser()
    else:
        candidates = sorted(search_root.rglob("list_attr_celeba.txt"))
        if len(candidates) != 1:
            raise FileNotFoundError(
                f"Found {len(candidates)} list_attr_celeba.txt files. "
                "Extract the attributes ZIP or specify ATTR_PATH."
            )
        attribute_file = candidates[0]
    if not attribute_file.is_file():
        raise FileNotFoundError(f"Attribute file not found: {attribute_file}")
    return image_directories[0].parent, attribute_file


class CelebADataset(Dataset):
    """HSE-LAMBDA dataset interface with explicit local images and attributes."""

    def __init__(self, root_dir="../data/celeba", transform=None, attributes_path=None):
        self.root_dir = Path(root_dir).expanduser()
        self.transform = transform
        self.dataset_folder = self.root_dir / "img_align_celeba"
        self.filenames = sorted(
            path.name for path in self.dataset_folder.glob("*")
            if path.is_file() and path.suffix.lower() in {".jpg", ".jpeg", ".png"}
        )
        if not self.filenames:
            raise FileNotFoundError(f"No images found in {self.dataset_folder}")
        attribute_file = (
            Path(attributes_path).expanduser() if attributes_path
            else self.root_dir / "annotations" / "list_attr_celeba.txt"
        )
        self.annotations = []
        with attribute_file.open(encoding="utf-8-sig") as stream:
            expected_count = int(next(stream).strip())
            self.header = next(stream).split()
            for line in stream:
                parts = line.split()
                if not parts:
                    continue
                values = [int(value) for value in parts[1:]]
                if len(values) != len(self.header) or not set(values).issubset({-1, 1}):
                    raise ValueError(f"Invalid attributes for {parts[0]}")
                self.annotations.append((parts[0], values))
        if len(self.annotations) != expected_count:
            raise ValueError("Attribute row count does not match the file header.")
        self.attr_map = {
            filename: torch.tensor(values, dtype=torch.long)
            for filename, values in self.annotations
        }
        if len(self.attr_map) != len(self.annotations):
            raise ValueError("Duplicate filenames in the attributes file.")
        missing = [name for name in self.filenames if name not in self.attr_map]
        if missing:
            raise ValueError(f"Missing attributes for {len(missing)} images, including {missing[:3]}")

    def __len__(self):
        return len(self.filenames)

    def __getitem__(self, idx):
        filename = self.filenames[idx]
        with Image.open(self.dataset_folder / filename) as source:
            image = source.convert("RGB")
        if self.transform is not None:
            image = self.transform(image)
        return image, {
            "filename": filename,
            "idx": idx,
            "attributes": self.attr_map[filename],
        }
