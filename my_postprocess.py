import queue
import shutil
from pathlib import Path


def make_copy(matched_images: queue.Queue[Path]) -> None:
    matches_directory = Path("matches")
    matches_directory.mkdir(exist_ok=True)
    while not matched_images.empty():
        image_path = matched_images.get()
        destination = matches_directory / image_path.name
        shutil.copy(image_path, destination)
