import logging
import os
import sys
from pathlib import Path

import numpy as np

os.environ["TF_USE_LEGACY_KERAS"] = "1"
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "2"
os.environ["TF_ENABLE_ONEDNN_OPTS"] = "0"

import cv2  # pylint: disable=wrong-import-position
from deepface import DeepFace  # pylint: disable=wrong-import-position


def target_emb_master(main_path: Path, use_cache: bool) -> list[np.ndarray]:
    target_directory = main_path

    if use_cache:
        for file in target_directory.iterdir():
            if file.is_file() and file.suffix.lower() in {".npy"}:
                master_target_embeddings = np.load(file=file, allow_pickle=False)
                logging.debug("Loaded target_emb cache")
                return master_target_embeddings
        logging.warning("No target_emb cache found. Calculating target_emb now")

    master_target_embeddings = []
    for directory in target_directory.iterdir():
        if directory.is_dir():
            target_embeddings = []
            for target_files in directory.iterdir():
                if target_files.is_file() and target_files.suffix.lower() in {
                    ".jpg",
                    ".jpeg",
                    ".png",
                }:
                    img_array = cv2.imread(target_files)
                    if img_array is None:
                        logging.warning(
                            "Target File %s was unable to be read by imread",
                            target_files,
                        )
                        continue

                    rep = DeepFace.represent(
                        align=True,
                        img_path=img_array,
                        model_name="ArcFace",
                        enforce_detection=False,
                        detector_backend="retinaface",
                        l2_normalize=True,
                    )

                    target_embeddings.append(np.array(rep[0]["embedding"]))  # type: ignore

            if len(target_embeddings) > 1:
                mean_embeddings = np.mean(target_embeddings, axis=0)
                master_target_embeddings.append(
                    mean_embeddings / np.linalg.norm(mean_embeddings)
                )
            elif len(target_embeddings) == 1:
                master_target_embeddings.append(target_embeddings[0])
    if not master_target_embeddings:
        logging.critical(
            "Master_target_embedding list is empty. Incorrect target directory structure"
        )
        sys.exit(1)
    np.save(target_directory / "master_emb.npy", master_target_embeddings)
    logging.debug("saved master_target_embedding to disk")
    return master_target_embeddings


def load_image(main_path: Path) -> list[Path]:
    """Loads the album images and returns list of Path

    Argument must be a path object, otherwise function will throw a uncaught exception

    Args:
        main_path (Path): Path object containing the directory of album files

    Returns:
        list[Path]: A list containing path objects of .jpg, .jpeg, .png types
    """

    image_directory = main_path
    valid_image = [
        p
        for p in image_directory.iterdir()
        if p.is_file() and p.suffix.lower() in {".jpg", ".jpeg", ".png"}
    ]
    return valid_image
