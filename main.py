import argparse
import logging
import os
import queue
import shutil
import sys
import threading
from pathlib import Path

import numpy as np
from dotenv import load_dotenv

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
        list[Path]: A list containing path objects of .jpg, .jpeg, .png types"""

    image_directory = main_path
    valid_image = [
        p
        for p in image_directory.iterdir()
        if p.is_file() and p.suffix.lower() in {".jpg", ".jpeg", ".png"}
    ]
    return valid_image


def make_copy(matched_images: queue.Queue[Path]) -> None:
    matches_directory = Path("matches")
    matches_directory.mkdir(exist_ok=True)
    while not matched_images.empty():
        image_path = matched_images.get()
        destination = matches_directory / image_path.name
        shutil.copy(image_path, destination)


def producer_comparison(img_directory: Path, image_queue: queue.Queue[tuple]) -> None:
    try:
        valid_image_path = load_image(img_directory)

        for image_path in valid_image_path:
            bgr_image_array = cv2.imread(image_path)
            if bgr_image_array is None:
                logging.warning(
                    "Album File %s was unable to be read by imread", image_path
                )
                continue
            image_queue.put((bgr_image_array, image_path))
    except Exception:
        logging.exception("Something went wrong in producer_comparison thread")
    finally:
        image_queue.put((None, None))


def consumer_comparison(
    target_embeddings: list,
    result_queue: queue.Queue[Path],
    image_queue: queue.Queue[tuple],
    threshold: float,
) -> None:

    while True:
        bgr_image_array, image_path = image_queue.get()
        if bgr_image_array is None:
            break

        try:
            rep = DeepFace.represent(
                align=True,
                img_path=bgr_image_array,
                model_name="ArcFace",
                enforce_detection=False,
                detector_backend="retinaface",
                l2_normalize=True,
            )

            is_match = False
            for face in rep:
                album_embedding = np.array(face["embedding"])  # type: ignore

                for target_face_embedding in target_embeddings:
                    cos_sim = np.dot(album_embedding, target_face_embedding)
                    distance = 1 - cos_sim
                    if distance <= threshold:
                        is_match = True
                        break
                if is_match:
                    break
            if is_match:
                result_queue.put(image_path)
                print(f"Match Found: {image_path.name}")
                logging.debug(f"Match Found: {image_path.name}")
        except Exception as e:
            logging.error("Some error in consumer comparison thread.")
            print(f"Error processing {image_path.name}: {e}")


def validate_cli_dir(cli_flag_value: Path | None, default_value: str | None) -> Path:
    if cli_flag_value is not None:
        try:
            return cli_flag_value.resolve(strict=True)
        except FileNotFoundError:
            logging.warning("Invalid CLI Argument, moving to default values(env)")
            print("Invalid CLI Argument, moving to default values(env)")
    if default_value is not None:
        try:
            return Path(default_value).resolve(strict=True)
        except FileNotFoundError:
            logging.critical("Invalid Default value in .env file. Exiting")
            print("Invalid Default value. Exiting")
            sys.exit(1)
    logging.critical("Exiting: Due to no cli & default arguments")
    print("Error: No cli argument and empty default. Exiting")
    sys.exit(1)


def main() -> int:
    logging.basicConfig(
        level=logging.DEBUG,
        format="%(asctime)s | %(levelname)-8s | %(name)s:%(funcName)s:%(lineno)d - %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
        filename="script.log",
        filemode="w",
    )

    load_dotenv()

    image_queue = queue.Queue(maxsize=20)
    result_queue = queue.Queue()

    target_directory_path = os.getenv("TARGET_DIR", default=None)
    album_directory_path = os.getenv("IMG_DIR", default=None)

    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--target",
        type=Path,
        help="Directory of Target Images",
    )
    parser.add_argument(
        "--album",
        type=Path,
        help="Directory of images to scan through",
    )
    parser.add_argument(
        "--threshold",
        type=float,
        default=0.50,
        help="Threshold value for confidence",
    )
    parser.add_argument(
        "--cache",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Enable or disable use of cached target embeddings",
    )

    args = parser.parse_args()

    args_target = validate_cli_dir(args.target, target_directory_path)
    args_album = validate_cli_dir(args.album, album_directory_path)

    target_embeddings = target_emb_master(args_target, args.cache)

    producer = threading.Thread(
        target=producer_comparison,
        args=(args_album, image_queue),
        daemon=True,
    )
    consumer = threading.Thread(
        target=consumer_comparison,
        args=(target_embeddings, result_queue, image_queue, args.threshold),
        daemon=True,
    )

    logging.debug("Pipeline starts now")
    print("Starting the face matching pipeline")

    producer.start()
    consumer.start()

    producer.join()
    consumer.join()

    print(f"Scanning complete! Found {result_queue.qsize()} matches")

    if not result_queue.empty():
        print("Creating copies to matches directory")
        make_copy(result_queue)
        print("Done")
    else:
        print("Final Matches was empty")
    return 0


if __name__ == "__main__":
    sys.exit(main())
