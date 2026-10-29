import argparse
import os
import queue
import shutil
import sys
import threading
from pathlib import Path

os.environ["TF_USE_LEGACY_KERAS"] = "1"
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "2"
os.environ["TF_ENABLE_ONEDNN_OPTS"] = "0"

import cv2
import numpy as np
from deepface import DeepFace
from dotenv import load_dotenv


def target_emb_master(main_path: Path) -> list[float]:
    target_directory = Path(main_path)
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

    return master_target_embeddings


def load_image(main_path: str) -> list[Path]:
    image_directory = Path(main_path)
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


def producer_comparison(img_directory: str, image_queue: queue.Queue[tuple]) -> None:
    try:
        valid_image_path = load_image(img_directory)

        for image_path in valid_image_path:
            bgr_image_array = cv2.imread(image_path)
            if bgr_image_array is None:
                continue
            image_queue.put((bgr_image_array, image_path))
    except Exception as e:
        print(f"Error: {e}")
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
        except Exception as e:
            print(f"Error processing {image_path.name}: {e}")


def validate_cli_dir(cli_flag_value: Path | None, default_value: str | None) -> Path:
    if cli_flag_value is not None:
        try:
            return cli_flag_value.resolve(strict=True)
        except FileNotFoundError:
            print("Invalid CLI Argument")
    if default_value is not None:
        try:
            return Path(default_value).resolve(strict=True)
        except FileNotFoundError:
            print("Invalid Default value. Exiting")
            sys.exit(1)
    print("Error: No cli argument and empty default. Exiting")
    sys.exit(1)


def main() -> int:
    load_dotenv()

    image_queue = queue.Queue(maxsize=20)
    result_queue = queue.Queue()

    target_directory_path = os.getenv("TARGET_DIR", default=None)
    album_directory_path = os.getenv("IMG_DIR", default=None)

    # if not target_directory_path or not album_directory_path:
    # raise ValueError("Check your .env file for TARGET_DIR & IMG_DIR paths")

    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--target",
        type=Path,
        # default=target_directory_path,
        help="Directory of Target Images",
    )
    parser.add_argument(
        "--album",
        type=Path,
        # default=album_directory_path,
        help="Directory of images to scan through",
    )
    parser.add_argument(
        "--threshold",
        type=float,
        default=0.50,
        help="Threshold value for confidence",
    )

    args = parser.parse_args()

    args_target = validate_cli_dir(args.target, target_directory_path)
    args_album = validate_cli_dir(args.album, album_directory_path)

    target_embeddings = target_emb_master(args_target)

    producer = threading.Thread(
        target=producer_comparison,
        args=(args_album, image_queue),
    )
    consumer = threading.Thread(
        target=consumer_comparison,
        args=(target_embeddings, result_queue, image_queue, args.threshold),
    )

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
