import os

os.environ["TF_USE_LEGACY_KERAS"] = "1"
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "2"
os.environ["TF_ENABLE_ONEDNN_OPTS"] = "0"

import queue
import shutil
import threading
from pathlib import Path

import cv2
import numpy as np
from deepface import DeepFace
from dotenv import load_dotenv

image_queue = queue.Queue(maxsize=20)
result_queue = queue.Queue()


def target_emb(dir: str) -> list:  # dir is directory of target images
    target_dir = Path(dir)
    target_emb_list = []
    for img in target_dir.iterdir():
        if img.is_file():
            img_array = cv2.imread(img)
            if img_array is None:
                continue

            rep = DeepFace.represent(
                align=True,
                img_path=img_array,
                model_name="ArcFace",
                enforce_detection=False,
                detector_backend="retinaface",
            )

            target_emb_list.append(np.array(rep[0]["embedding"]))  # type: ignore
    return target_emb_list


def target_emb_master(dir: str) -> list:
    target_dir = Path(dir)
    master_target_emd_list = []
    for directory in target_dir.iterdir():
        if directory.is_dir():
            target_emb_list = []
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
                    )

                    target_emb_list.append(np.array(rep[0]["embedding"]))  # type: ignore

            if len(target_emb_list) > 1:
                master_target_emd_list.append(np.mean(target_emb_list, axis=0))
            elif len(target_emb_list) == 1:
                master_target_emd_list.append(target_emb_list[0])

    return master_target_emd_list


def load_img(dir: str) -> list:
    img_dir = Path(dir)
    valid_img = [
        p
        for p in img_dir.iterdir()
        if p.is_file() and p.suffix.lower() in {".jpg", ".jpeg", ".png"}
    ]
    return valid_img


def makecopy(matched_images: queue.Queue):
    matches_dir = Path("matches")
    matches_dir.mkdir(exist_ok=True)
    while not matched_images.empty():
        img = matched_images.get()
        destination = matches_dir / img.name
        shutil.copy(img, destination)


def producer_comparison(img_dir: str):
    valid_img = load_img(img_dir)

    for img in valid_img:
        bgr_img = cv2.imread(img)
        if bgr_img is None:
            continue
        image_queue.put((bgr_img, img))
    image_queue.put((None, None))


def consumer_comparison(target_dir: str, results_queue: queue.Queue):
    target_embeddings = target_emb_master(target_dir)

    threshold = 0.50

    while True:
        bgr_img, img = image_queue.get()
        if bgr_img is None:
            break

        try:
            rep = DeepFace.represent(
                align=True,
                img_path=bgr_img,
                model_name="ArcFace",
                enforce_detection=False,
                detector_backend="retinaface",
            )

            is_match = False
            for face in rep:
                album_emb = np.array(face["embedding"])  # type: ignore

                for target_face_emb in target_embeddings:
                    cos_sin = np.dot(album_emb, target_face_emb) / (
                        np.linalg.norm(album_emb) * np.linalg.norm(target_face_emb)
                    )

                    distace = 1 - cos_sin
                    if distace <= threshold:
                        is_match = True
                        break
                if is_match:
                    break
            if is_match:
                result_queue.put(img)
                print(f"Match Found: {img.name}")
        except Exception as e:
            print(f"Error processing {img.name}: {e}")


if __name__ == "__main__":
    load_dotenv()

    target_dir_path = os.getenv("TARGET_DIR")
    album_dir_path = os.getenv("IMG_DIR")

    producer = threading.Thread(target=producer_comparison, args=(album_dir_path,))
    consumer = threading.Thread(
        target=consumer_comparison, args=(target_dir_path, result_queue)
    )

    print("Starting the face matching pipeline")

    producer.start()
    consumer.start()

    producer.join()
    consumer.join()

    print(f"Scaning complete! Found {result_queue.qsize()} matches")

    if result_queue:
        print("Creating copies to matches directory")
        makecopy(result_queue)
        print("Done")
    else:
        print("Final Matches was empty")
