import logging
import queue
from pathlib import Path

import cv2
import numpy as np
from deepface import DeepFace
from tqdm import tqdm

from my_preprocess import load_image


def producer_comparison(
    img_directory: Path, image_queue: queue.Queue[tuple], total_no_images: queue.Queue
) -> None:
    try:
        valid_image_path = load_image(img_directory)
        total_no_images.put(len(valid_image_path))

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
    total_no_queues: queue.Queue,
) -> None:

    total_images = total_no_queues.get()
    with tqdm(total=total_images, desc="GPU Inference", unit="img") as pbar:
        target_matrix = np.asarray(target_embeddings)
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

                pbar.update(1)
                pbar.set_postfix(qsize=image_queue.qsize(), refresh=False)

                is_match = False
                for face in rep:
                    album_embedding = np.array(face["embedding"])  # type: ignore
                    cos_sim_matrix = album_embedding @ target_matrix.T
                    distances = 1.0 - cos_sim_matrix
                    is_match = bool(np.any(distances <= threshold))

                    if is_match:
                        break
                if is_match:
                    result_queue.put(image_path)
                    logging.debug("Match Found: %s", image_path.name)
            except Exception:
                logging.error("Some error in consumer comparison thread.")
                logging.warning(
                    "Probably something wrong with image: %s", image_path.name
                )
