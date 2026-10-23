import os

os.environ["TF_USE_LEGACY_KERAS"] = "1"
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "2"
os.environ["TF_ENABLE_ONEDNN_OPTS"] = "0"

from pathlib import Path

import cv2
import numpy as np
from deepface import DeepFace


def target_emb(dir: str) -> list:  # dir is directory of target images
    target_dir = Path(dir)
    target_emb_list = []
    for img in target_dir.iterdir():
        if img.is_file():
            img_array = cv2.imread(img)
            rgb_img_array = cv2.cvtColor(img_array, cv2.COLOR_BGR2RGB)

            rep = DeepFace.represent(
                align=True,
                img_path=rgb_img_array,
                model_name="ArcFace",
                enforce_detection=False,
                detector_backend="retinaface",
            )

            target_emb_list.append(np.array(rep[0]["embedding"]))
    return target_emb_list


def load_img(dir: str) -> list:
    img_dir = Path(dir)
    valid_img = [
        p
        for p in img_dir.iterdir()
        if p.is_file() and p.suffix.lower() in {".jpg", ".jpeg", ".png"}
    ]
    return valid_img

def makecopy(main_dir: str, )


def comparison(target_dir: str, img_dir: str):
    valid_img = load_img(img_dir)
    face_emb = target_emb(target_dir)

    threshold = 0.50
    matched_count = 0

    for img in valid_img:
        bgr_img = cv2.imread(img)
        rgb_img = cv2.cvtcolor(bgr_img, cv2.COLOR_BGR2RGB)

        try:
            rep = DeepFace.represent(
                align=True,
                img_path=rgb_img,
                model_name="ArcFace",
                enforce_detection=False,
                detector_backend="retinaface",
            )

            is_match = False
            for face in rep:
                face_emb = np.array(face["embedding"])

                for target_face_emb in face_emb:
                    cos_sin = np.dot(face_emb, target_face_emb) / (
                        np.linalg.norm(face_emb) * np.linalg.norm(target_face_emb)
                    )

                    distace = 1 - cos_sin
                    if distace <= threshold:
                        is_match = True
                        break
                if is_match:
                    break
            if is_match:
                
                destination = 
                pass
        except Exception:
            pass
