import os

os.environ["TF_USE_LEGACY_KERAS"] = "1"
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "2"
os.environ["TF_ENABLE_ONEDNN_OPTS"] = "0"

from pathlib import Path

import cv2
import numpy as np
from deepface import DeepFace


def target_emb(dir: str):  # dir is directory of target images
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
