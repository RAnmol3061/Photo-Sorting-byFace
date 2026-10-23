# Automated Photo Sorter

A GPU-accelerated Python pipeline designed to automatically filter and sort high-resolution wedding album photos by matching them against target family members. 

By leveraging **DeepFace** and the **ArcFace** model, this script mathematically reduces false positives caused by family genetic similarities, accurately extracting specific individuals from massive unorganized photo dumps.

## Features
* **High-Accuracy Matching:** Utilizes ArcFace's angular margin loss to distinguish closely related family members (e.g., aunts vs. cousins) better than standard Facenet models.
* **Dynamic Target Loading:** Automatically scans and generates master embeddings from any reference images dropped into the `Target/` directory.
* **Secure Configuration:** Local network paths and sensitive directories are abstracted using `python-dotenv`.
* **Modular Pipeline:** Built with modern Python `pathlib` for robust, cross-OS file management.

## Tech Stack
* **Language:** Python 3
* **Computer Vision:** DeepFace, OpenCV (`cv2`), RetinaFace (Detection)
* **Data Processing:** NumPy
* **Environment Management:** `python-dotenv`

## Setup & Installation

**1. Clone the repository**
```bash
git clone git@github.com:RAnmol3061/Photo-Sorting-byFace.git
cd Photo-Sorting-byFace
```

**2. Install Dependencies**
```bash
pip install deepface opencv-python numpy python-dotenv
```

**3. Configure your environment variables**

Because local paths are kept private, you must create a `.env` file in the root directory and define your specific photo directories:
```.env
TARGET_DIR="/path/to/your/Target/folder"
IMG_DIR="/path/to/your/Photos/album"
```
## Usage

1. Place reference photos of the people you want to find inside your designated Target/ folder.

2. Run the main pipeline:
```python
python main.py  
```

3. The script will scan the album and automatically copy any verified matches into a newly generated `matches/` directory.