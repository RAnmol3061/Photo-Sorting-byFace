import argparse
import logging
import os
import queue
import sys
import threading
from pathlib import Path

from dotenv import load_dotenv

from comparison import consumer_comparison, producer_comparison
from input_validation import validate_cli_dir
from my_postprocess import make_copy
from my_preprocess import target_emb_master


def main() -> int:
    logging.basicConfig(
        level=logging.DEBUG,
        format="%(asctime)s | %(levelname)-8s | %(name)s:%(funcName)s:%(lineno)d - %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
        filename="script.log",
        filemode="w",
    )

    logger = logging.getLogger(__name__)
    logger.debug("Program started")
    load_dotenv()

    image_queue = queue.Queue(maxsize=20)
    result_queue = queue.Queue()
    total_no_images = queue.Queue()

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
        default=False,
        help="Enable or disable use of cached target embeddings",
    )

    args = parser.parse_args()

    args_target = validate_cli_dir(args.target, target_directory_path)
    args_album = validate_cli_dir(args.album, album_directory_path)

    if args_target is not Path:
        print("Value of args_target is not defined")
        return 1
    if args_album is not Path:
        print("Value of args_album not defined")
        return 1

    target_embeddings = target_emb_master(args_target, args.cache)

    producer = threading.Thread(
        target=producer_comparison,
        args=(args_album, image_queue, total_no_images),
        daemon=True,
    )
    consumer = threading.Thread(
        target=consumer_comparison,
        args=(
            target_embeddings,
            result_queue,
            image_queue,
            args.threshold,
            total_no_images,
        ),
        daemon=True,
    )

    logger.debug("Pipeline starts now")
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
