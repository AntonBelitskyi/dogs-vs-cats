import logging
import os
from functools import partial
from os.path import isfile, join, basename, exists, isdir

import PIL
import click
import tensorflow as tf
from PIL import Image
from tqdm.contrib.concurrent import process_map

from predicts import get_prediction_model
from utils import check_mimetype, WrongMimeTypeError


tf.get_logger().setLevel('ERROR')


def init_logger(predict_model):
    logging.basicConfig(filename=f'logs/{predict_model}.log',
                        filemode='a',
                        format='%(message)s',
                        level=logging.INFO)
    logger = logging.getLogger()
    return logger


def determine_image(prediction_model, input_path, logger, image_name):
    filepath = join(input_path, image_name)
    # Check image extension (it should be JPEG or PNG)
    try:
        check_mimetype(filepath)
    except WrongMimeTypeError:
        logger.info(f"{'*' * 50}\n{basename(image_name)} - unsupported_file\n{'*' * 50}")
    try:
        image = Image.open(filepath)
        result = prediction_model.predict(image)
        logger.info(f"{'*' * 50}\n{image_name} - {result}\n{'*' * 50}")
    except PIL.UnidentifiedImageError as e:
        logger.info(f"{'*' * 50}\n{basename(image_name)} - cannot identify your image\n{'*' * 50}")




@click.command()
@click.option("-i", "--input-path", required=True)
@click.option("-pm", "--predict-model", required=True)
def execute(predict_model, input_path):
    prediction_model = get_prediction_model(predict_model)
    if not prediction_model:
        raise Exception(f"Prediction model with the name of the '{predict_model}' does not exist")

    if not exists(input_path):
        raise Exception(f"Images path does not exist")
    logger = init_logger(predict_model)
    prediction_model = prediction_model()
    if isdir(input_path):
        # Get all files from directory
        images = [f for f in os.listdir(input_path) if isfile(join(input_path, f))]
        images.sort()
        process_map(
            partial(determine_image, *(prediction_model, input_path, logger)),
            images,
            max_workers=os.cpu_count()
        )
    else:
        try:
            check_mimetype(input_path)
            image = Image.open(input_path)
        except WrongMimeTypeError as e:
            logger.info(f"{'*' * 50}\n{basename(input_path)} - unsupported_file\n{'*' * 50}")
            raise e
        except PIL.UnidentifiedImageError as e:
            logger.info(f"{'*' * 50}\n{basename(input_path)} - cannot identify your image\n{'*' * 50}")
            raise e
        result = prediction_model.predict(image)
        logger.info(f"{'*' * 50}\n{basename(input_path)} - {result}\n{'*' * 50}")


if __name__ == "__main__":
    execute()
