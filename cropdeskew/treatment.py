"""
Date: 01/03/2021
Author: Axel Lebas
File: treatment.py

Image treatment.
"""

import imutils
import cv2
import numpy as np
from io import BytesIO


def resize_image(image):
    """
    Resizes the image to later preprocess it better, saving a copy of the original image and the ratio of resizing
    Attributes:
        orig: the original image
        ratio: the ratio of resizing
        image_resized: resized image
    :param image: an image
    :return: a dictionary containing the original image, the resized image and the ratio of resizing
    """

    RESCALED_HEIGHT = 500

    ratio = image.shape[0] / RESCALED_HEIGHT
    orig = image.copy()
    image_resized = imutils.resize(image, height=int(RESCALED_HEIGHT))

    return {"ratio": ratio, "orig": orig, "image": image_resized}


def grayscale(image):
    return cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)


def canny(image):
    return cv2.Canny(image, 0, 84)


def dilate(image):
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (9, 9))
    return cv2.morphologyEx(image, cv2.MORPH_CLOSE, kernel)


def blur(image):
    return cv2.GaussianBlur(image, (7, 7), 0)


def treat_image(image):
    """
    treat the image to later find the borders
    :param image: an image
    :return: the image but treated
    """
    image = grayscale(image)
    image = blur(image)
    image = dilate(image)
    image = canny(image)

    return image


def file_to_img(file):
    """ werkzeug file storage to image
    Converts a werkzeug file storage to an opencv image
    :param file: a file storage
    :return: an opencv image
    """

    file_str = file.read()
    np_img = np.fromstring(file_str, np.uint8)
    img = cv2.imdecode(np_img, cv2.IMREAD_COLOR)

    return img


def img_to_bytes(img):
    """ OpenCV image to bytes
    :param img: an opencv image
    :return: bytes
    """

    _, bytes_frame = cv2.imencode('.jpg', img)
    bytes_frame = BytesIO(bytes_frame.tostring())
    return bytes_frame
