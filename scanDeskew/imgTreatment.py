"""
Date: 01/03/2021
Author: Axel Lebas
File: imgTreatment.py

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
    ratio = image.shape[0] / 500
    orig = image.copy()
    image_resized = imutils.resize(image, height=500)

    return {
        "ratio": ratio,
        "orig": orig,
        "image": image_resized
    }


def grayscale(image):
    return cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)


def dilate(image, kernel=np.ones((1, 1), np.uint8)):
    return cv2.dilate(image, kernel, iterations=5)


def blur(image):
    return cv2.GaussianBlur(image, (3, 3), 0)


def erode(image, kernel=np.ones((1, 1), np.uint8)):
    return cv2.erode(image, kernel, iterations=5)


def treat_image(image):
    """
    treat the image to later find the borders
    :param image: an image
    :return: the image but treated
    """
    # return erode(blur(dilate(grayscale(image))))

    image = grayscale(image)
    image = dilate(image)
    image = blur(image)
    image = erode(image)

    return image


def file_to_img(file):
    """ filestorage to image
    Converts a filestorage to an opencv image
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
