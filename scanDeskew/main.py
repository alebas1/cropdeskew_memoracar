"""
Date: 1/03/2021
Author: Axel Lebas
File: main.py

Essential algorithms to deskew and scan files
"""

import numpy
import cv2
from io import BytesIO


def file_to_img(file):
    """ filestorage to image
    Converts a filestorage to an opencv image
    :param file: a file storage
    :return: an opencv image
    """
    file_str = file.read()
    np_img = numpy.fromstring(file_str, numpy.uint8)
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


def scan_and_deskew(file):
    """ Scan and deskew
    deskew and stretch a file
    :param file: a filestorage
    :return: the deskewed filestorage
    """

    # convert the filestorage to opencv img
    img = file_to_img(file)

    # TODO: integrate scanImg module
    # scan the opencv image
    scanned_img = img

    # convert the scanned opencv image to bytes
    scanned_file = img_to_bytes(scanned_img)

    return scanned_file
