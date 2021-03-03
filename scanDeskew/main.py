"""
Date: 1/03/2021
Author: Axel Lebas
File: main.py

Essential algorithms to deskew and scan files
"""

from scanDeskew.imgTreatment import file_to_img, resize_image, treat_image, img_to_bytes
from scanDeskew.scan import find_edges, transform_image_4_pts


def scan_and_deskew(file):
    """ Scan and deskew
    deskew and stretch a file
    :param file: a werkzeug file storage
    :return: the de-skewed werkzeug file storage
    """

    # convert the werkzeug file storage to opencv img
    image = file_to_img(file)

    # TODO: integrate scanImg module
    # scan the opencv image
    resize = resize_image(image)

    ratio = resize.get("ratio")
    orig = resize.get("orig")
    image = resize.get("image")

    image = treat_image(image)

    edges = find_edges(image)
    scanned_image = transform_image_4_pts(orig, edges * ratio)
    scanned_file = img_to_bytes(scanned_image)
    return scanned_file
