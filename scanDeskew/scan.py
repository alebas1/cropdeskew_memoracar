"""
Date: 22/02/2021
Author: Axel Lebas
File: scan.py

All the necessary algorithms to scan an image.
"""

import cv2
import numpy as np


def find_edges(processed_image):
    """ Find edges
    Find edges of a processed image
    Attributes:
        image_edged: image with a canny filter
        contours: all the contours found in the image
        epsilon: maximum distance from contour to approximated contour (accuracy parameter)
                 [https://opencv-python-tutroals.readthedocs.io/en/latest/py_tutorials/py_imgproc/py_contours/py_contour_features/py_contour_features.html#contour-approximation}
        approx: approximation of a contour with the Douglas-Peucker algorithm
        edges: found edges of the document
    :param processed_image: a processed image (this works with a treatment in main.py
    :return coords of the edges of the document in the image
    """
    image_edged = cv2.Canny(processed_image, 30, 200)

    contours = cv2.findContours(image_edged, cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE)[0]
    # Optimisation - On récupère les 4 plus grand éléments
    contours = sorted(contours, key=cv2.contourArea, reverse=True)[:5]

    for c in contours:
        # approximer le contour
        epsilon = 0.01 * cv2.arcLength(c, True)
        approx = cv2.approxPolyDP(c, epsilon, True)

        # si on trouve un quadrilatere, alors on a une approximation
        if len(approx) == 4:
            edges = approx
            break

    if 'edges' in locals():
        # print(edges)
        '''cv2.drawContours(processed_image, [edges], -1, (0, 255, 0), 2)
        cv2.imshow("step 2: trouver les contours", processed_image)
        cv2.waitKey(0)
        cv2.destroyAllWindows()'''

        return edges.reshape(4, 2)
    else:
        raise Exception("Aucun bord n'a été trouvé")


def order_points(pts):
    """ Order points.
    Order the points in a certain order to be conform to the transform_image_4_pts algorithm
    Relevant order: top-left (tl), top-right (tr), bot-right (br), bot-left (bl)
    Attribute:
        rect: 4 * 2 matrix defining the points
    :param pts: four points (four pairs)
    :return ordered points
    """

    # order: top-left (tl), top-right (tr), bot-right (br), bot-left (bl)
    rect = np.zeros((4, 2), dtype="float32")

    # top-left has the smallest sum
    # bot-right has the biggest sum
    s = pts.sum(axis=1)
    rect[0] = pts[np.argmin(s)]
    rect[2] = pts[np.argmax(s)]

    # top-right has the smallest diff
    # bot-left has the largest diff
    diff = np.diff(pts, axis=1)
    rect[1] = pts[np.argmin(diff)]
    rect[3] = pts[np.argmax(diff)]

    # returns ordered points
    return rect


def transform_image_4_pts(image, pts):
    """ Transform image 4 points
    Stretches a part of an image (defined by 4 points)
    to every edges
    Attribute:
        rect: 4 * 2 matrix defining the points
        tl: top-left point
        tr: top-right point
        br: bottom-right
        bl: bottom-left
        max_width: estimated width of the stretched image
        max_height: estimated height of the stretched image
        dst: an array that defines the four points of a rectangle with a width of max_width-1 (because it starts from 0)
             and an height of max_height-1
        PT: perspective transform from four pairs of the corresponding points.
    :param pts: four points defining the part of the image to be stretched
    :param image: original image
    :return stretched image
    """

    rect = order_points(pts)
    (tl, tr, br, bl) = rect

    # euclidean distance between br & bl
    width_a = np.linalg.norm(br - bl)
    # euclidean distance between tr & tl
    width_b = np.linalg.norm(tr - tl)
    max_width = max(int(width_a), int(width_b))

    # euclidean distance between tr & br
    height_a = np.linalg.norm(tr - br)
    # euclidean distance between tl & bl
    height_b = np.linalg.norm(tl - bl)

    max_height = max(int(height_a), int(height_b))

    dst = np.array([
        [0, 0],
        [max_width - 1, 0],
        [max_width - 1, max_height - 1],
        [0, max_height - 1]], dtype="float32")

    # transforms the perspective of the rectangle rect (ordered points of the given parameters)
    # to the perspective of dst
    PT = cv2.getPerspectiveTransform(rect, dst)

    return cv2.warpPerspective(image, PT, (max_width, max_height))