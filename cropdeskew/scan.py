"""
Date: 22/02/2021
Author: Axel Lebas
File: scan.py

All the necessary algorithms to scan an image.
"""
import itertools
import math
import cv2
import numpy as np
from pylsd.lsd import lsd

from scipy.spatial import distance as dist

MIN_QUAD_AREA_RATIO = 0.25
MAX_QUAD_ANGLE_RANGE = 40


def angle_between_vectors_degrees(u, v):
    """Returns the angle between two vectors in degrees"""
    return np.degrees(
        math.acos(np.dot(u, v) / (np.linalg.norm(u) * np.linalg.norm(v))))


def get_angle(p1, p2, p3):
    """
    Returns the angle between the line segment from p2 to p1
    and the line segment from p2 to p3 in degrees
    """
    a = np.radians(np.array(p1))
    b = np.radians(np.array(p2))
    c = np.radians(np.array(p3))

    avec = a - b
    cvec = c - b

    return angle_between_vectors_degrees(avec, cvec)


def angle_range(quad):
    """
    Returns the range between max and min interior angles of quadrilateral.
    The input quadrilateral must be a numpy array with vertices ordered clockwise
    starting with the top left vertex.
    """
    tl, tr, br, bl = quad
    ura = get_angle(tl[0], tr[0], br[0])
    ula = get_angle(bl[0], tl[0], tr[0])
    lra = get_angle(tr[0], br[0], bl[0])
    lla = get_angle(br[0], bl[0], tl[0])

    angles = [ura, ula, lra, lla]
    return np.ptp(angles)


def is_valid_contour(cnt, IM_WIDTH, IM_HEIGHT):
    """Returns True if the contour satisfies all requirements set at instantitation"""

    return (len(cnt) == 4 and cv2.contourArea(cnt) > IM_WIDTH * IM_HEIGHT * MIN_QUAD_AREA_RATIO
            and angle_range(cnt) < MAX_QUAD_ANGLE_RANGE)


def filter_corners(corners, min_dist=20):
    """Filters corners that are within min_dist of others"""

    def predicate(representatives, corner):
        return all(dist.euclidean(representative, corner) >= min_dist
                   for representative in representatives)

    filtered_corners = []
    for c in corners:
        if predicate(filtered_corners, c):
            filtered_corners.append(c)
    return filtered_corners


def get_corners(img):
    """
    Returns a list of corners ((x, y) tuples) found in the input image. With proper
    pre-processing and filtering, it should output at most 10 potential corners.
    This is a utility function used by get_contours. The input image is expected
    to be rescaled and Canny filtered prior to be passed in.
    """
    lines = lsd(img)

    # massages the output from LSD
    # LSD operates on edges. One "line" has 2 edges, and so we need to combine the edges back into lines
    # 1. separate out the lines into horizontal and vertical lines.
    # 2. Draw the horizontal lines back onto a canvas, but slightly thicker and longer.
    # 3. Run connected-components on the new canvas
    # 4. Get the bounding box for each component, and the bounding box is final line.
    # 5. The ends of each line is a corner
    # 6. Repeat for vertical lines
    # 7. Draw all the final lines onto another canvas. Where the lines overlap are also corners

    corners = []
    if lines is not None:
        # separate out the horizontal and vertical lines, and draw them back onto separate canvases
        lines = lines.squeeze().astype(np.int32).tolist()
        horizontal_lines_canvas = np.zeros(img.shape, dtype=np.uint8)
        vertical_lines_canvas = np.zeros(img.shape, dtype=np.uint8)
        for line in lines:
            x1, y1, x2, y2, _ = line
            if abs(x2 - x1) > abs(y2 - y1):
                (x1, y1), (x2, y2) = sorted(((x1, y1), (x2, y2)), key=lambda pt: pt[0])
                cv2.line(horizontal_lines_canvas, (max(x1 - 5, 0), y1), (min(x2 + 5, img.shape[1] - 1), y2), 255, 2)
            else:
                (x1, y1), (x2, y2) = sorted(((x1, y1), (x2, y2)), key=lambda pt: pt[1])
                cv2.line(vertical_lines_canvas, (x1, max(y1 - 5, 0)), (x2, min(y2 + 5, img.shape[0] - 1)), 255, 2)

        lines = []

        # find the horizontal lines (connected-components -> bounding boxes -> final lines)
        (contours, hierarchy) = cv2.findContours(horizontal_lines_canvas, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
        contours = sorted(contours, key=lambda c: cv2.arcLength(c, True), reverse=True)[:2]
        horizontal_lines_canvas = np.zeros(img.shape, dtype=np.uint8)
        for contour in contours:
            contour = contour.reshape((contour.shape[0], contour.shape[2]))
            min_x = np.amin(contour[:, 0], axis=0) + 2
            max_x = np.amax(contour[:, 0], axis=0) - 2
            left_y = int(np.average(contour[contour[:, 0] == min_x][:, 1]))
            right_y = int(np.average(contour[contour[:, 0] == max_x][:, 1]))
            lines.append((min_x, left_y, max_x, right_y))
            cv2.line(horizontal_lines_canvas, (min_x, left_y), (max_x, right_y), 1, 1)
            corners.append((min_x, left_y))
            corners.append((max_x, right_y))

        # find the vertical lines (connected-components -> bounding boxes -> final lines)
        (contours, hierarchy) = cv2.findContours(vertical_lines_canvas, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
        contours = sorted(contours, key=lambda c: cv2.arcLength(c, True), reverse=True)[:2]
        vertical_lines_canvas = np.zeros(img.shape, dtype=np.uint8)
        for contour in contours:
            contour = contour.reshape((contour.shape[0], contour.shape[2]))
            min_y = np.amin(contour[:, 1], axis=0) + 2
            max_y = np.amax(contour[:, 1], axis=0) - 2
            top_x = int(np.average(contour[contour[:, 1] == min_y][:, 0]))
            bottom_x = int(np.average(contour[contour[:, 1] == max_y][:, 0]))
            lines.append((top_x, min_y, bottom_x, max_y))
            cv2.line(vertical_lines_canvas, (top_x, min_y), (bottom_x, max_y), 1, 1)
            corners.append((top_x, min_y))
            corners.append((bottom_x, max_y))

        # find the corners
        corners_y, corners_x = np.where(horizontal_lines_canvas + vertical_lines_canvas == 2)
        corners += zip(corners_x, corners_y)

    # remove corners in close proximity
    corners = filter_corners(corners)
    return corners


def get_contours(processed_image):
    """ get the contours
    find the contours of a processed image
    Attributes:
        image_edged: image with a canny filter
        contours: all the contours found in the image
        epsilon: maximum distance from contour to approximated contour (accuracy parameter)
                 [https://opencv-python-tutroals.readthedocs.io/en/latest/py_tutorials/py_imgproc/py_contours/py_contour_features/py_contour_features.html#contour-approximation}
        approx: approximation of a contour with the Douglas-Peucker algorithm
        edges: found edges of the document
    :param processed_image: a processed image
    :return coords of the edges of the document in the image
    """

    IM_HEIGHT, IM_WIDTH = processed_image.shape

    test_corners = get_corners(processed_image)

    approx_contours = []

    if len(test_corners) >= 4:
        quads = []

        for quad in itertools.combinations(test_corners, 4):
            points = np.array(quad)
            points = order_points(points)
            points = np.array([[p] for p in points], dtype="int32")
            quads.append(points)

        # get the top five quadrilaterals by area
        quads = sorted(quads, key=cv2.contourArea, reverse=True)[:5]

        # sort candidate quadrilaterals by their angle range
        quads = sorted(quads, key=angle_range)

        approx = quads[0]

        if is_valid_contour(approx, IM_WIDTH, IM_HEIGHT):
            approx_contours.append(approx)

    (contours, hierarchy) = cv2.findContours(processed_image, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    # Optimisation - We recover the 4 largest contours
    contours = sorted(contours, key=cv2.contourArea, reverse=True)[:5]

    for c in contours:
        # approximate the contour
        approx = cv2.approxPolyDP(c, 80, True)
        if is_valid_contour(approx, IM_WIDTH, IM_HEIGHT):
            approx_contours.append(approx)
            break
    if not approx_contours:
        # XXX: raise exception?
        TOP_RIGHT = (IM_WIDTH, 0)
        BOTTOM_RIGHT = (IM_WIDTH, IM_HEIGHT)
        BOTTOM_LEFT = (0, IM_HEIGHT)
        TOP_LEFT = (0, 0)
        screen_cnt = np.array([[TOP_RIGHT], [BOTTOM_RIGHT], [BOTTOM_LEFT], [TOP_LEFT]])
    else:
        screen_cnt = max(approx_contours, key=cv2.contourArea)
    return screen_cnt.reshape(4, 2)


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
