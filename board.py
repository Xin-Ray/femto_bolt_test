"""ChArUco board definition shared by the demo scripts.

Defaults match boards/charuco_a4_7x5_38mm_4x4.pdf (printed at 100% on A4).
If you print at a different scale, measure one square with calipers and pass
--square / --marker (in metres) so the pose has the correct scale.
"""

import cv2

SQUARES_X = 7          # squares along the long edge
SQUARES_Y = 5          # squares along the short edge
SQUARE_LENGTH = 0.038  # metres
MARKER_LENGTH = 0.028  # metres
DICTIONARY = cv2.aruco.DICT_4X4_50


def add_board_args(parser):
    g = parser.add_argument_group("board")
    g.add_argument("--squares-x", type=int, default=SQUARES_X)
    g.add_argument("--squares-y", type=int, default=SQUARES_Y)
    g.add_argument("--square", type=float, default=SQUARE_LENGTH, help="square side in metres")
    g.add_argument("--marker", type=float, default=MARKER_LENGTH, help="marker side in metres")


def make_board(squares_x=SQUARES_X, squares_y=SQUARES_Y,
               square=SQUARE_LENGTH, marker=MARKER_LENGTH, dictionary=DICTIONARY):
    aruco_dict = cv2.aruco.getPredefinedDictionary(dictionary)
    return cv2.aruco.CharucoBoard((squares_x, squares_y), square, marker, aruco_dict)


def board_from_args(args):
    return make_board(args.squares_x, args.squares_y, args.square, args.marker)


def make_detector(board):
    charuco_params = cv2.aruco.CharucoParameters()
    detector_params = cv2.aruco.DetectorParameters()
    detector_params.cornerRefinementMethod = cv2.aruco.CORNER_REFINE_SUBPIX
    return cv2.aruco.CharucoDetector(board, charuco_params, detector_params)
