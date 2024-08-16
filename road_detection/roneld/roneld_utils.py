"""
Functions for RONELD Lane Detection method (ICPR 2020). Intended for use on deep learning lane
detection methods to improve lane detection accuracy by using information from preceding frames.
"""
import math
import warnings
import cv2
import numpy as np
import scipy.integrate as integrate
#from numba import jit, prange, njit
from scipy import interpolate
from typing import List

# ignore warnings
warnings.filterwarnings('ignore')

# set min straight and min curve lane length in terms of points
MIN_STRAIGHT_LANE_LENGTH = 3
MIN_CURVE_LANE_LENGTH = 9


def roneld_lane_detection(lane_images:list, prev_lanes, prev_curves, curve_mode=False,
                          center_ratio=0.5, lane_ratio=0, image=None):
    """
    Output processing on lane outputs from deep learning semantic segmentation maps.
    Uses lane tracking to detect lanes in the current frame.
    :param lane_images: probability map outputs from deep learning model (e.g. ENet-SAD, SCNN) in
    current frame
    :param prev_lanes: lanes from previous frames, use prev_lanes from previous image as actual
    parameter for this
    :param prev_curves: curves in previous frames, use prev_curves from previous image as actual
    parameter for this
    :param curve_mode: indicates if currently in curve mode, use curve_mode from previous image as
    actual parameter for this
    :param center_ratio: position of the center of image as a ratio of distance from left to the
    total width of image
    :param lane_ratio: position of the horizon on the image as a ratio of the distance from the
    bottom to the total height of the image
    :param image: image of driving scene (if included, will plot on driving scene image)
    :return: (output_images, prev_lanes, prev_curves, curve_mode) output_images with the lanes
    drawn on them which are the same size as the probability map inputs. Updated prev_lanes,
    prev_curves, curve_mode after the current frame which should be used as inputs for the call on
    the next frame
    """

    lane_images = np.array(lane_images)

    # get shape of 1 lane image
    lane_image_size = lane_images[0].shape[:2]

    # adaptive confidence threshold based on highest confidence for the image
    confidence_threshold = nb_max(lane_images.flatten()) // 3

    if confidence_threshold == 0:
        # need threshold to be at least 1
        confidence_threshold = 1

    # if lanes have a distance of less than 1/200 of image width in prev and curr frame, they will
    # be classified as the same lane
    merge_distance = lane_image_size[1] / 200

    # single channel image, based on label image size, 2 images, one for left lane and one for right
    # left lane in index 0 and right lane in index 1, same for label and scnn images
    output_images = [np.zeros(lane_image_size) for _ in range(2)]

    # get highest point of lane in image based on lane_ratio
    highest_point = lane_image_size[0] * (1 - lane_ratio)

    # get current lanes in the lane_images
    curr_lanes = get_all_lanes(lane_images, confidence_threshold)

    matchings = match_prev_curr(curr_lanes, prev_lanes, merge_distance, lane_image_size)

    # store the age for the current lanes, in the structure of prev_lanes which is different from
    # curr_lanes because this one doesn't store the image and only stores the lane and age of
    # the lane
    curr_lanes_age = sorted(get_curr_lanes_age(matchings, prev_lanes, curr_lanes),
                            key=lambda x: x["age"], reverse=True)

    display_lanes = [{}, {}]

    # segregating curves to left and right based on center_ratio
    # find one line on the left
    for lane in curr_lanes_age:
        x_intercept = find_x_intercept(lane["lane"][0], lane["lane"][1], lane_image_size[0])
        if x_intercept < center_ratio * lane_image_size[1] and (
                lane["points"][0][0] >= lane["points"][-1][0] or curve_mode):
            # highest age, x_intercept to the left of the center and top point is right of the
            # bottom point (i.e. lane is slanted right) or curve
            display_lanes[0] = lane
            break

    # find one line on the right
    for lane in curr_lanes_age:
        x_intercept = find_x_intercept(lane["lane"][0], lane["lane"][1], lane_image_size[0])
        if x_intercept > center_ratio * lane_image_size[1] and \
                (lane["points"][0][0] <= lane["points"][-1][0] or curve_mode):
            # highest age, x_intercept to the right of the center and top point is left of the
            # bottom point (i.e. lane is slanted left) or curve
            display_lanes[1] = lane
            break

    # set prev as curr
    prev_lanes = curr_lanes_age
    curve_counter = 0
    for lane in display_lanes:
        # count curves
        if lane != {}:
            curve_counter += lane["curve"]

    prev_curves = np.append(prev_curves[1:], curve_counter)
    # look at preceding frames to reduce false curve predictions
    if prev_curves.sum() >= 15:
        # more than 1.5 curves per frame on average (max is 2)
        curve_mode = True
    elif prev_curves.sum() <= 10:
        # less than 1 curve per frame on average (max is 2)
        curve_mode = False

    # take the highest point only if it exceeds the lane_ratio
    # min is taking the highest point (lower index is higher point on image)
    if display_lanes[0] != {} and display_lanes[1] != {}:
        highest_point = min(highest_point, min(display_lanes[0]["points"][0][1],
                                               display_lanes[1]["points"][0][1]))
    elif display_lanes[0] != {}:
        highest_point = min(highest_point, display_lanes[0]["points"][0][1])
    elif display_lanes[1] != {}:
        highest_point = min(highest_point, display_lanes[1]["points"][0][1])

    if curve_mode:
        # print (display_curve)
        plot_curves(display_lanes, output_images, lane_image_size, highest_point, image)

    else:
        # to draw lines when not in curve mode
        plot_lines(display_lanes, output_images, lane_image_size, highest_point, image)

    # return output images and prev lanes and prev curves for the next call
    return output_images, prev_lanes, prev_curves, curve_mode

#@jit(fastmath=True)
def plot_lines(display_lanes:list, output_images:list, lane_image_size:tuple, highest_point:int,
               image:np.array):
    """
    method to plot lines onto output_images
    :param display_lanes: parameters of lanes to be displayed
    :param output_images: images that the lanes are to be drawn on
    :param lane_image_size: size of the output_images
    :param highest_point: y coordinates of highest point lane is to be plotted
    :param image: image of driving scene (if included, will plot on driving scene image)
    :return: None (plots onto output_images)
    """
    for lane_index in range(len(display_lanes)):
        lane = display_lanes[lane_index]
        if lane != {}:
            # non-empty lane
            plot_line(lane, output_images[lane_index], lane_image_size, highest_point, image)

#@jit(fastmath=True)
def plot_line(lane:dict, output_image:np.ndarray, image_shape:tuple, highest_point:int,
              image:np.ndarray):
    """
    Plot a single lane onto one output image
    :param lane: lane to be plotted
    :param output_image: output image to be plotted on
    :param image_shape: shape of output image
    :param highest_point: highest point to plot the line to
    :param image: image of driving scene (if included, will plot on driving scene image)
    :return: None (plots onto output_image)
    """
    gradient = lane["lane"][0]
    y_intercept = lane["lane"][1]

    bottom_point, top_point = find_line_points(gradient, y_intercept, image_shape,
                                               top=highest_point)

    cv2.line(output_image, (int(bottom_point[0]), int(bottom_point[1])),
             (int(top_point[0]), int(top_point[1])), (255, 255, 255),
             int(30 * image_shape[0] / 288))
    if image is not None:
        #there's an input image, draw on it, line width 20, green
        cv2.line(image, (int(bottom_point[0] * image.shape[1] / output_image.shape[1]),
                                int (bottom_point[1] * image.shape[0] / output_image.shape[0])),
                 (int(top_point[0] * image.shape[1] / output_image.shape[1]),
                  int(top_point[1] * image.shape[0] / output_image.shape[0])), (0, 255, 0), 20)



#@jit(fastmath=True)
def plot_curves(display_lanes:list, output_images:list, lane_image_size:tuple,
                highest_point:int, image=np.ndarray):
    """
    Plots curves based on lane parameters onto output images

    :param display_lanes: parameters of 2 lanes to be displayed
    :param output_images: output images to draw lanes on for comparison
    :param lane_image_size: size of lane images, which should also be the size of the output images
    :param MIN_CURVE_LANE_LENGTH: minimum length of curved lane
    :param highest_point: y coordinate of the highest point to plot lane to (assumed to be horizon)
    :param image: image of driving scene (if included, will plot on driving scene image)
    :return: None
    """
    for lane_index in range(len(display_lanes)):
        lane = display_lanes[lane_index]
        if lane != {}:
            # lane is not empty

            # points that make up lane
            lane_points = lane["points"]

            if len(lane_points) <= MIN_CURVE_LANE_LENGTH:
                # only if a certain length then plot curve, below that plot line
                plot_line(lane, output_images[lane_index], lane_image_size, highest_point, image)
            else:
                # plot curve
                curve_points = []

                x_coords = lane_points[:, 0]
                y_coords = lane_points[:, 1]

                cs = interpolate.interp1d(y_coords, x_coords, kind="quadratic")
                for j in range(y_coords[0], y_coords[-1], 5):
                    # get points to draw curve
                    curve_points.append([cs(j), j])

                for point_index in range(len(curve_points) - 1):
                    cv2.line(output_images[lane_index],
                             (int(curve_points[point_index][0]), int(curve_points[point_index][1])),
                             (int(curve_points[point_index + 1][0]),
                              int(curve_points[point_index + 1][1])),
                             (255, 255, 255), int(30 * lane_image_size[0] / 288))
                    if image is not None:
                        cv2.line(image, (int(curve_points[point_index][0] * image.shape[1] /
                                             output_images[lane_index].shape[1]),
                                         int(curve_points[point_index][1] * image.shape[0] /
                                             output_images[lane_index].shape[0])),
                                 (int(curve_points[point_index + 1][0] * image.shape[1] /
                                      output_images[lane_index].shape[1]),
                                  int(curve_points[point_index + 1][1] * image.shape[0] /
                                      output_images[lane_index].shape[0])),
                                 (0, 255, 0), 20)

#@njit(fastmath=True)
def nb_max(array:np.ndarray):
    """
    finds max in array using parallel numba to speed it up
    :param array: array that we are searching
    :return: max in array
    """
    return np.max(array)

# same as linear_regression but in linear algebra form
#@jit(fastmath=True)
def linear_regression_matrix(points: np.ndarray, confidence: np.ndarray):
    """
    weighted least-squares linear regression to find line given points and their confidence
    :param points: array of points in (x,y) form
    :param confidence: confidence of points corresponding to points array
    :return: (gradient, y intercept) of line
    """
    if len(points) == 0 or len(points) == 1:
        # no points or only one points, redundant to calculate gradient and y intercept
        return None, None
    # print (points, confidence)
    x_coords = points[:, 0]
    if np.all(x_coords == x_coords[0]):
        # add a small random factor to vectors where everything is the same
        # which would otherwise lead to singular matrix that is non-invertible
        x_coords = x_coords + np.random.rand(len(points)) * 0.01
    y_coords = points[:, 1]

    # create weights matrix for weighted least squares linear regression based on confidence of
    # the points
    weights = np.diag(confidence)
    # design matrix, add vector of ones
    design_matrix = np.column_stack((np.array([1] * len(points)), x_coords))

    # lane_param = (design_matrix^(T) * weights * design_matrix)^(-1) * desing_matrix^(T) *
    # weights * y_coords
    lane_param = np.dot(np.dot(np.linalg.inv(design_matrix.transpose().dot(
        weights.dot(design_matrix))),design_matrix.transpose()), weights.dot(y_coords))

    if lane_param[1] == 0:
        # set a very small number of gradient to prevent zero division error later on
        lane_param[1] = 0.001
    return lane_param[1], lane_param[0]


# takes points as input and clears the outliers by doing least squares regression and dumping a
# portion of the furthest points in each iteration (by default, 5% of the points)
#@jit(fastmath=True)
def clear_outliers(points: np.ndarray, confidence:np.ndarray, iterations=1, to_dump=0.05):
    """
    clears outliers based on horizontal distance between point and regression line
    :param points: array of points in (x,y) form
    :param confidence: confidence of points corresponding to points array
    :param iterations: number of iterations to clear outliers
    :param to_dump: ratio of points to dump (will not go below min points for straight lane)
    :return: updated points and confidence arrays with outliers removed
    """
    for _ in range(iterations):

        # number of points to delete, length of points times proportion plus 1 (round up)
        number_deleted = int(len(points) * to_dump) + 1

        if len(points) <= MIN_STRAIGHT_LANE_LENGTH:
            # don't reduce below min_points
            return points, confidence

        # recalculate distances each time to get the updated distances
        gradient, y_intercept = linear_regression_matrix(points, confidence)
        point_distance = point_distance_wrapper((gradient, y_intercept))
        # get all the distances for the points form the current regression line
        distances = np.apply_along_axis(point_distance, 1, points)
        # sort by distance, need mergesort instead of unstable quicksort to ensure all indices are
        # represented, sorted in ascending order, so last points are to be deleted
        sorted_distances = np.argsort(distances, kind="mergesort")
        points = np.delete(points, sorted_distances[-number_deleted:], 0)
        confidence = np.delete(confidence, sorted_distances[-number_deleted:], 0)
    return points, confidence


def point_distance_wrapper(line:list):
    """
    wrapper function to return a function that finds the distance between point and the line
    :param line: line to be compared to (gradient, y_intercept)
    :return: function that takes in a point and finds the distance from the input point to the line
    """
    def point_distance(point:list):
        # calculates horizontal distance of point from line
        return abs(point[0] - (point[1] - line[1]) / line[0])
    return point_distance


#@jit(fastmath=True)
def correlation_linear(points: np.ndarray):
    """
    calculates coefficeint of determination of points
    :param points: arrays of points in (x,y) form
    :return: coefficient of determination of points using a linear model
    """
    if len(points) <= 1:
        return -1
    x_var = np.var(points[:,0])
    y_var = np.var(points[:,1])
    # covariance matrix should be 2 x 2, with [0][1] and [1][0] being cov(X,Y)
    # bias=True as we look at sample covariance (we look at sample variance as well)
    x_y_cov = np.cov(points.transpose(), bias=True)[0][1]
    if x_var == 0 or y_var == 0:
        # no variance along one axis, so its a vertical/horizontal line (in gopald, its vertical
        # since we don't consider horizontal lanes following our process flow)
        return 1
    return x_y_cov ** 2 / (x_var * y_var)

#@njit(fastmath=True)
def line_distance(line1, line2, lane_image_size):
    """
    returns root mean square distance between two lines within the image
    :param line1: lane parameters of first lane to be compared
    :param line2: lane parameters of second lane to be compared
    :param lane_image_size: size of the lane image
    :return: root mean square distance between lane1 and lane2 within lane_image_size
    """
    top = 0
    bottom = lane_image_size[0]

    # y coordinate where lines intersect
    if abs(line1[0] - line2[0]) <= 2 ** (-20):
        #higher threshold because this value is cubed later on
        #gradient is the same, parallel lines, calculate horizontal distance between them, square
        # it and multiply by length
        #x_delta = c_delta / gradient
        return math.sqrt(((line1[1] - line2[1]) / line2[0]) ** 2 * (bottom - top) / (bottom - top))

    intersect_x = (line2[1] - line1[1]) / (line1[0] - line2[0])
    intersect_y = line1[0] * intersect_x + line1[1]

    inverse_gradient_difference = 1 / line1[0] - 1 / line2[0]

    if top < intersect_y < bottom - 1:
        #above intersection
        above = abs((inverse_gradient_difference * bottom - (line1[1] / line1[0] - line2[1] / line2[0])) ** 3 -
                   (inverse_gradient_difference * intersect_y - (line1[1] / line1[0] - line2[1] / line2[0])) ** 3)
        below = abs((inverse_gradient_difference * intersect_y - (line1[1] / line1[0] - line2[1] / line2[0])) ** 3 -
                   (inverse_gradient_difference * top - (line1[1] / line1[0] - line2[1] / line2[0])) ** 3)
        #if the intersection happens within the image, then need to separate to calculate the integral
        return math.sqrt((above + below) / abs(inverse_gradient_difference) / 3 / (bottom - top))
    return math.sqrt(abs((inverse_gradient_difference * bottom - (line1[1] / line1[0] - line2[1] / line2[0])) ** 3 -
                   (inverse_gradient_difference * top - (line1[1] / line1[0] - line2[1] / line2[0])) ** 3) /
                     abs(inverse_gradient_difference) / 3 / (bottom - top))


def square_distance_wrapper(line1:list, line2:list):
    """
    wrapper function which returns a square_distance function that returns the horizontal distance
    between two lines at a particular y coordinate. Intended to avoid a lambda function in
    integrate.quad when calculating root mean square distance (lambda seems to have issues with
    numba jit)
    :param line1: first line to be compared (gradient, y_intercept)
    :param line2: second line to be compared (gradient, y_intercept)
    :return: square_distance(y_coord) function
    """
    # returns square_distance function that only takes in one parameter to be integrated
    # can't seem to use lambda with numba jit hence this workaround
    def square_distance(y_coord):
        # calculates distance between the two lines at y
        return abs((y_coord - line1[1]) / line1[0] - (y_coord - line2[1]) / line2[0]) ** 2

    return square_distance


# find the top and bottom point of lane in image (uses lane_ratio global variable)
#@jit(fastmath=True)
def find_line_points(gradient: float, y_intercept: float, lane_image_size: tuple, bottom=-1,
                     top=-1):
    """
    finding the top and bottom points of a straight lane (represented by a line)
    :param gradient: gradient of the lane
    :param y_intercept: y intercept of the lane
    :param lane_image_size: size of the lane image
    :param bottom: largest-valued row of the line (by default, the bottom of the lane image). This
    is also the lowest row in the image
    :param top: lowest-valued row of the line, which is also the highest row in the image
    :return: coordinates (x, y) of top and bottom point of the line in the image
    """
    if bottom == -1:
        # take bottom of the image by default
        bottom = lane_image_size[0] - 1

    if top == -1:
        # by default, the highest point to plot to should be based on the lane_ratio
        top = lane_image_size[0]

    # y axis positive actually points down because of the array
    # to find x intercept, need to find y = bottom_of_image
    x_intercept = find_x_intercept(gradient, y_intercept, bottom)

    if x_intercept == -1:
        # no x intercept, draw a line across the whole image
        top_point = (lane_image_size[1] - 1, y_intercept)
        bottom_point = (0, y_intercept)
        return bottom_point, top_point
    # bottom relative to the image (so actually the larger y value since axis of image for y
    # is inverted)
    bottom_point = (int(x_intercept), bottom)
    if bottom_point[0] < 0:
        # if x intercept is out of frame to the left, then use y intercept
        bottom_point = (0, y_intercept)
    elif bottom_point[0] >= lane_image_size[1]:
        # if x intercept is out of frame to the right, then find point where line means right of
        # the frame
        bottom_point = (lane_image_size[1] - 1, int(y_intercept + lane_image_size[1] * gradient))

    # need 1 - lane_ratio to get the location of the top point
    top_point = (int((top - y_intercept) / gradient), top)

    return bottom_point, top_point

#@jit(fastmath=True)
def find_highest_probability(lane_image:np.ndarray, start_row:int, end_row:int, left_bound=-1,
                             right_bound=-1):
    """
    find highest probability point in image in lane_image[start_row: end_row,
    left_bound:right_bound]
    :param lane_image: lane image to be searched
    :param start_row: start row to be searched (inclusive)
    :param end_row: end row to be searched (exclusive)
    :param left_bound: leftmost column to be searched (inclusive)
    :param right_bound: rightmost column to be searched (exclusive)
    :return: coordinate of highest point (column, row)
    """
    # to get the highest probability point in the image between start and end_row
    # (inclusive and exclusive respectively)
    highest_probability = (0, (-1, -1))
    if left_bound == -1:
        left_bound = 0
    if right_bound == -1:
        right_bound = lane_image.shape[1]
    if end_row > len(lane_image):
        # end_row out of the image, which has index until len(lane_image) - 1 only
        end_row = len(lane_image)
    for row in range(start_row, end_row):
        for column in range(left_bound, right_bound):
            if lane_image[row][column] > highest_probability[0]:
                # store the highest probability value and the location, (x, y)
                highest_probability = (lane_image[row][column], (column, row))
    # return the location of the highest proability

    return highest_probability[1]


# get lanes
#@jit(fastmath=True)
def get_all_lanes(lane_images:list, confidence_threshold:int):
    curr_lanes = [{} for _ in range(4)]
    for i in range(4):
        # the number in exists is present, so there should be a file to indicate the lane
        lane_image = lane_images[i]
        # (height, width, channels)
        lane_image_size = lane_image.shape
        longest_lane, confidence = get_lane(lane_image, row_stride=lane_image_size[0] // 20,
                                            confidence_threshold=confidence_threshold)
        avg_confidence = rms(confidence)
        # convert to numpy array, if done in get_lane, leads to IndexError, might be a numba issue
        # issue does not occur when jit decorator is commented out
        longest_lane = np.array(longest_lane)

        if len(longest_lane) <= MIN_STRAIGHT_LANE_LENGTH:
            # less than MIN_STRAIGHT_LANE_LENGTH, too short to be considered so go to next lane_
            # image
            continue

        curve = False

        # look at bottom points, number of points is equal to min length of straight lane
        initial_section = MIN_STRAIGHT_LANE_LENGTH

        # Check for curve length and whether it meets our curve criteria (initial section has lower
        # r^2 than whole lane)
        if len(longest_lane) >= MIN_CURVE_LANE_LENGTH and correlation_linear(longest_lane) < \
                correlation_linear(longest_lane[initial_section:]):
            curve = True

        else:
            # clear outliers if not a curve
            longest_lane, confidence = clear_outliers(longest_lane, confidence,
                                                      iterations=1, to_dump=0.2)
            avg_confidence = rms(confidence)

        gradient, y_intercept = linear_regression_matrix(longest_lane, confidence)
        curr_lanes[i] = {"lane": (gradient, y_intercept), "image": i, "curve": curve,
                         "points": longest_lane, "avg_confidence": avg_confidence,
                         "confidence": confidence}
    return curr_lanes


# get lane points for GOPALD method
#@jit(fastmath=True)
def get_lane(lane_image:np.ndarray, row_stride=20, confidence_threshold=1):
    curr_lane = []
    row = 0
    # store confidence of points
    confidence = []
    if confidence_threshold <= 0:
        # must always have at least some confidence, otherwise its just every point in the prob map
        confidence_threshold = 1
    # python for loop iterates through the output so changing row will not change the value for the
    # next iteration of the for loop
    left_bound = 0
    right_bound = lane_image.shape[1]

    while row < lane_image.shape[0]:
        for column in range(left_bound, right_bound):
            # confidence threshold
            if lane_image[row][column] >= confidence_threshold:
                # search for highest probability point in next 20 rows
                point = find_highest_probability(lane_image, row, row + row_stride, left_bound,
                                                 right_bound)
                # left and right bound set at 1/4 of the image width away from current point found
                # search 1/4 (if point is at left/right edge of image) to 1/2 of image width for
                # next point
                left_bound = int(point[0] - (lane_image.shape[1] * 0.25)) if (point[0] - (
                        lane_image.shape[1] * 0.25)) >= 0 else 0
                right_bound = int(point[0] + (lane_image.shape[1] * 0.25)) if (point[0] + (
                        lane_image.shape[1] * 0.25)) <= lane_image.shape[1] else lane_image.shape[1]
                curr_lane.append(point)
                confidence.append(lane_image[point[1]][point[0]])
                # skip 20 rows since we find highest probability for this row plus next 19 rows
                row += row_stride
                break
        row += 1

    # only 2 outputs, numba works noticebly slower with 3 outputs
    return curr_lane, np.array(confidence)


#@jit(fastmath=True)
def get_curr_lanes_age(matchings:list, prev_lanes:List[dict], curr_lanes:List[dict]):
    curr_lanes_age = []

    # set priority for active lane vs inactive lane, e^2 factor
    lane_priority = {0: 0, 1: (math.e ** 2 - 1), 2: (math.e ** 2 - 1), 3: 0}

    # stores current lanes that were matched to previous lanes
    matched_lanes = []

    # stores previous lanes that were matched to current lanes
    prev_matched_lanes = []

    # using the matched previous lanes to find the closest current one, then merging the current
    # lane with the previous one and letting the current one inherit the details of the previous
    # one (and make changes, e.g. increase age)
    for prev_index in range(len(matchings)):
        matching = matchings[prev_index]

        if matching != []:
            # this line is matched

            # if more than one matching, i.e. prev lane is matched to more than one lane
            for curr_index in matching:
                matched_lanes.append(curr_index)
                curr_lane = curr_lanes[curr_index]

                #calculate age
                age = prev_lanes[prev_index]["age"] + (1 + lane_priority[curr_lane["image"]]) * \
                          len(curr_lane["points"]) * curr_lane["avg_confidence"]

                # can change curr_lane since each curr lane only matches to one prev lane, and we
                # are processing this curr lane matched to this prev frame, so this curr lane won't
                # be used again in this run
                curr_lane["age"] = age

                # add more age if image is in the priority lane images
                # merge previous and current lane together for smoother transition between frames
                curr_lanes_age.append(curr_lane)

            prev_matched_lanes.append(prev_index)

    for curr_index in range(len(curr_lanes)):
        # curr lanes that were not matched and not empty (i.e. new lane)
        if curr_index not in matched_lanes and curr_lanes[curr_index] != {}:
            # higher priority lane gets extra age instantly added
            age = (1 + lane_priority[curr_lanes[curr_index]["image"]]) * \
                  len(curr_lanes[curr_index]["points"]) * \
                  curr_lanes[curr_index]["avg_confidence"]
            curr_lanes[curr_index]["age"] = age
            curr_lanes_age.append(curr_lanes[curr_index])

    # need to keep some of the old matched lines if they just missed like one or two frames
    for prev_index in range(len(prev_lanes)):
        # age very small, to clear memory and really old lanes
        if prev_index not in prev_matched_lanes and prev_lanes[prev_index]["age"] >= \
                math.e ** (-10):
            # around for more than 10 frames but not in current frame (otherwise would have been
            # deleted in earlier matchings for loop, penalize age by 5 frames
            prev_lanes[prev_index]["age"] = prev_lanes[prev_index]["age"] / math.e
            curr_lanes_age.append(prev_lanes[prev_index])

    return curr_lanes_age

# match lanes in prev frame to lanes in current frame
#@jit(fastmath=True)
def match_prev_curr(curr_lanes:List[dict], prev_lanes:List[dict], merge_distance:int,
                    lane_image_size:tuple):
    """
    Matches current lanes to previous lanes and returns an array with the matchings
    :param curr_lanes: list of current lanes
    :param prev_lanes: list of previous lanes
    :param merge_distance: distance to merge the lanes (if r.m.s. distance between lanes is smaller
    than this, the lanes are matched
    :param lane_image_size: size of lane image
    :return: the matchings of previous lane to current lane (2D array of same length as curr_lanes.)
    Each array in the 2D array represents the prev lane that matched to the curr lane of that index)
    """
    # matching current lanes (1D-index) to previous lanes (2D-index). The 2D-values are the distance
    # between the current and prev lane (1D- and 2D-index)
    curr_matchings = [{} for _ in curr_lanes]

    for i in range(len(curr_lanes)):
        if curr_lanes[i] != {}:
            # if lane is not empty
            for j in range(len(prev_lanes)):
                # compare one current lane and one prev lane
                if line_distance(curr_lanes[i]["lane"], prev_lanes[j]["lane"], lane_image_size) < \
                        merge_distance:
                    # distance between two lanes (r.m.s. distance between lanes within image) is
                    # less than merge_distance
                    curr_matchings[i][j] = line_distance(curr_lanes[i]["lane"],
                                                         prev_lanes[j]["lane"],
                                                         lane_image_size)

    # finding the current lanes that match to each previous lane (index). If current lane matches
    # to more than 1 prev lane in curr_matchings (i.e. more than one lane in preceding frame
    # matches current lane), then take the closest one
    matchings = [[] for _ in prev_lanes]
    # not prange as this part appends to matchines[closest_prev] and might have multiple current
    # lanes matching to the same prev lane hence race condition
    for i in range(len(curr_matchings)):
        # looking through each current lane
        matches = curr_matchings[i]
        if len(matches) != 0:
            # matched to a previous line
            get_from_matches = get_from_collection_wrapper(matches)
            # find closest prev_lane that curr is matched to
            closest_prev = sorted(matches.keys(), key=get_from_matches)[0]
            # each curr lane is only matched to closest prev lane, each curr has only one prev lane,
            # the closest prev lane (but at this stage, each prev can have multiple curr)
            matchings[closest_prev].append(i)

    return matchings

# accessor function to avoid lambda generator function in jit for match_prev_curr
def get_from_collection_wrapper(collection:dict):
    """
    wrapper function that returns a function which gets items from collection using key.
    Used to avoid a lambda function in key argument for sorted (seems to disrupt numba jit)
    :param collection: collection to be used
    :return: function that takes in key to access items in the collection
    """
    def get_from_collection(key):
        return collection[key]
    return get_from_collection


#@jit(fastmath=True)
def rms(array: np.ndarray):
    """
    return root mean square of values in the array
    :param array: input array
    :return: root mean square of values in input array
    """
    if array is not np.ndarray:
        array = np.array(array)
    return math.sqrt((array ** 2).sum() / len(array))

# find x_intercept for image with height, if gradient 0 then there is no x_intercept
#@jit(fastmath=True)
def find_x_intercept(gradient:int, y_intercept:int, height:int):
    """
    Find x intercept of the line with the bottom of the image from the line parameters
    :param gradient: gradient of line
    :param y_intercept: y intercept of line
    :param height: height of the image
    :return: x intercept of line with the bottom of the image
    """
    return (height - 1 - y_intercept) / gradient if gradient != 0 else -1
