# (c) 2021 ax Inc.

import math

import numpy as np

__all__ = [
    'poly_iou'
]


def crossing_number(point, polys):
    cn = 0
    pv = polys[-1]
    for pp in polys:
        if (pv[1] <= point[1] and pp[1] > point[1]) \
                or (pv[1] > point[1] and pp[1] <= point[1]):
            vt = (point[1] - pv[1]) / (pp[1] - pv[1])
            if point[0] < pv[0] + vt * (pp[0] - pv[0]):
                cn += 1
        pv = pp

    return cn


def on_edge(point, polys):
    pv = polys[-1]
    for pp in polys:
        v1 = point - pv
        v2 = pp - pv
        if np.linalg.norm(v1) <= np.linalg.norm(v2):
            theta = np.dot(v1, v2)
            theta /= np.sqrt(np.sum(v1 ** 2)) * np.sqrt(np.sum(v2 ** 2))
            theta = np.clip(theta, None, 1)
            if theta == 1:
                return True
        pv = pp

    return False


def crossing_point(g, p, eps=1e-8):
    ary = []
    for i, j in ((0, 3), (1, 0), (2, 1), (3, 2)):
        pv = g[i]
        pp = g[j]
        if abs(pp[0] - pv[0]) < eps:
            x0 = (pp[0] + pv[0]) / 2
            a = None
        else:
            # y = a * x + b
            a = (pp[1] - pv[1]) / (pp[0] - pv[0])
            b = pv[1] - a * pv[0]

        min_x0 = min(pv[0], pp[0])
        min_y0 = min(pv[1], pp[1])
        max_x0 = max(pv[0], pp[0])
        max_y0 = max(pv[1], pp[1])

        for i, j in ((0, 3), (1, 0), (2, 1), (3, 2)):
            pv = p[i]
            pp = p[j]
            if abs(pp[0] - pv[0]) < eps:
                x1 = (pp[0] + pv[0]) / 2
                c = None
            else:
                # y = c * x + d
                c = (pp[1] - pv[1]) / (pp[0] - pv[0])
                d = pv[1] - c * pv[0]

            if a is None:
                if c is None:
                    continue
                else:
                    x = x0
                    y = c * x + d
            else:
                if c is None:
                    x = x1
                    y = a * x + b
                elif abs(a - c) < eps:
                    continue
                else:
                    x = (d - b) / (a - c)
                    y = (a * d - b * c) / (a - c)

            min_x1 = min(pv[0], pp[0])
            min_y1 = min(pv[1], pp[1])
            max_x1 = max(pv[0], pp[0])
            max_y1 = max(pv[1], pp[1])

            if x < min_x0 or x > max_x0:
                continue
            if y < min_y0 or y > max_y0:
                continue
            if x < min_x1 or x > max_x1:
                continue
            if y < min_y1 or y > max_y1:
                continue

            ary.append(np.array([x, y]))

    return ary


def gift_wrapping(points):
    points = sorted(points, key=lambda x: (-x[0], -x[1]))

    s = getLargestVectorIndex(points[0], points)
    n = getLargestThetaIndex(points[0], points[s], points)

    paths = [s, n]
    while len(paths) < len(points):
        current = paths[-1]
        before = paths[-2]
        next = getLargestThetaIndex(points[before], points[current], points)
        paths.append(next)

    points = np.vstack([points[i] for i in paths])
    return points


def getLargestVectorIndex(x, points):
    max_i = 0
    max_value = 0
    for i in range(len(points)):
        d = np.sqrt(np.sum((points[i] - x) ** 2))
        if d > max_value:
            max_value = d
            max_i = i

    return max_i


def getLargestThetaIndex(before, current, points):
    max_i = 0
    max_value = 0
    v1 = before - current
    for i in range(len(points)):
        v2 = points[i] - current
        if np.linalg.norm(v2, ord=1) == 0:
            continue
        theta = np.dot(v1, v2)
        theta /= np.sqrt(np.sum(v1 ** 2)) * np.sqrt(np.sum(v2 ** 2))
        theta = np.clip(theta, -1, 1)
        theta = math.acos(theta)
        if theta > max_value:
            max_value = theta
            max_i = i

    return max_i


def area(points):
    pv = points[-1]
    s = 0
    for pp in points:
        s += (pv[0] + pp[0]) * (pv[1] - pp[1])
        pv = pp

    return abs(s) * 0.5


def isin(ary, p):
    for x in ary:
        if all(p == x):
            return True

    return False


def poly_iou(g, p):
    points = []

    # Extract inner points
    for x in g:
        cn = crossing_number(x, p)
        if cn > 0 and cn % 2 != 0:
            points.append(x)
    for x in p:
        cn = crossing_number(x, g)
        if cn > 0 and cn % 2 != 0:
            points.append(x)

    # Since the above judgment does not extract the points on the side,
    # extracted the contact points on the side.
    for x in g:
        if on_edge(x, p):
            points.append(x)
    for x in p:
        # Do not select the same vertex
        if isin(points, x):
            continue
        if on_edge(x, g):
            points.append(x)

    # intersection area points
    points.extend(crossing_point(g, p))

    if len(points) < 3:
        return 0

    # sort
    points = gift_wrapping(points)

    inter = area(points)
    union = area(g) + area(p) - inter
    if union == 0:
        return 0
    else:
        return inter / union


if __name__ == '__main__':
    g = np.array([
        811.2532959, 267.52346802, 871.93408203, 265.97250366,
        872.24749756, 278.23474121, 811.56671143, 279.78570557,
    ]).reshape(-1, 2)
    p = np.array([
        787.55474854, 264.10675049, 822.35186768, 263.85424805,
        822.45220947, 277.68392944, 787.65515137, 277.93643188,
    ]).reshape(-1, 2)

    print(poly_iou(g, p))
