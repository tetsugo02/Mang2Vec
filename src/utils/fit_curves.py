"""Python implementation of
Algorithm for Automatically Fitting Digitized Curves
by Philip J. Schneider
"Graphics Gems", Academic Press, 1990
"""

from __future__ import print_function

from numpy import dot, linalg, zeros

import src.utils.bezier as bezier


# Fit one (ore more) Bezier curves to a set of points
def fit_curve(points, max_error):
    left_tangent = normalize(points[1] - points[0])
    right_tangent = normalize(points[-2] - points[-1])
    return fit_cubic(points, left_tangent, right_tangent, max_error)


def fit_cubic(points, left_tangent, right_tangent, error):
    if len(points) == 2:
        dist = linalg.norm(points[0] - points[1]) / 3.0
        bez_curve = [
            points[0],
            points[0] + left_tangent * dist,
            points[1] + right_tangent * dist,
            points[1],
        ]
        return [bez_curve]
    u = chord_length_parameterize(points)
    bez_curve = generate_bezier(points, u, left_tangent, right_tangent)
    max_error, split_point = compute_max_error(points, bez_curve, u)
    if max_error < error:
        return [bez_curve]
    if max_error < error**2:
        for i in range(20):
            u_prime = reparameterize(bez_curve, points, u)
            bez_curve = generate_bezier(points, u_prime, left_tangent, right_tangent)
            max_error, split_point = compute_max_error(points, bez_curve, u_prime)
            if max_error < error:
                return [bez_curve]
            u = u_prime
    beziers = []
    center_tangent = normalize(points[split_point - 1] - points[split_point + 1])
    beziers += fit_cubic(points[: split_point + 1], left_tangent, center_tangent, error)
    beziers += fit_cubic(points[split_point:], -center_tangent, right_tangent, error)
    return beziers


def generate_bezier(points, parameters, left_tangent, right_tangent):
    bez_curve = [points[0], None, None, points[-1]]
    A = zeros((len(parameters), 2, 2))
    for i, u in enumerate(parameters):
        A[i][0] = left_tangent * 3 * (1 - u) ** 2 * u
        A[i][1] = right_tangent * 3 * (1 - u) * u**2
    C = zeros((2, 2))
    X = zeros(2)
    for i, (point, u) in enumerate(zip(points, parameters)):
        C[0][0] += dot(A[i][0], A[i][0])
        C[0][1] += dot(A[i][0], A[i][1])
        C[1][0] += dot(A[i][0], A[i][1])
        C[1][1] += dot(A[i][1], A[i][1])
        tmp = point - bezier.q([points[0], points[0], points[-1], points[-1]], u)
        X[0] += dot(A[i][0], tmp)
        X[1] += dot(A[i][1], tmp)
    det_C0_C1 = C[0][0] * C[1][1] - C[1][0] * C[0][1]
    det_C0_X = C[0][0] * X[1] - C[1][0] * X[0]
    det_X_C1 = X[0] * C[1][1] - X[1] * C[0][1]
    alpha_l = 0.0 if det_C0_C1 == 0 else det_X_C1 / det_C0_C1
    alpha_r = 0.0 if det_C0_C1 == 0 else det_C0_X / det_C0_C1
    seg_length = linalg.norm(points[0] - points[-1])
    epsilon = 1.0e-6 * seg_length
    if alpha_l < epsilon or alpha_r < epsilon:
        bez_curve[1] = bez_curve[0] + left_tangent * (seg_length / 3.0)
        bez_curve[2] = bez_curve[3] + right_tangent * (seg_length / 3.0)
    else:
        bez_curve[1] = bez_curve[0] + left_tangent * alpha_l
        bez_curve[2] = bez_curve[3] + right_tangent * alpha_r
    return bez_curve


def reparameterize(bezier_curve, points, parameters):
    return [
        newton_raphson_root_find(bezier_curve, point, u)
        for point, u in zip(points, parameters)
    ]


def newton_raphson_root_find(bez, point, u):
    d = bezier.q(bez, u) - point
    numerator = (d * bezier.qprime(bez, u)).sum()
    denominator = (bezier.qprime(bez, u) ** 2 + d * bezier.qprimeprime(bez, u)).sum()
    if denominator == 0.0:
        return u
    else:
        return u - numerator / denominator


def chord_length_parameterize(points):
    u = [0.0]
    for i in range(1, len(points)):
        u.append(u[i - 1] + linalg.norm(points[i] - points[i - 1]))
    for i, _ in enumerate(u):
        u[i] = u[i] / u[-1]
    return u


def compute_max_error(points, bez, parameters):
    max_dist = 0.0
    split_point = len(points) // 2
    for i, (point, u) in enumerate(zip(points, parameters)):
        dist = linalg.norm(bezier.q(bez, u) - point) ** 2
        if dist > max_dist:
            max_dist = dist
            split_point = i
    return max_dist, split_point


def normalize(v):
    return v / linalg.norm(v)
