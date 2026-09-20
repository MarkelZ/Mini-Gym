from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

import pygame
from pygame.math import Vector2


EPSILON = 1e-9


class Geometry:
    def intersects_with(self, other: "Geometry") -> bool:
        return intersection_point(self, other) is not None

    def intersection_point(self, other: "Geometry") -> Optional[Vector2]:
        return intersection_point(self, other)

class Circle(Geometry):
    center: Vector2
    radius: float

    def __init__(self, center, radius):
        self.center = center
        self.radius = radius

class LineSegment(Geometry):
    start: Vector2
    end: Vector2

    def __init__(self, start, end):
        self.start = start
        self.end = end

class Rectangle(Geometry):
    """
    Axis-aligned rectangle.

    x and y specify the top-left corner.
    """

    x: float
    y: float
    width: float
    height: float

    def __init__(self, x, y, width, height):
        self.x = x
        self.y = y
        self.width = width
        self.height = height

    @property
    def left(self) -> float:
        return self.x

    @property
    def right(self) -> float:
        return self.x + self.width

    @property
    def top(self) -> float:
        return self.y

    @property
    def bottom(self) -> float:
        return self.y + self.height

    @property
    def center(self) -> Vector2:
        return Vector2(
            self.x + self.width / 2,
            self.y + self.height / 2,
        )

    def to_pygame_rect(self) -> pygame.Rect:
        return pygame.Rect(self.x, self.y, self.width, self.height)

def intersection_point(
    a: Geometry,
    b: Geometry,
) -> Optional[Vector2]:
    """
    Procedural intersection function.

    Returns a representative intersection point, or None.
    """

    if isinstance(a, Rectangle) and isinstance(b, Rectangle):
        return rectangle_rectangle(a, b)

    if isinstance(a, Circle) and isinstance(b, Circle):
        return circle_circle(a, b)

    if isinstance(a, Rectangle) and isinstance(b, Circle):
        return rectangle_circle(a, b)

    if isinstance(a, Circle) and isinstance(b, Rectangle):
        return rectangle_circle(b, a)

    if isinstance(a, LineSegment) and isinstance(b, LineSegment):
        return line_line(a, b)

    if isinstance(a, LineSegment) and isinstance(b, Circle):
        return line_circle(a, b)

    if isinstance(a, Circle) and isinstance(b, LineSegment):
        return line_circle(b, a)

    if isinstance(a, LineSegment) and isinstance(b, Rectangle):
        return line_rectangle(a, b)

    if isinstance(a, Rectangle) and isinstance(b, LineSegment):
        return line_rectangle(b, a)

    raise TypeError(
        f"Unsupported geometry pair: "
        f"{type(a).__name__}, {type(b).__name__}"
    )


def rectangle_rectangle(
    a: Rectangle,
    b: Rectangle,
) -> Optional[Vector2]:
    left = max(a.left, b.left)
    right = min(a.right, b.right)
    top = max(a.top, b.top)
    bottom = min(a.bottom, b.bottom)

    if left <= right + EPSILON and top <= bottom + EPSILON:
        return Vector2(
            (left + right) / 2,
            (top + bottom) / 2,
        )

    return None


def circle_circle(
    a: Circle,
    b: Circle,
) -> Optional[Vector2]:
    delta = b.center - a.center
    distance_squared = delta.length_squared()
    radius_sum = a.radius + b.radius

    if distance_squared > radius_sum * radius_sum + EPSILON:
        return None

    # Coincident circles.
    if distance_squared <= EPSILON:
        return a.center + Vector2(a.radius, 0)

    center_distance = delta.length()
    direction = delta / center_distance

    # A representative point near the contact/overlap region.
    distance_from_a = min(a.radius, center_distance)
    return a.center + direction * distance_from_a


def rectangle_circle(
    rectangle: Rectangle,
    circle: Circle,
) -> Optional[Vector2]:
    closest = Vector2(
        max(rectangle.left, min(circle.center.x, rectangle.right)),
        max(rectangle.top, min(circle.center.y, rectangle.bottom)),
    )

    if closest.distance_to(circle.center) <= circle.radius + EPSILON:
        return closest

    return None


def line_line(
    a: LineSegment,
    b: LineSegment,
) -> Optional[Vector2]:
    p = a.start
    r = a.end - a.start

    q = b.start
    s = b.end - b.start

    denominator = r.cross(s)
    q_minus_p = q - p

    # Parallel lines.
    if abs(denominator) <= EPSILON:
        # Parallel but not collinear.
        if abs(q_minus_p.cross(r)) > EPSILON:
            return None

        r_length_squared = r.length_squared()

        # Both segments are points.
        if r_length_squared <= EPSILON:
            if p.distance_to(q) <= EPSILON:
                return p.copy()
            return None

        t0 = q_minus_p.dot(r) / r_length_squared
        t1 = t0 + s.dot(r) / r_length_squared

        overlap_start = max(0.0, min(t0, t1))
        overlap_end = min(1.0, max(t0, t1))

        if overlap_start <= overlap_end + EPSILON:
            t = (overlap_start + overlap_end) / 2
            return p + r * t

        return None

    t = q_minus_p.cross(s) / denominator
    u = q_minus_p.cross(r) / denominator

    if (
        -EPSILON <= t <= 1 + EPSILON
        and -EPSILON <= u <= 1 + EPSILON
    ):
        return p + r * t

    return None

def line_circle(
    line: LineSegment,
    circle: Circle,
) -> Optional[Vector2]:
    segment = line.end - line.start
    segment_length_squared = segment.length_squared()

    # Treat a zero-length line segment as a point.
    if segment_length_squared <= EPSILON:
        if line.start.distance_to(circle.center) <= circle.radius:
            return Vector2(line.start)
        return None

    # If the start point is already inside (or on) the circle, it's the closest point.
    if line.start.distance_to(circle.center) <= circle.radius + EPSILON:
        return Vector2(line.start)

    to_start = line.start - circle.center
    a = segment_length_squared
    b = 2 * to_start.dot(segment)
    c = to_start.length_squared() - circle.radius * circle.radius

    discriminant = b * b - 4 * a * c
    if discriminant < -EPSILON:
        return None
    discriminant = max(discriminant, 0.0)

    sqrt_discriminant = discriminant ** 0.5
    t1 = (-b - sqrt_discriminant) / (2 * a)
    t2 = (-b + sqrt_discriminant) / (2 * a)

    for t in (t1, t2):
        if -EPSILON <= t <= 1 + EPSILON:
            t = max(0.0, min(1.0, t))
            return line.start + segment * t

    return None

def line_rectangle(
    line: LineSegment,
    rectangle: Rectangle,
) -> Optional[Vector2]:
    """
    Liang-Barsky line-clipping algorithm.
    """

    dx = line.end.x - line.start.x
    dy = line.end.y - line.start.y

    t_min = 0.0
    t_max = 1.0

    boundaries = (
        (-dx, line.start.x - rectangle.left),
        (dx, rectangle.right - line.start.x),
        (-dy, line.start.y - rectangle.top),
        (dy, rectangle.bottom - line.start.y),
    )

    for p, q in boundaries:
        if abs(p) <= EPSILON:
            if q < -EPSILON:
                return None
            continue

        t = q / p

        if p < 0:
            if t > t_max + EPSILON:
                return None
            t_min = max(t_min, t)
        else:
            if t < t_min - EPSILON:
                return None
            t_max = min(t_max, t)

    if t_min <= t_max + EPSILON:
        t = (t_min + t_max) / 2
        return line.start + (line.end - line.start) * t

    return None
