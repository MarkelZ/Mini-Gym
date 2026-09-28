from math import cos, sin, atan2, pi

from pygame import Vector2


def clamp(x, a, b):
    return a if x <= a else b if x >= b else x

def angle_to_vec2(angle, length = 1):
    return length * Vector2(cos(angle), sin(angle))

def points_to_angle(p1: Vector2, p2: Vector2):
    dx = p2.x - p1.x
    dy = p2.y - p1.y
    rads = atan2(-dy, dx)
    rads %= 2 * pi
    return rads
