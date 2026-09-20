from math import cos, sin

from pygame import Vector2


def clamp(x, a, b):
    return a if x <= a else b if x >= b else x

def angle_to_vec2(angle, length = 1):
    return length * Vector2(cos(angle), sin(angle))