from minigym.kinetic_pointmass import KineticPointmass
from pygame import Vector2

class PhysicsEngine:
    entities: list[KineticPointmass]

    def __init__(self):
        self.entities = []

    def update(self, deltat: float):
        for p in self.entities:
            p._vel = p.friction * (p._vel + p._acc * deltat)
            p._pos += p._vel * deltat * p.friction
            p._acc = Vector2(0, 0)

    def add_pointmass(self, pm):
        self.entities.append(pm)
