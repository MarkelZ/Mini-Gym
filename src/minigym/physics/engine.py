from pygame import Vector2

from minigym.physics.kinetic_pointmass import KineticPointmass
from minigym.physics.link import Link


class PhysicsEngine:
    entities: list[KineticPointmass]
    links: list[Link]

    def __init__(self):
        self.entities = []
        self.links = []

    def update(self, deltat: float):
        for p in self.entities:
            vel = (p._pos - p._prev_pos) * p.friction
            new_pos = p._pos + vel + p._acc * deltat * deltat
            p._prev_pos = p._pos
            p._pos = new_pos
            p._acc = Vector2(0, 0)

        for link in self.links:
            link.constrain()

    def add_pointmass(self, pm):
        self.entities.append(pm)

    def add_link(self, link: Link):
        self.links.append(link)