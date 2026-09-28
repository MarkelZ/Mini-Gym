from minigym.physics.kinetic_pointmass import KineticPointmass


class Link:
    def __init__(self, a: KineticPointmass, b: KineticPointmass, length: float = None):
        self.a = a
        self.b = b
        self.length = length if length is not None else (a.pos - b.pos).length()

    def constrain(self):
        delta = self.b.pos - self.a.pos
        dist = delta.length()
        if dist == 0:
            return
        total_mass = self.a.mass + self.b.mass
        correction = delta * (1 - self.length / dist)
        self.a.pos += correction * (self.b.mass / total_mass)
        self.b.pos -= correction * (self.a.mass / total_mass)
