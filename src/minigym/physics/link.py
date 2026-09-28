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
        correction = delta * (1 - self.length / dist) * 0.5
        self.a.pos += correction
        self.b.pos -= correction


