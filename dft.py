#!/usr/bin/env python3

from cmath import exp, sqrt, pi

class Vector(list):

    @staticmethod
    def _cround(z):
        n = 8
        if round(z.imag, n) == 0:
            return round(z.real, n)
        return round(z.real, n) + round(z.imag, n) * 1j

    def __str__(self):
        return str(list(map(self._cround, self)))

    def dot(self, other):
        return sum(u * v for u, v in zip(self, other))

    def dft(self, s = -1):
        """Naive FT"""
        N = len(self)
        omega = s * 2 * pi * 1j / N
        x = [self.dot([exp(omega * n * k) for n in range(N)]) / sqrt(N) for k in range(N)]
        return Vector(x)

if __name__ == "__main__":
    f = Vector([0, 1, 0, -1, 0, 1, 0, -1])
    print(f)

    g = f.dft()
    print(g)

    f = g.dft(+1)
    print(f)

