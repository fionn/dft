#!/usr/bin/env python3

from collections.abc import Sequence
from cmath import exp, sqrt, pi

class Vector(Sequence):

    _sqrt2 = sqrt(2)

    def __init__(self, *args):
        super().__init__()
        self.N = len(args)
        self.value = args
        self.omega = 2j * pi / self.N

    @staticmethod
    def _cround(z):
        n = 8
        if round(z.imag, n) == 0:
            return round(z.real, n)
        return round(z.real, n) + round(z.imag, n) * 1j

    def __len__(self):
        return self.N

    def __getitem__(self, index):
        return self.value[index]

    def __str__(self):
        return str(tuple(map(self._cround, self.value)))

    def __eq__(self, other):
        return tuple(map(self._cround, self)) == tuple(map(self._cround, other))

    def __ne__(self, other):
        return not self.__eq__(other)

    def __mul__(self, other):
        return Vector(*tuple(x * other for x in self))

    def __rmul__(self, other):
        return self.__mul__(other)

    def __add__(self, other):
        return Vector(*tuple(x + y for x, y in zip(self, other)))

    def dot(self, other):
        return sum(u * v for u, v in zip(self, other))

    def dft(self, s = -1):
        '''Naive FT'''
        omega = s * self.omega
        x = [self.dot(Vector(*[exp(omega * n * k) for n in range(self.N)])) \
             / sqrt(self.N) for k in range(self.N)]
        return Vector(*x)

    def fft(self, s = -1):
        '''Radix-2 Cooley-Tukey'''

        if self.N <= 1:
            return self

        f_even = Vector(*self.value[::2])
        f_odd = Vector(*self.value[1::2])

        x_even = f_even.fft(s)
        x_odd = f_odd.fft(s)

        omega = s * self.omega

        F = [exp(omega * k) * x_odd[k] for k in range(self.N // 2)]

        x = [(x_even[k] + F[k]) / self._sqrt2 for k in range(self.N // 2)] \
            + [(x_even[k] - F[k]) / self._sqrt2 for k in range(self.N // 2)]

        return Vector(*x)

if __name__ == "__main__":
    f = Vector(0, 1, 0, -1)
    print(f)
    f_tilde = f.fft()
    print(f_tilde)
    f = f_tilde.fft(+1)
    print(f)

