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

    def __eq__(self, other):
        return list(map(self._cround, self)) == list(map(self._cround, other))

    def __ne__(self, other):
        return not self.__eq__(other)

    def dot(self, other):
        return sum(u * v for u, v in zip(self, other))

    def dft(self, s = -1):
        '''Naive FT'''
        N = len(self)
        omega = s * 2j * pi / N
        x = [self.dot([exp(omega * n * k) for n in range(N)]) / sqrt(N) for k in range(N)]
        return Vector(x)

    def fft(self, s = -1):
        '''Radix-2 Cooley-Tukey'''

        N = len(self)
        if N <= 1:
            return self

        f_even = Vector(self[::2])
        f_odd = Vector(self[1::2])
        x_even = f_even.fft(s)
        x_odd = f_odd.fft(s)

        omega = s * 2j * pi / N
        sqrt2 = sqrt(2)
        F = [exp(omega * k) * x_odd[k] for k in range(N // 2)]
        return Vector([(x_even[k] + F[k]) / sqrt2 for k in range(N // 2)] \
                      + [(x_even[k] - F[k]) / sqrt2 for k in range(N // 2)])

if __name__ == "__main__":
    f_0 = Vector([1, 1j, -1, -1j] * 128)

    g_1 = f_0.dft()
    f_1 = g_1.dft(+1)

    g_2 = f_0.fft()
    f_2 = g_2.fft(+1)

    assert f_1 == f_0
    assert f_2 == f_1
    
