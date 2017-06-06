#!/usr/bin/env python3

from cmath import exp, sqrt, pi

def fourier(f, s = -1):
    """Naive FT"""
    N = len(f)
    omega = s * 2 * pi * 1j / N
    x = [sum(f[n] * exp(omega * n * k) for n in range(N)) / sqrt(N) for k in range(N)]
    return x 

def cround(z):
    if round(z.imag, 9) == 0:
        return round(z.real, 9)
    return round(z.real, 9) + round(z.imag, 9) * 1j

def cprint(f):
    print(list(map(cround, f)))
    return True


if __name__ == "__main__":
    f = [0, 1, 0, -1, 0, 1, 0, -1]
    cprint(f)

    g = fourier(f)
    cprint(g)

    f = fourier(g, +1)
    cprint(f)

