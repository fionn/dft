#!/usr/bin/env python3

import unittest
from dft import Vector

class TestFourierTransformation(unittest.TestCase):

    def test_inverse_dft(self):
        x = [0, 1j, 0, -1j] * 128
        f = Vector(*x)
        g = f.dft()
        f_prime = g.dft(+1)
        self.assertEqual(f, f_prime)

    def test_inverse_fft(self):
        x = [0, 1j, 0, -1j] * 4096
        f = Vector(*x)
        g = f.fft()
        f_prime = g.fft(+1)
        self.assertEqual(f, f_prime)

    def test_dft_fft_equivalence(self):
        x = [0, 1j, 0, -1j] * 128
        f = Vector(*x)
        g_fast = f.fft()
        g_slow = f.dft()
        self.assertEqual(g_fast, g_slow)

    def test_fft_imaginary_structure(self):
        x = [0, 1j, 0, -1j] * 2048
        f = Vector(*x)
        f_reverse = Vector(*f[::-1])
        self.assertEqual(f, f.fft().fft().fft().fft())

    def test_fft_linearity(self):
        x, y = [2, 2, 5, 9] * 128, [1, 3, 7, 4] * 128
        a, b = 3, 4
        f, g = Vector(*x), Vector(*y)
        self.assertEqual((a * f + b * g).fft(), a * f.fft() + b * g.fft())

if __name__ == "__main__":
    unittest.main(verbosity = 2)

