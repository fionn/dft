#!/usr/bin/env python3

import unittest
from dft import Vector

class TestFourierTransformation(unittest.TestCase):

    def test_inverse_dft(self):
        '''DFT inverse is an inverse'''
        x = [0, 1j, 0, -1j] * 128
        f = Vector(*x)
        g = f.dft()
        f_prime = g.dft(+1)
        self.assertEqual(f, f_prime)

    def test_inverse_fft(self):
        '''FFT inverse is an inverse'''
        x = [0, 1j, 0, -1j] * 4096
        f = Vector(*x)
        g = f.fft()
        f_prime = g.fft(+1)
        self.assertEqual(f, f_prime)

    def test_dft_fft_equivalence(self):
        '''DFT and FFT are numerically equivalent'''
        x = [0, 1j, 0, -1j] * 128
        f = Vector(*x)
        self.assertEqual(f.fft(), f.dft())

    def test_fft_imaginary_structure(self):
        '''FFT ~ * i'''
        x = [0, 1j, 0, -1j] * 2048
        f = Vector(*x)
        self.assertEqual(f, f.fft().fft().fft().fft())

    def test_fft_linearity(self):
        '''Fourier transformations are linear'''
        x, y = [2, 2, 5, 9] * 2048, [1, 3, 7, 4] * 2048
        a, b = 3, 4
        f, g = Vector(*x), Vector(*y)
        self.assertEqual((a * f + b * g).fft(), a * f.fft() + b * g.fft())

    def test_fft_dirac_delta(self):
        '''FFT is consistent with the delta function'''
        x = [1] * 8192
        delta_tilde = Vector(*x)
        delta = delta_tilde.fft(+1)
        self.assertTrue(delta[0].real > 0)
        self.assertEqual(sum(delta[1:]), 0)

if __name__ == "__main__":
    unittest.main(verbosity = 2)

