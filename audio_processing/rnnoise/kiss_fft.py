import math

import numpy as np

MAXFACTORS = 8


class Complex:
    def __init__(self):
        self.r = 0.0
        self.i = 0.0

    def __repr__(self):
        return '{}{}{}j'.format(self.r, '-' if 0 > self.i else '+', abs(self.i))


class FFTState:
    nfft = 0
    scale = 0
    shift = 0
    factors = np.zeros(2 * MAXFACTORS)
    bitrev = None
    twiddles = None
    arch_fft = None


def compute_bitrev_table(Fout, f, fstride, in_stride, factors, st):
    p = int(factors[0])  # the radix
    m = int(factors[1])  # stage's fft length/p

    if m == 1:
        for j in range(p):
            f[0] = Fout + j
            f = f[fstride * in_stride:]
    else:
        for j in range(p):
            compute_bitrev_table(Fout, f, fstride * p, in_stride, factors[2:], st)
            f = f[fstride * in_stride:]
            Fout += m


def kf_factor(n, facbuf):
    p = 4
    stages = 0
    nbak = n

    while n > 1:
        while n % p:
            p = 2 if p == 4 else 3 if p == 2 else p + 2
            if p > 32000 or p * p > n:
                p = n
        n /= p
        if p > 5:
            return 0

        facbuf[2 * stages] = p
        if p == 2 and stages > 1:
            facbuf[2 * stages] = 4
            facbuf[2] = 2
        stages = stages + 1

    n = nbak

    for i in range(stages // 2):
        tmp = facbuf[2 * i]
        facbuf[2 * i] = facbuf[2 * (stages - i - 1)]
        facbuf[2 * (stages - i - 1)] = tmp

    for i in range(stages):
        n /= facbuf[2 * i]
        facbuf[2 * i + 1] = n

    return 1


def compute_twiddles(twiddles, nfft):
    for i in range(nfft):
        phase = (-2 * math.pi / nfft) * i
        twiddles[i].r = math.cos(phase)
        twiddles[i].i = math.sin(phase)


def opus_fft_alloc_twiddles(nfft):
    st = FFTState()

    st.nfft = nfft
    st.scale = 1. / nfft

    st.twiddles = twiddles = [Complex() for _ in range(nfft)]
    compute_twiddles(twiddles, nfft)
    st.shift = -1
    kf_factor(nfft, st.factors)

    # bitrev
    st.bitrev = bitrev = np.zeros(nfft, dtype=int)

    compute_bitrev_table(0, bitrev, 1, 1, st.factors, st)

    return st
