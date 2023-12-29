import math

import numpy as np

MAXFACTORS = 8


class Complex:
    def __init__(self):
        self.r = 0.0
        self.i = 0.0

    def __repr__(self):
        return '{:.6f}{}{:.6f}j'.format(self.r, '-' if 0 > self.i else '+', abs(self.i))


class FFTState:
    nfft = 0
    scale = 0
    shift = 0
    factors = np.zeros(2 * MAXFACTORS, dtype=int)
    bitrev = None
    twiddles = None
    arch_fft = None


def C_ADD(res, a, b):
    res.r = a.r + b.r
    res.i = a.i + b.i


def C_SUB(res, a, b):
    res.r = a.r - b.r
    res.i = a.i - b.i


def C_ADDTO(res, a):
    res.r = res.r + a.r
    res.i = res.i + a.i


def C_MUL(m, a, b):
    m.r = a.r * b.r - a.i * b.i
    m.i = a.r * b.i + a.i * b.r


def C_MULBYSCALAR(c, s):
    c.r *= s
    c.i *= s


def kf_bfly2(Fout, m, N):
    tw = 0.7071067812

    for i in range(N):
        Fout2 = Fout[4:]
        t = Fout2[0]

        C_SUB(Fout2[0], Fout[0], t)
        C_ADDTO(Fout[0], t)

        t.r = (Fout2[1].r + Fout2[1].i) * tw
        t.i = (Fout2[1].i - Fout2[1].r) * tw
        C_SUB(Fout2[1], Fout[1], t)
        C_ADDTO(Fout[1], t)

        t.r = Fout2[2].i
        t.i = -Fout2[2].r
        C_SUB(Fout2[2], Fout[2], t)
        C_ADDTO(Fout[2], t)

        t.r = (Fout2[3].i - Fout2[3].r) * tw
        t.i = -(Fout2[3].i + Fout2[3].r) * tw
        C_SUB(Fout2[3], Fout[3], t)
        C_ADDTO(Fout[3], t)

        Fout = Fout[8:]


def kf_bfly4(Fout, fstride, st, m, N, mm):
    if m == 1:
        # Degenerate case where all the twiddles are 1.
        for i in range(N):
            scratch0 = Complex()
            scratch1 = Complex()

            C_SUB(scratch0, Fout[0], Fout[2])
            C_ADDTO(Fout[0], Fout[2])
            C_ADD(scratch1, Fout[1], Fout[3])
            C_SUB(Fout[2], Fout[0], scratch1)
            C_ADDTO(Fout[0], scratch1)
            C_SUB(scratch1, Fout[1], Fout[3])

            Fout[1].r = scratch0.r + scratch1.i
            Fout[1].i = scratch0.i - scratch1.r
            Fout[3].r = scratch0.r - scratch1.i
            Fout[3].i = scratch0.i + scratch1.r
            Fout = Fout[4:]
    else:
        scratch = [Complex() for _ in range(6)]
        m2 = 2 * m
        m3 = 3 * m
        Fout_beg = Fout
        for i in range(N):
            Fout = Fout_beg[i * mm:]
            tw3 = tw2 = tw1 = st.twiddles
            # m is guaranteed to be a multiple of 4.
            for j in range(m):
                C_MUL(scratch[0], Fout[m], tw1[0])
                C_MUL(scratch[1], Fout[m2], tw2[0])
                C_MUL(scratch[2], Fout[m3], tw3[0])

                C_SUB(scratch[5], Fout[0], scratch[1])
                C_ADDTO(Fout[0], scratch[1])
                C_ADD(scratch[3], scratch[0], scratch[2])
                C_SUB(scratch[4], scratch[0], scratch[2])
                C_SUB(Fout[m2], Fout[0], scratch[3])
                tw1 = tw1[fstride:]
                tw2 = tw2[fstride * 2:]
                tw3 = tw3[fstride * 3:]
                C_ADDTO(Fout[0], scratch[3])

                Fout[m].r = scratch[5].r + scratch[4].i
                Fout[m].i = scratch[5].i - scratch[4].r
                Fout[m3].r = scratch[5].r - scratch[4].i
                Fout[m3].i = scratch[5].i + scratch[4].r

                Fout = Fout[1:]


def kf_bfly3(Fout, fstride, st, m, N, mm):
    m2 = 2 * m
    scratch = [Complex() for _ in range(5)]

    Fout_beg = Fout
    epi3 = st.twiddles[fstride * m]
    for i in range(N):
        Fout = Fout_beg[i * mm:]
        tw1 = tw2 = st.twiddles
        # For non-custom modes, m is guaranteed to be a multiple of 4.
        k = m
        while 0 < k:
            C_MUL(scratch[1], Fout[m], tw1[0])
            C_MUL(scratch[2], Fout[m2], tw2[0])

            C_ADD(scratch[3], scratch[1], scratch[2])
            C_SUB(scratch[0], scratch[1], scratch[2])
            tw1 = tw1[fstride:]
            tw2 = tw2[fstride * 2:]

            Fout[m].r = Fout[0].r - scratch[3].r / 2
            Fout[m].i = Fout[0].i - scratch[3].i / 2

            C_MULBYSCALAR(scratch[0], epi3.i)

            C_ADDTO(Fout[0], scratch[3])

            Fout[m2].r = Fout[m].r + scratch[0].i
            Fout[m2].i = Fout[m].i - scratch[0].r

            Fout[m].r = Fout[m].r - scratch[0].i
            Fout[m].i = Fout[m].i + scratch[0].r

            Fout = Fout[1:]
            k = k - 1


def kf_bfly5(Fout, fstride, st, m, N, mm):
    scratch = [Complex() for _ in range(13)]
    Fout_beg = Fout

    ya = st.twiddles[fstride * m]
    yb = st.twiddles[fstride * 2 * m]
    tw = st.twiddles
    for i in range(N):
        Fout = Fout_beg[i * mm:]
        Fout0 = Fout
        Fout1 = Fout0[m:]
        Fout2 = Fout0[2 * m:]
        Fout3 = Fout0[3 * m:]
        Fout4 = Fout0[4 * m:]

        # For non-custom modes, m is guaranteed to be a multiple of 4.
        for u in range(m):
            scratch[0].r = Fout0[0].r
            scratch[0].i = Fout0[0].i

            C_MUL(scratch[1], Fout1[0], tw[u * fstride])
            C_MUL(scratch[2], Fout2[0], tw[2 * u * fstride])
            C_MUL(scratch[3], Fout3[0], tw[3 * u * fstride])
            C_MUL(scratch[4], Fout4[0], tw[4 * u * fstride])

            C_ADD(scratch[7], scratch[1], scratch[4])
            C_SUB(scratch[10], scratch[1], scratch[4])
            C_ADD(scratch[8], scratch[2], scratch[3])
            C_SUB(scratch[9], scratch[2], scratch[3])

            Fout0[0].r = Fout0[0].r + (scratch[7].r + scratch[8].r)
            Fout0[0].i = Fout0[0].i + (scratch[7].i + scratch[8].i)

            scratch[5].r = scratch[0].r + ((scratch[7].r * ya.r) + (scratch[8].r * yb.r))
            scratch[5].i = scratch[0].i + ((scratch[7].i * ya.r) + (scratch[8].i * yb.r))

            scratch[6].r = (scratch[10].i * ya.i) + (scratch[9].i * yb.i)
            scratch[6].i = -((scratch[10].r * ya.i) + (scratch[9].r * yb.i))

            C_SUB(Fout1[0], scratch[5], scratch[6])
            C_ADD(Fout4[0], scratch[5], scratch[6])

            scratch[11].r = scratch[0].r + ((scratch[7].r * yb.r) + (scratch[8].r * ya.r))
            scratch[11].i = scratch[0].i + ((scratch[7].i * yb.r) + (scratch[8].i * ya.r))
            scratch[12].r = (scratch[9].i * ya.i) - (scratch[10].i * yb.i)
            scratch[12].i = (scratch[10].r * yb.i) - (scratch[9].r * ya.i)

            C_ADD(Fout2[0], scratch[11], scratch[12])
            C_SUB(Fout3[0], scratch[11], scratch[12])

            Fout0 = Fout0[1:]
            Fout1 = Fout1[1:]
            Fout2 = Fout2[1:]
            Fout3 = Fout3[1:]
            Fout4 = Fout4[1:]


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


def opus_fft_impl(st, fout):
    fstride = np.zeros(MAXFACTORS, dtype=int)

    # shift can be -1
    shift = st.shift if st.shift > 0 else 0

    fstride[0] = 1
    L = 0
    while True:
        p = st.factors[2 * L]
        m = st.factors[2 * L + 1]
        fstride[L + 1] = fstride[L] * p
        L += 1
        if m == 1:
            break

    m = st.factors[2 * L - 1]
    for i in range(L - 1, -1, -1):
        if i != 0:
            m2 = st.factors[2 * i - 1]
        else:
            m2 = 1

        x = st.factors[2 * i]
        if x == 2:
            kf_bfly2(fout, m, fstride[i])
        elif x == 4:
            kf_bfly4(fout, fstride[i] << shift, st, m, fstride[i], m2)
        elif x == 3:
            kf_bfly3(fout, fstride[i] << shift, st, m, fstride[i], m2)
        elif x == 5:
            kf_bfly5(fout, fstride[i] << shift, st, m, fstride[i], m2)

        m = m2


def opus_fft(st, fin, fout):
    scale = st.scale

    # Bit-reverse the input
    for i in range(st.nfft):
        x = fin[i]
        fout[st.bitrev[i]].r = scale * x.r
        fout[st.bitrev[i]].i = scale * x.i

    opus_fft_impl(st, fout)
