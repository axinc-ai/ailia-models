def _celt_lpc(lpc, ac, p):
    """
    lpc (out): [0...p-1] LPC coefficients
    ac (in):  [0...p] autocorrelation values
    """
    error = ac[0]

    for i in range(p):
        lpc[i] = 0

    if ac[0] != 0:
        for i in range(p):
            # Sum up this iteration's reflection coefficient
            rr = 0
            for j in range(i):
                rr += lpc[j] * ac[i - j]
            rr += ac[i + 1]
            r = -rr / error
            # Update LPC coefficients and total error
            lpc[i] = r

            for j in range((i + 1) >> 1):
                tmp1 = lpc[j]
                tmp2 = lpc[i - 1 - j]
                lpc[j] = tmp1 + (r * tmp2)
                lpc[i - 1 - j] = tmp2 + (r * tmp1)

            error = error - ((r * r) * error)
            # Bail out once we get 30 dB gain
            if error < .001 * ac[0]:
                break


def _celt_autocorr(x, ac, window, overlap, lag, n):
    """
    x: (in) [0...n-1] samples x
    ac: (out) [0...lag-1] ac values
    """

    fastN = n - lag
    xx = [0] * n
    if overlap == 0:
        xptr = x
    else:
        for i in range(n):
            xx[i] = x[i]
        for i in range(overlap):
            xx[i] = x[i] * window[i]
            xx[n - i - 1] = x[n - i - 1] * window[i]
        xptr = xx

    shift = 0
    celt_pitch_xcorr(xptr, xptr, ac, fastN, lag + 1)

    for k in range(lag + 1):
        d = 0
        for i in range(k + fastN, n):
            d = d + (xptr[i] * xptr[i - k])
        ac[k] += d

    return shift


def celt_fir5(x, num, y, N, mem):
    num0 = num[0]
    num1 = num[1]
    num2 = num[2]
    num3 = num[3]
    num4 = num[4]
    mem0 = mem[0]
    mem1 = mem[1]
    mem2 = mem[2]
    mem3 = mem[3]
    mem4 = mem[4]
    for i in range(N):
        _sum = x[i]
        _sum = _sum + num0 * mem0
        _sum = _sum + num1 * mem1
        _sum = _sum + num2 * mem2
        _sum = _sum + num3 * mem3
        _sum = _sum + num4 * mem4
        mem4 = mem3
        mem3 = mem2
        mem2 = mem1
        mem1 = mem0
        mem0 = x[i]
        y[i] = _sum

    mem[0] = mem0
    mem[1] = mem1
    mem[2] = mem2
    mem[3] = mem3
    mem[4] = mem4


def pitch_downsample(x, x_lp, _len, C):
    ac = [0] * 5
    tmp = 1.
    lpc = [0] * 4
    mem = [0] * 5
    lpc2 = [0] * 5
    c1 = .8

    for i in range(1, _len >> 1):
        x_lp[i] = .5 * (.5 * (x[0][(2 * i - 1)] + x[0][(2 * i + 1)]) + x[0][2 * i])
    x_lp[0] = .5 * (.5 * (x[0][1]) + x[0][0])
    if C == 2:
        for i in range(1, _len >> 2):
            x_lp[i] += .5 * (.5 * (x[1][(2 * i - 1)] + x[1][(2 * i + 1)]) + x[1][2 * i])
        x_lp[0] += .5 * (.5 * (x[1][1]) + x[1][0])

    _celt_autocorr(x_lp, ac, None, 0, 4, _len >> 1)

    # Noise floor -40 dB
    ac[0] *= 1.0001

    # Lag windowing
    for i in range(1, 4 + 1):
        ac[i] -= ac[i] * (.008 * i) * (.008 * i)

    _celt_lpc(lpc, ac, 4)
    for i in range(4):
        tmp = .9 * tmp
        lpc[i] = lpc[i] * tmp

    # Add a zero
    lpc2[0] = lpc[0] + .8
    lpc2[1] = lpc[1] + c1 * lpc[0]
    lpc2[2] = lpc[2] + c1 * lpc[1]
    lpc2[3] = lpc[3] + c1 * lpc[2]
    lpc2[4] = c1 * lpc[3]
    celt_fir5(x_lp, lpc2, x_lp, _len >> 1, mem)


def xcorr_kernel(x, y, _sum, _len):
    y_0 = y[0]
    y_1 = y[1]
    y_2 = y[2]
    y = y[3:]
    for j in range(0, _len - 3, 4):
        tmp = x[0]
        y_3 = y[0]
        x = x[1:]
        y = y[1:]
        _sum[0] = _sum[0] + tmp * y_0
        _sum[1] = _sum[1] + tmp * y_1
        _sum[2] = _sum[2] + tmp * y_2
        _sum[3] = _sum[3] + tmp * y_3
        tmp = x[0]
        y_0 = y[0]
        x = x[1:]
        y = y[1:]
        _sum[0] = _sum[0] + tmp * y_1
        _sum[1] = _sum[1] + tmp * y_2
        _sum[2] = _sum[2] + tmp * y_3
        _sum[3] = _sum[3] + tmp * y_0
        tmp = x[0]
        y_1 = y[0]
        x = x[1:]
        y = y[1:]
        _sum[0] = _sum[0] + tmp * y_2
        _sum[1] = _sum[1] + tmp * y_3
        _sum[2] = _sum[2] + tmp * y_0
        _sum[3] = _sum[3] + tmp * y_1
        tmp = x[0]
        y_2 = y[0]
        x = x[1:]
        y = y[1:]
        _sum[0] = _sum[0] + tmp * y_3
        _sum[1] = _sum[1] + tmp * y_0
        _sum[2] = _sum[2] + tmp * y_1
        _sum[3] = _sum[3] + tmp * y_2
    j += 4
    if j < _len:
        tmp = x[0]
        y_3 = y[0]
        x = x[1:]
        y = y[1:]
        _sum[0] = _sum[0] + tmp * y_0
        _sum[1] = _sum[1] + tmp * y_1
        _sum[2] = _sum[2] + tmp * y_2
        _sum[3] = _sum[3] + tmp * y_3
    j += 1
    if j < _len:
        tmp = x[0]
        y_0 = y[0]
        x = x[1:]
        y = y[1:]
        _sum[0] = _sum[0] + tmp * y_1
        _sum[1] = _sum[1] + tmp * y_2
        _sum[2] = _sum[2] + tmp * y_3
        _sum[3] = _sum[3] + tmp * y_0
    j += 1
    if j < _len:
        tmp = x[0]
        y_1 = y[0]
        _sum[0] = _sum[0] + tmp * y_2
        _sum[1] = _sum[1] + tmp * y_3
        _sum[2] = _sum[2] + tmp * y_0
        _sum[3] = _sum[3] + tmp * y_1


def celt_inner_prod(x, y, N):
    xy = 0
    for i in range(N):
        xy = xy + x[i] * y[i]
    return xy


def celt_pitch_xcorr(_x, _y, xcorr, _len, max_pitch):
    # The EDSP version requires that max_pitch is at least 1, and that _x is 32-bit aligned.
    # Since it's hard to put asserts in assembly, put them here.
    for i in range(0, max_pitch - 3, 4):
        _sum = [0, 0, 0, 0]
        xcorr_kernel(_x, _y + i, _sum, _len)
        xcorr[i] = _sum[0]
        xcorr[i + 1] = _sum[1]
        xcorr[i + 2] = _sum[2]
        xcorr[i + 3] = _sum[3]
    i += 4

    # In case max_pitch isn't a multiple of 4, do non-unrolled version.
    for i in range(i, max_pitch):
        _sum = celt_inner_prod(_x, _y[i:], _len)
        xcorr[i] = _sum
