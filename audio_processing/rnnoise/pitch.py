import math


def find_best_pitch(xcorr, y, _len, max_pitch, best_pitch):
    Syy = 1
    best_num = [-1, -1]
    best_den = [0, 0]

    best_pitch[0] = 0
    best_pitch[1] = 1
    for j in range(_len):
        Syy = Syy + (y[j] * y[j])
    for i in range(max_pitch):
        if xcorr[i] > 0:
            num = xcorr[i] * xcorr[i]
            if num * best_den[1] > best_num[1] * Syy:
                if num * best_den[0] > best_num[0] * Syy:
                    best_num[1] = best_num[0]
                    best_den[1] = best_den[0]
                    best_pitch[1] = best_pitch[0]
                    best_num[0] = num
                    best_den[0] = Syy
                    best_pitch[0] = i
                else:
                    best_num[1] = num
                    best_den[1] = Syy
                    best_pitch[1] = i

        Syy += (y[i + _len] * y[i + _len]) - (y[i] * y[i])
        Syy = max(1, Syy)


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


def dual_inner_prod(x, y01, y02, N):
    xy01 = xy02 = 0
    for i in range(N):
        xy01 = xy01 + x[i] * y01[i]
        xy02 = xy02 + x[i] * y02[i]
    return xy01, xy02


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
        xcorr_kernel(_x, _y[i:], _sum, _len)
        xcorr[i] = _sum[0]
        xcorr[i + 1] = _sum[1]
        xcorr[i + 2] = _sum[2]
        xcorr[i + 3] = _sum[3]
    i += 4

    # In case max_pitch isn't a multiple of 4, do non-unrolled version.
    for i in range(i, max_pitch):
        _sum = celt_inner_prod(_x, _y[i:], _len)
        xcorr[i] = _sum


def pitch_search(x_lp, y, _len, max_pitch):
    best_pitch = [0, 0]
    lag = _len + max_pitch

    x_lp4 = [0] * (_len >> 2)
    y_lp4 = [0] * (lag >> 2)
    xcorr = [0] * (max_pitch >> 1)

    # Downsample by 2 again
    for j in range(_len >> 2):
        x_lp4[j] = x_lp[2 * j]
    for j in range(lag >> 2):
        y_lp4[j] = y[2 * j]

    # Coarse search with 4x decimation

    celt_pitch_xcorr(x_lp4, y_lp4, xcorr, _len >> 2, max_pitch >> 2)

    find_best_pitch(xcorr, y_lp4, _len >> 2, max_pitch >> 2, best_pitch)

    # Finer search with 2x decimation
    for i in range(max_pitch >> 1):
        xcorr[i] = 0
        if abs(i - 2 * best_pitch[0]) > 2 and abs(i - 2 * best_pitch[1]) > 2:
            continue
        _sum = celt_inner_prod(x_lp, y[i:], _len >> 1)
        xcorr[i] = max(-1, _sum)
    find_best_pitch(xcorr, y, _len >> 1, max_pitch >> 1, best_pitch)

    # Refine by pseudo-interpolation
    offset = 0
    if 0 < best_pitch[0] < (max_pitch >> 1) - 1:
        a = xcorr[best_pitch[0] - 1]
        b = xcorr[best_pitch[0]]
        c = xcorr[best_pitch[0] + 1]
        if (c - a) > .7 * (b - a):
            offset = 1
        elif (a - c) > .7 * (b - c):
            offset = -1

    pitch = 2 * best_pitch[0] - offset
    return pitch


def compute_pitch_gain(xy, xx, yy):
    return xy / math.sqrt(1 + xx * yy)


second_check = [0, 0, 3, 2, 3, 2, 5, 2, 3, 2, 3, 2, 5, 2, 3, 2]


def remove_doubling(x, maxperiod, minperiod, N, T0_, prev_period, prev_gain):
    xcorr = [0] * 3

    minperiod0 = minperiod
    maxperiod //= 2
    minperiod //= 2
    T0_[0] //= 2
    prev_period //= 2
    N //= 2
    x0 = x
    x = x0[maxperiod:]
    if T0_[0] >= maxperiod:
        T0_[0] = maxperiod - 1

    T = T0 = T0_[0]
    yy_lookup = [0] * (maxperiod + 1)
    xx, xy = dual_inner_prod(x, x, x0[maxperiod - T0:], N)
    yy_lookup[0] = xx
    yy = xx
    for i in range(1, maxperiod + 1):
        yy = yy + (x0[maxperiod - i] * x0[maxperiod - i]) - (x[N - i] * x[N - i])
        yy_lookup[i] = max(0, yy)
    yy = yy_lookup[T0]
    best_xy = xy
    best_yy = yy
    g = g0 = compute_pitch_gain(xy, xx, yy)
    # Look for any pitch at T/k
    for k in range(2, 15 + 1):
        T1 = (2 * T0 + k) // (2 * k)
        if T1 < minperiod:
            break
        # Look for another strong correlation at T1b
        if k == 2:
            if T1 + T0 > maxperiod:
                T1b = T0
            else:
                T1b = T0 + T1
        else:
            T1b = (2 * second_check[k] * T0 + k) // (2 * k);

        xy, xy2 = dual_inner_prod(x, x0[maxperiod - T1:], x0[maxperiod - T1b:], N)
        xy = .5 * (xy + xy2)
        yy = .5 * (yy_lookup[T1] + yy_lookup[T1b])
        g1 = compute_pitch_gain(xy, xx, yy)
        if abs(T1 - prev_period) <= 1:
            cont = prev_gain
        elif abs(T1 - prev_period) <= 2 and 5 * k * k < T0:
            cont = .5 * prev_gain
        else:
            cont = 0
        thresh = max(.3, (.7 * g0) - cont)

        # Bias against very high pitch (very short period) to avoid false-positives
        # due to short-term correlation

        if T1 < 3 * minperiod:
            thresh = max(.4, (.85 * g0) - cont)
        elif T1 < 2 * minperiod:
            thresh = max(.5, (.9 * g0) - cont)
        if g1 > thresh:
            best_xy = xy
            best_yy = yy
            T = T1
            g = g1

    best_xy = max(0, best_xy)
    if best_yy <= best_xy:
        pg = 1.
    else:
        pg = best_xy / (best_yy + 1)

    for k in range(3):
        xcorr[k] = celt_inner_prod(x, x0[maxperiod - (T + k - 1):], N)
    if xcorr[2] - xcorr[0] > .7 * (xcorr[1] - xcorr[0]):
        offset = 1
    elif xcorr[0] - xcorr[2] > .7 * (xcorr[1] - xcorr[2]):
        offset = -1
    else:
        offset = 0

    if pg > g:
        pg = g
    T0_[0] = 2 * T + offset

    if T0_[0] < minperiod0:
        T0_[0] = minperiod0

    return pg
