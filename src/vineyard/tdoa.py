from typing import Callable

import numpy as np
from scipy.optimize import leastsq


def functions(x0, y0, x1, y1, x2, y2, d01, d02, d12):
    """Given observers at (x0, y0), (x1, y1), (x2, y2) and TDOA between
    observers d01, d02, d12, this closure returns a function that evaluates
    the system of three hyperbolae for given event x, y.
    """

    def fn(args):
        x, y = args
        a = (
            np.sqrt(np.power(x - x1, 2.0) + np.power(y - y1, 2.0))
            - np.sqrt(np.power(x - x0, 2.0) + np.power(y - y0, 2.0))
            - d01
        )
        b = (
            np.sqrt(np.power(x - x2, 2.0) + np.power(y - y2, 2.0))
            - np.sqrt(np.power(x - x0, 2.0) + np.power(y - y0, 2.0))
            - d02
        )
        c = (
            np.sqrt(np.power(x - x2, 2.0) + np.power(y - y2, 2.0))
            - np.sqrt(np.power(x - x1, 2.0) + np.power(y - y1, 2.0))
            - d12
        )
        return [a, b, c]

    return fn


def jacobian(x0, y0, x1, y1, x2, y2, d01, d02, d12):
    def fn(args):
        x, y = args
        adx = (x - x1) / np.sqrt(np.power(x - x1, 2.0) + np.power(y - y1, 2.0)) - (
            x - x0
        ) / np.sqrt(np.power(x - x0, 2.0) + np.power(y - y0, 2.0))
        bdx = (x - x2) / np.sqrt(np.power(x - x2, 2.0) + np.power(y - y2, 2.0)) - (
            x - x0
        ) / np.sqrt(np.power(x - x0, 2.0) + np.power(y - y0, 2.0))
        cdx = (x - x2) / np.sqrt(np.power(x - x2, 2.0) + np.power(y - y2, 2.0)) - (
            x - x1
        ) / np.sqrt(np.power(x - x1, 2.0) + np.power(y - y1, 2.0))
        ady = (y - y1) / np.sqrt(np.power(x - x1, 2.0) + np.power(y - y1, 2.0)) - (
            y - y0
        ) / np.sqrt(np.power(x - x0, 2.0) + np.power(y - y0, 2.0))
        bdy = (y - y2) / np.sqrt(np.power(x - x2, 2.0) + np.power(y - y2, 2.0)) - (
            y - y0
        ) / np.sqrt(np.power(x - x0, 2.0) + np.power(y - y0, 2.0))
        cdy = (y - y2) / np.sqrt(np.power(x - x2, 2.0) + np.power(y - y2, 2.0)) - (
            y - y1
        ) / np.sqrt(np.power(x - x1, 2.0) + np.power(y - y1, 2.0))

        return [[adx, ady], [bdx, bdy], [cdx, cdy]]

    return fn


def tdoa(x, y, t, speed, xp, yp) -> tuple[float, float, Callable]:
    x0, y0, t0 = x[0], y[0], t[0]
    x1, y1, t1 = x[1], y[1], t[1]
    x2, y2, t2 = x[2], y[2], t[2]
    
    F = functions(
        x0, y0, x1, y1, x2, y2, (t1 - t0) * speed, (t2 - t0) * speed, (t2 - t1) * speed
    )
    J = jacobian(
        x0, y0, x1, y1, x2, y2, (t1 - t0) * speed, (t2 - t0) * speed, (t2 - t1) * speed
    )

    pos, _ = leastsq(F, x0=[xp, yp], Dfun=J)
    return pos[0], pos[1], F
