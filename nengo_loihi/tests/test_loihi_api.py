import numpy as np
import pytest

from nengo.utils.numpy import rms

from nengo_loihi.loihi_api import overflow_signed
from nengo_loihi.loihi_api import decay_magnitude, decay_int


@pytest.mark.parametrize("b", (8, 16, 17, 23))
def test_overflow_signed(b):
    x = np.arange(-2**(b+2), 2**(b+2), dtype=np.int32)

    # compute check values
    b2 = 2**b
    z = x % b2
    zmask = np.right_shift(x, b) % 2  # sign bit, the b-th bit

    # if the sign bit is set, subtract 2**b to make it negative
    z -= np.left_shift(zmask, b)

    # compute whether it's overflowed
    q = (x < -b2) | (x >= b2)

    y, o = overflow_signed(x, bits=b)
    assert np.array_equal(y, z)
    assert np.array_equal(o, q)


@pytest.mark.parametrize('offset', (0, 1))
def test_decay_magnitude(offset, plt, logger):
    bits = 12
    # decays = np.arange(1, 2**bits)
    decays = np.arange(1, 2**bits, 7)
    ref = []
    emp = []
    est = []

    for decay in decays:
        def empirical_decay_magnitude(decay, x0, invert=False):
            x = x0 * np.ones(decay.shape, dtype=np.int32)
            y = x
            y_sum = np.zeros(decay.shape, dtype=np.int64)
            for i in range(100000):
                y_sum += y
                if (y <= 0).all():
                    break
                y = decay_int(y, decay, bits=bits, offset=offset)
            else:
                raise RuntimeError("Exceeded max number of iterations")

            return x0 / y_sum if invert else y_sum / x0

        # x0 = np.arange(2**10, 2**11, dtype=np.int32)
        x0 = np.arange(2**21 - 1000, 2**21, step=41, dtype=np.int32)
        # x0 = np.arange(2**21 - 1000, 2**21, dtype=np.int32)

        m = empirical_decay_magnitude(decay * np.ones_like(x0), x0)
        m0 = m.mean()
        emp.append(m0)

        # reference (naive) method, not accounting for truncation loss
        r = (2**bits - offset - decay) / 2**bits
        # Sx0 = 1. / (1 - r)  # sum_i^n x_i/x_0, where x_n = r**n*x_0 = 0
        # ref.append(Sx0)
        Sx1 = (1 - r/x0) / (1 - r)  # sum_i^n x_i/x_0, where x_n = r**n*x_0 = 1
        ref.append(Sx1.mean())

        m = decay_magnitude(decay, x0, bits=bits, offset=offset)
        m2 = m.mean()
        est.append(m2)

    ref = np.array(ref)
    emp = np.array(emp)
    est = np.array(est)
    rms_ref = rms(ref - emp) / rms(emp)
    rms_est = rms(est - emp) / rms(emp)
    logger.info("Ref rel RMSE: %0.3e, decay_magnitude rel RMSE: %0.3e" % (
        rms_ref, rms_est))

    abs_ref = np.abs(ref - emp)
    abs_est = np.abs(est - emp)

    # find places where ref is better than est
    relative_diff = (abs_est - abs_ref) / emp

    ax = plt.subplot(211)
    plt.plot(abs_ref, 'b')
    plt.plot(abs_est, 'g')
    ax.set_yscale('log')

    ax = plt.subplot(212)
    plt.plot(relative_diff.clip(0, None))

    assert np.all(relative_diff < 1e-6)
