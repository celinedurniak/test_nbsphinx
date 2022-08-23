import numpy as np
from typing import Union


try:
    import QENSmodels
except ImportError:
    print('Module QENSmodels not found')


def sqw_delta_two_lorentz(
        w: Union[float, list, np.ndarray],
        q: Union[float, list, np.ndarray],
        scale: float = 1,
        center: float = 0,
        fraction_immobile: Union[float, list, np.ndarray] = 1,
        amplitude_l1: Union[float, list, np.ndarray] = 1,
        hwhm1: Union[float, list, np.ndarray] = 1,
        hwhm2: Union[float, list, np.ndarray] = 1) -> Union[float, list, np.ndarray]:
    r"""
    Model corresponding to a delta representing a fraction p of
    fixed atoms and two Lorentzians corresponding to Brownian
    Translational diffusion model at different time scales for the remaining
    atoms.


    Model = fraction_immobile*delta
      + amplitude_L1 * Lorentzian1
      + (1-fraction_immobile-amplitude_L1)*Lorentzian2

    Parameters
    ----------
    w: float
        energy transfer (in 1/ps)

    q: float, list or :class:`~numpy:numpy.ndarray`
        momentum transfer (non-fitting, in 1/Angstrom)

    scale: float
        scale factor. Default to 1.

    center: float
        peak center. Default to 0.

    fraction_immobile: float, list or :class:`~numpy:numpy.ndarray` of the same size
        as q
        amplitude of the delta function. Default to 1.

    amplitude_l1: float, list or :class:`~numpy:numpy.ndarray` of the same size as q
        amplitude of the first Lorentzian. Default to 1.

    hwhm1: float, list or :class:`~numpy:numpy.ndarray` of the same size as q
        half-width half maximum of the first Lorentzian. Default to 1.

    hwhm2: float, list or :class:`~numpy:numpy.ndarray` of the same size as q
        half-width half maximum of the second Lorentzian. Default to 1.

    Return
    ------
    :class:`~numpy:numpy.ndarray`
        output array


    Examples
    --------
    >>> sqw = sqw_delta_two_lorentz([1, 2, 3], [0.1, 0.2], 1, 1, [1, 1], [1, 1], [0.01, 0.01], [0.01, 0.01])  # noqa: E501
    >>> round(sqw[0, 0])
    1
    >>> sqw[0, 1]
    0.0
    >>> sqw[0, 2]
    0.0
    >>> round(sqw[1, 0])
    1
    >>> sqw[1, 1]
    0.0
    >>> sqw[1, 2]
    0.0

    >>> sqw = sqw_delta_two_lorentz([1, 2, 3], [0.05, 0.3], 0.5, 2, [0.75, 0.5], [1, 2], [0.05, 0.04], [0.02, 0.03])  # noqa: E501
    >>> round(sqw[0, 0], 3)
    0.001
    >>> round(sqw[0, 1], 3)
    0.499
    >>> round(sqw[0, 2], 3)
    0.001
    >>> round(sqw[1, 0], 3)
    0.001
    >>> round(sqw[1, 1], 3)
    0.499
    >>> round(sqw[1, 2], 3)
    0.001

    Notes
    -----
    .. math::

        S(q, \omega) &= \text{fraction_immobile} \text{delta}(\omega - \text{center}) \\
        &+ amplitude_l1 \text{Lorentzian}(\omega, \text{scale}, \text{center},
        \text{hwhm}_1) \\
        &+ (1 - \text{fraction_immobile} - amplitude_l1) \text{Lorentzian}(\omega, \text{scale},
        \text{center}, \text{hwhm}_2)

    """

    # Input validation
    w = np.asarray(w)

    q = np.asarray(q, dtype=np.float32)

    fraction_immobile = np.asarray(fraction_immobile, dtype=np.float32)
    amplitude_l1 = np.asarray(amplitude_l1, dtype=np.float32)
    hwhm1 = np.asarray(hwhm1, dtype=np.float32)
    hwhm2 = np.asarray(hwhm2, dtype=np.float32)

    # Create output array
    sqw = np.zeros((q.size, w.size))

    # Model
    if q.size > 1:
        try:
            for i in range(q.size):
                sqw[i, :] = fraction_immobile[i] * QENSmodels.delta(w, scale, center)
                sqw[i, :] += amplitude_l1[i] * QENSmodels.lorentzian(
                    w,
                    scale,
                    center,
                    hwhm1[i]
                )
                sqw[i, :] += (1 - fraction_immobile[i] - amplitude_l1[i]) * QENSmodels.lorentzian(
                    w,
                    scale,
                    center,
                    hwhm2[i]
                )
        except TypeError as detail:
            msg = "At least one parameter has an incorrect type"
            raise TypeError(detail.__str__() + "\n" + msg)
        except IndexError as detail:
            msg = "At least one array has an incorrect size"
            raise IndexError(detail.__str__() + "\n" + msg)
    else:
        sqw[0, :] = fraction_immobile * QENSmodels.delta(
            w,
            scale,
            center
        )
        sqw[0, :] += amplitude_l1 * QENSmodels.lorentzian(
            w,
            scale,
            center,
            hwhm1
        )
        sqw[0, :] += (1. - fraction_immobile - amplitude_l1) * QENSmodels.lorentzian(
            w,
            scale,
            center,
            hwhm2
        )

    # For Bumps use (needed for final plotting)
    # Using a 'Curve' in bumps for each Q --> needs vector array
    if q.size == 1:
        sqw = np.reshape(sqw, w.size)

    return sqw


if __name__ == "__main__":
    import doctest
    doctest.testmod()
