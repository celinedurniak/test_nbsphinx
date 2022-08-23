import numpy as np
from typing import Union


try:
    import QENSmodels
except ImportError:
    print('Module QENSmodels not found')


def sqw_delta_lorentz(
        w: Union[float, list, np.ndarray],
        q: Union[float, list, np.ndarray],
        scale: float = 1.0,
        center: float = 0.0,
        fraction_immobile: Union[float, list, np.ndarray] = 0.0,
        hwhm: Union[float, list, np.ndarray] = 1.0) -> Union[float, list, np.ndarray]:
    r"""
    Model corresponding to a delta representing a fraction p of
    fixed atoms and a Lorentzian corresponding to a Brownian
    Translational diffusion model for the remaining (1 - p) atoms.

    Model = fraction_immobile*delta + (1-fraction_immobile)*Lorentz(Gamma)

    Parameters
    ----------
    w: float, list or :class:`~numpy:numpy.ndarray`
        energy transfer (in 1/ps)

    q: float, list or :class:`~numpy:numpy.ndarray`
        momentum transfer (non-fitting, in 1/Angstrom)

    scale: float
        scale factor. Default to 1.

    center: float
        peak center. Default to 0.

    fraction_immobile: float, list or :class:`~numpy:numpy.ndarray` of the same size as q
        proportion of immobile atoms, must be between 0 and 1. Default to 0.

    hwhm: float, list or :class:`~numpy:numpy.ndarray` of the same size as q
        half width half maximum. Default to 1.

    Return
    ------
    :class:`~numpy:numpy.ndarray`
        output array


    Examples
    --------
    >>> sqw = sqw_delta_lorentz([1, 2, 3], 0.1)
    >>> round(sqw[0], 1)
    0.2
    >>> round(sqw[1], 3)
    0.064
    >>> round(sqw[2], 3)
    0.032


    Notes
    -----
    The `sqw_delta_lorentz` is expressed as

    .. math::

        S(q, \omega) &= \text{fraction_immobile} \delta(\omega, \text{scale}, \text{center}) \\
        &+ (1 - \text{fraction_immobile}) \text{Lorentzian}(\omega, \text{scale}, \text{center},
                                    \text{hwhm})

    """
    w = np.asarray(w)

    # Input validation
    q = np.asarray(q, dtype=np.float32)

    fraction_immobile = np.asarray(fraction_immobile, dtype=np.float32)
    hwhm = np.asarray(hwhm, dtype=np.float32)

    # Create output array
    sqw = np.zeros((q.size, w.size))

    # Model
    if q.size > 1:

        # Validator for A0. We must have 0<= fraction_immobile <= 1
        if any(item > 1 or item < 0 for item in fraction_immobile):
            raise ValueError('fraction_immobile, the proportion of immobile atoms, '
                             'should be comprised between 0 and 1, included.')

        try:
            for i in range(q.size):
                sqw[i, :] = fraction_immobile[i] * QENSmodels.delta(w, scale, center)
                sqw[i, :] += (1 - fraction_immobile[i]) * QENSmodels.lorentzian(
                    w,
                    scale,
                    center,
                    hwhm[i])

        except TypeError as detail:
            msg = "At least one parameter has an incorrect type"
            raise TypeError(detail.__str__() + "\n" + msg)
        except IndexError as detail:
            msg = "At least one array has an incorrect size"
            raise IndexError(detail.__str__() + "\n" + msg)
    else:
        if fraction_immobile > 1 or fraction_immobile < 0:
            raise ValueError('fraction_immobile, the proportion of immobile atoms, '
                             'should be comprised between 0 and 1, included.')

        sqw[0, :] = fraction_immobile * QENSmodels.delta(w, scale, center)
        sqw[0, :] += (1 - fraction_immobile) * QENSmodels.lorentzian(w, scale, center, hwhm)

    # For Bumps use (needed for final plotting)
    # Using a 'Curve' in bumps for each Q --> needs vector array
    if q.size == 1:
        sqw = np.reshape(sqw, w.size)

    return sqw


if __name__ == "__main__":
    import doctest
    doctest.testmod()
