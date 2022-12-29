import numpy as np
from typing import Union


def delta(
        x: Union[float, list, np.ndarray],
        scale: Union[float, list, np.ndarray] = 1,
        center: Union[float, list, np.ndarray] = 0
) -> Union[float, list, np.ndarray]:
    r""" Dirac Delta function

    It is equal to zero except for the value of `x` closest to `center`.

    Parameters
    ----------
    x: list or :class:`~numpy:numpy.ndarray`
        domain of the function

    scale: float
        integrated intensity of the curve. Default to 1.

    center: float
        position of the peak. Default to 0.

    Return
    ------
    :class:`~numpy:numpy.ndarray`
        output array containing an impulse signal

    Examples
    --------
    >>> delta([0, 1, 2], 1, 0)
    array([1., 0., 0.])

    >>> delta([0, 1, 2, 3, 4], 5, 2)
    array([0., 0., 5., 0., 0.])


    Notes
    -----
    * A Delta (Dirac) function is defined as

    .. math::

        \text{Delta}(x, \text{scale}, \text{center}) = \text{scale}\
        \delta(x - \text{center})


    * For non-zero values, the amplitude of the Delta function is divided by
      the x-spacing.

    * **Equivalence between different implementations**

      +-------------+--------------------+
      | QENSmodels  | Mantid             |
      +=============+====================+
      | ``delta``   | ``DeltaFunction``  |
      +-------------+--------------------+
      | ``scale``   |  Height            |
      +-------------+--------------------+
      | ``center``  |  Centre            |
      +-------------+--------------------+

    """
    # Input validation
    if isinstance(x, (float, int)):
        x = [float(x)]

    x = np.asarray(x)

    model = np.zeros(x.size)

    try:
        if min(x) <= center <= max(x):
            # if center within x-range, delta is non-zero in this interval
            # otherwise do nothing
            idx = np.argmin(np.abs(x - center))
            if len(x) > 1:
                dx = (max(x) - min(x)) / (len(x) - 1)  # domain spacing
            else:
                dx = 1.
            model[idx] = scale / dx

    finally:
        return model
