import numpy as np
from typing import Union, Tuple

try:
    import QENSmodels
except ImportError:
    print('Module QENSmodels not found')


def hwhm_jump_translational_diffusion(
        q: Union[float, list, np.ndarray],
        diffusion_coeff: float = 0.23,
        residence_time: float = 1.25) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """ Returns some characteristics of `JumpTranslationalDiffusion` as functions
    of the momentum transfer `q`:
    the half-width half-maximum (`hwhm`), the elastic incoherent structure
    factor (`eisf`), and the quasi-elastic incoherent structure factor (`qisf`)

    Parameters
    ----------

    q: float, list or :class:`~numpy:numpy.ndarray`
        momentum transfer (non-fitting, in 1/Angstrom)

    diffusion_coeff: float
        diffusion coefficient (in Angstrom^2/ps). Default to 0.23.

    residence_time: float
        residence time (in ps). Default to 1.25.

    Returns
    -------

    hwhm: :class:`~numpy:numpy.ndarray`
        half-width half maximum

    eisf: :class:`~numpy:numpy.ndarray`
        elastic incoherent structure factor

    qisf: :class:`~numpy:numpy.ndarray`
        quasi-elastic incoherent structure factor


    Examples
    --------
    >>> hwhm, eisf, qisf = hwhm_jump_translational_diffusion([1., 2.], 0.5, 1.5)
    >>> round(hwhm[0], 3), round(hwhm[1], 3)
    (0.286, 0.5)
    >>> eisf
    array([0., 0.])
    >>> qisf
    array([1., 1.])


    Notes
    -----

    The default values for the fitting parameters come from the values
    for water at 298K and 1 atm, water has D=0.23 Angstrom^2/ps and
    ResTime=1.25 ps.

    """
    # Input validation
    if diffusion_coeff <= 0:
        raise ValueError("D, the diffusion coefficient, should be positive")
    if residence_time < 0:
        raise ValueError("resTime, the residence time, should be positive")

    q = np.asarray(q, dtype=np.float32)

    eisf = np.zeros(q.size)
    qisf = np.ones(q.size)
    hwhm = diffusion_coeff * q ** 2 / (1. + residence_time * diffusion_coeff * q ** 2)
    # Force hwhm to be numpy array, even if single value
    hwhm = np.asarray(hwhm, dtype=np.float32)
    hwhm = np.reshape(hwhm, hwhm.size)
    return hwhm, eisf, qisf


def sqw_jump_translational_diffusion(
        w: Union[float, list, np.ndarray],
        q: Union[float, list, np.ndarray],
        scale: float = 1.,
        center: float = 0.,
        diffusion_coeff: float = 0.23,
        residence_time: float = 1.25) -> Union[float, list, np.ndarray]:
    r"""
    Lorentzian model with half width half maximum equal to
    :math:`\frac{Dq^2}{1+ \text{resTime}Dq^2}`, where `D` is the diffusion coefficient

    It models a particle which performs jumps, randomly, between sites
    where it spends an average time `residence_time`. `residence_time` is very long
    compared to the duration of the jump.

    Parameters
    ----------

    w: float, list or :class:`~numpy:numpy.ndarray`
        energy transfer (in 1/ps)

    q: float, list or :class:`~numpy:numpy.ndarray`
        momentum transfer (non-fitting, in 1/Angstrom).

    scale: float
        scale factor. Default to 1.

    center: float
        center of peak. Default to 0.

    diffusion_coeff: float
        diffusion coefficient (in Angstrom :math:`^2` /ps). Default to 0.23.

    residence_time: float
        residence time (in ps). Default to 1.25.

    Return
    ------

    :class:`~numpy:numpy.ndarray`
        output array


    Examples
    --------

    >>> sqw = sqw_jump_translational_diffusion([1, 2, 3], 1, 1, 0, 1, 1)
    >>> round(sqw[0], 3)
    0.127
    >>> round(sqw[1], 3)
    0.037
    >>> round(sqw[2], 3)
    0.017

    >>> sqw = sqw_jump_translational_diffusion(1, 1, 1, 0, 1, 1)
    >>> round(sqw[0], 3)
    0.127


    Notes
    -----

    * The `sqw_jump_translational_diffusion` is expressed as

      .. math::

         S(q, \omega) = \text{Lorentzian}(\omega, \text{scale}, \text{center},
         \frac{D q^2}{ 1 + \text{resTime}\ D q^2})

      where `D` is the diffusion coefficient


    * The default values for the fitting parameters come from the values for
      water at 298K and 1 atm, water has `D` = 0.23 Angstrom^2/ps and
      `resTime` = 1.25 ps.


    * If `residence_time` is equal to 0, this model reduces to
      `sqw_brownian_translational_diffusion`.


    * At small `q`, `hwhm` is similar to the
      `hwhm_brownian_translational_diffusion`, *i.e.* equivalent to
      :math:`Dq^2`. And at large `q`, `hwhm` :math:`\propto` 1/`residence_time`.


    References
    ----------

    J. Teixeira, M.-C. Bellissent-Funel, S.H. Chen, and A.J, Dianoux,
    *Phys. Rev. A* **31**, 1913-1917 (1985)
    `link <https://journals.aps.org/pra/abstract/10.1103/PhysRevA.31.1913>`__

    """
    # Input validation
    w = np.asarray(w)

    q = np.asarray(q, dtype=np.float32)

    # Create output array
    sqw = np.zeros((q.size, w.size))

    # Get widths, EISFs and QISFs of model
    hwhm, eisf, qisf = hwhm_jump_translational_diffusion(
        q,
        diffusion_coeff,
        residence_time
    )

    # Model
    for i in range(q.size):
        sqw[i, :] = QENSmodels.lorentzian(w, scale, center, hwhm[i])

    # For Bumps use (needed for final plotting)
    # Using a 'Curve' in bumps for each Q --> needs vector array
    if q.size == 1:
        sqw = np.reshape(sqw, w.size)

    return sqw


if __name__ == "__main__":
    import doctest
    doctest.testmod()
