import numpy as np
from typing import Union, Tuple

try:
    import QENSmodels
except ImportError:
    print('Module QENSmodels not found')


def hwhm_chudley_elliott_diffusion(
        q: Union[float, list, np.ndarray],
        diffusion_coeff: float = 0.23,
        jump_length: float = 1.0) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """ Returns some characteristics of `Chudley Elliott Diffusion` as functions
    of the momentum transfer `q`:
    the half-width half-maximum (`hwhm`), the elastic incoherent structure
    factor (`eisf`), and the quasi-elastic incoherent structure factor (`qisf`)

    Parameters
    ----------

    q: float, list or :class:`~numpy:numpy.ndarray`
        momentum transfer (non-fitting, in 1/Angstrom)

    diffusion_coeff: float
        diffusion coefficient (in Angstrom^2/ps). Default to 0.23.

    jump_length: float
        jump length (in Angstrom). Default to 1.0.


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
    >>> hwhm, eisf, qisf = hwhm_chudley_elliott_diffusion([1., 2.], 0.5, 1.5)
    >>> round(hwhm[0], 3), round(hwhm[1], 3)
    (0.447, 1.271)
    >>> eisf
    array([0., 0.])
    >>> qisf
    array([1., 1.])

    """
    # input validation
    if diffusion_coeff <= 0:
        raise ValueError('The diffusion coefficient should be positive')
    if jump_length <= 0:
        raise ValueError('The jump length should be positive')

    q = np.asarray(q, dtype=np.float32)

    eisf = np.zeros(q.size)
    qisf = np.ones(q.size)
    hwhm = 6. * diffusion_coeff * (1. - np.sinc(q * jump_length / np.pi)) / jump_length ** 2

    # Force hwhm to be numpy array, even if single value
    hwhm = np.asarray(hwhm, dtype=np.float32)
    hwhm = np.reshape(hwhm, hwhm.size)

    return hwhm, eisf, qisf


def sqw_chudley_elliott_diffusion(
        w: Union[float, list, np.ndarray],
        q: Union[float, list, np.ndarray],
        scale: float = 1,
        center: float = 0,
        diffusion_coeff: float = 0.23,
        jump_length: float = 1.0) -> Union[float, list, np.ndarray]:
    r""" Lorentzian model with half width half maximum equal to

     .. math::

      \frac{6\text{diffusion_coeff}}{\text{jump_length}^2}
      (1 - \frac{sin(Q\text{jump_length}/pi)}{Q\text{jump_length}/pi})


    It is a model originally developed for jump diffusion in
    liquids. But it can also be applied to diffusion in
    crystalline lattices.

    Atoms or molecules are `caged` by other atoms and jump into
    a neighbouring cage from time to time.

    The jump length is identical for all sites.


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
        diffusion coefficient (in Angstrom^2/ps). Default to 0.23.

    jump_length: float
        jump distance (in Angstrom). Default to 1.0.

    Return
    ------

    :class:`~numpy:numpy.ndarray`
        output array


    Examples
    --------
    >>> sqw = sqw_chudley_elliott_diffusion(([1, 2, 3], 1, 1, 0, 1, 1)
    >>> round(sqw[0], 3)
    0.159
    >>> round(sqw[1], 3)
    0.062
    >>> round(sqw[2], 3)
    0.031

    >>> sqw = sqw_chudley_elliott_diffusion(1, 1, 1, 0, 1, 1)
    >>> round(sqw[0], 3)
    0.159


    Notes
    -----

    * The `sqw_chudley_elliott_diffusion` is expressed as

      .. math::

          S(q, \omega) = \text{Lorentzian}(\omega, \text{scale}, \text{center},
          \frac{6D}{l^2}(1 - \frac{sin(Ql/pi)}{Ql/pi}))

    * Note that an equivalent expression is

      .. math::

          S(q, \omega) = \text{Lorentzian}(\omega, \text{scale}, \text{center},
          \frac{1}{\tau}(1 - \frac{sin(Ql/pi)}{Ql/pi}))

      with :math:`\tau=\frac{l^2}{6D}`.


    References
    ----------

    * R. Hempelmann, Quasielastic Neutron Scattering and Solid State Diffusion
     (Oxford, 2000).

    * C. T. Chudley and R. J. Elliott,  *Proc. Phys. Soc.* **77**,
      353-361 (1961)
`link <https://iopscience.iop.org/article/10.1088/0370-1328/77/2/319/meta>`_

    """
    # Input validation
    w = np.asarray(w)

    q = np.asarray(q, dtype=np.float32)

    # Create output array
    sqw = np.zeros((q.size, w.size))

    # Get widths, EISFs and QISFs of model
    hwhm, eisf, qisf = hwhm_chudley_elliott_diffusion(q, diffusion_coeff, jump_length)

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
