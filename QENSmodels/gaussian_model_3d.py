import numpy as np
import math
from typing import Union, Tuple

try:
    import QENSmodels
except ImportError:
    print('Module QENSmodels not found')


def hwhm_gaussian_model3D(
        q: Union[float, list, np.ndarray],
        diffusion_coeff: float = 1.,
        variance_ux: float = 1.) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Returns some characteristics of `Gaussian Model 3D` as functions
    of the momentum transfer `q`:
    the half-width half-maximum (`hwhm`), the elastic incoherent structure
    factor (`eisf`), and the quasi-elastic incoherent structure factor (`qisf`)

    Parameters
    ----------

    q: float, list or :class:`~numpy:numpy.ndarray`

        momentum transfer (non-fitting, in 1/Angstrom)

    diffusion_coeff: float
        diffusion coefficient (in Angstrom**2/ps). Default to 1.

    variance_ux: float
        variance <u_x**2> of Gaussian random variable `u_x`
        (in Angstrom**2), displacement from the origin.
        Default to 1.

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
    >>> hwhm, eisf, qisf = hwhm_gaussian_model3D([1., 2.], 0.5, 1.5)
    >>> round(hwhm[0,10], 3), round(hwhm[1, 10], 3)
    (3.333, 3.333)
    >>> round(hwhm[0,99], 3), round(hwhm[1, 99], 3)
    (33.0, 33.0)
    >>> round(eisf[0], 3)
    0.223
    >>> round(eisf[1], 3)
    0.002
    >>> round(qisf[0, 1], 4)
    0.3347
    >>> round(qisf[1, 1], 4)
    0.0149

    """
    # Input validation
    if diffusion_coeff <= 0:
        raise ValueError("The diffusion coefficient, should be positive")
    if variance_ux <= 0:
        raise ValueError("variance_ux, the variance, should be "
                         "strictly positive")

    q = np.asarray(q, dtype=np.float64)

    number_lorentz = 100

    qisf = np.zeros((q.size, number_lorentz))
    hwhm = np.zeros((q.size, number_lorentz))
    al = np.zeros((q.size, number_lorentz))

    arg = q**2 * variance_ux

    if q.size == 1:
        for i in range(number_lorentz):
            if arg > 0:
                al[:, i] = np.exp(-arg) * arg ** i / math.factorial(i)
            else:
                if i == 0:
                    al[:, 0] = 1.
                else:
                    al[:, i] = 0.
    else:
        al[:, 0] = [np.exp(-item) if item > 0 else 1. for item in arg]

        for i in range(1, number_lorentz):
            al[:, i] = [np.exp(-item) * item ** i / math.factorial(i)
                        if item > 0 else 0. for item in arg]

    eisf = al[:, 0]

    for i in range(1, number_lorentz):
        hwhm[:, i] = np.repeat(i * diffusion_coeff / variance_ux, q.size)
        qisf[:, i] = al[:, i]

    return hwhm, eisf, qisf


def sqw_gaussian_model3D(
        w: Union[float, list, np.ndarray],
        q: Union[float, list, np.ndarray],
        scale: float = 1,
        center: float = 0,
        diffusion_coeff: float = 1.,
        variance_ux: float = 1.) -> Union[float, list, np.ndarray]:
    r"""
    Model based on Gaussian statistics

    It describes localized diffusive translational motion in 1, 2 or 3D

    Considering a particle moving along the direction x about a
    fixed point taken as the origin and u_x being the displacement
    from the origin, the model assumes that u_x is a Gaussian random
    variable with variance <u_x^2>, which quantifies the size of the
    region of confinement.

    For the 3D case, the model assumes also <u_x^2> = <u_y^2> = <u_z^2>.


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
        diffusion coefficient (in Angstrom**2/ps). Default to 1.

    variance_ux: float
        variance :math:`<u_x^2>` of Gaussian random variable u_x
        (in Angstrom^2), displacement from the origin.
        Default to 1.

    Return
    ------

    :class:`~numpy:numpy.ndarray`
        output array


    Examples
    --------
    >>> sqw = sqw_gaussian_model3D([1, 2, 3], 1, 1, 0, 1, 1)
    >>> round(sqw[0], 3)
    0.089
    >>> round(sqw[1], 3)
    0.044
    >>> round(sqw[2], 3)
    0.025

    >>> sqw = sqw_gaussian_model3D(1, 1, 1, 0, 1, 1)
    >>> round(sqw[0], 3)
    0.089


    Notes
    -----

    * The `sqw_gaussian_model3D` is expressed as

        .. math::

            S(q, \omega) = \text{delta}(\omega, A_0(q), \text{center} )
            + \sum_{i=1}^{N-1} A_i(Q)
            \text{Lorentzian}(\omega, A_i(Q)\Gamma_i, \text{center},
          \Gamma_i)

      where

      .. math::

         A_i(Q) &= \exp(-q^2<u_x^2>) \frac{(q^2<u_x^2>)^i}{i!} \\
         \Gamma_i &= \frac{i D}{<u_x^2>}

    * The number of terms in the infinite sum is limited to 100.
      According to Volino's paper, as a rule of thumb, the number of
      terms to be considered in practical calculations must be (much)
      larger than :math:`Q^2<u_x^2>`. Therefore this condition should be
      checked when using this model.

    References
    ----------

    F. Volino, J.-C. Perrin, and S. Lyonnard, *J. Phys. Chem. B* **110**,
    11217-11223 (2006) `link <https://pubs.acs.org/doi/10.1021/jp061103s>`__

    """
    # Input validation
    w = np.asarray(w)

    q = np.asarray(q, dtype=np.float64)

    # Create output array
    sqw = np.zeros((q.size, w.size))

    # Get widths, EISFs and QISFs of model
    hwhm, eisf, qisf = hwhm_gaussian_model3D(q, diffusion_coeff, variance_ux)

    # # Number of Lorentzians used to represent the infinite sum in R
    number_lorentz = hwhm.shape[1]

    # Sum of Lorentzians
    for i in range(q.size):
        sqw[i, :] = eisf[i] * QENSmodels.delta(w, scale, center)
        for j in range(1, number_lorentz):
            sqw[i, :] += qisf[i, j] * QENSmodels.lorentzian(w,
                                                            scale,
                                                            center,
                                                            hwhm[i, j])

    # For Bumps use (needed for final plotting)
    # Using a 'Curve' in bumps for each Q --> needs vector array
    if q.size == 1:
        sqw = np.reshape(sqw, w.size)

    return sqw


if __name__ == "__main__":
    import doctest
    doctest.testmod()
