import numpy as np
from typing import Union

try:
    import QENSmodels
except ImportError:
    print('Module QENSmodels not found')


def sqw_water_teixeira(
        w: Union[float, list, np.ndarray],
        q: Union[float, list, np.ndarray],
        scale: float = 1,
        center: float = 0,
        diffusion_coeff: float = 0.23,
        residence_time: float = 1.25,
        radius: float = 1,
        rot_diffusion_coeff: float = 1) -> Union[float, list, np.ndarray]:
    r"""
    Model corresponding to the convolution of `Jump Translational
    diffusion` (model T) and `Isotropic rotational diffusion` (model R)

    Model = convolution(T, R), where

    T = Jump Translational diffusion = Lorentz(Gamma_T)

    R = Isotropic rotational diffusion = A0 + A1*L1 + A2*L2 + ...

    This results in

    Model = A0*Lorentz(Gamma_T) + A1*Lorentz(Gamma_T+Gamma_1)
    + A2*Lorentz(Gamma_T+Gamma_2) + ...

    Parameters
    ----------
    w: float, list or :class:`~numpy:numpy.ndarray`
        energy transfer (in 1/ps)

    q: float, list or :class:`~numpy:numpy.ndarray`
        momentum transfer (non-fitting, in 1/Angstrom)

    scale: float
        scale factor. Default to 1.

    center: float
        center of peak. Default to 0.

    diffusion_coeff: float
        Diffusion coefficient (in Angstrom^2/ps). Default to 1.

    residence_time: float
        Residence time (in ps). Default to 1.

    radius: float
        radius of rotation (in Angstrom). Default to 1.

    rot_diffusion_coeff: float
        rotational diffusion coefficient (in 1/ps). Default to 1.


    Return
    ------

    :class:`~numpy:numpy.ndarray`
        output array


    Examples
    --------

    >>> result = sqw_water_teixeira(1, 1, 1, 1, 1, 1, 1, 1)
    >>> round(result[0], 3)
    0.486


    Notes
    -----

    The default values for the fitting parameters come from the values
    for water at 298K and 1 atm, water has diffusion_coeff=0.23 Angstrom^2/ps
    and residence_time=1.25 ps.

    References
    ----------

    J. Teixeira, M.-C. Bellissent-Funel, S.H. Chen, and A.J, Dianoux,
    *Phys. Rev. A* **31**, 1913-1917 (1985)
    `link <https://journals.aps.org/pra/abstract/10.1103/PhysRevA.31.1913>`__

    """
    # Input validation
    w = np.asarray(w, dtype=np.float32)

    q = np.asarray(q, dtype=np.float32)

    # Create output array
    sqw = np.zeros((q.size, w.size))

    # Get widths, EISFs and QISFs of each model
    hwhm1, eisf1, qisf1 = QENSmodels.jump_translational_diffusion.\
        hwhm_jump_translational_diffusion(q, diffusion_coeff, residence_time)
    hwhm2, eisf2, qisf2 = QENSmodels.isotropic_rotational_diffusion.\
        hwhm_isotropic_rotational_diffusion(q, radius, rot_diffusion_coeff)

    # Number of Lorentzians used to represent the infinite sum in R
    number_lorentz = hwhm2.shape[1]

    # Sum of Lorentzians giving the full model
    for i in range(q.size):
        sqw[i, :] = eisf2[i] * QENSmodels.lorentzian(
            w,
            scale,
            center,
            hwhm1[i]
        )
        for j in range(1, number_lorentz):
            sqw[i, :] += qisf2[i, j] * QENSmodels.lorentzian(
                w,
                scale,
                center,
                hwhm1[i] + hwhm2[i, j]
            )

    # For Bumps use (needed for final plotting)
    # Using a 'Curve' in bumps for each Q --> needs vector array
    if q.size == 1:
        sqw = np.reshape(sqw, w.size)

    return sqw


if __name__ == "__main__":
    import doctest
    doctest.testmod()
