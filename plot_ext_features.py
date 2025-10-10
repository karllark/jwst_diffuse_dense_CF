import argparse
import matplotlib.pyplot as plt
import matplotlib.patches as patches
import numpy as np
import astropy.units as u
from astropy.modeling.models import Drude1D, Polynomial1D, PowerLaw1D, Gaussian1D

from dust_extinction.baseclasses import BaseExtRvModel
from dust_extinction.helpers import _smoothstep
from dust_extinction.shapes import _modified_drude, FM90
from dust_extinction.parameter_averages import G23

x_range_G23 = [1.0 / 32.0, 1.0 / 0.0912]

# modify G23 models to get just continuum and/or features
class G23_nofeatures(BaseExtRvModel):
    r"""
    Gordon et al. (2023) Milky Way R(V) dependent model

    Parameters
    ----------
    Rv: float
        R(V) = A(V)/E(B-V) = total-to-selective extinction

    Raises
    ------
    InputParameterError
       Input Rv values outside of defined range

    Notes
    -----
    From Gordon et al. (2023, ApJ, in press)

    Example showing CCM89 curves for a range of R(V) values.

    .. plot::
        :include-source:

        import numpy as np
        import matplotlib.pyplot as plt
        import astropy.units as u

        from dust_extinction.parameter_averages import G23

        fig, ax = plt.subplots()

        # generate the curves and plot them
        lam = np.logspace(np.log10(0.0912), np.log10(30.0), num=1000) * u.micron

        Rvs = [2.5, 3.1, 4.0, 4.75, 5.5]
        for cur_Rv in Rvs:
           ext_model = G23(Rv=cur_Rv)
           ax.plot(lam,ext_model(lam),label='R(V) = ' + str(cur_Rv))

        ax.set_xscale('log')
        ax.set_yscale('log')

        ax.set_xlabel('$\lambda$ [$\mu$m]')
        ax.set_ylabel(r'$A(x)/A(V)$')

        ax.legend(loc='best')
        plt.show()
    """

    Rv_range = [2.3, 5.6]
    x_range = x_range_G23

    def evaluate(self, x, Rv):
        """
        G23 function

        Parameters
        ----------
        in_x: float
           expects either x in units of wavelengths or frequency
           or assumes wavelengths in wavenumbers [1/micron]

           internally wavenumbers are used

        Returns
        -------
        axav: np array (float)
            A(x)/A(V) extinction curve [mag]

        Raises
        ------
        ValueError
           Input x values outside of defined range
        """
        # setup the a & b coefficient vectors
        self.a = np.zeros(x.shape)
        self.b = np.zeros(x.shape)

        # define the ranges
        ir_indxs = np.where(np.logical_and(1.0 / 35.0 <= x, x < 1.0 / 1.0))
        opt_indxs = np.where(np.logical_and(1.0 / 1.1 <= x, x < 1.0 / 0.3))
        uv_indxs = np.where(np.logical_and(1.0 / 0.3 <= x, x <= 1.0 / 0.09))

        # overlap ranges
        optir_waves = [0.9, 1.1]
        optir_overlap = (x >= 1.0 / optir_waves[1]) & (x <= 1.0 / optir_waves[0])
        uvopt_waves = [0.3, 0.33]
        uvopt_overlap = (x >= 1.0 / uvopt_waves[1]) & (x <= 1.0 / uvopt_waves[0])

        # NIR/MIR
        # fmt: off
        # (scale, alpha1, alpha2, swave, swidth), sil1, sil2
        ir_a = [0.38526, 1.68467, 0.78791, 4.30578, 4.78338,
                0.0, 9.8434, 2.21205, -0.24703,
                0.0 , 19.58294, 17., -0.27]
        # fmt: on
        ir_b = [-1.01251, 1.0, -1.06099]
        self.a[ir_indxs] = self.nirmir_intercept(x[ir_indxs], ir_a)

        irpow = PowerLaw1D()
        irpow.parameters = ir_b
        self.b[ir_indxs] = irpow(x[ir_indxs])

        # optical
        # fmt: off
        # polynomial coeffs, ISS1, ISS2, ISS3
        opt_a = [-0.35848, 0.7122 , 0.08746, -0.05403, 0.00674,
                 0.0, 2.288, 0.243,
                 0.0, 2.054, 0.179,
                 0.0, 1.587, 0.243]
        opt_b = [0.0, -2.68335, 2.01901, -0.39299, 0.03355,
                 0.0, 2.288, 0.243,
                 0.0, 2.054, 0.179,
                 0.0 , 1.587, 0.243]
        # fmt: on
        m20_model_a = Polynomial1D(4) + Drude1D() + Drude1D() + Drude1D()
        m20_model_a.parameters = opt_a
        self.a[opt_indxs] = m20_model_a(x[opt_indxs])
        m20_model_b = Polynomial1D(4) + Drude1D() + Drude1D() + Drude1D()
        m20_model_b.parameters = opt_b
        self.b[opt_indxs] = m20_model_b(x[opt_indxs])

        # overlap between optical/ir
        # weights = (1.0 / optir_waves[1] - x[optir_overlap]) / (
        #     1.0 / optir_waves[1] - 1.0 / optir_waves[0]
        # )
        weights = _smoothstep(
            1.0 / x[optir_overlap], x_min=optir_waves[0], x_max=optir_waves[1], N=1
        )
        self.a[optir_overlap] = (1.0 - weights) * m20_model_a(x[optir_overlap])
        self.a[optir_overlap] += weights * self.nirmir_intercept(x[optir_overlap], ir_a)
        self.b[optir_overlap] = (1.0 - weights) * m20_model_b(x[optir_overlap])
        self.b[optir_overlap] += weights * irpow(x[optir_overlap])

        # Ultraviolet
        uv_a = [0.81297, 0.2775, 0.0, 0.0, 4.60, 0.99]
        uv_b = [-2.97868, 1.89808, 0.0, 0.0, 4.60, 0.99]

        fm90_model_a = FM90()
        fm90_model_a.parameters = uv_a
        self.a[uv_indxs] = fm90_model_a(x[uv_indxs] / u.micron)
        fm90_model_b = FM90()
        fm90_model_b.parameters = uv_b
        self.b[uv_indxs] = fm90_model_b(x[uv_indxs] / u.micron)

        # overlap between uv/optical
        # weights = (1.0 / uvopt_waves[1] - x[uvopt_overlap]) / (
        #     1.0 / uvopt_waves[1] - 1.0 / uvopt_waves[0]
        # )
        weights = _smoothstep(
            1.0 / x[uvopt_overlap], x_min=uvopt_waves[0], x_max=uvopt_waves[1], N=1
        )
        self.a[uvopt_overlap] = (1.0 - weights) * fm90_model_a(
            x[uvopt_overlap] / u.micron
        )
        self.a[uvopt_overlap] += weights * m20_model_a(x[uvopt_overlap])
        self.b[uvopt_overlap] = (1.0 - weights) * fm90_model_b(
            x[uvopt_overlap] / u.micron
        )
        self.b[uvopt_overlap] += weights * m20_model_b(x[uvopt_overlap])

        # return A(x)/A(V)
        return self.a + self.b * (1 / Rv - 1 / 3.1)

    @staticmethod
    def nirmir_intercept(x, params):
        """
        Functional form for the NIR/MIR intercept term.
        Based on modifying the G21 shape model to have two power laws instead
        of one with a break wavelength.

        Parameters
        ----------
        x: float
           expects x in wavenumbers [1/micron]
        params: floats
           paramters of function

        Returns
        -------
        axav: np array (float)
            A(x)/A(V) extinction curve [mag]
        """
        wave = 1 / x

        # fmt: off
        (scale, alpha, alpha2, swave, swidth,
            sil1_amp, sil1_center, sil1_fwhm, sil1_asym,
            sil2_amp, sil2_center, sil2_fwhm, sil2_asym) = params
        # fmt: on

        # broken powerlaw with a smooth transition
        axav_pow1 = scale * (wave ** (-1.0 * alpha))

        norm_ratio = swave ** (-1.0 * alpha) / swave ** (-1.0 * alpha2)
        axav_pow2 = scale * norm_ratio * (wave ** (-1.0 * alpha2))

        # use smoothstep to smoothly transition between the two powerlaws
        weights = _smoothstep(
            wave, x_min=swave - swidth / 2, x_max=swave + swidth / 2, N=1
        )
        axav = axav_pow1 * (1.0 - weights) + axav_pow2 * weights

        # silicate feature drudes
        axav += _modified_drude(wave, sil1_amp, sil1_center, sil1_fwhm, sil1_asym)
        axav += _modified_drude(wave, sil2_amp, sil2_center, sil2_fwhm, sil2_asym)

        return axav
    

def fwhm_to_stddev(fwhm):
    """
    Function to convert the FWHM of a Gaussian to the standard deviation
    Gaussian1D model stddev: FWHM = 2 * stddev * sqrt(2 * ln(2))
    stddev = FWHM / 2 / sqrt(2 * ln(2))

    Parameters
    ----------
    fwhm : float
        FWHM of the Gaussian

    Returns
    -------
    stddev : float
        standard deviation of the Gaussian
    """
    return fwhm / 2 / np.sqrt(2 * np.log(2))


def gauss_model(amplitudes, means, stddevs):
    """
    Function to create an Astropy model with the sum of multiple Gaussians

    Parameters
    ----------
    amplitudes : np.ndarray
        Amplitudes of the Gaussians

    means : np.ndarray
        Central wavelengths of the Gaussians

    stddev : np.ndarray
        Standard deviations of the Gaussians

    Returns
    -------
    Astropy CompoundModel with the sum of the Gaussians
    """
    # create the first Gaussian
    model_sum = Gaussian1D(
        amplitude=amplitudes[0],
        mean=means[0],
        stddev=stddevs[0],
        fixed={"mean": True, "stddev": True},
    )
    # add the rest of the Gaussians
    for i in range(1, len(amplitudes)):
        model_sum += Gaussian1D(
            amplitude=amplitudes[i],
            mean=means[i],
            stddev=stddevs[i],
            fixed={"mean": True, "stddev": True},
        )

    return model_sum


if __name__ == "__main__":
    parser = argparse.ArgumentParser()   
    parser.add_argument("--notprop", help="save figure as a png file", action="store_true")
    parser.add_argument("--png", help="save figure as a png file", action="store_true")
    parser.add_argument("--pdf", help="save figure as a pdf file", action="store_true")
    args = parser.parse_args()

    fontsize = 16

    font = {"size": fontsize}

    plt.rc("font", **font)

    plt.rc("lines", linewidth=2)
    plt.rc("axes", linewidth=2)
    plt.rc("xtick.major", width=2)
    plt.rc("xtick.minor", width=2)
    plt.rc("ytick.major", width=2)
    plt.rc("ytick.minor", width=2)

    figsize = (10, 6)
    fig, ax = plt.subplots(2, figsize=figsize, sharex=True)

    extmod = G23()
    extmod_nofeat = G23_nofeatures()
    waves = np.logspace(np.log10(0.0912), np.log10(30.), 10000) * u.micron
    waves1 = np.logspace(np.log10(0.0912), np.log10(0.3), 10000) * u.micron
    waves2 = np.logspace(np.log10(0.33), np.log10(30.), 10000) * u.micron
    ext = extmod(waves)
    ext_nofeat = extmod_nofeat(waves)
    ax[0].plot(waves, ext, "k-")
    ax[0].set_ylabel(r"A($\lambda$)/A(V)")
    ax[0].text(35.0, 5.1, "Milky Way R(V)=3.1", ha="right", fontsize=0.8*fontsize)
    ax[0].text(35.0, 4.5, "Features + Continuum", ha="right")
    ax[0].plot([0.0912, 30.], [0.0, 0.0], "k--", alpha=0.7)
    ax[0].set_ylim(-0.3, 6.0)
    # ax[0].set_title("Extinction Features+Continuum")

    if not args.notprop:
        ccol = "tab:red"
        ax[0].text(8.5, 1.5, "JWST MEAD: NIRCam + MIRI Spectroscopy", ha="center", 
                fontsize=0.6*fontsize, color=ccol, alpha=0.5)
        rect = patches.Rectangle((2.5, 0.7), 26, 0.6, linewidth=1, edgecolor=ccol, facecolor=ccol, alpha=0.3)
        ax[0].add_patch(rect)

        ccol = "tab:purple"
        ax[0].text(0.16, 1.8, "FUSE+IUE Spectra", ha="center", 
                fontsize=0.6*fontsize, color=ccol, alpha=0.5)
        ax[0].text(0.16, 1.45, "HST MEAD: STIS", ha="center", 
                fontsize=0.6*fontsize, color=ccol, alpha=0.5)
        rect = patches.Rectangle((0.0912, 0.7), 0.21, 0.6, linewidth=1, edgecolor=ccol, facecolor=ccol, alpha=0.3)
        ax[0].add_patch(rect)

        ccol = "tab:green"
        ax[0].text(0.55, 4.1, "Proposed", ha="center", 
                fontsize=0.7*fontsize, color=ccol)
        ax[0].text(0.55, 3.65, "STIS Optical Spectra", ha="center", 
                fontsize=0.7*fontsize, color=ccol)
        rect = patches.Rectangle((0.29, 2.5), 1.027 - 0.29, 1.0, linewidth=1, edgecolor=ccol, facecolor=ccol, alpha=0.75)
        ax[0].add_patch(rect)

    ax[1].plot(waves1, extmod(waves1) - extmod_nofeat(waves1), "k-")

    Av_cyg12 = 10.
    # manually add 3.4 C-H feature
    model_profile34 = gauss_model(
        np.array([0.0364, 0.0424, 0.0324, 0.0261, 0.0262]) / Av_cyg12,
        np.array([3.376, 3.420, 3.474, 3.520, 3.289]),
        fwhm_to_stddev(np.array([0.05, 0.05, 0.05, 0.05, 0.09])),
    )
    # manually add 6.2 C=C feature
    model_profile62 = gauss_model(
        np.array([0.0169, 0.00974]) / Av_cyg12,
        np.array([6.19, 6.25]),
        fwhm_to_stddev(np.array([0.06, 0.16])),
    )
    featext = (extmod(waves2) - extmod_nofeat(waves2)) + model_profile34(waves2.value) + model_profile62(waves2.value)

    ivals = (waves2 >= 1.0 * u.micron)
    ax[1].plot(waves2[ivals], featext[ivals] * 10, "k-")
    ivals = (waves2 < 1.0 * u.micron)
    ax[1].plot(waves2[ivals], featext[ivals] * 10, "g-", alpha=0.75)
    #ax[1].plot([0.0912, 30.], [0.0, 0.0], "k--", alpha=0.7)

    ax[1].plot([0.315, 0.315], [0.0, 1.6], "k:")
    ax[1].text(35.0, 1.3, "Features Only", ha="right")
    ax[1].text(0.29, 1.4, "x1", ha="right")
    ax[1].text(0.2175, 0.5, r"2175 $\mathrm{\AA}$", ha="center", va="top", fontsize=0.7*fontsize, rotation=90.)
    ax[1].text(0.1, 0.5, r"FUV Rise", ha="center", va="center", fontsize=0.7*fontsize, rotation=90.)

    ax[1].text(0.35, 1.4, "x10")
    ax[1].text(1./2.288 - 0.01, 0.55, "ISS1", ha="center", fontsize=0.7*fontsize, rotation=90., color="tab:green")
    ax[1].text(1./2.054 + 0.01, 0.5, "ISS2", ha="center", fontsize=0.7*fontsize, rotation=90., color="tab:green")
    ax[1].text(1./1.587, 0.3, "ISS3", ha="center", fontsize=0.7*fontsize, rotation=90., color="tab:green")
    ax[1].text(3.2, 0.1, r"C-H 3.3 $\mu$m", ha="center", fontsize=0.6*fontsize, rotation=90.)
    ax[1].text(3.5, 0.15, r"C-H 3.4 $\mu$m", ha="center", fontsize=0.6*fontsize, rotation=90.)
    ax[1].text(6.2, 0.1, r"C=C 6.2 $\mu$m", ha="center", fontsize=0.6*fontsize, rotation=90.)
    ax[1].text(10.0, 0.8, r"Si-O 10 $\mu$m", ha="center", fontsize=0.7*fontsize)
    ax[1].text(20.0, 0.4, r"Si-O 20 $\mu$m", ha="center", fontsize=0.7*fontsize)
    ax[1].set_xlabel(r"$\lambda$ [$\mu$m]")
    ax[1].set_ylabel("A($\lambda$)/A(V)")
    ax[1].set_xscale("log")
    ax[1].set_xlim(1.0, 5.0)
    ax[1].set_ylim(0.0, 0.1)

    ax[0].set_ylim(0.0, 0.5)

    def tenx(x):
        return x*10.

    def itenx(x):
        return x/10.

    secax = ax[1].secondary_yaxis('right', functions=(itenx, tenx))
    secax.set_ylabel(r"A($\lambda$)/A(V)")

    fig.tight_layout()

    save_str = "mw_ext_features"
    if args.png:
        fig.savefig(f"{save_str}.png")
    elif args.pdf:
        fig.savefig(f"{save_str}.pdf")
    else:
        plt.show()
