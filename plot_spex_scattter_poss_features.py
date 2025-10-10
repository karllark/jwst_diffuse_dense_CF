import argparse
import matplotlib.pyplot as plt
import matplotlib.patches as patches
import numpy as np

from astropy.io import fits
from astropy.stats import sigma_clipped_stats
import astropy.units as u

from measure_extinction.extdata import ExtData
from dust_extinction.parameter_averages import G23
from plot_ext_features import G23_nofeatures, fwhm_to_stddev, gauss_model


def drude_asym(x, scale, x_o, gamma_o, asym):
    """
    Drude to play with
    """
    gamma = 2.0 * gamma_o / (1.0 + np.exp(asym * (x - x_o)))
    y = scale * ((gamma / x_o) ** 2) / ((x / x_o - x_o / x) ** 2 + (gamma / x_o) ** 2)
    return y


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--notprop", help="save figure as a png file", action="store_true"
    )
    parser.add_argument("--png", help="save figure as a png file", action="store_true")
    parser.add_argument("--pdf", help="save figure as a pdf file", action="store_true")
    args = parser.parse_args()

    inpath = "Spex/Ext_curves/"
    ice = False
    fs = 16
    outpath = "./"

    # def plot_ave_res(inpath, outpath, ice=False):
    """
    Plot the residuals of the average curve fit separately

    Parameters
    ----------
    inpath : string
        Path to the data files

    outpath : string
        Path to save the plot

    ice : boolean [default=False]
        Whether or not to use the average fit with a fixed ice feature

    Returns
    -------
    Plot with the residuals of the average curve fit
    """
    # read in the average extinction curve
    filename = "average_ext.fits"
    if ice:
        filename = filename.replace(".", "_ice.")
    average = ExtData(inpath + filename)

    # plot the residuals
    plt.rc("axes", linewidth=0.8)
    fig, ax = plt.subplots(
        3,
        figsize=(14, 7),
        sharex=True,
        gridspec_kw={
            "height_ratios": [1, 1, 3],
            "hspace": 0,
        },
    )
    waves = average.model["waves"]
    residuals = average.model["residuals"]
    ax[2].scatter(waves, np.absolute(residuals), s=1.5, color="k", alpha=0.2)

    # calculate the standard deviation of the residuals in different wavelength ranges
    ranges = [(0.79, 1.37), (1.4, 1.82), (1.92, 2.54), (2.85, 4.05), (4.55, 5.0)]
    for range in ranges:
        mask = (waves > range[0]) & (waves < range[1])
        mean, median, stddev = sigma_clipped_stats(residuals[mask])
        ax[2].hlines(
            y=(stddev, -stddev),
            xmin=range[0],
            xmax=range[1],
            colors="black",
            ls="--",
            lw=2,
            zorder=5,
            alpha=0.5,
        )

    # indicate hydrogen lines and jumps
    # ax[2].annotate(
    #     "Pa\njump",
    #     xy=(0.85, 0.0185),
    #     xytext=(0.85, 0.023),
    #     fontsize=0.7 * fs,
    #     ha="center",
    #     va="center",
    #     color="blue",
    #     arrowprops=dict(arrowstyle="-[, widthB=.5, lengthB=.5", lw=1, color="blue"),
    # )
    # ax[2].annotate(
    #     "Br\njump",
    #     xy=(1.46, 0.0165),
    #     xytext=(1.46, 0.021),
    #     fontsize=0.7 * fs,
    #     ha="center",
    #     va="center",
    #     color="blue",
    #     arrowprops=dict(arrowstyle="-[, widthB=.55, lengthB=.5", lw=1, color="blue"),
    # )

    # lines = [
    #     0.9017385,
    #     0.9231547,
    #     0.9548590,
    #     1.0052128,
    #     1.0941090,
    #     1.282159,
    #     1.5264708,
    #     1.5443139,
    #     1.5560699,
    #     1.5704952,
    #     1.5884880,
    #     1.6113714,
    #     1.6411674,
    #     1.6811111,
    #     1.7366850,
    #     2.166120,
    #     2.3544810,
    #     2.3828230,
    #     2.3924675,
    #     2.4163852,
    #     3.0392022,
    # ]

    # ax[2].vlines(x=lines, ymin=-0.017, ymax=0.017, color="blue", lw=0.5, alpha=0.5)

    # add in the predicted feature strengths
    extmod = G23()
    extmod_nofeat = G23_nofeatures()
    waves2 = np.logspace(np.log10(0.33), np.log10(30.0), 10000) * u.micron

    # manually add 3.0 ice feature
    model_profile30 = drude_asym(waves2.value, 0.0034, 3.0, 0.50, -3.0)

    Av_cyg12 = 10.0
    # manually add 3.4 C-H feature
    model_profile33 = gauss_model(
        np.array([0.0137]) / Av_cyg12,
        np.array([3.289]),
        fwhm_to_stddev(np.array([0.09])),
    )
    # manually add 3.4 C-H feature
    model_profile34 = gauss_model(
        np.array([0.0364, 0.0424, 0.0324, 0.0261]) / Av_cyg12,
        np.array([3.376, 3.420, 3.474, 3.520]),
        fwhm_to_stddev(np.array([0.05, 0.05, 0.05, 0.05])),
    )
    print(fwhm_to_stddev(np.array([0.05, 0.05, 0.05, 0.05])))
    # manually add 4.4 aromatic C-D feature
    model_profile44 = gauss_model(
        np.array([0.0025]),
        np.array([4.4]),
        fwhm_to_stddev(np.array([0.0265])),
    )
    # manually add 4.65 aromatic C-D feature
    print("blah")
    model_profile465 = gauss_model(
        np.array([0.005]),
        np.array([4.65]), 
        fwhm_to_stddev(np.array([0.0265])),
    )
    ax[2].plot(waves2, model_profile30, "b--", alpha=0.5, lw=3)
    ax[2].plot(waves2, model_profile33(waves2.value), "g--", alpha=0.5, lw=3)
    ax[2].plot(waves2, model_profile34(waves2.value), "r--", alpha=0.5, lw=3)
    ax[2].plot(
        waves2, model_profile44(waves2.value), color="tab:olive", alpha=0.5, lw=3
    )
    ax[2].plot(
        waves2, model_profile465(waves2.value), color="tab:orange", alpha=0.5, lw=3
    )

    featext = (
        model_profile30
        + model_profile33(waves2.value)
        + model_profile34(waves2.value)
        + model_profile44(waves2.value)
        + model_profile465(waves2.value)
    )
    ax[2].plot(waves2, featext, "c-", lw=3)

    dave = ExtData(filename="Spex/Ext_curves/average_ext.fits")
    dave.plot(ax[1])

    # ax[1].plot(waves2, featext + extmod(waves2), "k-", alpha=0.7, label="G23 R(V)=3.1 + features")
    ax[1].set_ylim(0.01, 0.7)
    ax[1].set_yscale("log")
    ax[1].set_ylabel(r"$A(\lambda)/A(V)$")
    # ax[1].legend(fontsize=0.5*fs)

    ax[2].text(
        3.0,
        0.0075,
        r"H$_2$O ice 3.0 $\mu$m",
        ha="center",
        fontsize=0.6 * fs,
        rotation=90.0,
        color="b",
        backgroundcolor="white"
    )
    ax[2].text(
        3.3,
        0.01,
        r"C-H 3.3 $\mu$m",
        ha="center",
        fontsize=0.6 * fs,
        rotation=90.0,
        color="g",
        backgroundcolor="white"
    )
    ax[2].text(
        3.42,
        0.01,
        r"C-H 3.4 $\mu$m",
        ha="center",
        fontsize=0.6 * fs,
        rotation=90.0,
        color="r",
        backgroundcolor="white"
    )
    ax[2].text(
        4.4,
        0.0075,
        r"C-D 4.4 $\mu$m",
        ha="center",
        fontsize=0.6 * fs,
        rotation=90.0,
        color="tab:olive",
        backgroundcolor="white"
    )

    ax[2].text(
        4.65,
        0.0075,
        r"C-D 4.65 $\mu$m",
        ha="center",
        fontsize=0.6 * fs,
        rotation=90.0,
        color="tab:orange",
        backgroundcolor="white"
    )

    # add the atmospheric transmission curve
    hdulist = fits.open("Spex/atran2000.fits")
    data = hdulist[0].data
    ax[0].plot(data[0], data[1], color="k", alpha=0.7, lw=0.5)

    fontsize = fs
    # add observing info
    ccol = "tab:blue"
    ax[2].text(3.2, 0.0205, "NIRCam grism/F322W2", ha="center", 
            fontsize=0.6*fontsize, color=ccol, alpha=0.7)
    rect = patches.Rectangle((2.45, 0.02), 3.95 - 2.45, 0.002, linewidth=1, edgecolor=ccol, facecolor=ccol, alpha=0.3)
    ax[2].add_patch(rect)

    ccol = "tab:green"
    ax[2].text(4.5, 0.0235, "NIRCam grism/F444W", ha="center", 
            fontsize=0.6*fontsize, color=ccol, alpha=0.7)
    rect = patches.Rectangle((3.93, 0.023), 4.95 - 3.93, 0.002, linewidth=1, edgecolor=ccol, facecolor=ccol, alpha=0.3)
    ax[2].add_patch(rect)

    # finalize and save the plot
    ax[2].axhline(ls="-", c="k", alpha=0.5, lw=1.5)
    ax[0].tick_params(width=1)
    ax[2].tick_params(width=1)
    ax[0].set_xlim(2.4, 5.0)
    plt.ylim(0.0, 0.025)
    ax[0].set_ylabel("Atmospheric\ntransmission", fontsize=0.6 * fs)
    ax[2].set_xlabel(r"$\lambda$ [$\mu m$]", fontsize=fs)
    ax[2].set_ylabel("$A(\lambda)/A(V)$ scatter", fontsize=fs)
    fig.savefig(
        outpath + filename.replace("ext", "res").replace("fits", "pdf"),
        bbox_inches="tight",
    )
