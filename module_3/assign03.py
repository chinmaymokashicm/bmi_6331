"""
Author: Luca Giancardo
Date: 2017-09-28
Version: 1.0
"""

# import libraries
import matplotlib
matplotlib.use('TkAgg')
import matplotlib.pyplot as plt

import numpy as np
import skimage as sk
import skimage.io as skio
import skimage.transform as sktr
import skimage.filters as skfl
import skimage.feature as skft
import skimage.color as skcol
import skimage.exposure as skexp


def sin2(x, freq=1, phase=0, amplitude=1):
    """
    Compute sine wave with given frequency, phase and amplitude

    :param x: array of the domain expressed in sine cycles, i.e. x=1 represent the sine of 360 degrees (or 2pi),
    :param freq: sine frequency
    :param phase: sine phase
    :param amplitude: sine amplitude
    :return: array of values
    """
    y = amplitude * np.sin(x * 2 * np.pi * freq - (phase * 2 * np.pi))

    return y


def fft1D( xIn, yIn ):
    """
    Given a signal y(x) compute the FFT of a 1D signal and return the magnitude and phase at each frequency
    :param xIn: x of signal
    :param yIn: y of signal
    :return: (freqArray, spMag, spPhase)
    """
    # initialization
    domain = (min(xIn),max(xIn))
    fs = len(xIn)/2

    # Compute FFT
    sp = np.fft.fft(yIn) / len(xIn)
    # Compute frequency array
    freqArray = np.fft.fftfreq(len(xIn), (domain[1] - domain[0]) / 2 / fs)
    # Shift FFT to interpretable format
    sp = np.fft.fftshift(sp)
    freqArray = np.fft.fftshift(freqArray)

    # compute magnitude and angle of complex number
    spMag = np.absolute(sp)
    spPhase = np.angle(sp)

    return freqArray, spMag, spPhase

def plotFFT2d(imgIn, sizeFig=(12, 6)):
    """
    Compute the FFT of a 2D signal and diplay magnitude component
    :param imgIn: 2d signal
    :param sizeFig: optional tuple with plot size
    :return: none
    """
    # resolution
    M, N = imgIn.shape

    # compute FFT
    F = np.fft.fftn(imgIn)
    # compute magnitude and shift FFT to interpretable format
    F_magnitude = np.abs(F)
    F_magnitude = np.fft.fftshift(F_magnitude)

    # plot
    plt.subplots(figsize=sizeFig)
    plt.imshow(np.log(1 + F_magnitude), cmap='viridis', extent=(-N // 2, N // 2, -M // 2, M // 2))
    plt.colorbar()
    plt.title('Spectrum magnitude');
    plt.show()

if __name__ == "__main__":
    pass
