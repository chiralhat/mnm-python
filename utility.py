# -*- coding: utf-8 -*-
"""
Created on Fri Apr 29 09:25:29 2016

@author: ccollett
"""

import numpy as np
import scipy as sp
import pandas as pd
# import scipy.constants as sc
import scipy.optimize as sop
import zipfile
import os
import fnmatch
import glob
import re
import matplotlib.pyplot as plt
from functools import partial
import scipy.constants as sc

# Field offsets
# Offset when sweep rate is 10.8 Oe/s
h_off_10 = -93.5

# Regexp pattern for conflict files
conf_patt = '(.*) \(SFConflict.*'


def plot(*args, **kwargs):
    if len(args[0]) == 2:
        args = list(args)
        args.insert(1, args[0][1])
        args[0] = args[0][0]
        args = tuple(args)
    if 'figsize' not in kwargs:
        fsize = (10, 8)
    else:
        fsize = kwargs['figsize']
        del kwargs['figsize']
    fig, ax = plt.subplots(figsize=fsize)
    if 'xlim' in kwargs:
        ax.set_xlim(kwargs['xlim'])
        del kwargs['xlim']
    if 'ylim' in kwargs:
        ax.set_ylim(kwargs['ylim'])
        del kwargs['ylim']
    ax.plot(*args, **kwargs)
    plt.tight_layout()
    return ax


def get_zip_names(path, dir, infold=True):
    """Loads the filenames for the data matching 'path' in the location 'dir'
    ('dir' does not have to end with '.zip').

    Returns a list of filenames and the zip archive."""
    folder = os.path.split(dir)[1] if infold else ''
    if not dir.endswith('.zip'):
        dir += '.zip'
    else:
        folder = folder[:-4]
    archive = zipfile.ZipFile(dir, 'r')
    names = [n for n in archive.namelist() if
             fnmatch.fnmatch(n, os.path.join(folder, path))]
    return names, archive


def get_folder_names(path, dir):
    """Loads the filenames for the data matching 'path' in the location 'dir'
    ('dir' does not have to end with '.zip').

    Returns a list of filenames and the zip archive."""
    names = glob.glob(os.path.join(dir, path))
    conflicts = [re.match(conf_patt, name) for name in names]
    c_inds = np.where(conflicts)[0]
    if len(c_inds) > 0:
        conflicts = [conflicts[i].groups()[0] for i in c_inds]
        conf_names = [conf + '.txt' for conf in conflicts]
        [os.remove(os.path.join(dir, name)) for name in conf_names]
        [os.rename(names[i], new) for i, new in zip(c_inds, conf_names)]
        names = glob.glob(os.path.join(dir, path))
    return names


def get_zip_data(path, dir, skip=0):
    """Loads the data matching 'path' in the location 'dir'
    ('dir' does not have to end with '.zip').

    Returns a list of datasets and a list of filenames."""
    names, archive = get_zip_names(path, dir)
    data = [np.loadtxt(archive.open(n), skiprows=skip).transpose()
            for n in names]
    archive.close()
    return data, names


def get_data_names(path, dir, skip=0):
    """Loads the data matching 'path' in the location 'dir'
    ('dir' can either be a folder or a zip archive).

    Returns a list of datasets and a list of filenames."""
    if os.path.isdir(dir):
        names = get_folder_names(path, dir)
        data = np.array([np.loadtxt(n, skiprows=skip).transpose()
                         for n in names])
    else:
        data, names = get_zip_data(path, dir, skip)
    return data, names


def get_data_exclude_names(path, dir, excl, skip=0):
    """Loads the data matching 'path' in the location 'dir'
    ('dir' can either be a folder or a zip archive),
    which does not end with 'excl'.

    Returns a list of datasets and a list of filenames."""
    if os.path.isdir(dir):
        prenames = get_folder_names(path, dir)
        names = [x for x in prenames if not x.endswith(excl)]
        data = np.array([np.loadtxt(n, skiprows=skip).transpose()
                         for n in names])
        archive = 0
    else:
        prenames, archive = get_zip_names(path, dir)
        names = [x for x in prenames if not x.endswith(excl)]
        data = np.array([np.loadtxt(archive.open(n),
                                    skiprows=1).transpose() for n in names])
    return data, names, archive


def get_data(path, dir, skip=0):
    return get_data_names(path, dir, skip)[0]


def get_one_data(path, dir, skip=0):
    """Loads a single dataset matching 'path' in the location 'dir'
    ('dir' can either be a folder or a zip archive).

    Returns the dataset."""
    return get_data(path, dir, skip)[0]

# Fitting functions


def lorentzian(x, b, a, w, f):
    return b + a * (w / 2)**2 / ((x - f)**2 + (w / 2)**2)


def lorentziannoback(x, a, w, f):
    return a * (w / 2)**2 / ((x - f)**2 + (w / 2)**2)


def lorentzianslope(x, b, a, w, f, slope, gf):
    return b + a * (w / 2)**2 / ((x - f)**2 + (w / 2)**2) + slope * (x - gf)


def exponential(x, c, a, t):
    return a * np.exp(-(x) / t) + c


def exponentialnoback(x, a, t):
    return a * np.exp(-(x) / t)


def gaussian(x, b, a, w, f):
    return b + a * np.exp(-(2 / w)**2 * (x - f)**2)


def gaussiannoback(x, a, w, f):
    return a * np.exp(-(2 / w)**2 * (x - f)**2)


def gaussianslope(x, b, a, w, f, slope, gf):
    return b + a * np.exp(-(2 / w)**2 * (x - f)**2) + slope * (x - gf)


def sinefit(x, b, a, w, t):
    return b + a * np.sin(w * x + t)


def quadlorfit(x, b, a, w, f, c, d):
    return lorentzian(x, b, a, w, f) + c * x + d * x**2

# Voigt profile calculation


def V(x, GW, LW):
    alpha = GW / 2
    gamma = LW / 2
    sigma = alpha / np.sqrt(2 * np.log(2))
    return np.real(sp.special.wofz((x + 1j * gamma) / sigma /
                                   np.sqrt(2))) / sigma / np.sqrt(2 * np.pi)


def voigt(x, b, a, f, GW, LW):
    return b + a * V(x - f, GW, LW)


def voigtslope(x, b, a, f, GW, LW, slope, gf):
    return b + a * V(x - f, GW, LW) + slope * (x - gf)


def voigtnoback(x, a, f, GW, LW):
    return a * V(x - f, GW, LW)


def atanfit(x, b, a, w, f):
    return b + a * np.arctan(-2 * (x - f) / w)


def atanslope(x, b, a, w, f, slope, gf):
    return atanfit(x, b, a, w, f) + slope * (x - gf)


def atannoback(x, a, w, f):
    return atanfit(x, 0, a, w, f)

# utility functions
# Power conversion between mW and dBm


def mw_to_dbm(mW):
    return 10 * np.log10(mW) + 30


def dbm_to_mw(dBm):
    return 10**((dBm - 30) / 10)


def find_nearest_idx(array, value):
    """Finds the element in 'array' that is nearest to 'value'.

    Returns the index of the nearest element."""
    if type(value) != list and type(value) != np.ndarray:
        return (np.abs(array - value)).argmin()
    else:
        def mfun(val):
            return find_nearest_idx(array, val)
        return np.array(list(map(mfun, value)))


def find_nearest_values_df(df, value, width=1):
    """Finds the elements in a dataframe that are nearest to the index 'value'.
    If 'width' is greater than 1, it takes the mean of the nearest 2*width
    values.

    Returns an array of the nearest elements in each dataframe member."""
    names = list(df)
    if width == 1:
        inds = [np.abs(df[name].dropna().index.values - value).argmin()
                for name in names]
        out = np.array([df[name].dropna().iloc[ind]
                        for ind, name in zip(inds, names)])
    else:
        wids = np.array([-width+1, width])
        inds = [np.abs(df[name].dropna().index.values - value).argmin() + wids
                for name in names]
        value_array = np.array([df[name].dropna().iloc[ind[0]:ind[1]]
                                for ind, name in zip(inds, names)])
        out = np.mean(value_array, axis=1)
    return np.array([float(o) for o in out])


def find_min(data):
    return data[1].min()


def derep(pars, rep):
    """Average the fit parameters for repeated fields.

    'pars' is an array of arrays of fit parameters;
    'rep' is the number of repetitions at each field.

    Returns an array of arrays of averaged fit parameters."""
    pars = np.array(pars)
    outpars = [np.zeros(len(pars[i]) / rep) for i in np.arange(len(pars))]
    for i in np.arange(len(pars)):
        outpars[i] = [np.mean(pars[i][j * rep:j * rep + rep])
                      for j in np.arange(len(outpars[i]))]
    return np.array(outpars)


def rolling_ave(data, num=10):
    half = int(np.ceil(num / 2))
    st, en = half, len(data) - half
    dout = np.zeros(en - st)
    for i, j in enumerate(np.arange(st, en)):
        dout[i] = np.mean(data[j - half:j + half])
    return dout


def rolling_mean(data, num=10):
    half = int(np.ceil(num / 2))
    xdat = data[0][half:len(data[0]) - half]
    ydat = [rolling_ave(dat, num) for dat in data[1:]]
    return np.array(np.concatenate(([xdat], ydat), axis=0))


def rolling_mean_df(df, num=10):
    names = list(df)
    aves = [df[name].dropna().rolling(num).mean().dropna() for name in names]
    return pd.concat(aves, axis=1)


def r2bar(data, fit, numfitpars):
    """Calculates the adjusted R-squared value for a fit based on the data.

    R2=1-np.sum(data-fit)**2/np.sum(data-np.mean(data))**2
    r2bar=1-(1-R2)*(len(data)-1)/(len(data)-numfitpars-1)

    'data' is an array of y-axis data;
    'fit' is an array of fit data corresponding to 'data';
    'numfitpars' is the number of fit parameters used in the fit.

    Returns the adjusted R-squared value."""
    R2 = 1 - np.sum([(data[i] - fit[i])**2 for i in np.arange(len(data))]) / \
        np.sum([(data[i] - np.mean(data))**2 for i in np.arange(len(data))])
    r2_adj = 1 - (1 - R2) * (len(data) - 1) / (len(data) - numfitpars - 1)
    return r2_adj


def func_fit(func, data, guess, **kwargs):
    """Fits any function to a dataset.

    'func' is the fitting function to use;
    'data' is a data array;
    'guess' is an array of initial guess parameters.

    Returns a list with three elements: an array with the fit parameters;
    an array with the uncertainties in those parameters;
    the adjusted R-squared for the fit."""
    try:
        fit = sop.curve_fit(func, data[0], data[1], guess, **kwargs)
        err = np.sqrt(np.diag(fit[1]))
        fitdat = func(data[0], *fit[0])
        R2b = r2bar(data[1], fitdat, len(fit[0]))
    except RuntimeError:
        n = len(guess)
        fit = [np.zeros(n)]
        err = np.zeros(n)
        R2b = 0
    return [fit[0], err, R2b]


def plot_func_fit(func, data, guess, plt=plt, fit=0, cols=0, datlab='Data',
                  fitlab='Fit', **kwargs):
    """Fits any function to a dataset and plots the result.

    'func' is the fitting function to use;
    'data' is a data array;
    'guess' is an array of initial guess parameters;
    'plt' is the plotting environment to use,
    defaulting to pyplot (imported as plt).

    Returns an array with the fit parameters."""
    if fit == 0:
        fit = func_fit(func, data, guess)[0]
    if cols == 0:
        cols = ['b', 'g']
    plt.plot(data[0], data[1], label=datlab, c=cols[0], **kwargs)
    plt.plot(data[0], func(data[0], *fit), '--', label=fitlab, c=cols[1])

    return fit


def sub_func_fit(func, data, guess):
    fit = func_fit(func, data, guess)[0]
    return np.array([data[0], data[1] - func(data[0], *fit)])


def plot_sub_func_fit(func, data, guess, plt=plt):
    subdat = sub_func_fit(func, data, guess)
    plt.plot(subdat[0], subdat[1])


def guess_pars(data, skew, full):
    """Algorithmically produces the initial guess parameters for a Lorentzian fit.

    'data' is a data array with elements [x-axis data,y-axis data];
    'skew' is 1 to subtract a linear background, and 0 to use no slope;
    'full' is 1 to fit the whole dataset, and 0 to only fit between the maxima
    on either side of the peak.
    Works for VNA and pulsed data, where the peak goes down and is centered.

    Returns an array of guess parameters: Background, Amplitude, Width,
    Frequency, slope, fitting points."""
    datLength = len(data[1])
    minPos = data[1].argmin()
    GuessB = data[1, 1:].max()
    GuessA = data[1].min() - GuessB
    left_midpoint = find_nearest_idx(data[1, minPos:], GuessB + GuessA / 2)
    right_midpoint = find_nearest_idx(data[1, :minPos], GuessB + GuessA / 2)
    GuessW = data[0][left_midpoint] - data[0][right_midpoint]
    GuessF = data[0, minPos]
    if full == 0:
        fitp = [data[1, :minPos].argmax(), data[1, minPos:].argmax() + minPos]
    else:
        fitp = [0, datLength - 1]
    if skew == 1:
        slope = -(data[1][fitp[0]] - data[1][fitp[1]]) * minPos / \
            datLength / (data[0][fitp[1]] - data[0][fitp[0]])
    else:
        slope = 0
    return [GuessB, GuessA, GuessW, GuessF, slope, fitp]


def lor_fit(data, skew=1, full=1):
    """Fits a Lorentzian function to a dataset.

    'data' is a data array with elements [x-axis data,y-axis data];
    'skew' is 1 (default) to subtract a linear background,
    and 0 to use no slope;
    'full' is 1 (default) to fit the whole dataset,
    and 0 to only fit between the maxima on either side of the peak.
    Works for VNA and pulsed data, where the peak goes down and is centered.

    Returns an array with: an array with the fit parameters (background,
    amplitude, width, frequency); the uncertainty in those parameters;
    the slope (if any) of the linear background;
    the adjusted R-squared of the fit."""
    [GuessB, GuessA, GuessW, GuessF, slope,
     fitp] = guess_pars(data, skew, full)

    def lorentzian(x, b, a, w, f):
        return lorentzianslope(x, b, a, w, f, slope, GuessF)
    fitdata = np.array([data[0][fitp[0]:fitp[1]], data[1][fitp[0]:fitp[1]]])
    guess = [GuessB, GuessA, GuessW, GuessF]
    fit, err, R2b = func_fit(lorentzian, fitdata, guess)
    return [fit, err, slope, R2b]


def plot_lor_fit(data, skew=1, full=1, plt=plt):
    """Fits a Lorentzian function to a dataset and plot the result.

    'data' is a data array with elements [x-axis data,y-axis data];
    'skew' is 1 (default) to subtract a linear background,
    and 0 to use no slope;
    'full' is 1 (default) to fit the whole dataset,
    and 0 to only fit between the maxima on either side of the peak;
    'plt' is the plotting environment to use, defaulting to pyplot
    (imported as plt).
    Works for VNA and pulsed data, where the peak goes down and is centered.

    Returns an array with: an array with the fit parameters (background,
    amplitude, width, frequency), the Q of the peak."""
    [fit, err, slope, _] = lor_fit(data, skew, full)
    plt.plot(data[0], data[1], data[0], lorentzianslope(
        data[0], fit[0], fit[1], fit[2], fit[3], slope, fit[3]))
    return [fit, np.abs(fit[3] / fit[2])]


def lor_fit_range(fulldata, low, high, skew=1):
    """Fits a Lorentzian function to a dataset in a specified window.

    'fulldata' is a data array with elements [x-axis data,y-axis data];
    'low' is the lowest x-axis point to fit;
    'high' is the highest x-axis point to fit;
    'skew' is 1 to subtract a linear background, and 0 to use no slope.
    Works for VNA and pulsed data, where the peak goes down.

    Returns an array with: an array with the fit parameters (background,
    mplitude, width, frequency); the uncertainty in those parameters;
    the slope (if any) of the linear background;
    the adjusted R-squared of the fit."""
    points = find_nearest_idx(fulldata[0], [low, high])
    data = np.array([np.array(x[points[0]:points[1]]) for x in fulldata])
    [fit, err, slope, R2b] = lor_fit(data, skew, 1)
    return [fit, err, slope, R2b]


def plot_lor_fit_range(data, low, high, skew=1, sub=0, plt=plt):
    """Fits a Lorentzian function to a dataset and plot the result.

    'fulldata' is a data array with elements [x-axis data,y-axis data];
    'low' is the lowest x-axis point to fit;
    'high' is the highest x-axis point to fit;
    'skew' is 1 to subtract a linear background, and 0 to use no slope;
    'plt' is the plotting environment to use, defaulting to pyplot
    (imported as plt).
    Works for VNA and pulsed data, where the peak goes down.

    Returns an array with: an array with the fit parameters (background,
    amplitude, width, frequency), the Q of the peak."""
    [fit, err, slope, _] = lor_fit_range(data, low, high, skew)
    if sub == 1:
        points = find_nearest_idx(data[0], [low, high])
    else:
        points = [0, len(data[0])]
    data = np.array(data).transpose()[points[0]:points[1]].transpose()
    plt.plot(data[0], data[1], data[0], lorentzianslope(
        data[0], fit[0], fit[1], fit[2], fit[3], slope, fit[3]))
    return [fit, np.abs(fit[3] / fit[2])]

# has not yet been updated to calculated the R-squared


def gauss_fit_abs_range(fulldata, low, high, skew=1):
    """Fits a Lorentzian function to a dataset in a specified window,
    taking the absolute value of x.

    'fulldata' is a data array with elements [x-axis data,y-axis data];
    'low' is the lowest x-axis point to fit;
    'high' is the highest x-axis point to fit;
    'skew' is 1 to subtract a linear background, and 0 to use no slope.
    Works for VNA and pulsed data, where the peak goes down.

    Returns an array with: an array with the fit parameters
    (background, amplitude, width, frequency),
    an array with the uncertainty in those parameters,
    the slope (if any) of the linear background."""
    points = find_nearest_idx(fulldata[0], [low, high])
    data = np.array([np.array(x[points[0]:points[1]]) for x in fulldata])
    [GuessB, GuessA, GuessW, GuessF, slope, fitp] = guess_pars(data, skew, 1)
    GuessW = (high - low) / 2

    def gaussian(x, b, a, w, f):
        return gaussianslope(np.abs(x), b, a, w, f, slope, GuessF)
    fit = sop.curve_fit(gaussian, data[0][fitp[0]:fitp[1]], data[1][
                        fitp[0]:fitp[1]], [GuessB, GuessA, GuessW, GuessF])
    return [fit[0], np.sqrt(np.diag(fit[1])), slope]


def plot_gauss_fit_abs_range(data, low, high, skew=1, plt=plt):
    """Fits a Lorentzian function to a dataset and plot the result,
    taking the absolute value of x.

    'fulldata' is a data array with elements [x-axis data,y-axis data];
    'low' is the lowest x-axis point to fit;
    'high' is the highest x-axis point to fit;
    'skew' is 1 to subtract a linear background, and 0 to use no slope;
    'plt' is the plotting environment to use, defaulting to pyplot
    (imported as plt).
    Works for VNA and pulsed data, where the peak goes down.

    Returns an array with: an array with the fit parameters (background,
    amplitude, width, frequency), the Q of the peak."""
    [fit, err, slope] = gauss_fit_abs_range(data, low, high, skew)
    plt.plot(data[0], data[1], data[0], gaussianslope(
        np.abs(data[0]), fit[0], fit[1], fit[2], fit[3], slope, fit[3]))
    return [fit, np.abs(fit[3] / fit[2])]


def exp_fit(data, freq, start, end, up, coefs=0):
    """Fits an exponential decay function to a dataset.

    'data' is a data array with elements [time,in-phase,quadrature];
    'start' is the lowest time point in the fitting window;
    'end' is the highest time point in the fitting window;
    'up' is 1 if the exponential is decaying down (towards -infinity),
    and 0 if it is decaying up (towards +infinity);
    'coefs' is 0 (default) if the initial guess parameters should be
    calculated from the dataset, and an array of guess parameters otherwise.

    Returns a list with three elements: an array with the fit parameters
    (background, amplitude, decay time, Q);
    an array with the uncertainties in those parameters;
    the adjusted R-squared for the fit."""
    points = find_nearest_idx(data[0], [start, end])
    if coefs == 0:
        guess = []
        guess.append(data[1, points[0]:points[1]].min())
        guess.append((data[1, points[0]:points[1]].max() -
                      data[1, points[0]:points[1]].min()) * up)
        guess.append(np.diff(data[0, points])[0] / 10)
    else:
        guess = coefs
    fit, err, R2b = func_fit(exponential, [data[0, points[0]:points[1]], data[
                             1, points[0]:points[1]]], guess)
    Q = np.pi * freq * fit[2]
    Qerr = np.pi * freq * err[2]
    fitout = np.append(fit, Q)
    errout = np.append(err, Qerr)
    return [fitout, errout, R2b]


def exp_no_back_fit(data, back, start, end, up, coefs=0):
    """Fits an exponential decay function with no background to a dataset.

    'data' is a data array with elements [time,in-phase,quadrature];
    'start' is the lowest time point in the fitting window;
    'end' is the highest time point in the fitting window;
    'up' is 1 if the exponential is decaying down (towards -infinity),
    and 0 if it is decaying up (towards +infinity);
    'coefs' is 0 (default) if the initial guess parameters should be
    calculated from the dataset, and an array of guess parameters otherwise.

    Returns an array with the fit parameters (amplitude, decay time)."""
    points = find_nearest_idx(data[0], [start, end])
    if coefs == 0:
        guess = []
        guess.append((data[1, points[0]:points[1]].max() -
                      data[1, points[0]:points[1]].min() - back) * up)
        guess.append(np.diff(data[0, points])[0] / 10)
    else:
        guess = coefs
    fit = sop.curve_fit(exponentialnoback, data[0, points[0]:points[1]], data[
                        1, points[0]:points[1]] - back, guess)
    return fit[0]
# return [fit[0],np.sqrt(np.diag(fit[1]))];


def plot_exp_fit(data, freq, start, end, up=1, coefs=0, plt=plt, col=0,
                 datlab='Data', fitlab='Fit'):
    """Fits an exponential decay function to a dataset and plot the result.

    'data' is a data array with elements [time,in-phase,quadrature];
    'start' is the lowest time point in the fitting window;
    'end' is the highest time point in the fitting window;
    'up' is 1 (default) if the exponential is decaying down
    (towards -infinity), and 0 if it is decaying up (towards +infinity);
    'coefs' is 0 (default) if the initial guess parameters should be
    calculated from the dataset, and an array of guess parameters otherwise;
    'plt' is the plotting environment to use, defaulting to pyplot
    (imported as plt).

    Returns an array with the fit parameters (background, amplitude,
    decay time, Q)."""
    points = find_nearest_idx(data[0], [start, end])
    [fit, err, _] = exp_fit(data, freq, start, end, up, coefs)
    if col == 0:
        plt.plot(data[0, points[0]:points[1]], data[
                 1, points[0]:points[1]], '-', label=datlab)
        plt.plot(data[0, points[0]:points[1]], exponential(
            data[0, points[0]:points[1]], fit[0], fit[1], fit[2]), '--',
            label=fitlab)
    else:
        plt.plot(data[0, points[0]:points[1]], data[
                 1, points[0]:points[1]], '-', c=col[0], label=datlab)
        plt.plot(data[0, points[0]:points[1]], exponential(
                data[0, points[0]:points[1]], fit[0], fit[1], fit[2]), '--',
                c=col[1], label=fitlab)
    return fit


def sub_data_guess(fulldata, low, high, down, ind):
    """Algorithmically produces the initial guess parameters and fitting
    window for a Lorentzian fit; more generally applicable than guess_pars.

    'fulldata' is a data array with x-axis data as the first element;
    'low' is the lowest x-axis point to fit;
    'high' is the highest x-axis point to fit;
    'ind' is the index of the y-axis data in fulldata;
    'down' is 1 if the peak goes down and 0 if it points up.

    Returns an array with: the windowed data, amplitude guess,
    frequency guess, window width."""
    points = find_nearest_idx(fulldata[0], [low, high])
    data = np.array([np.array(x[points[0]:points[1]]) for x in fulldata])
    gf = low
    df = high - low
    if down == 1:
        ga = data[ind].min() - data[ind][0]
    else:
        ga = data[ind].max() - data[ind][0]
    return [data, ga, gf, df]


def process_coefs(coefs, errs, R2bs, mins, fields, temps):
    """Takes various data parameters from a series of Lorentzian fits and
    collects them into the appropriate form for exporting.

    'coefs' is a list of arrays of Lorentzian fit coefficients;
    'errs' is a list of arrays of the uncertainties in those coefficients;
    'R2bs' is an array of adjusted R-squared values for the fits;
    'mins' is an array of the minima of the data;
    'fields' is an array of the field at each dataset;
    'temps' is an array of the temperature at each dataset.

    Returns an array whose elements are as follows: field, background offset,
    background offset uncertainty, amplitude, amplitude uncertainty,
    peak width, peak width uncertainty, frequency, frequency shift,
    frequency uncertainty, adjusted R-squared for the fit, Q, Q uncertainty,
    peak minimum, temperature."""
    Qs = np.abs(coefs[3] / coefs[2])
    Qerrs = Qs * np.sqrt((errs[3] / coefs[3])**2 + (errs[2] / coefs[2])**2)
    fitparerrs = np.concatenate(
        ([[coefs[i], errs[i]] for i in np.arange(len(coefs))]), axis=0)
    fullpars = np.insert(fitparerrs, 7, coefs[3] - coefs[3][0], axis=0)
    out = np.concatenate(([fields], fullpars,
                          [R2bs, Qs, Qerrs, mins, temps]), axis=0)
    return out

# Helpful functions for rescaling data


def divide_endpoint(data):
    def div_end(dat):
        return dat / dat[-1]
    out = np.array(list(map(div_end, data[1:])))
    return np.concatenate(([data[0]], out))


def div_end_df(df, last=True):
    if last:
        inds = [df[col].dropna().last_valid_index() for col in df]
    else:
        inds = [df[col].dropna().first_valid_index() for col in df]
    ends = np.diag(df.loc[inds].values)
    return df.divide(ends)


def sub_end_df(df, last=True):
    if last:
        inds = [df[col].dropna().last_valid_index() for col in df]
    else:
        inds = [df[col].dropna().first_valid_index() for col in df]
    ends = np.diag(df.loc[inds].values)
    return df.subtract(ends)


def div_back_df(df, ind, backind):
    divdf = div_end_df(df)
    return divdf[ind].interpolate() / divdf[backind].interpolate().values


def div_back_dfs(df, backind):
    inds = [ind for ind, _ in df.iteritems() if ind != backind]
    divdf = pd.concat([div_back_df(df, ind, backind) for ind in inds], axis=1)
    return divdf.dropna()


def subtract_endpoint(data):
    def div_end(dat):
        return dat - dat[-1]
    out = np.array(list(map(div_end, data[1:])))
    return np.concatenate(([data[0]], out))


def divide_background(indata, back):
    data = divide_endpoint(indata)

    def div_back(ind):
        bpt = find_nearest_idx(data[0], back[0, ind])
        return data[1:, bpt] / back[1:, ind]
    out = np.array(list(map(div_back, np.arange(len(back[0]))))).transpose()
    return np.concatenate((np.array([back[0]]), out))


def divide_backgrounds(data):
    back = divide_endpoint(data[0])
    mapfun = partial(divide_background, np.array(back))
    return list(map(mapfun, data[1:]))


def subtract_background(indata, inback):
    data = divide_endpoint(indata)
    back = divide_endpoint(inback)

    def sub_back(ind):
        bpt = find_nearest_idx(data[0], back[0, ind])
        return data[1:, bpt] - back[1:, ind]
    out = np.array(list(map(sub_back, np.arange(len(back[0]))))).transpose()
    return np.concatenate((np.array([back[0]]), out))


def subtract_backgrounds(data):
    back = divide_endpoint(data[0])
    mapfun = partial(subtract_background, inback=np.array(back))
    return list(map(mapfun, data[1:]))


def power_to_string(pows):
    """Converts an array of power values in dBm to strings.

    'pows' is an array of power values.

    Returns an array of corresponding strings."""
    def form_pow(pow):
        return "{:.1f}".format(pow)
    return list(map(form_pow, pows))

# Conversion functions
# This function converts a wavenumber in 1/cm to a frequency in GHz


def ktof(k):
    return sc.c * 100 * k * 1e-9
# This function converts a frequency in GHz to a wavenumber in 1/cm


def ftok(f):
    return f * 1e9 / sc.c / 100
# This function converts a temperature in K to a frequency in GHz


def ttof(t):
    return sc.k * t / sc.h * 1e-9
# This function converts a frequency in GHz to a temperature in K


def ftot(f):
    return f * 1e9 / sc.k * sc.h
# This function converts an energy in J to eV


def JtoeV(J):
    return J * 6.242e18
# This function converts a temperature in K to an energy in eV


def ttoeV(t):
    return JtoeV(sc.k * t)
# This function converts a wavenumber in 1/cm to a temperature in K


def ktot(k):
    return ktof(k) * 1e9 * sc.h / sc.k
# This function converts a wavenumber in 1/cm to an energy in J


def ktoJ(k):
    return ktof(k) * 1e9 * sc.h
# This function converts an energy in J to a temperature in K


def Jtot(J):
    return J / sc.k
# temperature in K to wavenumber in 1/cm


def ttok(t):
    return ftok(ttof(t))


# Old fitting trials for 3D cavity data
def Doublelor_fit_range(fulldata, low, high, ind=1, down=1):
    data, ga, gf, df = sub_data_guess(fulldata, low, high, ind, down)

    def doublelor(x, b, a, w, f, aa, ww, ff, slope):
        wslope = lorentzianslope(x, b, a, w, f, slope, gf)
        noback = lorentziannoback(x, aa, ww, ff)
        return wslope + noback
    guess = [data[ind][0], ga, df / 4, gf + df / 4, ga, df / 4,
             gf + 3 * df / 4, (data[ind][-1] - data[ind][1]) / df]
    fit = sop.curve_fit(doublelor, data[0], data[ind], guess)
    return [fit[0], data]


def PlotDoublelor_fit_range(data, low, high, ind=1, down=1, plt=plt):
    fit, dat = Doublelor_fit_range(data, low, high, ind, down)

    def doublelor(x, b, a, w, f, aa, ww, ff, slope):
        wslope = lorentzianslope(x, b, a, w, f, slope, low)
        noback = lorentziannoback(x, aa, ww, ff)
        return wslope + noback
    plt.plot(dat[0], dat[ind], dat[0], doublelor(dat[0], *fit))
    return fit


def DoubleGaussFitRange(fulldata, low, high, ind=1, down=1):
    data, ga, gf, df = sub_data_guess(fulldata, low, high, ind, down)

    def doublegauss(x, b, a, w, f, aa, ww, ff, slope):
        wslope = lorentzianslope(x, b, a, w, f, slope, gf)
        noback = lorentziannoback(x, aa, ww, ff)
        return wslope + noback
    guess = [data[ind][0], ga, df / 4, gf + df / 4, ga, df / 4,
             gf + 3 * df / 4, (data[ind][-1] - data[ind][1]) / df]
    fit = sop.curve_fit(doublegauss, data[0], data[ind], guess)
    return [fit[0], data]


def PlotDoubleGaussFitRange(data, low, high, ind=1, down=1, plt=plt):
    fit, dat = DoubleGaussFitRange(data, low, high, ind, down)

    def doublegauss(x, b, a, w, f, aa, ww, ff, slope):
        wslope = lorentzianslope(x, b, a, w, f, slope, low)
        noback = lorentziannoback(x, aa, ww, ff)
        return wslope + noback
    plt.plot(dat[0], dat[ind], dat[0], doublegauss(dat[0], *fit))
    return fit


def VoigtFitRange(fulldata, low, high, ind=1, down=1):
    data, ga, gf, df = sub_data_guess(fulldata, low, high, ind, down)

    def v(x, b, a, f, GW, LW, slope):
        return voigtslope(x, b, a, f, GW, LW, slope, gf)
    guess = [data[ind][0], ga * 1e3, gf + df / 2,
             df / 2, 1, (data[ind][-1] - data[ind][1]) / df]
    fit = sop.curve_fit(v, data[0], data[ind], guess)
    return [fit[0], data]


def PlotVoigtFitRange(data, low, high, ind=1, down=1, plt=plt):
    [fit, dat] = VoigtFitRange(data, low, high, ind, down)
    plt.plot(dat[0], dat[ind], dat[0], voigtslope(
        dat[0], fit[0], fit[1], fit[2], fit[3], fit[4], fit[5], low))
    return fit


def DoubleVoigtFitRange(fulldata, low, high, ind=1, down=1):
    data, ga, gf, df = sub_data_guess(fulldata, low, high, ind, down)

    def doublev(x, b, a, f, GW, LW, aa, ff, GWw, LWw, slope):
        wslope = voigtslope(x, b, a, f, GW, LW, slope, low)
        noback = voigtnoback(x, aa, ff, GWw, LWw)
        return wslope + noback
    guess = [data[ind][0], ga * 1e3, gf + df / 4, df / 4, 1, ga * 1e3,
             gf + 3 * df / 4, df / 4, 1, (data[ind][-1] - data[ind][1]) / df]
    fit = sop.curve_fit(doublev, data[0], data[ind], guess)
    return [fit[0], data]


def PlotDoubleVoigtFitRange(data, low, high, ind=1, down=1, plt=plt):
    fit, dat = DoubleVoigtFitRange(data, low, high, ind, down)

    def doublev(x, b, a, f, GW, LW, aa, ff, GWw, LWw, slope):
        wslope = voigtslope(x, b, a, f, GW, LW, slope, low)
        noback = voigtnoback(x, aa, ff, GWw, LWw)
        return wslope + noback
    plt.plot(dat[0], dat[ind], dat[0], doublev(dat[0], *fit))
    return fit


def ATanFitRange(fulldata, low, high, ind=2, down=1, indf=3):
    data, ga, gf, df = sub_data_guess(fulldata, low, high, ind, down)

    def atans(x, b, a, w, f, slope):
        return atanslope(x, b, a, w, f, slope, gf)
    guess = [data[ind][0], ga * 1e3, df / 4, gf + df / 4,
             (fulldata[indf][-1] - fulldata[indf][1]) / (fulldata[0][-1] -
             fulldata[0][1])]
    fit = sop.curve_fit(atans, data[0], data[indf], guess)
    return [fit[0], data]


def PlotATanFitRange(data, low, high, indf=2, plt=plt):
    fit, dat = ATanFitRange(data, low, high)

    def atans(x, b, a, w, f, slope):
        return atanslope(x, b, a, w, f, slope, low)
    plt.plot(dat[0], dat[indf], dat[0], atans(
        dat[0], fit[0], fit[1], fit[2], fit[3], fit[4]))
    return fit


def DoubleATanFitRange(fulldata, low, high, ind=2, indf=3):
    points = find_nearest_idx(fulldata[0], [low, high])
    data = np.array([np.array(x[points[0]:points[1]]) for x in fulldata])
    gf = low
    df = high - low
    ga = data[ind].min() - data[ind][0]

    def doubleatan(x, b, a, w, f, aa, ww, ff, slope):
        return atanslope(x, b, a, w, f, slope, gf) + atannoback(x, aa, ww, ff)
    guess = [data[ind][0], ga * 1e3, df / 4, gf + df / 4, ga * 1e3,
             df / 4, gf + 3 * df / 4, (data[indf][-1] - data[indf][1]) / df]
    fit = sop.curve_fit(doubleatan, data[0], data[indf], guess)
    return [fit[0], data]


def PlotDoubleATanFitRange(data, low, high, indf=2, plt=plt):
    fit, dat = DoubleATanFitRange(data, low, high)

    def doubleatan(x, b, a, w, f, aa, ww, ff, slope):
        return atanslope(x, b, a, w, f, slope, low) + atannoback(x, aa, ww, ff)
    plt.plot(dat[0], dat[indf], dat[0], doubleatan(dat[0], *fit))
    return fit


def PDVFRComp(data, low, high, ind=1, down=1, plt=plt):
    fit = PlotDoubleVoigtFitRange(data, low, high, ind, down, plt)
    GW = np.abs(fit[[3, 7]])
    LW = np.abs(fit[[4, 8]])
    VW = 0.5346 * LW + np.sqrt(0.2166 * LW**2 + GW**2)
    return {'Width': VW, 'Lor/Gauss': LW / GW, 'Center': fit[[2, 6]],
            'Temp': data.T[0]}


def PDLFRComp(data, low, high, ind=1, down=1, plt=plt):
    fit = PlotDoublelor_fit_range(data, low, high, ind, down, plt)
    W = np.abs(fit[[2, 5]])
    f = np.abs(fit[[3, 6]])
    return {'Width': W, 'Center': f, 'Temp': data.T[0]}


def PDGFRComp(data, low, high, plt=plt):
    fit = PlotDoubleGaussFitRange(data, low, high, plt)
    W = np.abs(fit[[2, 5]])
    f = np.abs(fit[[3, 6]])
    return {'Width': W, 'Center': f, 'Temp': data.T[0]}


def PVFRComp(data, low, high, plt=plt):
    fit = PlotVoigtFitRange(data, low, high, plt)
    GW = np.abs(fit[3])
    LW = np.abs(fit[4])
    VW = 0.5346 * LW + np.sqrt(0.2166 * LW**2 + GW**2)
    return {'Width': VW, 'Lor/Gauss': LW / GW, 'Center': fit[2],
            'Temp': data.T[0]}
