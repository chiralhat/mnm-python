import utility as ut
import glob
import re
import csv
import os
import numpy as np
import pandas as pd
import scipy.optimize as sop
import matplotlib.pyplot as plt
from functools import partial

# These regular expressions are used to extract information from the names
# of the data files
h_patt = '([+-]?\d+(?:\.\d+)?)Oe'
t_patt = '(\d+(?:\.\d+)?)K'
p_patt = '([+-]?\d+(?:\.\d+)?dBm)'
n_patt = '(\d+(?:\.\d+)?).txt'
f_patt = '(\d+(?:\.\d+)?)MHz'
pd_patt = 'P(\d+(?:\.\d+)?)D(\d+(?:\.\d+)?)'
defNSpatt = '.* ' + t_patt + ' ' + n_patt
defVNAoutpatt = '(.* 100).* ' + p_patt + ' *'
defpatt = '.* ' + h_patt + '.* ' + t_patt + ' .*.txt'
defFSFSpatt = '.* ' + h_patt + '.* ' + t_patt + ' .* ' + f_patt + '.*.txt'
defPulsepatt = '.* ' + t_patt + '.* ' + f_patt + '* ' + h_patt + '.*.txt'
defPulseFSpatt = '.* ' + f_patt + '*.txt'
defPulseoutpatt = '(.*Sweep).* (\d+Ave) ([+-]?\d+(?:\.\d+)?dBm)*'
defPSoutpatt = '(.*us) .* (\d+Ave) ([+-]?\d+(?:\.\d+)?dBm)*'
defechopatt = '.* ' + h_patt + '.* ' + t_patt + ' ' + pd_patt + '.* ' + n_patt
filepatt = '^K  ([+-]?\\d+(?:\\.\\d+)?)*'

# This section is for data gathered from loop-gap resonators using the VNA

# Names of the columns in an exported dataset
colnames = ["Magnetic Field", "Base", "B-error", "Peak Amplitude", "A-error",
            "FWHH", "W-error", "Central Frequency", "Frequency Shift",
            "F-error", "R-Squared", "Q-Value", "Q-error" "Minimum",
            "Temperature"]

# Functions for VNA fitting


def xp_to_quad(data, mod=1):
    """Converts VNA data from amplitude (x) and phase (p) to in-phase (i) and
    quadrature (q).

    'data' is a data array with elements [frequency (f), x, p];
    'mod' is 1 (default) to convert to [f, q, i], 3 to convert to [f, i, q],
    5 to convert to [f, -x, p], 0 to not convert and just pass [f, x, p].

    Returns one of the above arrays."""
    f = data[0]
    x = data[1]
    p = data[2]
    q = (-x * np.cos(p / 180 * np.pi) / (x.max() - x.min()) + x.mean()) * mod
    i = 180 / np.pi * x * np.sin(p / 180 * np.pi)

    def outsw(x):
        return {
            3: [f, np.sign(mod) * i, q],
            5: [f, -x, p],
            0: [f, x, p]
        }.get(x, [f, q, i])
    return np.array(outsw(np.abs(mod)))


def test_vna_xp(path, dir, skew, full, mod):
    """Loads a VNA dataset and fits it and its converted counterpart to
    Lorentzians, plotting the result to two separate subplots.

    The fit function is:
    y(x) = b + a * (w / 2)**2 / ((x - f)**2 + (w / 2)**2)

    b is the background offset, a is the peak amplitude, w is the peak width,
    and f is the peak frequency.

    'path' is a string corresponding to the desired datafile;
    'dir' is the data directory;
    'skew' is 1 to subtract a linear background, and 0 to use no slope;
    'full' is 1 to fit the whole dataset, and 0 to only fit between the
    maxima on either side of the peak;
    for details on 'mod' see 'xp_to_quad'."""
    name = glob.glob(os.path.join(dir, path))[0]
    data = np.loadtxt(name, skiprows=1).transpose()
    xdata = xp_to_quad(data, mod)
    f, axarr = plt.subplots(2, sharex=True, figsize=(8, 6))
    ut.plot_lor_fit(data, skew, full, axarr[0])
    ut.plot_lor_fit(xdata, skew, 0, axarr[1])


def load_vna_data(path, dir, mod=0, patt=defNSpatt):
    """Loads all datasets from field sweeps using the VNA corresponding to a
    given string.

    'path' is a string corresponding to the desired datafile;
    'dir' is the data directory;
    'mod' controls whether to convert the amplitude and phase data into
    in-phase and quadrature data, see documentation for
    'xp_to_quad' for full details (default:0, no conversion);
    'patt' is a regular expression pattern to extract temperature and sequence
    information from the filename.

    Returns a list with: a list of datasets; an array of experimental fields;
    an array of experimental temperatures; one of the matching filenames."""
    namex = ('Min.txt', 'Fit.txt')
    data, names, archive = ut.get_data_exclude_names(path + '*', dir,
                                                     namex, 1)

    def matchnum(name):
        [temp, num] = re.match(patt, name).groups()
        return [float(temp), int(num)]
    [temps, nums] = np.array(list(map(matchnum, names))).transpose()

    def fields1(path):
        if archive == 0:
            with open(path) as f:
                first_line = f.readline()
        else:
            with archive.open(path) as f:
                first_line = f.readline().decode('utf-8')
        return float(re.match(filepatt, first_line).groups()[0])
    fields = np.array([fields1(nam) for nam in names])
    sortarr = np.argsort(nums)
    [data, names, fields, temps] = [np.array(data)[sortarr], np.array(names)[
        sortarr], fields[sortarr], temps[sortarr]]
    if mod != 0:
        data = np.array([xp_to_quad(dat, mod) for dat in data])
    if archive != 0:
        archive.close()
    return [data, fields, temps, os.path.split(names[0])[1]]


def load_and_fit_vna(path, dir, skew=1, full=1, mod=0, patt=defNSpatt,
                     **kwargs):
    """Loads all datasets from field sweeps using the VNA corresponding to a
    given string and fits them to a Lorentzian.

    The fit function is:
    y(x) = b + a * (w / 2)**2 / ((x - f)**2 + (w / 2)**2)

    b is the background offset, a is the peak amplitude, w is the peak width,
    and f is the peak frequency.

    'path' is a string corresponding to the desired datafile;
    'dir' is the data directory;
    'skew' is 1 (default) to subtract a linear background,
    and 0 to use no slope;
    'full' is 1 (default)  to fit the whole dataset, and 0 to only fit
    between the maxima on either side of the peak;
    'mod' controls whether to convert the amplitude and phase data into
    in-phase and quadrature data, see documentation for
    'xp_to_quad' for full details (default:0, no conversion);
    'patt' is a regular expression pattern to extract temperature and sequence
    information from the filename.

    Returns a list with: an array whose elements are as follows:
    field, background offset, background offset uncertainty, amplitude,
    amplitude uncertainty, peak width, peak width uncertainty, frequency,
    frequency shift, frequency uncertainty, adjusted R-squared for the fit,
    Q, Q uncertainty, peak minimum, temperature;
    one of the matching filenames."""
    [data, fields, temps, name] = load_vna_data(path, dir, mod, patt)
    [fits, fiterrs, slopes, R2bs] = np.array(
        [ut.lor_fit(dat, skew, full) for dat in data]).transpose()
    coefs = np.array([f for f in fits]).transpose()
    errs = np.array([f for f in fiterrs]).transpose()
    mins = np.array(list(map(ut.find_min, data)))
    out = ut.process_coefs(coefs, errs, R2bs, mins, fields, temps)
    return[out, name]

# This will eventually be converted to a more general function, like
# cav_load_fit_out


def load_fit_vna_out(path, dir, outdir, skew=1, full=1, mod=0, patt=defNSpatt,
                     outpatt=defVNAoutpatt, **kwargs):
    """Loads all datasets from field sweeps using the VNA corresponding to a
    given string, fits them to a Lorentzian, and saves the fit parameters to a
    CSV file.

    The fit function is:
    y(x) = b + a * (w / 2)**2 / ((x - f)**2 + (w / 2)**2)

    b is the background offset, a is the peak amplitude, w is the peak width,
    and f is the peak frequency.

    'path' is a string corresponding to the desired datafile;
    'dir' is the data directory;
    'outdir' is the fit output directory;
    'skew' is 1 (default)  to subtract a linear background,
    and 0 to use no slope;
    'full' is 1 (default)  to fit the whole dataset, and 0 to only fit
    between the maxima on either side of the peak;
    'mod' controls whether to convert the amplitude and phase data into
    in-phase and quadrature data, see documentation for
    'xp_to_quad' for full details (default:0, no conversion);
    'patt' is a regular expression pattern to extract temperature and sequence
    information from the filename; 'outpatt' is a regular expression pattern to
    extract the filename prefix and the applied power from the filename.

    Returns the data array from 'load_and_fit_vna'."""
    [out, name] = load_and_fit_vna(path, dir, skew, full, mod, patt)
    [dir1, dat] = os.path.split(dir)
    [dir2, res] = os.path.split(dir1)
#    [tmp,proj]=os.path.split(dir2);
    # Get the initial part of the path and the power from the name
    [pref, power] = re.match(outpatt, name).groups()
    tout = out.transpose()
    outname = res + ' ' + dat + ' ' + pref + ' ' + power + \
        ' ' + str(tout[0][-1]) + 'K Fit Analysis.csv'
    outpath = os.path.join(outdir, outname)
    outnames = ["Magnetic Field", "Base", "B-error", "Peak Amplitude",
                "A-error", "FWHH", "W-error", "Central Frequency",
                "Frequency Shift", "F-error", "R-Squared", "Q-Value",
                "Q-error", "Minimum", "Temperature"]
    data_save_out(outpath, outnames, tout)
    return out


# This section is for data gathered from loop-gap resonators using the
# pulsed setup


def iq_to_xp(data, mod=1):
    """Converts pulsed data from in-phase (i) and quadrature (q) to
    amplitude (x) and phase (p).

    'data' is a data array with elements [frequency (f), i, q];
    'mod' is 1 (default) to convert to [f, x, p], -3 to convert to [f, -x, p],
    4 to conver to [f, q, i], 5 to convert to [f, -i, q],
    0 to not convert and just pass [f, i, q].

    Returns one of the above arrays."""
    t = data[0]
    i = data[1]
    q = data[2]
    x = np.sqrt(i**2 + q**2)
    p = np.arccos((i**2 - q**2) / (i**2 + q**2)) / 2

    def outsw(m):
        return {
            3: [t, x * np.sign(mod), p],
            4: [t, q, i],
            5: [t, -i, q],
            0: [t, i, q]
        }.get(m, [t, x, p])
    return np.array(outsw(mod))

# Various fitting functions


def fit_exp_decay(data, freq, timoff=0, fitind=1):
    """Fits an exponential decay function to a pulsed dataset.

    The fit function is:
    y(x) = a * np.exp(-x / t) + c

    c is the background offset, a is the amplitude, t is the decay time.

    'data' is a list of data arrays, each of which has elements
    [time,decay data,...];
    'freq' is the microwave frequency in Hz (or an array of frequencies);
    'timoff' is the time in us after the maximum of the signal to start the
    fit window (default: 0);
    'fitind' is the index of the data array corresponding to the
    decay data to fit.

    Returns a list with three elements: an array with the fit parameters
    (the background, the amplitude, the decay time, the Q);
    an array with the uncertainties in those parameters;
    the adjusted R-squared for the fit."""
    if type(freq) == float or type(freq) == int:
        fit = np.array([ut.exp_fit(dat, freq, dat[0, dat[fitind].argmax()] +
                                   timoff, dat[0, -1], 1, 0) for dat in data])
    else:
        fit = np.array([ut.exp_fit(dat, freq[i], dat[0, dat[fitind].argmax()] +
                                   timoff, dat[0, -1], 1, 0)
                        for i, dat in enumerate(data)])
    [fits, fiterrs, R2bs] = fit.transpose()
    coefs = np.array([f for f in fits]).transpose()
    errs = np.array([f for f in fiterrs]).transpose()
    return [coefs, errs, R2bs]


def plot_fit_exp_decay(data, freq, timoff=0.028, fitind=1, inv=0, plt=plt):
    """Fits an exponential decay function to a pulsed dataset and plots it.

    The fit function is:
    y(x) = a * np.exp(-x / t) + c

    c is the background offset, a is the amplitude, t is the decay time.

    'data' is a list of data arrays, each of which has elements
    [time,decay data,...];
    'freq' is the microwave frequency in Hz;
    'rep' is the number of repetitions at each experimental parameter
    (usually field);
    'timoff' is the time in us after the maximum of the signal to start the
    fit window (default: 0);
    'fitind' is the index of the data array corresponding to the
    decay data to fit;
    'plt' is the plotting environment to use, defaulting to pyplot
    (imported as plt).

    Returns an array with the fit parameters
    (the background, the amplitude, the decay time, the Q)."""
    stpt = data[fitind].argmax()
    fstpt = np.abs(data[0] - (data[0][stpt] + timoff)).argmin()
    fit = fit_exp_decay([data], freq, timoff, fitind)[0][:, 0]
    plt.plot(data[0][stpt:], data[1][stpt:])
    plt.plot(data[0][fstpt:], ut.exponential(data[0][fstpt:], *fit[:-1]), 'r')
    return fit


def fit_exp_rep_decay(data, freq, rep, timoff=0, fitind=1):
    """Fits an exponential decay function to a pulsed repetition dataset.

    The fit function is:
    y(x) = a * np.exp(-x / t) + c

    c is the background offset, a is the amplitude, t is the decay time.

    'data' is a list of data arrays, each of which has elements
    [time,decay data,...];
    'freq' is the microwave frequency in Hz;
    'rep' is the number of repetitions at each experimental parameter
    (usually field);
    'timoff' is the time in us after the maximum of the signal to
    start the fit window (default: 0);
    'fitind' is the index of the data array corresponding to the
    decay data to fit.

    Returns a list with three elements: an array with the fit parameters
    (the background, the amplitude, the decay time, the Q);
    an array with the uncertainties in those parameters;
    the adjusted R-squared for the fit."""
    fit = np.array([ut.exp_fit(dat, freq, dat[0, dat[fitind].argmax()] +
                               timoff, dat[0, -1], 1, 0) for dat in data])
    [fits, fiterrs, R2bs] = fit.transpose()
    coefs = np.array([f for f in fits]).transpose()
    errs = np.array([f for f in fiterrs]).transpose()
    fits = []
    fiterrs = []
    fitR2bs = np.array([np.mean(R2bs[j * rep:j * rep + rep])
                        for j in np.arange(len(R2bs) / rep)])
    for i in np.arange(len(coefs)):
        fits.append(np.array([np.mean(coefs[i][j * rep:j * rep + rep])
                              for j in np.arange(int(len(coefs[i]) / rep))]))
        fiterrs.append(np.array([np.mean(errs[i][j * rep:j * rep + rep])
                                 for j in np.arange(int(len(errs[i])/rep))]))
    fits = np.array(fits)
    fiterrs = np.array(fiterrs)
    return [fits, fiterrs, fitR2bs]

# Data loading functions; often the main differentiating feature is the
# number and order of parameters matched from the filenames


def load_pulse(path, dir, patt=defPulsepatt, inv=0):
    """Loads all datasets from field sweeps using the pulsed setup
    corresponding to a given string.

    'path' is a string corresponding to the desired datafile;
    'dir' is the data directory;
    'patt' is a regular expression pattern to extract temperature and
    field information from the filename;
    'inv' is nonzero if one of the arrays should be endpoint-subtracted
    and inverted, with the array index being the value
    (default: 0, do nothing).

    Returns an array with: a list of datasets;
    an array of experimental fields; an array of experimental temperatures;
    one of the matching filenames."""
    fpath = path + '*.txt'
    data, names = ut.get_data_names(fpath, dir)

    def matchnum(name):
        [temp, freq, field] = re.match(patt, name).groups()
        return [float(field), float(freq), float(temp)]
    [fields, freqs, temps] = np.array(list(map(matchnum, names))).transpose()
    sortarr = np.argsort(fields)
    [data, names, fields, freqs] = [np.array(data)[sortarr], np.array(names)[
        sortarr], fields[sortarr], freqs[sortarr]]
    if inv != 0:
        for dat in data:
            dat[1] = (dat[1] - dat[1][-1])
            dat[2] = (dat[2] - dat[2][-1])
            dat[inv] = -dat[inv]
    return [data, fields, freqs, temps, os.path.split(names[0])[1]]


def load_pulse_freq_sweep(path, dir, patt=defPulseFSpatt):
    """Loads all datasets from frequency sweeps using the pulsed setup
    corresponding to a given string.

    'path' is a string corresponding to the desired datafile;
    'dir' is the data directory;
    'patt' is a regular expression pattern to extract frequency
    information from the filename.

    Returns an array with: a list of datasets;
    an array of experimental frequencies;
    one of the matching filenames."""
    fpath = path + '*.txt'
    data, names = ut.get_data_names(fpath, dir)

    def matchnum(name):
        freq = re.match(patt, name).groups()[0]
        return float(freq)
    freqs = np.array(list(map(matchnum, names)))
    sortarr = np.argsort(freqs)
    [data, names, freqs] = [np.array(data)[sortarr], np.array(names)[
        sortarr], freqs[sortarr]]
    return [data, freqs, os.path.split(names[0])[1]]


def load_pulse_step(path, dir, patt=defpatt):
    """Loads all datasets from stepped field sweeps using the
    pulsed setup corresponding to a given string.

    'path' is a string corresponding to the desired datafile;
    'dir' is the data directory;
    'patt' is a regular expression pattern to extract temperature
    and field information from the filename.

    Returns an array with: a list of datasets;
    an array of experimental fields;
    an array of experimental temperatures;
    one of the matching filenames."""
    fpath = path + '*.txt'
    data, names = ut.get_data_names(fpath, dir)

    def matchnum(name):
        [field, temp] = re.match(patt, name).groups()
        return [float(field), float(temp)]
    [fields, temps] = np.array(list(map(matchnum, names))).transpose()
    sortarr = np.argsort(fields)
    [data, names, fields] = [np.array(data)[sortarr], np.array(names)[
        sortarr], fields[sortarr]]
    return [data, fields, temps, os.path.split(names[0])[1]]


def load_pulse_freq_sweep_field_step(path, dir, patt=defFSFSpatt):
    """Loads all datasets from stepped field sweeps with frequency sweeps
    using the pulsed setup corresponding to a given string.

    'path' is a string corresponding to the desired datafile;
    'dir' is the data directory;
    'patt' is a regular expression pattern to extract temperature
    and field information from the filename.

    Returns an array with: a list of datasets;
    an array of experimental fields;
    an array of experimental temperatures;
    an array of experimental powers;
    one of the matching filenames."""
    fpath = path + '*.txt'
    data, names = ut.get_data_names(fpath, dir)

    def matchnum(name):
        [field, temp, freq] = re.match(patt, name).groups()
        return [float(field), float(temp), float(freq)]
    [fields, temps, freqs] = np.array(list(map(matchnum, names))).transpose()
    sortarr = np.argsort([int(f) for f in fields])
    [data, names, fields, freqs] = [np.array(data)[sortarr], np.array(names)[
        sortarr], fields[sortarr], freqs[sortarr]]
    return [data, fields, temps, freqs, os.path.split(names[0])[1]]


def load_pp(path, dir, freq, patt=defechopatt, inv=0):
    """Loads all datasets from field sweeps using the pump probe setup
    corresponding to a given string.

    'path' is a string corresponding to the desired datafile;
    'dir' is the data directory;
    'freq' is the microwave frequency in MHz;
    'patt' is a regular expression pattern to extract temperature, field,
    pulse length, delay length, and sequence information from the filename;
    'inv' is nonzero if one of the arrays should be endpoint-subtracted and
    inverted, with the array index being the value (default: 0, do nothing).

    Returns an array with: a list of datasets;
    an array of experimental fields;
    an array of experimental temperatures;
    an array of pulse lengths;
    an array of delay lengths;
    one of the matching filenames."""
    fpath = path + '*' + str(freq) + '*.txt'
    data, names = ut.get_data_names(fpath, dir)

    def matchnum(name):
        [field, temp, pulse, delay, num] = re.match(patt, name).groups()
        return [float(field), float(temp), int(pulse), int(delay), int(num)]
    [fields, temps, pulses, delays, nums] = np.array(
        list(map(matchnum, names))).transpose()
    sortarr = np.argsort(delays)
    sortarrs = [np.array(data)[sortarr], np.array(names)[sortarr],
                fields[sortarr], pulses[sortarr], delays[sortarr]]
    [data, names, fields, pulses, delays] = sortarrs
    if inv != 0:
        for dat in data:
            dat[1] = (dat[1] - dat[1][-1])
            dat[2] = (dat[2] - dat[2][-1])
            dat[inv] = -dat[inv]
    return [data, fields, temps, pulses, delays, os.path.split(names[0])[1]]

# Functions to load and fit datasets


def load_fit_pulse(path, dir, timoff=0, xpconv=0, fitind=1,
                   patt=defPulsepatt, inv=0, **kwargs):
    """Loads all datasets from field sweeps using the pulsed setup
    corresponding to a given string and fits them to an exponential decay.

    'path' is a string corresponding to the desired datafile;
    'dir' is the data directory;
    'freq' is the microwave frequency in MHz;
    'timoff' is the number of points after the maximum of the signal to start
    the fit window (default: 0);
    'xpconv' controls whether to convert the data from in-phase and quadrature
    to amplitude and phase, see the documentation for 'iq_to_xp'
    (default:0, no conversion);
    'fitind' is the index of the data array corresponding to the
    decay data to fit (default:1);
    'patt' is a regular expression pattern to extract temperature and sequence
    information from the filename;
    'inv' is nonzero if one of the arrays should be endpoint-subtracted and
    inverted, with the array index being the value (default: 0, do nothing).

    Returns an array with: an array whose elements are as follows:
    the field, the background offset, the background offset uncertainty,
    the amplitude, the amplitude uncertainty, the decay time,
    the decay time uncertainty, the Q, the Q uncertainty,
    the adjusted R-squared for the fit, the temperature;
    one of the matching filenames."""
    [data, fields, freqs, temps, name] = load_pulse(path, dir, patt, inv)
    if xpconv != 0:
        mfun = partial(iq_to_xp, mod=xpconv)
        data = list(map(mfun, data))
    [fits, fiterrs, R2bs] = fit_exp_decay(data, freqs, timoff, fitind)
    fitparerrs = np.concatenate([[fits[i], fiterrs[i]]
                                 for i in np.arange(len(fits))], axis=0)
    out = np.concatenate(([fields], fitparerrs, [freqs, R2bs, temps]), axis=0)
    return[out, name]


def load_fit_pulse_rep_step(path, dir, freq, reps=1, xpconv=0, timoff=0,
                            fitind=1, patt=defpatt, **kwargs):
    """Loads all datasets from stepped field sweeps using the pulsed setup
    corresponding to a given string and fits them to an exponential decay.

    The fit function is:
    y(x) = a * np.exp(-x / t) + c

    c is the background offset, a is the amplitude, t is the decay time.

    'path' is a string corresponding to the desired datafile;
    'dir' is the data directory;
    'freq' is the microwave frequency in MHz;
    'reps' is the number of repeated measurements at each field (default:1);
    'timoff' is the number of points after the maximum of the signal to start
    the fit window, when left at 0 it is determined algorithmically by finding
    the number of points that produces the smallest deviation from the fit
    (default: 0);
    'xpconv' controls whether to convert the data from in-phase and quadrature
    to amplitude and phase, see the documentation for 'iq_to_xp'
    (default:0, no conversion);
    'fitind' is the index of the data array corresponding to the
    decay data to fit (default:1);
    'patt' is a regular expression pattern to extract temperature and sequence
    information from the filename.

    Returns an array with: an array whose elements are as follows:
    the field, the background offset, the background offset uncertainty,
    the amplitude, the amplitude uncertainty, the decay time,
    the decay time uncertainty, the Q, the Q uncertainty, the temperature;
    one of the matching filenames; an array with all the loaded datasets."""
    [data, fields, temps, name] = load_pulse_step(path, dir, patt)
    if xpconv != 0:
        mfun = partial(iq_to_xp, mod=xpconv)
        data = list(map(mfun, data))
    fields = fields[::reps]
    temps = temps[::reps]
    if timoff == 0:
        timoff = np.array(list(map(lambda n: np.std(fit_exp_decay(
            data[:reps], freq, n, fitind)[0][2]), np.arange(60)))).argmin()
    [fits, fiterrs, R2bs] = fit_exp_rep_decay(
        data, freq, reps, timoff, fitind)
    fitparerrs = np.concatenate([[fits[i], fiterrs[i]]
                                 for i in np.arange(len(fits))], axis=0)
    out = np.concatenate(([fields], fitparerrs, [R2bs, temps]), axis=0)
    return[out, name, data]


def process_max_q(fits, fiterrs, R2bs, fields, temps, freqs, nsweep):
    """Finds the maximum Q in a set of frequency sweeps with 'nsweep' members.
    Returns arrays of the input values at the maximum index."""
    max_Q_index = np.array([fits[3][i*nsweep:(i+1)*nsweep].argmax()+i*nsweep
                            for i in np.arange(len(fits[3]) // nsweep)])
    fits = np.array([fit[max_Q_index] for fit in fits])
    fiterrs = np.array([fit[max_Q_index] for fit in fiterrs])
    R2bs, fields, temps, freqs = np.array(
        [arr[max_Q_index] for arr in [R2bs, fields, temps, freqs]])
    return fits, fiterrs, R2bs, fields, temps, freqs


def load_fit_pulse_freq_sweep_field_step(path, dir, xpconv=0, timoff=0,
                                         fitind=1, patt=defFSFSpatt,
                                         **kwargs):
    """Loads all datasets from stepped field sweeps with frequency sweeps
    using the pulsed setup corresponding to a given string and fits them
    to an exponential decay.

    The fit function is:
    y(x) = a * np.exp(-x / t) + c

    c is the background offset, a is the amplitude, t is the decay time.

    'path' is a string corresponding to the desired datafile;
    'dir' is the data directory;
    'xpconv' controls whether to convert the data from in-phase and quadrature
    to amplitude and phase, see the documentation for 'iq_to_xp'
    (default:0, no conversion);
    'timoff' is the number of points after the maximum of the signal to start
    the fit window, when left at 0 it is determined algorithmically by finding
    the number of points that produces the smallest deviation from the fit
    (default: 0);
    'fitind' is the index of the data array corresponding to the
    decay data to fit (default:1);
    'patt' is a regular expression pattern to extract temperature
    and sequence information from the filename.

    Returns an array with: an array whose elements are as follows:
    the field, the background offset, the background offset uncertainty,
    the amplitude, the amplitude uncertainty, the decay time,
    the decay time uncertainty, the Q, the Q uncertainty, the temperature;
    one of the matching filenames;
    an array with all the loaded datasets."""
    [data, fields, temps, freqs, name] = load_pulse_freq_sweep_field_step(
        path, dir, patt)
    # REMOVE AFTER FIXING LABVIEW
    data = np.array([dat.transpose() for dat in data])
    ###
    if xpconv != 0:
        mfun = partial(iq_to_xp, mod=xpconv)
        data = list(map(mfun, data))
    [fits, fiterrs, R2bs] = fit_exp_decay(data, freqs, timoff, fitind)
    nsweep = np.unique(fields, return_counts=True)[1][0]
    # Find the maximum Q for each frequency sweep, and return its index
    fits, fiterrs, R2bs, fields, temps, freqs = process_max_q(
        fits, fiterrs, R2bs, fields, temps, freqs, nsweep)
    fitparerrs = np.concatenate([[fits[i], fiterrs[i]]
                                 for i in np.arange(len(fits))], axis=0)
    out = np.concatenate(([fields], fitparerrs, [freqs, R2bs, temps]), axis=0)
    return[out, name]


def load_fit_pp(path, dir, freq, timoff=0, xpconv=0, fitind=1,
                patt=defechopatt, inv=0, **kwargs):
    """Loads all datasets from pump probe measurements using the pulsed setup
    corresponding to a given string and fits them to a Lorentzian.

    'path' is a string corresponding to the desired datafile;
    'dir' is the data directory;
    'freq' is the microwave frequency in MHz;
    'timoff' is the number of points after the maximum of the signal to start
    the fit window (default: 0);
    'xpconv' controls whether to convert the data from in-phase and quadrature
    to amplitude and phase, see the documentation for 'iq_to_xp'
    (default:0, no conversion);
    'fitind' is the index of the data array corresponding to the
    decay data to fit (default:1);
    'patt' is a regular expression pattern to extract temperature, field,
    pulse length, delay length, and sequence information from the filename;
    'inv' is nonzero if one of the arrays should be endpoint-subtracted and
    inverted, with the array index being the value (default: 0, do nothing).

    Returns an array with: an array whose elements are as follows:
    the delay between pulses, the background offset,
    the background offset uncertainty, the amplitude,
    the amplitude uncertainty, the decay time, the decay time uncertainty,
    the Q, the Q uncertainty, the pulse length, the field,
    the adjusted R-squared for the fit, the temperature;
    one of the matching filenames."""
    [data, fields, temps, pulses, delays, name] = load_pp(
        path, dir, freq, patt, inv)
    if xpconv != 0:
        mfun = partial(iq_to_xp, mod=xpconv)
        data = list(map(mfun, data))
    [fits, fiterrs, R2bs] = fit_exp_decay(data, freq, timoff, fitind)
    fitparerrs = np.concatenate([[fits[i], fiterrs[i]]
                                 for i in np.arange(len(fits))], axis=0)
    out = np.concatenate(
        ([delays], fitparerrs, [pulses, fields, R2bs, temps]), axis=0)
    return[out, name]


def load_fit_pp_rep_step(path, dir, freq, reps, xpconv=0, timoff=0, fitind=1,
                         patt=defpatt, **kwargs):
    """Loads all datasets from repeated pump probe measurements using the
    pulsed setup corresponding to a given string and fits them to a
    Lorentzian, averaging the results for each point in parameter space.

    'path' is a string corresponding to the desired datafile;
    'dir' is the data directory;
    'freq' is the microwave frequency in MHz;
    'reps' is the number of repetitions at each experimental parameter;
    'timoff' is the number of points after the maximum of the signal to start
    the fit window (default: 0);
    'xpconv' controls whether to convert the data from in-phase and quadrature
    to amplitude and phase, see the documentation for 'iq_to_xp'
    (default:0, no conversion);
    'fitind' is the index of the data array corresponding to the decay data to
    fit (default:1);
    'patt' is a regular expression pattern to extract temperature, field,
    pulse length, delay length, and sequence information from the filename.

    Returns an array with: an array whose elements are as follows:
    the delay between pulses, the background offset,
    the background offset uncertainty, the amplitude,
    the amplitude uncertainty, the decay time, the decay time uncertainty,
    the Q, the Q uncertainty, the pulse length, the field,
    the adjusted R-squared for the fit, the temperature;
    one of the matching filenames."""
    [data, fields, temps, pulses, delays, name] = load_pp(path, dir, freq,
                                                          patt)
    if xpconv != 0:
        mfun = partial(iq_to_xp, mod=xpconv)
        data = list(map(mfun, data))
    fields = fields[::reps]
    temps = temps[::reps]
    pulses = pulses[::reps]
    delays = delays[::reps]
    if timoff == 0:
        offsets = np.linspace(0, .01, 50)
        timoff = data[0][0, np.array(list(map(lambda n: np.std(fit_exp_decay(
            data[:reps], freq, n, fitind)[2]), offsets))).argmin()]
    [fits, fiterrs, R2bs] = fit_exp_rep_decay(
        data, freq, reps, timoff, fitind)
    fitparerrs = np.concatenate([[fits[i], fiterrs[i]]
                                 for i in np.arange(len(fits))], axis=0)
    out = np.concatenate(
        ([delays], fitparerrs, [pulses, fields, R2bs, temps]), axis=0)
    return[out, name]

# Functions to load, fit, and save datasets


def load_fit_pulse_out(path, dir, outdir, timoff=0, xpconv=0, fitind=1,
                       patt=defPulsepatt, outpatt=defPulseoutpatt, **kwargs):
    """Loads all datasets from field sweeps using the pulsed setup
    corresponding to a given string, fits them to an exponential decay,
    and saves the parameters to a CSV file.

    The fit function is:
    y(x) = a * np.exp(-x / t) + c

    c is the background offset, a is the amplitude, t is the decay time.

    'path' is a string corresponding to the desired datafile;
    'dir' is the data directory;
    'outdir' is the fit output directory;
    'freq' is the microwave frequency in MHz;
    'timoff' is the number of points after the maximum of the signal to start
    the fit window (default: 0);
    'xpconv' controls whether to convert the data from in-phase and quadrature
    to amplitude and phase, see the documentation for 'iq_to_xp'
    (default:0, no conversion);
    'fitind' is the index of the data array corresponding to the decay data to
    fit (default:1);
    'patt' is a regular expression pattern to extract temperature and sequence
    information from the filename;
    'outpatt' is a regular expression pattern to extract the filename prefix
    and the applied power from the filename.

    Returns an array with: the field, the amplitude,
    the amplitude uncertainty, the Q of the decay, the Q uncertainty,
    the frequency, the background offset, the background offset uncertainty,
    the temperature."""
    [out, name] = load_fit_pulse(path, dir, timoff, xpconv, fitind, patt)
    out = np.array(out).transpose()
    [dir1, dat] = os.path.split(dir)
    [tmp, res] = os.path.split(dir1)
    # Get the initial part of the path and the power from the name
    [pref, ave, power] = re.match(outpatt, name).groups()
    outname = res + ' ' + dat + ' ' + pref + \
        (' %.2fMHz ' % out[0][-3]) + ave + ' ' + power + \
        ' ' + str(out[0][-1]) + 'K Fit Analysis.csv'
    outpath = os.path.join(outdir, outname)
    with open(outpath, 'w') as csvfile:
        fout = csv.writer(csvfile, lineterminator='\n')
        fout.writerow(["Magnetic Field", "Base", "B-error", "Peak Amplitude",
                       "A-error", "Tau", "T-error", "Q-Value", "Q-error",
                       "Frequency", "R-Squared", "Temperature"])
        for row in out:
            fout.writerow(row)
    return out.transpose()


def load_fit_rep_ps_out(path, dir, outdir, freq, reps=1, timoff=0, xpconv=0,
                        fitind=1, patt=defpatt, outpatt=defPSoutpatt,
                        **kwargs):
    """Loads all datasets from field sweeps using the pulsed setup
    corresponding to a given string, fits them to an exponential decay,
    and saves the parameters to a CSV file.

    The fit function is:
    y(x) = a * np.exp(-x / t) + c

    c is the background offset, a is the amplitude, t is the decay time.

    'path' is a string corresponding to the desired datafile;
    'dir' is the data directory;
    'outdir' is the fit output directory;
    'freq' is the microwave frequency in MHz;
    'reps' is the number of repeated measurements at each field (default:1);
    'timoff' is the number of points after the maximum of the signal to start
    the fit window (default: 0);
    'xpconv' controls whether to convert the data from in-phase and quadrature
    to amplitude and phase, see the documentation for 'iq_to_xp'
    (default:0, no conversion);
    'fitind' is the index of the data array corresponding to the decay data to
    fit (default:1);
    'patt' is a regular expression pattern to extract temperature and sequence
    information from the filename;
    'outpatt' is a regular expression pattern to extract the filename prefix
    and the applied power from the filename.

    Returns an array with: the field, the amplitude,
    the amplitude uncertainty, the Q of the decay, the Q uncertainty,
    the background offset, the background offset uncertainty,
    the temperature."""
    [pout, name, _] = load_fit_pulse_rep_step(
        path, dir, freq, reps, xpconv, fitind, patt)
    out = np.array(pout).transpose()
    [dir1, dat] = os.path.split(dir)
    [tmp, res] = os.path.split(dir1)
    # Get the initial part of the path and the power from the name
    [pref, ave, power] = re.match(outpatt, name).groups()
    outname = res + ' ' + dat + ' ' + pref + ' ' + ave + ' ' + \
        power + ' ' + str(out[0][-1]) + 'K Fit Analysis.csv'
    outpath = os.path.join(outdir, outname)
    with open(outpath, 'w') as csvfile:
        fout = csv.writer(csvfile, lineterminator='\n')
        fout.writerow(["Magnetic Field", "Amplitude",
                       "Q-Value", "Shift", "Temperature"])
        for row in out:
            fout.writerow(row)
    return pout


# The following section is for 3D cavity datasets
def poly_back(data, snip, deg=2):
    """Fits the background to a polynomial.

    The fit function is:
    y(x) = b + c * x + d * x**2

    b is the background offset, c is the linear term,
    and d is the quadratic term.

    'data' is a data array;
    'snip' is an array containing two frequencies, [f1,f2], such that the
    background fit will exclude the data in between those frequencies;
    'deg' is the degree of the polynomial (default: 2).

    Returns an array with the polynomial fit parameters,
    in order of polynomial degree, starting with the highest degree."""
    snippoints = ut.find_nearest_idx(data[0], snip)
    subdata = np.array([list(dat[:snippoints[0]]) +
                        list(dat[snippoints[1]:]) for dat in data])
    fit = np.polyfit(subdata[0], subdata[1], deg)
    return fit


def poly_lor_fit(fulldata, snip, span, deg=2, guess=0):
    """Fits the background to a polynomial, then uses that fit as the initial
    parameters for a combined polynomial and Lorentzian fit of the data.

    The background fit function is:
    y(x) = b + c * x + d * x**2

    b is the background offset, c is the linear term,
    and d is the quadratic term.

    The combined fit function is:
    y(x) = b + a * (w / 2)**2 / ((x - f)**2 + (w / 2)**2) + c * x + d * x**2

    a is the peak amplitude, w is the peak width, and f is the peak frequency.

    'fulldata' is a data array;
    'snip' is an array containing two frequencies, [f1,f2], such that the
    background fit will exclude the data in between those frequencies;
    'span' is an array containing two other frequencies, [f3,f4], such that
    nothing outside the range [f3,f4] will be fit;
    'deg' is the degree of the polynomial (default: 2);
    'guess' is an array of initial Lorentzian fit guesses
    (default: 0, generate guesses automatically).

    Returns a list with two elements:
    an array with the polynomial and Lorentzian fit parameters:
    the background offset, peak amplitude, peak width, peak frequency,
    linear term, quadratic term;
    and an array with the uncertainties for those fit parameters."""
    spanpoints = ut.find_nearest_idx(fulldata[0], span)
    data = (fulldata.transpose()[spanpoints[0]:spanpoints[1]]).transpose()
    pfit = poly_back(data, snip, deg)
    if guess == 0:
        # Currently this only works for polynomials of degree 2, but that's
        # all we need
        guess = [pfit[2], -.1, 10,
                 np.mean(fulldata[0][[0, len(fulldata[0]) - 1]]), pfit[1],
                 pfit[0]]
    else:
        guess = [pfit[2], guess[0], guess[1], guess[2], pfit[1], pfit[0]]
    fit, fiterr = sop.curve_fit(ut.quadlorfit, data[0], data[1], guess)
    fitdat = ut.quadlorfit(data[0], *fit)
    R2b = ut.r2bar(data[1], fitdat, len(fit))
    return [fit, np.sqrt(np.diag(fiterr)), R2b]


def qlfout(x, fit):
    """Returns fit data matching the input x data.

    'x' is an array of x data (usually frequency);
    'fit' is an array of fit parameters for a combined Lorentzian and
    quadratic fit.

    Returns an array of y data corresponding to the fit at each x point."""
    return ut.quadlorfit(x, fit[0], fit[1], fit[2], fit[3], fit[4], fit[5])


def plot_poly_lor_fit(fulldata, snip, span, deg=2, guess=0, plt=plt):
    """Fits the background to a polynomial, then uses that fit as the initial
    parameters for a combined polynomial and Lorentzian fit of the data,
    and plots that fit along with the data.

    The background fit function is:
    y(x) = b + c * x + d * x**2

    b is the background offset, c is the linear term,
    and d is the quadratic term.

    The combined fit function is:
    y(x) = b + a * (w / 2)**2 / ((x - f)**2 + (w / 2)**2) + c * x + d * x**2

    a is the peak amplitude, w is the peak width, and f is the peak frequency.

    'fulldata' is a data array;
    'snip' is an array containing two frequencies, [f1,f2], such that the
    background fit will exclude the data in between those frequencies;
    'span' is an array containing two other frequencies, [f3,f4], such that
    nothing outside the range [f3,f4] will be fit;
    'deg' is the degree of the polynomial (default: 2);
    'guess' is an array of initial Lorentzian fit guesses
    (default: 0, generate guesses automatically);
    'plt' is the plotting environment to use, defaulting to pyplot
    (imported as plt).

    Returns the Q of the peak."""
    fit = poly_lor_fit(fulldata, snip, span, deg, guess)[0]
    plt.plot(fulldata[0], fulldata[1], fulldata[0], qlfout(fulldata[0], fit))
    return fit[3] / fit[2]


def sub_pl_fit_pre_back(fulldata, pfit, span, endbase=0, deg=2):
    """Subtracts a previously-fit background from the data,
    then fits a Lorentzian to the data.

    The fit function is:
    y(x) = a * (w / 2)**2 / ((x - f)**2 + (w / 2)**2)

    a is the peak amplitude, w is the peak width, and f is the peak frequency.

    'fulldata' is a data array;
    'pfit' is an array of polynomial background fit parameters;
    'span' is an array containing two points, [p1,p2], such that the data will
    be fit in the data range [p1:p2];
    'endbase' is the final y-value of the first dataset, used to remove a slow
    background drift from the data (default:0, no shift);
    'deg' is the degree of the polynomial (default: 2).

    Returns a list with two elements:
    an array with the Lorentzian fit parameters:
    the background offset, peak amplitude, peak width, peak frequency;
    and an array with the uncertainties for those fit parameters."""
#    data=(fulldata.transpose()[span[0]:span[1]]).transpose()
    data = (fulldata.transpose()[span[0]:span[1]]).transpose()
    if endbase == 0:
        endbase = fulldata[1][-1]
    yoff = fulldata[1][-1] - endbase
    ydat = data[1] - np.poly1d([pfit[5], pfit[4], pfit[0]])(data[0]) - yoff
    guess = [pfit[1], pfit[2], pfit[3]]
    fit, fitcov = sop.curve_fit(ut.lorentziannoback, data[0], ydat, guess)
    fitdat = ut.lorentziannoback(data[0], *fit)
    R2b = ut.r2bar(data[1], fitdat, len(fit))
    return [fit, np.sqrt(np.diag(fitcov)), R2b]
#    return np.array([np.abs(fit[2]/fit[1]),fit[2],fit[1],fit[0]])


def plot_sub_pl_fit_pre_back(fulldata, pfit, span, endbase=0, deg=2, plt=plt):
    """Subtracts a previously-fit background from the data, then fits a
    Lorentzian to the data and plots the fit and the data.

    The fit function is:
    y(x) = a * (w / 2)**2 / ((x - f)**2 + (w / 2)**2)

    a is the peak amplitude, w is the peak width, and f is the peak frequency.

    'fulldata' is a data array;
    'pfit' is an array of polynomial background fit parameters;
    'span' is an array containing two points, [p1,p2], such that the data will
    be fit in the data range [p1:p2];
    'endbase' is the final y-value of the first dataset, used to remove a slow
    background drift from the data (default:0, no shift);
    'deg' is the degree of the polynomial (default: 2);
    'plt' is the plotting environment to use, defaulting to pyplot
    (imported as plt).

    Returns an array with the Lorentzian fit parameters:
    the background offset, peak amplitude, peak width, peak frequency."""
    fit = sub_pl_fit_pre_back(fulldata, pfit, span, endbase, deg)[0]
    if endbase == 0:
        endbase = fulldata[1][-1]
    plt.plot(fulldata[0], fulldata[1] - (fulldata[1][-1] - endbase),
             fulldata[0], qlfout(fulldata[0], [pfit[0], fit[1], fit[2],
                                               fit[3], pfit[4], pfit[5]]))
    return fit

field_patt = 'H=([+-]?\\d+(?:\\.\\d+)?)*'
cpatt = '.*[_\\s](\\d+(?:\\.\\d+)?).?K.*' + field_patt
clockinpatt = '.* ([+-]?\d+(?:\.\d+)?)Oe.* (\d+(?:\.\d+)?)K.*.txt'
creppatt = '.* Step (\\d+(?:\\.\\d+)?) (\\d+(?:\\.\\d+)?)K.*' + field_patt


def cav_load(path, dir, patt=cpatt, end=9994):
    """Loads all datasets from 3D cavity field sweeps corresponding to a given
    string.

    'path' is a string corresponding to the desired datafile;
    'dir' is the data directory;
    'patt' is a regular expression pattern to extract temperature and field
    information from the filename;
    'end' is the point number at which the data should be cut off
    (default:9994, any dataset shorter will not be cut off at all).

    Returns an array with: a list of datasets;
    an array of experimental fields;
    an array of experimental temperatures."""
    data, names = ut.get_data_names(path, dir)

    def matchnum(name):
        [temp, field] = re.match(patt, name).groups()
        return [float(temp), int(field)]
    temps, fields = np.array([matchnum(x) for x in names]).transpose()
    sortarr = np.argsort(fields)
    [data, names, fields, temps] = [np.array(data)[sortarr], np.array(
        names)[sortarr], np.array(fields)[sortarr], np.array(temps)[sortarr]]
    data = [dat.transpose()[:end].transpose() for dat in data]
    return [data, fields, temps]


def cav_rep_load(path, dir, patt=creppatt, end=9994):
    """Loads all datasets from 3D cavity stepped field sweeps corresponding to
    a given string.

    'path' is a string corresponding to the desired datafile;
    'dir' is the data directory;
    'patt' is a regular expression pattern to extract sequence, temperature,
    and field information from the filename;
    'end' is the point number at which the data should be cut off
    (default:9994, any dataset shorter will not be cut off at all).

    Returns an array with: a list of datasets;
    an array of experimental fields;
    an array of experimental temperatures."""
    names = glob.glob(os.path.join(dir, path))

    def matchnum(name):
        [num, temp, field] = re.match(patt, name).groups()
        return [int(num), float(temp), int(field)]
    nums, temps, fields = np.array([matchnum(x) for x in names]).transpose()
    sortarr = np.argsort(nums)
    [names, fields, temps] = [np.array(names)[sortarr], np.array(fields)[
        sortarr], np.array(temps)[sortarr]]
    data = [np.loadtxt(n)[:end].transpose() for n in names]
    return [data, fields, temps]


def cav_load_first_fit(path, dir, snip, span, ffit=0, deg=2, patt=cpatt,
                       end=9994, **kwargs):
    """Loads all datasets from 3D cavity field sweeps corresponding to a given
    string, and fits them to a Lorentzian, subtracting the polynomial
    background using either a previous fit or the first dataset.

    The background fit function is:
    y(x) = b + c * x + d * x**2

    b is the background offset, c is the linear term,
    and d is the quadratic term.

    The combined fit function is:
    y(x) = b + a * (w / 2)**2 / ((x - f)**2 + (w / 2)**2) + c * x + d * x**2

    a is the peak amplitude, w is the peak width, and f is the peak frequency.

    'path' is a string corresponding to the desired datafile;
    'dir' is the data directory;
    'snip' is an array containing two frequencies, [f1,f2], such that the
    background fit will exclude the data in between those frequencies;
    'span' is an array containing two points, [p1,p2], such that the data will
    be fit in the data range [p1:p2];
    'ffit' is an array of polynomial background fit parameters
    (default:0, fit the background from the first dataset);
    'deg' is the degree of the polynomial (default: 2);
    'patt' is a regular expression pattern to extract temperature and field
    information from the filename;
    'end' is the point number at which the data should be cut off
    (default:9994, any dataset shorter will not be cut off at all).

    Returns an array whose elements are as follows: field, background offset,
    background offset uncertainty, amplitude, amplitude uncertainty,
    peak width, peak width uncertainty, frequency, frequency shift,
    frequency uncertainty, adjusted R-squared for the fit, Q, Q uncertainty,
    peak minimum, temperature."""
    data, fields, temps = cav_load(path, dir, patt, end)
    endbase = data[0][1][-1]
    if type(ffit) == int:
        ffit = poly_lor_fit(data[0], snip, span, deg)[0]
    fit, fiterr, R2bs = np.array([sub_pl_fit_pre_back(
        dat, ffit, span, endbase, deg) for dat in data]).transpose()
    coefs = fit.transpose()
    errs = fiterr.transpose()
    mins = np.array(list(map(ut.find_min, data)))
    out = ut.process_coefs(coefs, errs, R2bs, mins, fields, temps)
    return out
#    return np.array([fields,Qs,freqs,As,temps])


def cav_load_all_fit(path, dir, snip, span, deg=2, patt=cpatt, end=9994,
                     **kwargs):
    """Loads all datasets from 3D cavity field sweeps corresponding to a given
    string, and fits them to a combined quadratic background and Lorentzian.

    The background fit function is:
    y(x) = b + c * x + d * x**2

    b is the background offset, c is the linear term,
    and d is the quadratic term.

    The combined fit function is:
    y(x) = b + a * (w / 2)**2 / ((x - f)**2 + (w / 2)**2) + c * x + d * x**2

    a is the peak amplitude, w is the peak width, and f is the peak frequency.

    'path' is a string corresponding to the desired datafile;
    'dir' is the data directory;
    'snip' is an array containing two frequencies, [f1,f2], such that the
    background fit will exclude the data in between those frequencies;
    'span' is an array containing two points, [p1,p2], such that the data will
    be fit in the data range [p1:p2];
    'deg' is the degree of the polynomial (default: 2);
    'patt' is a regular expression pattern to extract temperature and field
    information from the filename;
    'end' is the point number at which the data should be cut off
    (default:9994, any dataset shorter will not be cut off at all).

    Returns an array whose elements are as follows: field, background offset,
    background offset uncertainty, amplitude, amplitude uncertainty,
    peak width, peak width uncertainty, frequency, frequency shift,
    frequency uncertainty, adjusted R-squared for the fit, Q, Q uncertainty,
    peak minimum, temperature."""
    data, fields, temps = cav_load(path, dir, patt, end)
    fit, fiterr, R2bs = list(
        zip(*[poly_lor_fit(dat, snip, span, deg) for dat in data]))
    coefs = np.array(fit).transpose()[:4]
    errs = np.array(fiterr).transpose()[:4]
    mins = np.array(list(map(ut.find_min, data)))
    out = ut.process_coefs(coefs, errs, R2bs, mins, fields, temps)
    return out
# return np.array([fields,fit[3]/fit[2],fit[3],fit[1],temps])


def derep(pars, rep):
    """Average the fit parameters for repeated fields.

    'pars' is an array of arrays of fit parameters;
    'rep' is the number of repetitions at each field.

    Returns an array of arrays of averaged fit parameters."""
    outpars = [np.zeros(len(pars[i]) / rep) for i in np.arange(len(pars))]
    for i in np.arange(len(pars)):
        outpars[i] = [np.mean(pars[i][j * rep:j * rep + rep])
                      for j in np.arange(len(outpars[i]))]
    return np.array(outpars)


def cav_rep_load_all_fit(path, dir, rep, snip, span, deg=2, patt=creppatt,
                         end=9994, **kwargs):
    """Loads all datasets from 3D cavity repeated stepped field sweeps
    corresponding to a given string, and fits them to a combined quadratic
    background and Lorentzian.

    The background fit function is:
    y(x) = b + c * x + d * x**2

    b is the background offset, c is the linear term,
    and d is the quadratic term.

    The combined fit function is:
    y(x) = b + a * (w / 2)**2 / ((x - f)**2 + (w / 2)**2) + c * x + d * x**2

    a is the peak amplitude, w is the peak width, and f is the peak frequency.

    'path' is a string corresponding to the desired datafile;
    'dir' is the data directory;
    'rep' is the number of repetitions at each field;
    snip' is an array containing two frequencies, [f1,f2], such that the
    background fit will exclude the data in between those frequencies;
    'span' is an array containing two points, [p1,p2], such that the data will
    be fit in the data range [p1:p2];
    'deg' is the degree of the polynomial (default: 2);
    'patt' is a regular expression pattern to extract temperature and field
    information from the filename;
    'end' is the point number at which the data should be cut off
    (default:9994, any dataset shorter will not be cut off at all).

    Returns an array with: field, Q (f/w), peak frequency, peak amplitude,
    temperature."""
    data, field, temps = cav_rep_load(path, dir, patt, end)
    fit = np.array([poly_lor_fit(dat, snip, span, deg)
                    for dat in data]).transpose()
    As, Qs, freqs = [fit[1], fit[3] / fit[2], fit[3]]
    field = field[::rep]
    temps = temps[::rep]
    Q, freq, A = derep([Qs, freqs, As], rep)
    return [field, Q, freq, A, temps]


def cav_rep_load_fit(path, dir, rep, snip, span, ffit=0, deg=2, end=9994,
                     patt=cpatt, **kwargs):
    """Loads all datasets from 3D cavity repeated stepped field sweeps
    corresponding to a given string, and fits them to a Lorentzian,
    subtracting the polynomial background using either a previous fit or the
    first dataset.

    The background fit function is:
    y(x) = b + c * x + d * x**2

    b is the background offset, c is the linear term,
    and d is the quadratic term.

    The combined fit function is:
    y(x) = b + a * (w / 2)**2 / ((x - f)**2 + (w / 2)**2) + c * x + d * x**2

    a is the peak amplitude, w is the peak width, and f is the peak frequency.

    'path' is a string corresponding to the desired datafile;
    'dir' is the data directory;
    'rep' is the number of repetitions at each field;
    'snip' is an array containing two frequencies, [f1,f2], such that the
    background fit will exclude the data in between those frequencies;
    'span' is an array containing two points, [p1,p2], such that the data will
    be fit in the data range [p1:p2];
    'ffit' is an array of polynomial background fit parameters
    (default:0, fit the background from the first dataset);
    'deg' is the degree of the polynomial (default: 2);
    'patt' is a regular expression pattern to extract temperature and field
    information from the filename;
    'end' is the point number at which the data should be cut off
    (default:9994, any dataset shorter will not be cut off at all).

    Returns an array with: field, Q (f/w), peak frequency, peak amplitude,
    temperature."""
    field, Qs, freqs, As, temps = cav_load_first_fit(
        path, dir, snip, span, deg, patt, end)
    field = field[::rep]
    temps = temps[::rep]
    Q, freq, A = derep([Qs, freqs, As], rep)
    return [field, Q, freq, A, temps]


def cav_rep_load_fit_no_back(path, dir, rep, snip, patt=cpatt, end=9994,
                             **kwargs):
    """Loads all datasets from 3D cavity repeated stepped field sweeps
    corresponding to a given string, and fits them to a Lorentzian.

    The fit function is:
    y(x) = b + a * (w / 2)**2 / ((x - f)**2 + (w / 2)**2) + c * x + d * x**2

    b is the background offset, a is the peak amplitude, w is the peak width,
    and f is the peak frequency.

    'path' is a string corresponding to the desired datafile;
    'dir' is the data directory;
    'rep' is the number of repetitions at each field;
    'snip' is an array containing two frequencies, [f1,f2], such that the fit
    applies to data in between those frequencies;
    'patt' is a regular expression pattern to extract temperature and field
    information from the filename;
    'end' is the point number at which the data should be cut off
    (default:9994, any dataset shorter will not be cut off at all).

    Returns an array with: field, Q (f/w), peak frequency, peak amplitude,
    temperature."""
    data, field, temps = cav_load(path, dir, patt, end)
    field = field[::rep]
    temps = temps[::rep]
    pars = np.array(
        [ut.lor_fit_range(dat, snip[0], snip[1], -1)[0] for dat in data])
    _, As, Ws, freqs = pars.transpose()
    Q, freq, A = derep([freqs / Ws, freqs, As], rep)
    return [field, Q, freq, A, temps]


def cav_load_fit_out(path, postpath, dir, func, outdir, *args, **kwargs):
    """Loads all datasets from 3D cavity field sweeps corresponding to a given
    string, fits them, and saves the fit parameters to a CSV file. See the
    documentation for the different fitting functions for details on the fit.

    'path' is a string corresponding to the desired datafile;
    'postpath' is a string corresponding to a later portion of the filename,
    usually the temperature;
    dir' is the data directory;
    'func' is the fitting function to use;
    'outdir' is the fit output directory. Further arguments, including keyword
    arguments, needed by the various fitting functions can be passed at this
    point, and will be passed to the function specified by 'func'.

    Returns an array with: field, Q (f/w), peak frequency, peak amplitude,
    temperature."""
    out = np.array(func(path+'*[ _]'+postpath+'*', dir, *args, **kwargs))
    [dir1, dat] = os.path.split(dir)
    [dir2, res] = os.path.split(dir1)
#    [tmp,proj]=os.path.split(dir2);
    # Get the initial part of the path and the power from the name
    tout = out.transpose()
    outname = res + ' ' + dat + ' ' + path + \
        ' %.1fK Fit Analysis.csv' % float(tout[0][-1])
    outpath = os.path.join(outdir, outname)
    data_save_out(outpath, colnames, tout)
    return out


def data_save_out(path, names, out):
    """Saves data to a CSV file.

    'path' is the full path of the file to save;
    'names' is an array of column names corresponding to the data;
    'out' is an array of output data arrays."""
    with open(path, 'w') as csvfile:
        fout = csv.writer(csvfile, lineterminator='\n')
        fout.writerow(names)
        for row in out:
            fout.writerow(row)

name_dict = {'Magnetic Field': 'H', 'Base': 'B', 'B-error': 'dB',
             'Peak Amplitude': 'A', 'A-error': 'dA', 'FWHH': 'W',
             'W-error': 'dW', 'Central Frequency': 'f', 'Frequency': 'f',
             'Frequency Shift': 'fs', 'F-error': 'df', 'R-Squared': 'R2',
             'Q-Value': 'Q', 'Q-value': 'Q', 'Q-error': 'dQ',
             'Minimum': 'min', 'Temperature': 'T', 'Amplitude': 'A',
             'Shift': 'fs', 'Width': 'W', 'Frequency': 'f', 'Tau': 't',
             'T-error': 'dt'}


def lfp(fitpath, offset=0):
    with open(fitpath) as f:
        colnames = f.readline()[:-1].split(sep=',')
    ColNames = [name_dict[n] for n in colnames]
    data = np.loadtxt(fitpath, delimiter=',', skiprows=1).transpose()
    data[0] += offset  # This is equivalent to data[0]=data[0]+offset

    df = pd.DataFrame(data=data.transpose(), index=data[0].transpose(),
                      columns=ColNames)
    df = df[~df.index.duplicated(keep='first')]
    return df


def load_fit(path, dir, offset=0):
    fitpath = os.path.join(dir, path)
    return lfp(fitpath, offset)


def load_pulse_fits(paths, dir, iters, offset=0):
    dfs = [load_fit(path, dir, offset) for path in paths]
    outpan = pd.Panel({label: df for label, df in zip(iters, dfs)})
    return outpan
