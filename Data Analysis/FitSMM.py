# Importing useful modules.
# os gives functions to interact with the underlying Operating System
# glob gives functions to find files matching certain string patterns
import os
import sys
# This section defines where to find the Research folder on my computers,
# you will want to change it to reflect your directory structure
rpath = os.path.join('C:\\', 'Research')

sys.path.insert(0, os.path.join(rpath, 'Python'))

# You should recognize these from the tutorial
import utility as ut
import smmesr as sm
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

# Experimentally-found field offset for experiments sweeping at 10.8 Oe/s
HOffset10 = -93.5

project_folder = os.path.join(rpath, 'LGR Data')
fit_folder = os.path.join(rpath, 'Fits')


date = 'Name of the date directory or zip archive for the data'
res = 'Name of the resonator director for the data'
data_folder = os.path.join(project_folder, res, date)
data_prefix = 'Beginning of the filename for the data you want'
data_power = np.arange(-20, 1, 5)

"""Instead of calling sm.load_and_fit_vna directly, we can assign it to a more
generic function Func, and then call it using Func().
This is not strictly necessary, but as you will sometimes want to call
sm.load_and_fit_vna and sometimes sm.load_fit_vna_out it is convenient.
I recommend splitting any function or parameter that you will frequently
change out into its own line, as done here, it helps to keep track."""
func = sm.load_fit_vna_out

sdat = [sm.proc_data(func(data_prefix+'* %.2fdBm*.txt' % p, dir=data_folder,
                          outdir=fit_folder), HOffset10) for p in data_power]

# This will produce two plots with aligned x-axes, plotting the field
# dependence of the Q and the f over each power
fig, ax = plt.subplots(2, sharex=True, figsize=(8, 6))
ax[0].set_title(date+' '+data_prefix)
ut.div_end(sdat.minor_xs('Q')).plot(ax=ax[0], marker='.')
ut.div_end(sdat.minor_xs('Q')).plot(ax=ax[1], marker='.')
