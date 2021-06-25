# -*- coding: utf-8 -*-
"""
Created on Tue Jun 18 18:26:45 2019

@author: mstan
"""
# %%

import spym
import matplotlib.pyplot as plt
GDD = 0.0
TOD = -10000.0
FOD = 0.0
NLEffects = True
Length = 1.2
beam_dia = 11.7*4.5/20.0
ftl_duration = 30.0

# %%
pulse_0 = spym.generateIntensityGaussianSpectrum(freq_fwhm = 441.0/ftl_duration,center_freq=299700.0/800.0,phase_coef=[GDD,TOD,FOD],EPP=0.001)
fiber = spym.createFSFiber(Length=Length, beam_dia=beam_dia)
y,AW,AT,pulse_1 = spym.runNLOSim(pulse_0,fiber,Raman=NLEffects,Steep=NLEffects)

# %%
plt.close('all')

fig,ax = plt.subplots(ncols=2)
ax = pulse_0.plot(ax=ax,cutoff_percent=0.0001,linestyles=['b','b--'],labels=['Initial','Initial','Initial Phase'],phase=1,normalize=1,x_axis_wave=1)
ax = pulse_1.plot(ax=ax,cutoff_percent=0.0001,linestyles=['r','r--'],labels=['Final','Final','Final Phase'],phase=1,normalize=1,x_axis_wave=1)

fig_ftl,ax_ftl = plt.subplots(ncols=2)
ax_ftl = pulse_0.ftl(output_pulse=1).plot(ax=ax_ftl,cutoff_percent=0.0001,labels=['Initial FTL','Initial FTL'],linestyles=['b','b--'],phase=0,normalize=1,x_axis_wave=1)
ax_ftl = pulse_1.ftl(output_pulse=1).plot(ax=ax_ftl,cutoff_percent=0.0001,labels=['Final FTL','Final FTL'],linestyles=['r','r--'],phase=0,normalize=1,x_axis_wave=1)

plt.show()

# %%
print('\n\n')
print('Initial Temporal FWHM (fs): %f'%pulse_0.fwhm_t())
print('Initial FTL (fs): %f'%pulse_0.ftl())
print('Initial Spectral FWHM (THz): %f'%pulse_0.fwhm_f())
print('\n')

print('Final Temporal FWHM (fs): %f'%pulse_1.fwhm_t())
print('Final FTL (fs): %f'%pulse_1.ftl())
print('Final Spectral FWHM (THz): %f'%pulse_1.fwhm_f())
