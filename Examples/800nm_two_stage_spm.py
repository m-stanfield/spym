# -*- coding: utf-8 -*-
"""
Created on Fri Jun 21 15:20:45 2019

@author: mstan
"""

# %%
import spym
import numpy as np
import matplotlib.pyplot as plt
import glob
# %%
frgDir = r'D:\SPM Experiment\2017-11-21_8mmFSSPM\FROG_TRACES\FROG-2017-11-21-03-24-26800_nm_initial_5285_1_hires\cleaned'
sim_pulse = True

# %%
freq_fwhm = 15.0
center_freq = 374.7
EPP = 0.00075

initialGDD = 0.0
initialTOD = -15000.0
initialFOD = 0.0

# %%
Length = 1.0
beamDia1 = 1.2
beamDia2 = 6.0

# %%

pass1CM = -90.0
pass2CM = 0.0

# %%
beta2 = 36.2
beta3 = 27.5
beta4 = -11.4

# %%
number_bins = 2**14
freq_range = 4000.0
Steps = 20
NLEffects = True
flip_phase = 0

# %%
cutoff_percent = 0.005
normalize= 1
x_axis_wave = True
# %%

if sim_pulse == True:
    pulse_0 = spym.generateIntensityGaussianSpectrum(number_bins = number_bins,
                 freq_range = freq_range, freq_fwhm = freq_fwhm,center_freq = center_freq,
                 phase_coef=[initialGDD,initialTOD,initialFOD],EPP=EPP)
else:
    pulse_0 = spym.load_frog(glob.glob(frgDir + '\\*.Speck.*')[0],EPP=EPP,cen_wave=299700.0/center_freq,
                                       freq_range=freq_range,number_of_bins=number_bins,flip_phase=flip_phase)
print('Pass 1 Target Intensity: %.2E TW/cm^2'%(pulse_0.peakI(beam_dia=beamDia1)*10**(-12)))

fsPass1 = spym.createFiberNLO(central_wavelength=299700.0/center_freq,Length=Length,
                              beam_dia=beamDia1,beta2=beta2,beta3=beta3,beta4=beta4)
fsPass2 = spym.createFiberNLO(central_wavelength=299700.0/center_freq,Length=Length,
                              beam_dia=beamDia2,beta2=beta2,beta3=beta3,beta4=beta4)

# %%
reflection = 0.97
pulse_0.set_epp(pulse_0.calc_epp()*reflection)
y1,AW1,AT1,pulse_1 = spym.runNLOSim(pulse_0,fsPass1,Steps=Steps,Raman=NLEffects,Steep=NLEffects)
pulse_1.set_epp(pulse_1.calc_epp()*reflection**2)

pulse_1.add_phase([pass1CM])
y1,AW1,AT1,pulse_2 = spym.runNLOSim(pulse_1,fsPass2,Steps=Steps,Raman=NLEffects,Steep=NLEffects)
# %%
pulse_3 = pulse_2.copy()
_,finalGDD = pulse_3.fit_phase(coef='taylor')
pulse_3.add_phase([-finalGDD[0]])

# %%
plt.close('all')
fig, ax = plt.subplots(ncols=2)

ax = pulse_0.plot(ax=ax,cutoff_percent=cutoff_percent,phase=1,linestyles=['r','r--'],
             normalize=normalize,labels=['Initial','Initial','Initial Phase'],
             x_axis_wave=x_axis_wave)

ax = pulse_1.plot(ax=ax,cutoff_percent=cutoff_percent,phase=1,linestyles=['b','b--'],
             normalize=normalize,labels=['Pass 1','Pass 1','Pass 1 Phase'],
             x_axis_wave=x_axis_wave)

ax = pulse_3.plot(ax=ax,cutoff_percent=cutoff_percent,phase=1,linestyles=['k','k--'],
             normalize=normalize,labels=['Final','Final','Final Phase'],
             x_axis_wave=x_axis_wave)

# %%

print('\n')

print('Initial FWHM: %f fs'%pulse_0.fwhm_t())
print('Initial FWHM: %f THz'%pulse_0.fwhm_f())
print('Initial FTL FWHM: %f fs'%pulse_0.ftl())
print('\n')
print('Pass 1 FWHM: %f fs'%pulse_1.fwhm_t())
print('Pass 1 FWHM: %f THz'%pulse_1.fwhm_f())
print('Pass 1 FTL FWHM: %f fs'%pulse_1.ftl())
print('Pass 1 B Integral: %f'%pulse_1.calc_B())
print('Pass 1 Target Intensity: %.2E TW/cm^2'%(pulse_0.peakI(beam_dia=beamDia1)*10**(-12)))

print('\n')
print('Pass 2 FWHM: %f fs'%pulse_2.fwhm_t())
print('Pass 2 FWHM: %f THz'%pulse_2.fwhm_f())
print('Pass 2 FTL FWHM: %f fs'%pulse_2.ftl())
print('Pass 2 B Integral: %f'%pulse_2.calc_B())
print('Pass 2 Target Intensity: %.2E TW/cm^2'%(pulse_1.peakI(beam_dia=beamDia2)*10**(-12)))

print('\n')
print('Compressed FWHM: %f fs'%pulse_3.fwhm_t())
print('Compressed FTL FWHM: %f fs'%pulse_3.ftl())

