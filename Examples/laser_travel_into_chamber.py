# -*- coding: utf-8 -*-
"""
Created on Wed Jun 19 08:27:21 2019

@author: mstan
"""

import spym
import matplotlib.pyplot as plt

central_wavelength = 800.0  #Central Wavelength of pulse
beam_dia = 11.7*2.0 #1/e^2 beam diameter in mm
EPP = 0.007 #Energy of pulse in Joules
ftl_duration = 30.0 #FTL FWHM pulse duration in fs

initialGDD = 250.0 #initial GDD in fs^2
initialTOD = -10000.0 #initial third order dispersion in fs^3
initialFOD = 0.0 #initial fourth order dispersion in fs^4

fs_length = 3.0 #Length of fused silica media in mm
air_length = 4000.0  #length of air 

CM_GDD = -500.0 #GDD applied from chirped mirror inside the chamber


# %%
#Default values should be roughly correct values for 800 nm
#If central wavelength changes will need to edit

fs_beta2 = 36.2 #GVD in fs^2/mm
fs_beta3 = 27.5 #Third order phase in fs^3/mm
fs_beta4 = -11.43 #fourth order phase in fs^4/mm
fs_n2 = 2.7*10**(-20) #Kerr index in m^2/W

air_beta2 = 0.021233
air_beta3 = 0.0
air_beta4 = 0.0
air_n2 = 2*10**(-23)  #Very rough estimate. Very likely only order of magnitude

# %%
#simulation and plotting parameters
Steps = 20  #number of steps per each media
NLEffects = True #Include Self-Steepening and Intrapulse Raman effect
normalize = True #If plots are normalized
plot_phase = True
# %%
#Creating gaussian pulse with given specifications
pulse_0 = spym.generateIntensityGaussianSpectrum(freq_fwhm = 441.0/ftl_duration,center_freq=299700.0/central_wavelength,phase_coef=[initialGDD,initialTOD,initialFOD],EPP=EPP)

# %%
#Creating instances for the media that the pulse will propegate through
air = spym.createFiberNLO(central_wavelength=central_wavelength,Length=air_length,beam_dia = beam_dia, n2=air_n2,beta2=air_beta2,beta3=air_beta3,beta4=air_beta4)
fusedSilica = spym.createFiberNLO(central_wavelength=central_wavelength,Length=fs_length,beam_dia = beam_dia, n2=fs_n2,beta2=fs_beta2,beta3=fs_beta3,beta4=fs_beta4)

# %%
#Simulating the pulse through media
y,AW,AT,pulse_air = spym.runNLOSim(pulse_0,air,Raman=False,Steep=NLEffects,Steps=Steps) 
y,AW,AT,pulse_FS = spym.runNLOSim(pulse_air,fusedSilica,Raman=NLEffects,Steep=NLEffects,Steps=Steps)
# %%
pulse_cm = pulse_FS.copy()
pulse_cm.add_phase([CM_GDD])
# %%
#Plotting pulses
plt.close('all')

fig,ax = plt.subplots(ncols=2)
ax = pulse_0.plot(ax=ax,cutoff_percent=0.0001,linestyles=['b','b--'],labels=['Initial','Initial','Initial Phase'],phase=plot_phase,normalize=normalize,x_axis_wave=1)
ax = pulse_air.plot(ax=ax,cutoff_percent=0.0001,linestyles=['r','r--'],labels=['After Air','After Air','After Air Phase'],phase=plot_phase,normalize=normalize,x_axis_wave=1)
ax = pulse_FS.plot(ax=ax,cutoff_percent=0.0001,linestyles=['k','k--'],labels=['After FS','After FS','After FS Phase'],phase=plot_phase,normalize=normalize,x_axis_wave=1)


fig,ax = plt.subplots(ncols=2)
ax = pulse_FS.plot(ax=ax,cutoff_percent=0.0001,linestyles=['k','k--'],labels=['After FS','After FS','After FS Phase'],phase=plot_phase,normalize=normalize,x_axis_wave=1)
ax = pulse_cm.plot(ax=ax,cutoff_percent=0.0001,linestyles=['b','b--'],labels=['Final','Final','Final Phase'],phase=plot_phase,normalize=normalize,x_axis_wave=1)

fig_ftl,ax_ftl = plt.subplots(ncols=2)
ax_ftl = pulse_0.ftl(output_pulse=1).plot(ax=ax_ftl,cutoff_percent=0.0001,labels=['Initial FTL','Initial FTL'],linestyles=['b','b--'],phase=0,normalize=normalize,x_axis_wave=1)
ax_ftl = pulse_air.ftl(output_pulse=1).plot(ax=ax_ftl,cutoff_percent=0.0001,labels=['After Air FTL','After Air FTL'],linestyles=['r','r--'],phase=0,normalize=normalize,x_axis_wave=1)
ax_ftl = pulse_FS.ftl(output_pulse=1).plot(ax=ax_ftl,cutoff_percent=0.0001,labels=['After FS FTL','After FS FTL'],linestyles=['k','k--'],phase=0,normalize=normalize,x_axis_wave=1)

plt.show()

# %%
print('\n\n')
print('Initial Temporal FWHM: %f fs'%pulse_0.fwhm_t())
print('Initial FTL: %f fs'%pulse_0.ftl())
print('Initial Spectral FWHM: %f THz'%pulse_0.fwhm_f())
print('Initial Peak I0: %E W/cm^2'%(pulse_0.peakI(beam_dia=beam_dia)))

print('\n')

print('After Air Temporal FWHM: %f fs'%pulse_air.fwhm_t())
print('After Air FTL: %f fs'%pulse_air.ftl())
print('After Air Spectral FWHM: %f THz'%pulse_air.fwhm_f())
print('After Air Peak I0: %E W/cm^2'%(pulse_air.peakI(beam_dia=beam_dia)))
print('After Air B Integral: %f'%pulse_air.calc_B())

print('\n')

print('After FS Temporal FWHM: %f fs'%pulse_FS.fwhm_t())
print('After FS FTL: %f fs'%pulse_FS.ftl())
print('After FS Spectral FWHM: %f THz'%pulse_FS.fwhm_f())
print('After FS Peak I0: %E W/cm^2'%(pulse_FS.peakI(beam_dia=beam_dia)))
print('After FS B Integral: %f'%pulse_FS.calc_B())

print('\n')

print('After CM Temporal FWHM: %f fs'%pulse_cm.fwhm_t())
print('After CM FTL: %f fs'%pulse_cm.ftl())
print('After CM Spectral FWHM: %f THz'%pulse_cm.fwhm_f())
print('After CM Peak I0: %E W/cm^2'%(pulse_cm.peakI(beam_dia=beam_dia)))

