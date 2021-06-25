# -*- coding: utf-8 -*-
"""
Created on Mon Jul 15 13:00:10 2019

@author: mstan
"""
import matplotlib

font = {'family' : 'normal',
        'weight' : 'bold',
        'size'   : 22}

matplotlib.rc('font', **font)
import spym
import matplotlib.pyplot as plt
import numpy as np

# %%

saveDir = ''
# %%

# 1 mJ beamline setup. 1 mm material
#EPP = 0.00085
#beam_dia1 = 1.4*np.sqrt(2)*0.95
#beam_dia2 = 2.1*np.sqrt(2)*0.8
#Length1 = 1.2
#Length2 = 1.2
# 1 mJ beamline setup. 0.5 mm material
EPP = 0.0009*0.98
beam_dia1 = 1.4
beam_dia2 = 2.1
Length1 = 1.2*0.5
Length2 = 1.2*0.5
## 7 mJ beamline setup. 0.5 mm material
#
#EPP = 0.0068
#beam_dia1 = 3.96
#beam_dia2 = 5.94
#Length1 = 1.2*0.5
#Length2 = 1.2*0.5
# %%

initialBeamDia = 11.7
focalLength = 200.0

freq_fwhm = 15.0

GDD = 0.0
TOD = -10000.0
FOD = 0.0

n2 = 2.7*10**(-20)

CM1Phase = [-90.0]
CM2Phase = [0.0]

# %%
number_bins = 2**14
freq_range = 4000.0
phase_coef = [GDD,TOD,FOD]

NLEffect = True

# %%
figSize = [10,8]
plt_wavelength = True
log_scale = False
# %%
pulse_0 = spym.generateIntensityGaussianSpectrum(number_bins=number_bins,freq_fwhm=freq_fwhm,freq_range=freq_range,center_freq=299700.0/800.0,phase_coef=phase_coef,EPP=EPP,output_pulse=1)
stage1 = spym.createFSFiber(Length=Length1,beam_dia=beam_dia1,n2=n2)
stage2 = spym.createFSFiber(Length=Length2,beam_dia=beam_dia2,n2=n2)

y, AW, AT, pulse_1 = spym.runNLOSim(pulse_0,stage1,Steps=20,Raman=NLEffect,Steep=NLEffect)
pulse_1chirped = pulse_1.copy()
pulse_1.add_phase(CM1Phase)
y,AW,AT,pulse_2 = spym.runNLOSim(pulse_1,stage2,Steps=20,Raman=NLEffect,Steep=NLEffect)

if not(np.all(np.asarray(CM2Phase)==0.0)):
    pulse_2.add_phase(CM2Phase)
    
pulse_2zeroGDD = pulse_2.copy()
_,phase_fit = pulse_2.fit_phase(min_intensity=0.01,fit_order=5)
pulse_2zeroGDD.add_phase([-phase_fit[0]])


# %%
plt.close('all')

fig, ax = plt.subplots(ncols=2,figsize=figSize)
pulse_2.plot(ax = ax,cutoff_percent=0.001,phase=0,linestyles=['k','k--'],labels=['Stage 2','Stage 2','Stage 2 Phase'],x_axis_wave=plt_wavelength)
pulse_1.plot(ax = ax,cutoff_percent=0.001,phase=0,linestyles=['b','b--'],labels=['Stage 1','Stage 1','Stage 1 Phase'],x_axis_wave=plt_wavelength)
pulse_0.plot(ax = ax,cutoff_percent=0.001,phase=0,linestyles=['r','r--'],labels=['Initial','Initial','Initial Phase'],x_axis_wave=plt_wavelength)
ax[1].set_xlim([500,1100])
if not(saveDir == ''):
    plt.savefig(saveDir + '\\multipleStages_firststage.png',dpi=150,transpart=True)
    
fig, ax = plt.subplots(ncols=2,figsize=figSize)
pulse_2.plot(ax = ax,cutoff_percent=0.001,phase=0,linestyles=['k','k--'],labels=['Stage 2','Stage 2','Stage 2 Phase'],x_axis_wave=plt_wavelength)
pulse_0.plot(ax = ax,cutoff_percent=0.001,phase=0,linestyles=['r','r--'],labels=['Initial','Initial','Initial Phase'],x_axis_wave=plt_wavelength)
ax[1].set_xlim([400,1200])
if not(saveDir == ''):
    plt.savefig(saveDir + '\\startend.png',dpi=150,transpart=True)

fig, ax = plt.subplots(ncols=2,figsize=figSize)
pulse_2.plot(ax = ax,cutoff_percent=0.001,phase=1,linestyles=['r','r--'],labels=['Final Uncompressed','Final Uncompressed','Final Uncompressed Phase'],x_axis_wave=plt_wavelength)
pulse_2zeroGDD.plot(ax = ax,cutoff_percent=0.001,phase=1,linestyles=['b','b--'],labels=['Final Compressed','Final Compressed','Final Compressed Phase'],x_axis_wave=plt_wavelength)
if not(saveDir == ''):
    plt.savefig(saveDir + '\\finalStageCompressed.png',dpi=150,transpart=True)


if log_scale == True:
    ax[1].set_yscale('log')
    minScale = np.min(np.log(np.abs(pulse_2.AW[np.abs(pulse_2.AW)**2>0.0])**2))
    ax[1].set_ylim([10**-10,1])
plt.show()

print('~~~~~~~~~~')
print('Initial FWHM: %0.2f fs'%(pulse_0.fwhm_t()))
print('Initial FTL: %0.2f fs'%(pulse_0.ftl()))
print('Initial FWHM: %0.2f THz'%(pulse_0.fwhm_f()))
print('\n')
print('Initial Peak Intensity: %E'%(pulse_0.peakI(beam_dia=beam_dia1)))
print('Distance from CM: %0.2f mm'%(beam_dia1/initialBeamDia*focalLength))
print('Stage 1 FWHM After CM: %0.2f fs'%(pulse_1.fwhm_t()))
print('Stage 1 FWHM Before CM: %0.2f fs'%(pulse_1chirped.fwhm_t()))
print('Stage 1 FTL: %0.2f fs'%(pulse_1.ftl()))
print('Stage 1 FWHM: %0.2f THz'%(pulse_1.fwhm_f()))
print('Stage 1 B-Integral: %0.2f'%(pulse_1.calc_B()))

print('\n')
print('Stage 2 Peak Intensity: %E'%(pulse_1.peakI(beam_dia=beam_dia2)))
print('Distance from CM: %0.2f mm'%(beam_dia2/initialBeamDia*focalLength))

print('Stage 2 FWHM: %0.2f fs'%(pulse_2.fwhm_t()))
print('Stage 2 FWHM Zero GDD: %0.2f fs'%(pulse_2zeroGDD.fwhm_t()))
print('Stage 2 FTL: %0.2f fs'%(pulse_2.ftl()))
print('Stange 2 FWHM: %0.2f THz'%(pulse_2.fwhm_f()))
print('Stage 2 B-Integral: %0.2f'%(pulse_2.calc_B() - pulse_1.calc_B()))
print('Total B-Integral: %0.2f'%(pulse_2.calc_B()))




