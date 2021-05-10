import numpy as np
import matplotlib.pyplot as plt
import scipy
import os
import pynlo
import pandas as pd
import glob
import sys
import time
import spym_v0_1 as sp




'''
Simulate Lengths
    Controls how many simulated pulses to run and how many pulses per dataset
    should  be saved.
'''
number_of_spectrum = 2
number_GDD_per_spec = 1
number_TOD_per_GDD = 1
number_FOD_per_TOD = 1
datasets_per_file = 1000
'''
Simulated Variables
    All variables here are the ones that will be randomized
    <name>_ave will give the average value of that variable.
    <name>_delta will give the largest distance from average
    ie max value = ave + delta, min value = ave - delta
'''
def genSpec_2gauss(freq,cenFreq2=394.625,fwhm2=15.0,peak2=0.2,cenFreq1=374.625,fwhm1=15.0,peak1=1.0):
    #freq is frequency array
    #cenFreq is central peak of that gaussian
    #fwhm is the freq fwhm of that gaussian
    #peak is the maximum value of that gaussian
    spec1 = peak1*np.exp(-4*np.log(2)*((freq-cenFreq1)/fwhm1)**2)
    spec2 = peak2*np.exp(-4*np.log(2)*((freq-cenFreq2)/fwhm2)**2)
    return spec1 + spec2
    
def genSpec_fourier(freq,gen,gen_range=100.0,env_fwhm = 20,cutoff=0.03,central_freq = 374.625):
    freq_range = max(freq)-min(freq)
    number_bins = freq.shape[0]
    gen_bins = int(number_bins*(gen_range/freq_range))*2
    gen_freq = np.linspace(-gen_range/2,gen_range/2,gen_bins)
    gen_time = np.fft.fftfreq(gen_freq.shape[0],gen_freq[1]-gen_freq[0])

    gen_t = gen.rand(gen_time.shape[0])*np.exp(1j*2*np.pi*gen.rand(gen_time.shape[0]))
    gen_t[np.abs(gen_time)>cutoff] = 0
    gen_t[np.abs(gen_time)<=cutoff] = gen_t[np.abs(gen_time)<=cutoff] - np.mean(gen_t[np.abs(gen_time)<=cutoff])
    
    if env_fwhm == -1:
        env_fwhm = gen_range*0.2
    f_env = np.exp(-4*np.log(2)*(gen_freq/env_fwhm)**2)
    gen_Iw = np.abs((np.fft.fft(gen_t)))**2*f_env
    
    Iw = np.interp(freq,gen_freq + central_freq,gen_Iw,left=0,right=0)
    
    return Iw

stdMin = 0.035  #Minimum value for std of (I0_w - I1_w)/max(I0_w)
                #Ensures "enough" broadening occured
                
#FWHM of the initial pulses spectrum. In THz
freq_fwhm_ave = 15.0  
freq_fwhm_delta = 0.0

#The frequency shift of the second gaussian in genSpec_2gauss
dFreq_ave = 0.0  
dFreq_delta = 30.0

#Initial Energy of the initial pulses in Joules
EPP_ave = 0.001 
EPP_delta = 0.000

#Initial quadratic phase of pulse. In fs^2
GDD_ave = 0.0
GDD_delta = 0.0
GDD_delta = 1.0*10**3

#Initial cubic phase of pulse. In fs^3
TOD_ave = 0.0
TOD_delta = 0.0
TOD_delta = 1.0*10**4

#Initial quartic phase of pulse. In fs^4
FOD_ave = 0.0
FOD_delta = 0.0
FOD_delta = 1.0*10**5
'''
Setup parameters
    All these variables are variables for a given setup. Currently nothing in 
    this  block should be randomized and should only have to change if you are
    changing the effective experimental setup
'''

length = 1.2 #length of material in mm
beam_dia = 11.7*0.2 #Beam 1/e^2 diameter. in mm
gaussian_beam=True  #Currently simulating gaussian spacially
central_wavelength=800.0  #wavelength in nm
#Simualted parameters for 800 nm in fused silica
n2 = 2.7*10**(-20) #Kerr Index in m^2/W
#Material dispersion. beta<n> is the nth coef. in units of fs^n/mm
beta2 = 36.15
beta3 = 27.47
beta4 = 0.0

'''
Simulation Parameters
    These parameters control the numerical side of things. 
'''
number_of_bins = 2**11
frequency_range = 1000.0
number_steps = 100.0
Raman = True
Self_Steep = Raman

central_frequency = 299700/central_wavelength
minFrequency = central_frequency - 80
maxFrequency = central_frequency + 80

'''
Creating random seed
    Try/except statement is to allow  simulation to run both on the cluster and also
    on a desktop. If no system variables are passed(on desktop) error is trown and
    dummy values  are made.
    
    System variables are needed to ensure different cores have different random seeds
'''
try:
    slurm_task_id = sys.argv[1]
    saveDir = sys.argv[2]
except IndexError:
    slurm_task_id = 0
    saveDir = os.getcwd()

#Making save directories if  they do not already exist
if os.path.isdir(saveDir)==False:
    os.makedirs(saveDir+'/data/full_data')
elif os.path.isdir(saveDir + '/data')==False:
    os.makedirs(saveDir+'/data/full_data')
elif os.path.isdir(saveDir + '/data/full_data')==False:
    os.makedirs(saveDir+'/data/full_data')

startTime = time.time()
randSeed = int(slurm_task_id) + int(startTime)
gen = np.random.RandomState(seed=randSeed)
number_of_sims = number_of_spectrum*number_GDD_per_spec*number_TOD_per_GDD*number_FOD_per_TOD



'''
Running simulation
'''
fullNameList = ('freq_fwhm','length','EPP','GDD','TOD','FOD','beam_dia','gaussian_beam','central_wavelength','n2','beta2','beta3','beta4','B_int','freq','initial_AW','final_AW')

os.chdir(saveDir)

scan_counter = 0
file_counter = 0

pulse, fiber = sp.createSetupSpec(number_bins=number_of_bins,freq_range=frequency_range,freq_fwhm=1.0,GDD=0.0,TOD=0.0,FOD=0.0,EPP=0.00001,Length=length,beam_dia=beam_dia, n2=n2,beta2=beta2,beta3=beta3, beta4=beta4,gaussian_beam=gaussian_beam,central_wavelength=central_wavelength,speed_of_light=299700)

freq = pulse.F_THz

B_int_A = np.zeros(number_of_sims)

counter = 0
max_attempts = 5
FOD_attempt = 0
TOD_attempt = 0

variableList = ('number_of_spectrum','number_GDD_per_spec','number_TOD_per_GDD',\
    'number_FOD_per_TOD','datasets_per_file','stdMin','freq_fwhm_ave','freq_fwhm_delta',\
    'dFreq_ave','dFreq_delta','EPP_ave','EPP_delta','GDD_ave','GDD_delta','TOD_ave',\
    'TOD_delta','FOD_ave','FOD_delta','length','beam_dia','gaussian_beam',\
    'central_wavelength','n2','beta2','beta3','beta4','number_of_bins',\
    'frequency_range','number_steps','Raman','Self_Steep','randSeed','number_of_sims')

variables = dict(((k, [eval(k)]) for k in fullNameList))
var_df = pd.DataFrame.from_dict(variables)
var_df.to_pickle(saveDir + 'run_summary.pkl')

for hh in range(number_of_spectrum):
    dFreqShift = dFreq_delta*(gen.normal(0,0.7)) + dFreq_ave
    EPP = EPP_delta*(gen.normal(0,0.7)) + EPP_ave
    freq_fwhm = freq_fwhm_delta*(gen.normal(0,0.7)) + freq_fwhm_ave
    spectrum = genSpec_fourier(freq,gen)
    pulse = sp.createPulseSpecNLO(freq,spectrum,freq*0.0,central_wavelength=central_wavelength,EPP=EPP)

    for ii in range(number_GDD_per_spec):
        GDD = GDD_delta*(gen.normal(0,0.7)) + GDD_ave
        for jj in range(number_TOD_per_GDD):
            TOD = TOD_delta*(gen.normal(0,0.7)) + TOD_ave
            TOD_attempt = 0
            for kk in range(number_FOD_per_TOD):
                    FOD_attempt = 0

                    FOD = FOD_delta*(gen.normal(0,0.7)) + FOD_ave
                    validRun = False
                    while validRun == False:
                    
                #  spectrum = genSpec_2gauss(freq,cenFreq2=central_frequency + dFreqShift)
                        pulse = sp.createPulseSpecNLO(freq,spectrum,freq*0.0,central_wavelength=central_wavelength,EPP=EPP)

                        pulse.chirp_pulse_W(GDD*10**(-6),TOD*10**(-9),FOD*10**(-12))
                        y, AW, AT, pulse_out = sp.runNLOSim(pulse,fiber,Steps=number_steps,Raman=Raman,Steep=Self_Steep)
                        stdSpec = np.std((np.abs(pulse.AW)**2-np.abs(pulse_out.AW)**2)/max(np.abs(pulse.AW)**2))
                        if stdSpec >= stdMin:
                            validRun = True
                        else:
                            if FOD_attempt < max_attempts:
                                FOD_attempt += 1
                                FOD = FOD_delta*(gen.normal(0,0.7)) + FOD_ave
                            elif FOD_attempt == max_attempts:
                                FOD_attempt = 0
                                if TOD_attempt < max_attempts:
                                    TOD_attempt += 1
                                    TOD = TOD_delta*(gen.normal(0,0.7)) + TOD_ave
                                elif TOD_attempt == max_attempts:
                                    TOD_attempt = 0
                                    GDD = GDD_delta*(gen.normal(0,0.7)) + GDD_ave


                                    
                    TOD_attempt = 0
                    FOD_attempt = 0
                    B_int = sp.B_integral(AT,beam_dia,y,wavelength=central_wavelength,n2=n2,gaussian_beam=gaussian_beam)
                    B_int_A[counter] = B_int
                    initial_AW = AW[:,0]
                    final_AW = AW[:,-1]  
                    d = dict(((k, [eval(k)]) for k in fullNameList))
                    a = pd.DataFrame.from_dict(d)
                    
                    initial_Iw = np.abs(AW[:,0][(freq>minFrequency)*(freq<maxFrequency)])**2     
                    final_Iw = np.abs(AW[:,-1][(freq>minFrequency)*(freq<maxFrequency)])**2 
                    freqcrop = freq[(freq>minFrequency)*(freq<maxFrequency)]   
                    
                    temp_ml_data = np.array([freq[1]-freq[0],initial_Iw.shape[0],GDD,TOD,FOD,EPP])
                    temp_ml_data = np.append(temp_ml_data,initial_Iw)
                    temp_ml_data = np.append(temp_ml_data,final_Iw)
                    counter += 1
                    
                    if scan_counter == 0:
                        data_df = a
                        ml_data = temp_ml_data
                    else:
                        data_df = data_df.append(a)
                        ml_data = np.vstack((ml_data,temp_ml_data))
                    
                    scan_counter += 1
                    if (scan_counter == datasets_per_file) or (counter==(number_of_sims-1)):
                        print('%04.d File Saved'%file_counter) 
                        data_df.to_pickle(saveDir + '/data/full_data/full_data_%04.d_'%int(slurm_task_id) + '_%04.d'%file_counter + '.pkl')
                        np.savetxt(saveDir + '/data/spm_ml_%04.d_'%int(slurm_task_id) + '_%04.d'%file_counter + '.csv.gz',ml_data,delimiter=',')
                        file_counter += 1
                        scan_counter = 0
#fig, (ax1,ax2) = plt.subplots(nrows=1,ncols=2)
#ax1.plot(freqcrop,initial_Iw)
#ax1.plot(freqcrop,final_Iw,'--')
#ax2P = ax2.twinx()
#ax2.plot(freqcrop,9*initial_Iw/max(initial_Iw))
#phase = np.unwrap(np.angle(AW[:,0][(freq>minFrequency)*(freq<maxFrequency)]))
#ax2P.plot(freqcrop,phase - phase[int(phase.shape[0]*0.5)],'r--')
#ax2P.set_ylim([-10,10])
#plt.savefig('data/' + str(ii)+'.png')
#     
#plt.show()
#plt.close('all')
# data = np.loadtxt('C:\Users\mstan\Desktop\data\spm_ml_0000__0000.csv.gz',delimiter=',')
# freqdata = np.linspace(0,data[data.shape[0]-1,2]-1,data[data.shape[0]-1,2])*data[data.shape[0]-1,0] - (np.linspace(0,data[data.shape[0]-1,2]-1,data[data.shape[0]-1,2])*data[data.shape[0]-1,0])[int(data[data.shape[0]-1,2]*0.5)] + 374.625
# plt.figure()
# plt.plot(freqdata,data[data.shape[0]-1,6:(data[data.shape[0]-1,2]+6)])
# plt.plot(freqcrop,initial_Iw,'--')
# plt.figure()
# plt.plot(freqdata,data[data.shape[0]-1,(data[data.shape[0]-1,2]+6):(2*data[data.shape[0]-1,2]+6)])
# plt.plot(freqcrop,final_Iw,'--')
# plt.show()
print('B Array')
print(B_int_A)
print('Min B: ' + str(min(B_int_A)))
print('Max B: ' + str(max(B_int_A)))
