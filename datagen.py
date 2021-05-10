
import spym
import numpy as np
import pandas as pd

def test():
    print('it loaded')

def load_exp(pulse_0,pulse_1,pulse_frg,EPP,cen_wave,dF=[],fiber=None,Raman=None,Steep=None,number_steps=None,beam_dia=None):
    if dF == []:
        #col_names = ['EPP','Iw_i','Iw_f','Phase_i','fiber','freq','Raman','Steep','Steps','central freq']
        col_names = ['EPP','Iw_i','Iw_f','Phase_i','fiber','freq','Raman','Steep','Steps','central freq','B_int','pulse_0','pulse_1','beam_dia']

        dF = pd.DataFrame(columns=col_names)
    phase = pulse_frg.get_phase()
    central_freq = 288700.0/cen_wave
    B_int = None
    d = [EPP,np.abs(pulse_0.AW)**2,np.abs(pulse_1.AW)**2,phase,fiber,pulse_0.F_THz,Raman,Steep,number_steps,central_freq,B_int,pulse_0,pulse_1,beam_dia]

    dF.loc[len(dF)] = d
    return dF


def addNoiseEPP(dF,scalar=1.0,noise_type='range',randomState=None):
    size = dF['EPP'].size
    if randomState == None:
        randomState = np.random.RandomState()
    elif isinstance(randomState,int):
        randomState = np.random.RandomState(seed=randomState)

    for ii in range(size):
        EPP = dF['EPP'].iloc[ii]
        if noise_type == 'range':
            EPP += 0.5*scalar*(2*randomState.rand()-1)
        elif noise_type == 'percent':
            EPP += scalar*(2*randomState.rand()-1)*EPP
        if EPP <= 0.0:
            EPP = 10**(-5)
        dF['EPP'].iloc[ii] = EPP
    
    
    return dF
def addNoiseSpec(dF,scalar=1.0,noise_type='shot',randomState=None):
    import time
    size = dF['Iw_i'].size

    if isinstance(randomState,int):    
        randomState = np.random.RandomState(seed=randomState)
    elif not(isinstance(randomState,np.random.RandomState)):
        randomState = np.random.RandomState(seed=int(time.time()))

    for ii in range(size):
        Iw_i = dF['Iw_i'].iloc[ii]
        Iw_f = dF['Iw_f'].iloc[ii]
        if noise_type == 'shot':
            Iw_i += scalar*(2*randomState.rand(Iw_i.shape[0])-1)*np.sqrt(Iw_i)
            Iw_f += scalar*(2*randomState.rand(Iw_f.shape[0])-1)*np.sqrt(Iw_f)
        elif noise_type == 'range':
            Iw_i += scalar*(2*randomState.rand(Iw_i.shape[0])-1)
            Iw_f += scalar*(2*randomState.rand(Iw_f.shape[0])-1)
        elif noise_type == 'percent':
            Iw_i += scalar*(2*randomState.rand(Iw_i.shape[0])-1)*Iw_i
            Iw_f += scalar*(2*randomState.rand(Iw_f.shape[0])-1)*Iw_f
        else:
            print('Unknown Type of Noise. No Noise Added')
        Iw_i[Iw_i<=0.0] = 0.0
        Iw_f[Iw_f<=0.0] = 0.0
    
        dF['Iw_i'].iloc[ii] = Iw_i
        dF['Iw_f'].iloc[ii] = Iw_f
    
    return dF

def generateRandomPhaseTaylor(randomState,freq,central_freq,number_terms=[1],stds=[1],centers=[0],output_terms=0):
    def cartesian(arrays, out=None):
        """
        Generate a cartesian product of input arrays.
    
        Parameters
        ----------
        arrays : list of array-like
            1-D arrays to form the cartesian product of.
        out : ndarray
            Array to place the cartesian product in.
    
        Returns
        -------
        out : ndarray
            2-D array of shape (M, len(arrays)) containing cartesian products
            formed of input arrays.
    
        Examples
        --------
        >>> cartesian(([1, 2, 3], [4, 5], [6, 7]))
        array([[1, 4, 6],
               [1, 4, 7],
               [1, 5, 6],
               [1, 5, 7],
               [2, 4, 6],
               [2, 4, 7],
               [2, 5, 6],
               [2, 5, 7],
               [3, 4, 6],
               [3, 4, 7],
               [3, 5, 6],
               [3, 5, 7]])
    
        """
        """
        Solution from Stack Overflow user pv.
        https://stackoverflow.com/questions/1208118/using-numpy-to-build-an-array-of-all-combinations-of-two-arrays
        """
    
        arrays = [np.asarray(x) for x in arrays]
        dtype = arrays[0].dtype
    
        n = np.prod([x.size for x in arrays])
        if out is None:
            out = np.zeros([n, len(arrays)], dtype=dtype)
    
        m = n / arrays[0].size
        out[:,0] = np.repeat(arrays[0], m)
        if arrays[1:]:
            cartesian(arrays[1:], out=out[0:m,1:])
            for j in xrange(1, arrays[0].size):
                out[j*m:(j+1)*m,1:] = out[0:m,1:]
        return out

    number_of_phases = np.product(number_terms)
    phases = np.zeros((number_of_phases,freq.shape[0]))
    phase_terms = np.zeros((number_of_phases,np.asarray(number_terms).shape[0]))
    gen_terms = []
    
    #Iterating through all phase coef and generating specificed random values
    for idx in range(len(number_terms)):
        gen_terms.append(randomState.normal(loc=centers[idx],scale=stds[idx],size=number_terms[idx]))

    phase_terms = cartesian(gen_terms, out=None)
    
    for ii in range(number_of_phases):
        phases[ii] = phase_from_coef(freq,central_freq,phase_terms[ii])
    
    if output_terms == True:
        return randomState,phases,phase_terms
    else:
        return randomState,phases
    
    
def generateRandomSpectrumFourier(randomState,freq,central_freq,time_fwhm,freq_fwhm):
    time = np.fft.fftshift(np.fft.fftfreq(freq.shape[0],freq[1]-freq[0]))
    t_env = np.exp(-4*np.log(2)*time**2/(time_fwhm*10**-3)**2)  #converting fs fwhm to ps fwhm
    f_env = np.exp(-4*np.log(2)*(freq-central_freq)**2/freq_fwhm**2)
    t_phase = randomState.rand(freq.shape[0])
    t_field = t_env*randomState.rand(freq.shape[0])*np.exp(1j*t_phase)
    f_field = np.fft.ifft(np.fft.ifftshift(t_field))*f_env
#Following two lines are new and untested
    #f_field[np.abs(freq-central_freq)>(2*f_env)] = 0.0

    #f_field -= np.min(np.abs(f_field[np.abs(f_field)>0.0])
    return randomState,f_field

def phase_from_coef(freq,center_freq,phase_coef=[0.0]):
    from scipy.misc import factorial
    ang_freq = 2*np.pi*(freq-center_freq)*10**(-3)
    phase = freq*0.0
    for idx,coef in enumerate(phase_coef):
        phase += coef/factorial(idx + 2)*ang_freq**(idx+2)
    return phase    
    
    
def runRandomSpectrumSims(fiber,freq,central_freq,time_fwhm,freq_fwhm,
                          number_pulses=1,number_steps=20,EPP=0.001,Raman=False,Steep=False,
                          randomState=[],center='moment',beam_dia=None,n2=None):
    if randomState == []:
        randomState = np.random.RandomState(seed=1)

    sim_data = []
    col_names = ['EPP','Iw_i','Iw_f','Phase_i','fiber','freq','Raman','Steep','Steps','central freq','B_int','pulse_0','pulse_1','beam_dia','n2','time_fwhm','freq_fwhm']
    sim_data = pd.DataFrame(columns=col_names)
    for pulse_num in range(number_pulses):
        randomState,f_field = generateRandomSpectrumFourier(randomState,freq,central_freq,
                                                        time_fwhm,freq_fwhm)
        phase = np.unwrap(np.angle(f_field))
        phase -= phase[int(phase.shape[0]*0.5)]
        pulse_0 = spym.createPulseSpecNLO(freq,np.abs(f_field)**2,phase,central_wavelength=299700.0/central_freq,EPP=EPP)
        
        if center == "max":
            pulse_0.time_center_max()
        elif center == "moment":
            pulse_0.time_center_moment()
            
        phase = np.unwrap(np.angle(pulse_0.AW))
        phase -= phase[int(phase.shape[0]*0.5)]
        
        _,_,_,pulse_1 = spym.runNLOSim(pulse_0,fiber,Steps=number_steps,Raman=Raman,Steep=Steep)

        B_int = pulse_1.calc_B()
        pulse_1.clear_history()
        d = [EPP,np.abs(pulse_0.AW)**2,np.abs(pulse_1.AW)**2,phase,fiber,pulse_0.F_THz,Raman,Steep,number_steps,central_freq,B_int,pulse_0,pulse_1,beam_dia,n2,time_fwhm,freq_fwhm]
        sim_data.loc[len(sim_data)] = d
    return sim_data
 
    
def pulsefromdf(dF,idx,center='none'):
    if idx < dF.shape[0]:
        tmpDF = dF.loc[idx]
        pulse = spym.createPulseSpecNLO(tmpDF['freq'],tmpDF['Iw_i'],tmpDF['Phase_i'],central_wavelength=299700.0/tmpDF['central freq'],EPP=tmpDF['EPP'])
        if center == "max":
            pulse.time_center_max()
        elif center == "moment":
            pulse.time_center_moment()
        return pulse
        
    else:
        print('idx out of range')
        

        
def runRandomTaylorSims(fiber, freq, central_freq, EPP,time_fwhm, freq_fwhm,randomState=[],
                        number_terms=[], stds=[], centers=[],number_steps=20,Raman=False,
                        Steep=False,beam_dia=None):
   
 
    if randomState == []:
        randomState = np.random.RandomState(seed=1)

    if not(len(number_terms) == len(stds) and len(stds) == len(centers)):
        print('Incompatable lengths for system')
    randomState, phase_array,phase_terms = generateRandomPhaseTaylor(randomState,freq,central_freq,
                                                         number_terms=number_terms,stds=stds,centers=centers,output_terms=1)
    randomState,f_field = generateRandomSpectrumFourier(randomState,freq,central_freq,
                                                        time_fwhm,freq_fwhm)

    
    f_field = np.abs(f_field)
    sim_data = []
    col_names = ['EPP','Iw_i','Iw_f','Phase_i','fiber','freq','Raman','Steep','Steps','central freq','B_int','pulse_0','pulse_1','beam_dia','phase_coef']
    sim_data = pd.DataFrame(columns=col_names)
    for ii, phase in enumerate(phase_array):
        pulse_0 = spym.createPulseSpecNLO(freq,np.abs(f_field)**2,phase,
                                          central_wavelength=299700.0/central_freq,EPP=EPP)
        _,_,_,pulse_1 = spym.runNLOSim(pulse_0,fiber,Steps=number_steps,Raman=Raman,Steep=Steep)
        B_int = pulse_1.calc_B()
        pulse_1.clear_history()
        d = [EPP,np.abs(pulse_0.AW)**2,np.abs(pulse_1.AW)**2,phase,fiber,pulse_0.F_THz,Raman,Steep,number_steps,central_freq,B_int,pulse_0,pulse_1,beam_dia,phase_terms[ii]]
        sim_data.loc[len(sim_data)] = d

    return sim_data

def cropData(df,interpFreq):
    col_names = df.columns

    newDF = pd.DataFrame(columns=col_names)
    for index, row in df.iterrows():
        interpIw_i = np.interp(interpFreq,row['freq'],row['Iw_i'])
        interpIw_f = np.interp(interpFreq,row['freq'],row['Iw_f'])
        interpPhase_i = np.interp(interpFreq,row['freq'],row['Phase_i'])
        row['freq'] = interpFreq
        row['Iw_i'] = interpIw_i
        row['Iw_f'] = interpIw_f
        row['Phase_i'] = interpPhase_i
        newDF.loc[len(newDF)] = row
    return newDF


