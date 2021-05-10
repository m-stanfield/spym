
import numpy as np
import matplotlib.pyplot as plt
import scipy
import os
import pynlo
from copy import deepcopy



class spym_pulse(pynlo.light.PulseBase.Pulse):
    def __init__(self):
        pynlo.light.PulseBase.Pulse.__init__(self)
        self.fiber_hist = []
        self.AW_hist = []
        self.AT_hist = []
        self.y_hist = []
        self.B_int = [0.0]

    def compress(self,GDDRange=1000,bins=256):
        GDD_A = np.linspace(-0.5*GDDRange,0.5*GDDRange,bins)
        peakI_val = GDD_A*0.0
        for idx, GDD in enumerate(GDD_A):
            self.add_phase([GDD])
            peakI_val[idx] = self.peakI()
            self.add_phase([-GDD])
        index = np.argmax(peakI_val)
        self.add_phase([GDD_A[index]])            




    def calc_B(self,piecewise=False):
        '''
        The total B-Integral that this pulse has undergone.

        Parameters:
            piecewise (bool): Forces program to output list of individual B-Int if multiple stages of SPM have been done.

        Returns:
            B_int(list): The calcualted B integral for the system. If piecewise is true it will have the B integral per stage seperated out. If false it will automatically sum it for you.
        '''
        self.B_int = []
        if len(self.fiber_hist) > 0:
            for ii in range(len(self.AT_hist)):
                power = np.abs(self.AT_hist[ii])**2
                max_power = np.amax(power,axis=0)
                gamma = self.fiber_hist[ii].get_gamma()
                self.B_int.append(gamma*np.inner(max_power,np.diff(self.y_hist[ii])))
            if piecewise == True:
                return np.array(self.B_int)
            else:
                return np.array([np.sum(self.B_int)])
        else:
            return [0.0]

    def copy(self):
        '''
        Creates a copy of pulse object with the same frequency axis, spectrum, phase central frequency, and energy.

        Returns:
            Copy of the pulse structure
        '''
        pulse_copy = createPulseSpecNLO(self.F_THz,np.abs(self.AW)**2,np.angle(self.AW),central_wavelength=299700.0/self.center_frequency_THz,EPP=self.calc_epp())
        return pulse_copy

    def zero_phase(self):
        '''
        Sets the phase to be zero for the entire spectrum. Creates FTL pulse.

        '''
        self.set_AW(np.abs(self.AW))


    def add_phase(self,phase_coef=[0.0]):
        '''
        Adds spectral phase to the pulse. Starts with GDD and will allow for arbitrary number of phase terms.

        Inputs:
            phase_coef (list of floats): List of n taylor expansion terms starting with the quardatic term(GDD).
        '''
        phase = self.phase_from_coef(phase_coef=phase_coef)
        self.set_AW(self.AW*np.exp(1j*phase))

    def set_phase(self,phase_coef=[0.0]):
        '''
        Sets spectral phase to the pulse. Starts with GDD and will allow for arbitrary number of phase terms.

        Inputs:
            phase_coef (list of floats): List of n taylor expansion terms starting with the quardatic term(GDD).
        '''
        phase = self.phase_from_coef(phase_coef=phase_coef)
        self.set_AW(np.abs(self.AW)*np.exp(1j*phase))

    def flip_phase(self):
        '''
        Flips the spectral phase of the pulse. Allows for easy testing of SHG FROG resconstructions.
        '''
        self.set_AW(np.conj(self.AW))

    def time_shift(self,t_shift=0.0,units='fs'):
        '''
        Shifts the temporal profile by a set amount of time.

        Inputs:
            t_shift (float): Amount of time that the pulse will be shifted.

            units (str): determines units of the t_shift. Allowed inputs are fs, ps, or s
        '''
        if units == 'fs':
            t_shift = t_shift/1000.0
        elif units == 'ps':
            pass # do nothing but want to put error function at the end incase no units are given
        elif units == 's':
            t_shift = t_shift*10**(-12)
        else:
            print("Invalid Unit, please give fs, ps, or s")
            return
        rot_AT = np.roll(self.AT,1*int(t_shift/self.dT_ps))

        self.set_AT(rot_AT)

    def time_center_max(self):
        '''
        Shifts temporal profile so maximum of the profile is at t=0
        '''
        self.time_shift(-1.0*self.T_ps[np.argmax(np.abs(self.AT))],units='ps')

    def time_center_moment(self):
        '''
        Centers pulse on the second moment.

        '''
        self.time_shift(-1.0*np.sum(self.T_ps*np.abs(self.AT)**2)/np.sum(np.abs(self.AT)**2),units='ps')


    def spectral_std(self,center_freq=None):
        '''
        Calculates spectral standard deviation of the pulse.
        Inputs:
            center_freq (float): Calculates the spectral deviation based around this value. If no value is given the default value for hte pulse is used.
        '''
        if center_freq == None:
            center_freq = np.sum(self.F_THz*np.abs(self.AW)**2)/np.sum(np.abs(self.AW)**2)
            sigma = np.sqrt(np.abs(np.sum((self.F_THz-center_freq)**2*np.abs(self.AW)**2)/np.sum(np.abs(self.AW)**2)))
        return sigma


    def temporal_std(self,center_time=None):
        '''
        Calculates temporal standard deviation of the pulse.
        Inputs:
            center_time (float): Calculates the spectral deviation based around this value. If no value is given the default value for hte pulse is used.
        '''
        if center_time == None:
            center_time = np.sum(self.T_ps*np.abs(self.AT)**2)/np.sum(np.abs(self.AT)**2)
            sigma = np.sqrt(np.abs(np.sum((self.T_ps-center_time)**2*np.abs(self.AT)**2)/np.sum(np.abs(self.AT)**2)))
        return sigma

    def TBP(self,method='fwhm'):
        '''
        Calculates time bandwidth produce of the pulse.
        Inputs:
            method (str): If fwhm will base calcuation off of FWHM value. If std bases it off of standard deviation.
        Returns:
            TBP (float): Time bandwidth produce calculated
        '''
        if method== 'fwhm':
            TBP = self.fwhm_t()*self.fwhm_f()/1000.0 #factor of 1000 due to fwhm_t() being in fs
        elif method == 'std':
            TBP = self.spectral_std()*self.temporal_std()
        return TBP

    def get_phase(self,domain='freq'):
        '''
        Returns phase of pulse with the center element set to zero.

        Retuns:
            phase (numpy array):
        '''
        if domain == 'freq':
            phase = np.unwrap(np.angle(self.AW))
            phase = (phase - phase[int(phase.shape[0]*0.5)])
        elif domain == 'time':
            phase = np.unwrap(np.angle(self.AT))
            phase = (phase - phase[int(phase.shape[0]*0.5)])
        else:
            print('Unknown domain')
        return phase

    def dF(self):
        '''
        The frequency resolution in THz

        Returns:
            dF (flaot): frequency resolution
        '''
        return self.F_THz[1] - self.F_THz[0]

    def dT(self):
        '''
        The temporal resolution in fs

        Returns:
            dT (flaot): temporal resolution
        '''
        return 1000*(self.T_ps[1] - self.T_ps[0])

    def W_nm(self,sol=299700.0):
        '''
        Calculates wavelength axis for pulse
        Inputs:
            sol (float): speed of light used for calculation.
        Returns:
            wavelength axis (numpy array): wavelength axis
        '''
        return sol/self.F_THz[::-1]

    def I_nm(self,sol=299700.0):
        '''
        Calculates intensity for wavelength axis for pulse
        Inputs:
            sol (float): speed of light used for calculation.
        Returns:
            I (numpy array): Spectral intensity ni wavelength space
        '''
        I = np.abs(self.AW)**2*self.F_THz**2/sol
        I = I[::-1]
        return I
    def zero_If(self,lower_freq=None,upper_freq=None):
        '''
        Zeros spectrum outside set frequency ranges

        Inputs:
            lower_Freq (float): any value below this point is set to zero
            upper_freq (float): any value above this point is set to zero
        '''
        spectrum = self.AW*1.0
        if not(lower_freq == None):
            print('here')
            spectrum[self.F_THz<lower_freq] = 0.0
        if not(upper_freq == None):
            spectrum[self.F_THz>upper_freq] = 0.0
        self.set_AW(spectrum)

    def zero_Iw(self,lower_wave=None,upper_wave=None):
        '''
        Zeros spectrum outside set wavelength ranges

        Inputs:
            lower_wave (float): any value below this point is set to zero
            upper_wave (float): any value above this point is set to zero
        '''
        if not(lower_wave == None):
            self.zero_If(lower_freq=299700.0/upper_wave,upper_freq=None)
        if not(upper_wave == None):
            self.zero_If(lower_freq=None,upper_freq=299700.0/lower_wave)


    def save_fields(self,path,dT=0.003,number_bins=2**14):
        #dT in ps
        T = np.linspace(-0.5*(number_bins-1)*dT,0.5*(number_bins-1)*dT,number_bins)
        AT_re = np.interp(T,self.T_ps,np.real(self.AT),left=0.0,right=0.0)
        AT_im = np.interp(T,self.T_ps,np.imag(self.AT),left=0.0,right=0.0)
        phase =  np.interp(T,self.T_ps,np.unwrap(np.angle(self.AT)),left=0.0,right=0.0)
        It = np.abs(AT_re+1j*AT_im)**2
        output = np.array([1000.0*T,It,phase,AT_re,AT_im]).T
        np.savetxt(path,output)


    def peakI(self,beam_dia=11.7,gaussian_beam=True):
        '''
        Calculates peak intensity of the pulse given specific beam diameter

        Inputs:
            beam_dia (float): Beam diameter of the beam. 1/e^2 for gaussian or width for flattop
            gaussian_beam (bool): If true treat beam as gaussian. If false treat as flat top.

        Returns:
            peak_intensity (float): Returns peak intensity in W/cm^2
        '''

#Calculates peak intensity of pulse. In W/cm^2
#beam_dia is in mm
        peakPower = np.max(np.abs(self.AT)**2)
        area = np.pi*(0.5*beam_dia*10**(-1))**2 #converting mm to cm
        if gaussian_beam == True:
            return 2*peakPower/area
        else:
            return peakPower/area


    def plot(self,ax=[],cutoff_percent=0.0,phase=0,linestyles=['b-','r--'],normalize=1,labels=['Temporal','Spectrum','Phase'],x_axis_wave=False):
        freq = self.F_THz[self.F_THz>0.0]    #x-axis for frequency domain

        spectrum = (np.abs(self.AW)**2)[self.F_THz>0.0]          #Calculating spectral intensity

        cropIndex_spec = (spectrum/np.max(spectrum)>cutoff_percent) #finding indices for cropped spectrum


        if len(linestyles)<2:
            linestyleI = 'b-'
            linestyleP = 'r--'
        else:
            linestyleI = linestyles[0]
            linestyleP = linestyles[1]

        if len(labels)<2:
            labels=[]
            labels.append('Temporal')
            labels.append('Spectrum')
            print('Improper number of labels')

        temporal = np.abs(self.AT)**2
        cropIndex_time = (temporal/max(temporal)>cutoff_percent)
        time = 1000*self.T_ps

        if normalize ==1:
            spectrum = spectrum/np.max(spectrum)
            temporal = temporal/np.max(temporal)

        if len(ax)<2:
            fig, ax = plt.subplots(ncols=2)

        ax[0].plot(time,temporal,linestyleI,label=labels[0])
        ax[0].set_xlabel('Time (fs)')
        ax[0].set_ylabel('Intensity (a.u.)')
        ax[0].set_xlim([time[cropIndex_time][0],time[cropIndex_time][-1]])
        ax[0].set_ylim([0,1.05*np.max(temporal)])
        if x_axis_wave == 1:
            if normalize == 1:
                spectrum = spectrum/np.max((spectrum*freq**2))
            ax[1].plot((299700.0/freq)[::-1],(spectrum*freq**2)[::-1],linestyleI,label=labels[1])
            ax[1].set_xlabel('Wavelength (nm)')
            ax[1].set_ylabel('Intensity (a.u.)')
            ax[1].set_ylim([0,1.05*np.max((spectrum*freq**2))])

            ax[1].set_xlim([299700.0/(freq[cropIndex_spec][-1]),299700.0/(freq[cropIndex_spec][0])])

        else:
            ax[1].plot(freq,spectrum,linestyleI,label=labels[1])
            ax[1].set_xlabel('Frequency (THz)')
            ax[1].set_ylabel('Intensity (a.u.)')
            ax[1].set_ylim([0,1.05*np.max(spectrum)])

            ax[1].set_xlim([freq[cropIndex_spec][0],freq[cropIndex_spec][-1]])




        if phase == 1:
            if len(labels)<3:
                labels.append('Phase')



            phase = np.unwrap(np.angle(self.AW))
            phase = (phase - phase[int(phase.shape[0]*0.5)])[self.F_THz>0]

            if len(ax)<3:
                ax = np.append(ax,ax[1].twinx())

            if x_axis_wave == 1:
                ax[1].plot((299700.0/freq)[::-1],spectrum*0-10.0,linestyleP,label=labels[2])
                ax[2].plot((299700.0/freq)[::-1],phase[::-1],linestyleP,label=labels[2])
                ax[1].set_xlim([299700.0/(freq[cropIndex_spec][-1]),299700.0/(freq[cropIndex_spec][0])])

            else:
                ax[1].plot(freq,spectrum*0-1000.0,linestyleP,label=labels[2])
                ax[2].plot(freq,phase,linestyleP,label=labels[2])
                ax[1].set_xlim([freq[cropIndex_spec][0],freq[cropIndex_spec][-1]])

            ax[2].set_ylim([-15,15])
            ax[0].legend()
            ax[1].legend()
            return ax
        else:
            ax[0].legend()
            ax[1].legend()
            return ax

    def ftl(self,output_pulse=False,min_percent=0.0,units='time'):
        '''
        Calculates the FTL of the pulse.



        Parameters
        ----------
        output_pulse : Bool, optional
            If true, the output is the FTL pulse. Otherwise the output is the output fwhm. The default is 0.
        min_percent : TYPE, optional
            Zeros the spectrum below min_percent. The default is 0.0.
        units : TYPE, optional
            DESCRIPTION. The default is 'time'.

        Returns
        -------
        float/pulse class
            If output_pulse is false the temporal fwhm is return, otherise the FTL pulse object is returned.

        '''
        spectrum = np.abs(self.AW)**2
        spectrum[spectrum/max(spectrum)<min_percent] = 0
        spectrum = spectrum - spectrum[spectrum>0][0]
        spectrum[spectrum<0] = 0
        ftl_pulse = createPulseSpecNLO(self.F_THz,spectrum,self.F_THz*0,central_wavelength=self.center_wavelength_nm,EPP=self.calc_epp())
        t_int = np.abs(ftl_pulse.AT)**2/max(np.abs(ftl_pulse.AT)**2)
        fwhm = 1000*(ftl_pulse.T_ps[t_int>0.5][-1] - ftl_pulse.T_ps[t_int>0.5][0])
        if output_pulse == 1:
            return ftl_pulse
        else:
            if units == 'time':
                return fwhm
            elif units =='cycles':
                return fwhm*self.center_frequency_THz/1000.0
    def db20_f(self):
        """
        Calculates the dB -20 (1%) width of the spectrum.

        Returns
        -------
        FLOAT
            The dB -20 width.
        """
        freq = self.F_THz
        intensity = np.abs(self.AW)**2/max(np.abs(self.AW)**2)
        return freq[intensity>0.01][-1] - freq[intensity>0.01][0]

    def fwhm_t(self,units='time'):
        """
        Returns the full width at half maximum for the temporal profile

        Parameters
        ----------
        units : str, optional
            If 'time' the output is returned in units of fs. If 'cycles' the fwhm is normalized to number of cycles. The default is 'time'.

        Returns
        -------
        float
            The temporal fwhm in desired units.

        """

        time = self.T_ps*1000
        intensity = np.abs(self.AT)**2/max(np.abs(self.AT)**2)
        if units == 'time':
            return time[intensity>0.5][-1] - time[intensity>0.5][0]
        elif units == 'cycles':
            return (time[intensity>0.5][-1] - time[intensity>0.5][0])*self.center_frequency_THz/1000.0


    def fwhm_f(self):
        """
        Returns the full width at half maximum for the spectral profile

        Returns
        -------
        float
            The spectral fwhm in desired units.

        """
        freq = self.F_THz
        intensity = np.abs(self.AW)**2/max(np.abs(self.AW)**2)
        return freq[intensity>0.5][-1] - freq[intensity>0.5][0]

    def fit_phase(self,min_intensity=0.01,fit_order=5, coef='taylor'):
        """
        Fits the spectral phase to a taylor expansion of a desired order.

        Parameters
        ----------
        min_intensity : float, optional
            Defines the region of that the taylor expnasion is taken over. The default is 0.01.
        fit_order : int, optional
            Determines the number of phase terms to fit over.. The default is 5.
        coef : TYPE, optional
            DESCRIPTION. The default is 'taylor'.

        Returns
        -------
        TYPE
            DESCRIPTION.
        TYPE
            DESCRIPTION.

        """
        #If coef == 'taylor' returns the coef in taylor expansion form
        #If coef == 'poly' it returns it as a polynormial(aka with factorial backed in)
        from scipy.special import factorial
        cropIndex = np.abs(self.AW)**2>max(min_intensity*np.abs(self.AW)**2)
        phase = np.unwrap(np.angle(self.AW))
        phase = phase - phase[int(phase.shape[0]*0.5)]
        phase = phase[cropIndex]
        z = np.polyfit(2*np.pi*(self.F_THz[cropIndex]-self.center_frequency_THz),phase,fit_order+1)
        p_fit = np.poly1d(z)
        z = z[::-1]
        if coef == 'poly':
            return p_fit(2*np.pi*(self.F_THz-self.center_frequency_THz)), z[2:]
        elif coef == 'taylor':
            phi_coef = z*0
            for ii in range(z.shape[0]):
                if ii == 0 or ii == 1:
                    phi_coef[ii] = 0
                else:
                    phi_coef[ii] = z[ii]*factorial(ii)*10**(3*ii)
            return p_fit(2*np.pi*(self.F_THz-self.center_frequency_THz)), phi_coef[2:]
        else:
            return p_fit(2*np.pi*(self.F_THz-self.center_frequency_THz))

    def phase_from_coef(self,phase_coef=[0.0]):
        from scipy.special import factorial
        ang_freq = 2*np.pi*(self.F_THz-self.center_frequency_THz)*10**(-3)
        phase = ang_freq*0.0
        for idx,coef in enumerate(phase_coef):
            phase += coef/factorial(idx + 2)*ang_freq**(idx+2)
        return phase

    def save_spectrum(self,fname='spectrum.csv',delimiter=','):
        data = np.zeros((3,self.F_THz.shape[0]))
        data[0] = self.F_THz
        data[1] = np.abs(self.AW)**2
        data[2] = np.unwrap(np.angle(self.AW))
        np.savetxt(fname,data,delimiter=delimiter)


    def clear_history(self):
        """
        Clears history variables. Reset to default.

        """
        self.fiber_hist = []
        self.AW_hist = []
        self.AT_hist = []
        self.y_hist = []
        self.B_int = [0.0]

"""Intensity Generating Functions"""




def generateIntensityGaussianSpectrum(number_bins = 2**14,freq_range = 4000.0 , freq_fwhm = 15.0 ,center_freq = 374.7, peakPower = 1,phase_coef=[0.0,0.0,0.0],EPP=0.001,output_pulse=1):
    """
    Generates a Gaussian Intensity Spectrum over a give frequency range. Units
    are defined by user but need to be self-consistent. Default Units are THz
    Parameters
    ----------
    freq_range : int
        Range of the spectrum array. Will create a spectrum that has a range give
       around the central frequency. This and the centeral frequency define the
        minimum and maximum frequencies of the spectrum.

    number_bins : int
        Number of elements in the spectrum. Needs to match the number of phase
        elements.
    freq_fwhm : float
        The spectual full width half maximum for the intensity spectrum.

    center_freq : float
        Defines Central frequency of the spectrum. Creates symmetric gaussian pulse
        around this point. Defaults to 374.7 [THz] which corresponds to the frequency
        of 800 nm light.

    peakPower : float
        Defines peak power of spectrum


    Returns
    -------
    array
        First array returned is the frequency array of the pulse
    array
        Second array returned is the spectual intensity of the pulse as a function
        of frequency. Shape should be
    """
    freq = np.linspace(-freq_range/2,freq_range/2,number_bins) + center_freq

    phase = phase_from_coef(freq,center_freq,phase_coef=phase_coef)
    intensity = np.exp(-2.7725887*(freq - center_freq)**2/freq_fwhm**2)
    if output_pulse == 1:
        return createPulseSpecNLO(freq,intensity,phase,central_wavelength=299700.0/center_freq,EPP=EPP)
    else:
        return freq,intensity,phase

def phase_from_coef(freq,center_freq,phase_coef=[0.0]):
    from scipy.special import factorial
    ang_freq = 2*np.pi*(freq-center_freq)*10**(-3)
    phase = freq*0.0
    for idx,coef in enumerate(phase_coef):
        phase += coef/factorial(idx + 2)*ang_freq**(idx+2)
    return phase

def generateIntensityGaussianTime(number_bins = 128,time_range = 100 , time_fwhm = 10 , peakI = 1):
    """
    Generates a Gaussian Intensity Spectrum over a give frequency range. Units
    are defined by user but need to be self-consistent. Default Units are THz
    Parameters
    ----------
    freq_range : int
        Range of the spectrum array. Will create a spectrum that has a range give
       around the central frequency. This and the centeral frequency define the
        minimum and maximum frequencies of the spectrum.

    number_bins : int
        Number of elements in the spectrum. Needs to match the number of phase
        elements.
    freq_fwhm : float
        The spectual full width half maximum for the intensity spectrum.

    center_freq : float
        Defines Central frequency of the spectrum. Creates symmetric gaussian pulse
        around this point. Defaults to 374.7 [THz] which corresponds to the frequency
        of 800 nm light.

    peakI : float
        Defines central


    Returns
    -------
    array
        First array returned is the frequency array of the pulse
    array
        Second array returned is the spectual intensity of the pulse as a function
        of frequency. Shape should be
    """
    time = np.linspace(-time_range/2,time_range/2,number_bins)
    return time,peakI*np.exp(-2.7725887*(time)**2/time_fwhm**2)

"""PyNLO Functions"""


def createPulseSpecNLO(freq,intensity,phase,central_wavelength=800.0,EPP=0.007):
    field = np.sqrt(intensity)*np.exp(phase*1.j)
    pulse = spym_pulse()
    pulse.set_NPTS(freq.shape[0])
    pulse.set_center_wavelength_nm(central_wavelength)
 #   pulse.set_frep_mks(1000)


    pulse.set_frequency_window_THz((max(freq)-min(freq)))


    pulse.set_AW(field)

    pulse.set_epp(EPP)


    return pulse

def createPulseTimeNLO(time,intensity,phase,central_wavelength=800,EPP=0.007,time_window_ps=-1):
    field = np.sqrt(intensity)*np.exp(phase*1.j)
    pulse = spym_pulse()
    pulse.set_NPTS(time.shape[0])
    pulse.set_center_wavelength_nm(central_wavelength)
 #   pulse.set_frep_mks(1000)

    if time_window_ps == -1:
        pulse.set_time_window_ps((max(time)-min(time))/1000)
    else:
        pulse.set_time_window_ps(time_window_ps)

    pulse.set_AT(field)

    pulse.set_epp(EPP)


    return pulse

def createFSFiber(central_wavelength=800,Length=1.0, beam_dia=1.0, n2=2.6*10**(-20),beta2=36.1,beta3=27.49,beta4=-11.4335,gaussian_beam=True,alpha=0):
    fiber = createFiberNLO(central_wavelength=central_wavelength,Length=Length, beam_dia=beam_dia, n2=n2,beta2=beta2,beta3=beta3,beta4=beta4,gaussian_beam=gaussian_beam,alpha=alpha)
    return fiber

def createFiberNLO(central_wavelength=800,Length=1.0, beam_dia=1.0, n2=2.7*10**(-20),beta2=0.0,beta3=0.0,beta4=0,gaussian_beam=True,alpha=0):
    """
    central_wavelength : float
        Sets the central wavelength of the media.

    length: float
        Sets the length of the media in millimeters.

    beam_dia : float
        1/e^2 diameter of a gaussian beam or the diameter of a flat top beam in millimeters

    n2 : float
        Kerr index value. In m^2/W

    betas : float
        Dispersion Terms for fiber. Units are fs^n/mm

    gaussian_beam : bool
        Contains boolean that tracks if beam is flat top vs gaussian. For the same
            energy and temporal profile the peak power of a gaussian is twice of that
            of a flat top with the same beam_dia
                #https://www.rp-photonics.com/effective_mode_area.html
    Returns
    -------
    pulse
        A PyNLO pulse class that contains all the infomation from above.
    """
    #Converting betas to ps^n/km
    #beta2 = beta2    fs^2/km == ps^2/mm
    beta3 = beta3*10**(-3)   #needs one conversion of fs to ps
    beta4 = beta4*10**(-6)   #needs two conversion of fs to ps
    #https://www.rp-photonics.com/effective_mode_area.html
    beam_radius = beam_dia/2000 #Radius of the beam in meters

    Aeff = np.pi*(beam_radius)**2 #Effective Area of the beam(RP Photonics)(m^2)


    Gamma   = 2*np.pi*n2/((central_wavelength*10**(-12))*Aeff)

    if gaussian_beam == True:
        #Due to peak intensity of gaussian being twice the peak intensity of a
        #equivlent flattop beam
        Gamma = 2*Gamma

    Alpha   = 0.0     # attentuation coefficient (dB/cm)
    alpha = np.log((10**(Alpha * 0.1))) * 100  # convert from dB/cm to 1/m

    fiber1 = pynlo.media.fibers.fiber.FiberInstance()
    fiber1.generate_fiber(Length * 1e-3, center_wl_nm=central_wavelength, betas=(beta2, beta3, beta4),
                              gamma_W_m=Gamma * 1e-3, gvd_units='ps^n/km', gain=-alpha)
    #Gamma = 0
    return fiber1


def createSetupSpec(number_bins=2**10,freq_range=1000,freq_fwhm=15.0,GDD=0.0,TOD=0.0,FOD=0.0,EPP=0.001,Length=(1.2),beam_dia=11.7*0.2, n2=2.7*10**(-20),beta2=36.0,beta3=27.47,beta4=0.0,gaussian_beam=True,central_wavelength=800,speed_of_light=299700):
    #profile =g for gaussian f for flat top
    freq, intensity = generateIntensityGaussianSpectrum(number_bins = number_bins,freq_range = freq_range, freq_fwhm = freq_fwhm,center_freq = speed_of_light/central_wavelength)

    pulse = createPulseSpecNLO(freq=freq,intensity=intensity,phase=freq*0.0,central_wavelength=central_wavelength,EPP=EPP)
    pulse.chirp_pulse_W(GDD*10**(-6),TOD*10**(-9),FOD*10**(-12))
    fiber = createFiberNLO(central_wavelength=central_wavelength,Length=Length, beam_dia=beam_dia, n2=n2,beta2=beta2,beta3=beta3*0.001,beta4=beta4*0.001**2,gaussian_beam=gaussian_beam)

    return pulse, fiber

def runNLOSim(pulse,fiber,Steps=100,Raman=False,Steep=False,USE_SIMPLE_RAMAN=True):

    evol = pynlo.interactions.FourWaveMixing.SSFM.SSFM(local_error=0.005, USE_SIMPLE_RAMAN=USE_SIMPLE_RAMAN,
                 disable_Raman              = np.logical_not(Raman),
                 disable_self_steepening    = np.logical_not(Steep))

    y, AW, AT, pulse_out = evol.propagate(pulse_in=pulse, fiber=fiber, n_steps=Steps)
    pulse_out.__class__ = spym_pulse
    pulse_out.__init__()
    pulse_out.AW_hist = deepcopy(pulse.AW_hist)
    pulse_out.AT_hist = deepcopy(pulse.AT_hist)
    pulse_out.y_hist = deepcopy(pulse.y_hist)
    pulse_out.fiber_hist = deepcopy(pulse.fiber_hist)

    pulse_out.AW_hist.append(AW)
    pulse_out.AT_hist.append(AT)
    pulse_out.y_hist.append(y)
    pulse_out.fiber_hist.append(fiber)
    return y, AW, AT, pulse_out

def runNLOArray(pulse,fiber,Steps=100,Raman=False,Steep=False,USE_SIMPLE_RAMAN=True):

    if type(fiber) == list:
        pulse_out = pulse
        for fiber_step in fiber:
            y,AW,AT,pulse_out = runNLOSim(pulse_out,fiber_step,Steps=Steps,Raman=Raman,Steep=Steep,USE_SIMPLE_RAMAN=USE_SIMPLE_RAMAN)

    else:
        y,AW,AT,pulse_out = runNLOSim(pulse,fiber,Steps=Steps,Raman=Raman,Steep=Steep,USE_SIMPLE_RAMAN=USE_SIMPLE_RAMAN)


    return y, AW, AT, pulse_out

"""General Funtions"""

def get_B(AT, beam_dia, y, wavelength=800, n2=2.7*10**(-20),gaussian_beam=True ):
    if gaussian_beam==True:
        intensity = 2*np.abs(AT)**2/(np.pi*(beam_dia/2000)**2)
    else:
        intensity = np.abs(AT)**2/(np.pi*(beam_dia/2000)**2)

    max_intensity = np.amax(intensity,axis=0)
    wavelength = wavelength*10**(-9)
    return 2*np.pi/(wavelength)*(n2)*np.inner(max_intensity,np.diff(y))

def ftl_calc(pulse,output_pulse=0,calc_fwhm=1,min_percent=0.0):
    spectrum = np.abs(pulse.AW)**2
    spectrum[spectrum/max(spectrum)<min_percent] = 0
    spectrum = spectrum - spectrum[spectrum>0][0]
    spectrum[spectrum<0] = 0
    ftl_pulse = createPulseSpecNLO(pulse.F_THz,spectrum,pulse.F_THz*0,EPP=pulse.calc_epp())
    t_int = np.abs(ftl_pulse.AT)**2/max(np.abs(ftl_pulse.AT)**2)
    fwhm = 1000*(ftl_pulse.T_ps[t_int>0.5][-1] - ftl_pulse.T_ps[t_int>0.5][0])
    if output_pulse == 1:
        if calc_fwhm == 1:
            return ftl_pulse,fwhm
        else:
            return ftl_pulse
    else:
        return fwhm

def plot_spectrum(pulse,ax=0,axP=0,cutoff_percent=0.0,phase=0):
    spectrum = np.abs(pulse.AW)**2
    cropIndex = (spectrum/max(spectrum)>cutoff_percent)

    spectrum = spectrum[cropIndex]
    freq = pulse.F_THz[cropIndex]
    if ax==0:
        fig, ax = plt.subplots()
    ax.plot(freq,spectrum,label='Spectrum')

    if phase == 1:
        phase = np.unwrap(np.angle(pulse.AW))[cropIndex]
        print(phase)
        phase = phase - phase[int(phase.shape[0]*0.5)]
        if axP == 0:
            axP = ax.twinx()

        axP.plot(freq,phase,'--',label='Phase')
        axP.set_ylim([-15,15])
        return ax,axP
    return ax

def load_frog(fileName,EPP=0.001,cen_wave=800,freq_range=2000.0,number_of_bins=2**11,flip_phase = 0):
    sol = 299700.0
    data = np.loadtxt(fileName)
    wavelength = data[data[:,0]>0,0][::-1]
    w_intensity = data[data[:,0]>0,1][::-1]
    w_phase = data[data[:,0]>0,2][::-1]

    freq = (sol/wavelength)[::-1]
    f_intensity = (w_intensity*wavelength**2)[::-1]
    f_intensity = f_intensity/max(f_intensity)
    f_phase = (w_phase)[::-1]
    f_phase = f_phase - f_phase[int(f_phase.shape[0]*0.5)]

    i_freq = np.linspace(sol/cen_wave - freq_range/2,sol/cen_wave + freq_range/2,number_of_bins)
    i_intensity = np.interp(i_freq,freq,f_intensity,left=0,right=0)
    i_intensity = i_intensity - i_intensity[i_intensity>0][0]
    i_intensity[i_intensity<0]=0
    i_phase = np.interp(i_freq,freq,f_phase)
    if flip_phase == 0:
        pulse = createPulseSpecNLO(i_freq,i_intensity,i_phase,central_wavelength=cen_wave,EPP=EPP)
    else:
        pulse = createPulseSpecNLO(i_freq,i_intensity,-i_phase,central_wavelength=cen_wave,EPP=EPP)

    return pulse

def flame_calib():
    import pkgutil
    import glob

    data = pkgutil.get_data(__name__, 'data/FlameTotalCalibration.txt').decode('utf8')


    data = data.split('\n')  #needs to be \r\n for windows \n for linux

    newdata = []
    for idx in range(len(data)-1):
        newdata.append(np.asarray(data[idx].split(' ')).astype(float))

    newdata = np.asarray(newdata)


    return newdata

def load_spec(file_list,bkg_list=[],boxcar=1,wavelength_limits=[0.0,1000.0],wavelength_shift=0.0,percent_cutoff=0.0,freq_range=2000.0,number_of_bins=2**11,EPP=0.001,cen_wave=800.0,calibration=True):
    #boxcar is distance it averages to one side. ie if boxcar =1 then it is average over 3 pixels.(+- 1 pixel)
    #Loading calibration files for flame spectrometer



    #Averaging over all spectrum files
    for idx, name in enumerate(file_list):
        data = np.loadtxt(name,skiprows=14)
        rawspec = data[:,1]
        if idx == 0:
            wavelength = data[:,0]
            spec = 0.0*data[:,0]
            if len(bkg_list) > 0:
                bkg = 0.0*data[:,0]


    	#Does boxcar averaging to help smooth data
        if boxcar > 0:
            boxcar = int(boxcar)
            temp_spec = rawspec*0.0
            for ii in range(temp_spec.shape[0]-2*boxcar):
                temp_spec[ii+boxcar] = np.mean(rawspec[(ii):(ii+2*boxcar)])

            rawspec = temp_spec
        spec += rawspec
    	#Interpulating calibration file to flame output
    spec = spec/len(file_list)

    	#If background  is given, finds average background
    if len(bkg_list)>0:
        for idx, name in enumerate(bkg_list):
            data = np.loadtxt(name,skiprows=14)
            rawbkg = data[:,1]



        	#Does boxcar averaging to help smooth data
            if boxcar > 0:
                boxcar = int(boxcar)
                temp_bkg = rawbkg*0.0
                for ii in range(temp_bkg.shape[0]-2*boxcar):
                    temp_bkg[ii+boxcar] = np.mean(rawbkg[(ii):(ii+2*boxcar)])

                rawbkg = temp_bkg
            bkg += rawbkg
        bkg = bkg/len(bkg_list)
        spec -= bkg


#        for idx, name in enumerate(bkg_list):
#            if idx == 0:
#                bkg = np.loadtxt(name,skiprows=14)[:,1]
#            else:
#                bkg += np.loadtxt(name,skiprows=14)[:,1]
#
##Does boxcar averaging to help smooth data
#
#
#    if boxcar > 0:
#        boxcar = int(boxcar)
#        temp_spec = spec*0.0
#        if len(bkg_list) > 0:
#            temp_bkg = bkg*0.0
#        for ii in range(spec.shape[0]-2*boxcar):
#            temp_spec[ii+boxcar] = np.mean(spec[(ii):(ii+2*boxcar)])
#            if  len(bkg_list) > 0:
#                temp_bkg[ii+boxcar] = np.mean(bkg[(ii):(ii+2*boxcar)])
#        spec = temp_spec
#        bkg = temp_bkg

    if not(wavelength_shift == 0):
        wavelength += wavelength_shift

    spec -= np.max(spec)*percent_cutoff
    spec[wavelength<np.min(wavelength_limits)] = 0.0
    spec[wavelength>np.max(wavelength_limits)] = 0.0
    spec[spec<=0.0] = 0.0

    if calibration == True:
        calib_data = flame_calib()
        print(calib_data.shape)
        calib_wave = calib_data[:,0]
        calib = np.interp(wavelength,calib_wave,calib_data[:,1],left=0.0,right=0.0)
        spec = spec*calib

    freq = 299700.0/wavelength[::-1]
    fspec = (spec*wavelength**2)[::-1]

    iFreq = np.linspace(-freq_range*0.5,freq_range*0.5,number_of_bins) + 299700.0/cen_wave
    ispec = np.interp(iFreq,freq,fspec,left=0.0,right=0.0)

    pulse = createPulseSpecNLO(iFreq,ispec,iFreq*0.0,central_wavelength=cen_wave,EPP=EPP)
    return pulse



def load_flame(fileName,bkgName=-1,EPP=0.001,cen_wave=800,wave_range=[700,900],freq_range=2000.0,number_of_bins=2**11,noise_cutoff=0.005,wavelength_shift=0.0,average_bins = 0):
    print('~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~\n~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~\n~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~')
    print('This function is DEFUNCT use load_spec instead')
    print('~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~\n~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~\n~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~')

    sol = 299700.0
    average_bins = int(average_bins)
    fileName = np.asarray(fileName)
    bkgName = np.asarray(bkgName)
    w_intensity = 0
    if not(fileName.shape == ()):
        for ii in range(fileName.shape[0]):
            data = np.loadtxt(fileName[ii],skiprows=14)
            w_intensity = w_intensity + data[:,1]
        w_intensity = w_intensity/fileName.shape[0]
    else:
        data = np.loadtxt(fileName,skiprows=14)
        w_intensity = data[:,1]
    wavelength = data[:,0]
    if not(wavelength_shift == 0):
        wavelength = wavelength + wavelength_shift
    freq = (sol/wavelength)[::-1]


    if (bkgName.dtype == 'S66'):
        #imports bkg file and subtracts out
        bkg = 0
        if not(bkgName.shape == ()):
            for ii in range(bkgName.shape[0]):
                data = np.loadtxt(bkgName[ii],skiprows=14)
                bkg = bkg + data[:,1]
            bkg = bkg/bkgName.shape[0]
            w_intensity = w_intensity - bkg
        else:
            data = np.loadtxt(bkgName[ii],skiprows=14)
            bkg = bkg + data[:,1]
    elif bkgName == 0:
        #If no bkg file then take average of
        bkg_ave = np.mean(w_intensity[wavelength<300])
        w_intensity = w_intensity - bkg_ave
    elif (type(bkgName) == int) or (type(bkgName) == float):
        #if passing nonzero number it will subtract out that number
        w_intensity = w_intensity - bkgName

    #Cleaning up the data a bit.

    if average_bins > 0:
        ave = w_intensity
        print(w_intensity.shape[0])
        for ii in range(w_intensity.shape[0]-2*average_bins):
            print(ii,ii+2*average_bins)
            ave[ii+average_bins] = np.mean(w_intensity[(ii):(ii+2*average_bins)])
        print(w_intensity.shape)
        w_intensity = ave

   #removes all values below a noise_cutoff percent
    w_intensity[(w_intensity/max(w_intensity))<noise_cutoff]=0
    #zeros intensity outside of given range
    w_intensity[wavelength<wave_range[0]] = 0
    w_intensity[wavelength>wave_range[1]] = 0
    #sets lowest intensity = 0 to prevent a step
    w_intensity = w_intensity - min(w_intensity > 0)
    #removes any leftover negative counts and sets to zero
    w_intensity[w_intensity<=0] = 0
    f_intensity = (w_intensity*wavelength**2/sol)[::-1]

    i_freq = np.linspace(sol/cen_wave - freq_range/2,sol/cen_wave + freq_range/2,number_of_bins)
    i_intensity = np.interp(i_freq,freq,f_intensity,left=0,right=0)
    i_intensity = i_intensity - i_intensity[i_intensity>0][0]
    i_intensity[i_intensity<0]=0

    pulse = createPulseSpecNLO(i_freq,i_intensity,i_freq*0.0,central_wavelength=cen_wave,EPP=EPP)

    return pulse

def compare_pulse(pulse1,pulse2,phase=1,output=0,normalize=0,wavelength=0,axis = -1,linestyle1='-',linestyle2='--',label1='pulse 1',label2='pulse 2',linewidth=1):
    freq1 = pulse1.F_THz
    wave1 = 299700/freq1[freq1>0]
    time1 = pulse1.T_ps*1000
    freq2 = pulse2.F_THz
    wave2 = 299700/freq2[freq2>0]
    time2 = pulse2.T_ps*1000

    t_int1 = np.abs(pulse1.AT)**2
    t_int2 = np.abs(pulse2.AT)**2
    f_int1 = np.abs(pulse1.AW)**2
    w_int1 = f_int1[freq1>0]*freq1[freq1>0]**2/299700
    f_int2 = np.abs(pulse2.AW)**2
    w_int2 = f_int2[freq2>0]*freq2[freq2>0]**2/299700
    phase1 = np.unwrap(np.angle(pulse1.AW))
    phase2 = np.unwrap(np.angle(pulse2.AW))
    phase1 = phase1 - phase1[int(phase1.shape[0]*0.5)]
    phase2 = phase2 - phase2[int(phase2.shape[0]*0.5)]

    if normalize == 1:
        t_int1 = t_int1/max(t_int1)
        t_int2 = t_int2/max(t_int2)
        f_int1 = f_int1/max(f_int1)
        f_int2 = f_int2/max(f_int2)
        w_int1 = w_int1/max(w_int1)
        w_int2 = w_int2/max(w_int2)
    if axis == -1:
        fig, (ax1,ax2) = plt.subplots(ncols=2)
    else:
        ax1 = axis[0]
        ax2 = axis[1]

    ax1.plot(time1,t_int1,'r',linestyle = linestyle1 ,label=label1,linewidth=linewidth)
    ax1.plot(time2,t_int2,'b',linestyle = linestyle2, label=label2,linewidth=linewidth)
    ax1.set_title('Temporal Profile')
    ax1.set_xlabel('Time (fs)')
    ax1.set_ylabel('Intensity (a.u.)')
    if wavelength == 1:
        ax2.plot(wave1,w_int1,'r',linestyle = linestyle1,label=label1,linewidth=linewidth)
        ax2.plot(wave2,w_int2,'b',linestyle = linestyle2,label=label2,linewidth=linewidth)

    else:
        ax2.plot(freq1,f_int1,'r',linestyle = linestyle1,label=label1,linewidth=linewidth)
        ax2.plot(freq2,f_int2,'b',linestyle = linestyle2,label=label2,linewidth=linewidth)
    if phase == 1:
        ax2P = ax2.twinx()

        ax2P.plot(freq1,phase1,'r--',linewidth=linewidth)
        ax2P.plot(freq2,phase2,'b--',linewidth=linewidth)

        ax2P.set_title('Spectral Profile')
        ax2P.set_xlabel('Frequency (THz)')
        ax2P.set_ylabel('Intensity (a.u.)')
        if output == 1:
            return ax1,ax2,ax2P

    else:
        ax2.set_title('Spectral Profile')
        if wavelength == 1:
            ax2.set_xlabel('Wavelength (nm)')
        else:
            ax2.set_xlabel('Frequency (THz)')
        ax2.set_ylabel('Intensity (a.u.)')
        if output == 1:
            return ax1,ax2



def flame_ftl(spmFiles,nospmFiles,bkgFiles,EPP=0.001,cen_wave=800,wave_range=[700,900],freq_range=2000.0,number_of_bins=2**11,noise_cutoff=0.005,output = 0,wavelength=0,normalize=1):
    import glob
    spmPulse =  load_flame(glob.glob(spmFiles),bkgName=glob.glob(bkgFiles),EPP=EPP,cen_wave=cen_wave,wave_range=wave_range,freq_range=freq_range,number_of_bins=number_of_bins,noise_cutoff=noise_cutoff)
    nospmPulse =  load_flame(glob.glob(nospmFiles),bkgName=glob.glob(bkgFiles),EPP=EPP,cen_wave=cen_wave,wave_range=wave_range,freq_range=freq_range,number_of_bins=number_of_bins,noise_cutoff=noise_cutoff)


    ax1,ax2 =  compare_pulse(spmPulse,nospmPulse,phase=0,output=1,normalize=normalize,wavelength=wavelength)
    plt.show()

    print('No SPM Pulse: ' + str( fwhm_t(nospmPulse)) + ' fs')
    print('SPM Pulse: ' + str( fwhm_t(spmPulse)) + ' fs')

    print('No SPM Pulse: ' + str( fwhm_f(nospmPulse)) + ' fs')
    print('SPM Pulse: ' + str( fwhm_f(spmPulse)) + ' fs')
    if output == 1:
        return (ax1,ax2), ( fwhm_t(nospmPulse), fwhm_t(spmPulse)),( fwhm_f(nospmPulse), fwhm_f(spmPulse))

def fwhm_t(pulse):
    time = pulse.T_ps*1000
    intensity = np.abs(pulse.AT)**2/max(np.abs(pulse.AT)**2)
    return time[intensity>0.5][-1] - time[intensity>0.5][0]

def fwhm_f(pulse):
    freq = pulse.F_THz
    intensity = np.abs(pulse.AW)**2/max(np.abs(pulse.AW)**2)
    return freq[intensity>0.5][-1] - freq[intensity>0.5][0]


def get_phase(pulse):
    phase = np.unwrap(np.angle(pulse.AW))
    phase = phase - phase[int(phase.shape[0]*0.5)]
    return phase
def fit_phase(pulse,center_freq = 374.625,min_intensity=0.01,fit_order=6, coef=-1):
    #If coef == 'taylor' returns the coef in taylor expansion form
    #If coef == 'poly' it returns it as a polynormial(aka with factorial backed in)
    from scipy.special import factorial
    cropIndex = np.abs(pulse.AW)**2>max(min_intensity*np.abs(pulse.AW)**2)
    phase = get_phase(pulse)
    phase = phase[cropIndex]
    z = np.polyfit(2*np.pi*(pulse.F_THz[cropIndex]-center_freq),phase,fit_order)
    p_fit = np.poly1d(z)
    z = z[::-1]
    if coef == 'poly':
        return p_fit(2*np.pi*(pulse.F_THz-center_freq)), z[2:]
    elif coef == 'taylor':
        phi_coef = z*0
        for ii in range(z.shape[0]):
            if ii == 0 or ii == 1:
                phi_coef[ii] = 0
            else:
                phi_coef[ii] = z[ii]*factorial(ii)*10**(3*ii)
        return p_fit(2*np.pi*(pulse.F_THz-center_freq)), phi_coef[2:]
    else:
        return p_fit(2*np.pi*(pulse.F_THz-center_freq))


def waveToFreqConverter(wavelength, I_wave, speed_of_light=299.7):
    """
    Converts Wavelength Array and corresponding Spectual Intensity array into
    frequency and the corresponding spectral intensity arrays.
    Parameters
    ----------
    wavelength: array size N
        Contains the values of the wavelength for each element of the array.
        For Default speed_of_light value needs to be in nanometers

    w_intensity: array size N
        Contains the values for the intensity corresponding to the wavelength given
        by the same element in the wavelength array.

    speed_of_light: float
        Conversion factor between wavelength and frequency. Must have same length
        units are your wavelength array and whatever your time units are will be
        the inverse units of your frequency.
        By default speed_of_light is given in nm/fs. So your wavelength array
        needs to be in nm and your output frequency will be in fs.




    Returns
    -------
    array
        First array returned is the frequency array of the pulse
    array
        Second array returned is the spectual intensity of the pulse as a function
        of frequency.
    """
    freq = speed_of_light/wavelength
    I_freq = I_wave*wavelength**2/(speed_of_light)

    return freq, I_freq

def save_pulse(pulse=[],name=['pynlo_pulse']):
    import pickle
    for idx, pulse_obj in enumerate(pulse):
        if len(name) == len(pulse):
            pulse_file = open(name[idx] + '.pulse','wb')
            print(pulse[idx])
            pickle.dump(pulse[idx],pulse_file)
        else:
            pulse_file = open(name[0] + '_%d.pulse'%(idx),'wb')
            pickle.dump(pulse[idx],pulse_file)


def load_pulse(filename=[]):
    import pickle
    pulses = []

    for name in filename:
        pulse_obj = open(name,'rb')
        pulses.append(pickle.load(pulse_obj))

    return pulses

def save_fiber(fiber=[],name=['pynlo_fiber']):
    import pickle
    for idx, pulse_obj in enumerate(fiber):
        if len(name) == len(fiber):
            fiber_file = open(name[idx] + '.fiber','wb')
            print(fiber[idx])
            pickle.dump(fiber[idx],fiber_file)
        else:
            fiber_file = open(name[0] + '_%d.fiber'%(idx),'wb')
            pickle.dump(fiber[idx],fiber_file)

def load_fiber(filename=[]):
    import pickle
    fiber = []

    for name in filename:
        fiber_obj = open(name,'rb')
        fiber.append(pickle.load(fiber_obj))

    return fiber

def save_sim(pulseList=[],fiberList=[],name=['pynlo_sim']):
    import pickle
    for idx, pulses in enumerate(pulseList):
        print(pulses)

        if len(fiberList) == len(pulseList):
            fiber = fiberList[idx]
        else:
            fiber = fiberList[0]

        if len(pulses) == 1:
            simulation = (pulses[0],fiber)
        elif len(pulses) == 2:
            simulation = (pulses[0],pulses[1],fiber)

        if len(name) == len(pulseList):
            sim_file = open(name[idx] + '_%d.pynlo'%(idx),'wb')
            pickle.dump(simulation,sim_file)
        else:
            sim_file = open(name[0] + '_%d.pynlo'%(idx),'wb')
            pickle.dump(simulation,sim_file)


def TBP(freq,f_intensity,phase):
    time = np.fft.fftshift(np.fft.fftfreq(freq.shape[0],freq[1]-freq[0]))
    ang_freq = 2*np.pi*freq

    f_field = np.sqrt(f_intensity)*np.exp(1j*phase)
    t_intensity = np.abs(np.fft.fftshift(np.fft.fft(f_field)))**2

    mean_freq = np.mean(ang_freq*f_intensity)/np.mean(f_intensity)
    freqB = (np.mean((ang_freq-mean_freq)**2*f_intensity)/np.mean(f_intensity))
    mean_time= np.mean(time*t_intensity)/np.mean(t_intensity)
    timeB = (np.mean((time-mean_time)**2*t_intensity)/np.mean(t_intensity))
    #return time,t_intensity
    return timeB,freqB

def test():
    print('it loaded2')

def _testPulse():
    freq, intensity = generateIntensityGaussianSpectrum(number_bins = 2**14,freq_range = 2000 , freq_fwhm = 10 ,center_freq = 374.7, peakI = 1)
    phase = freq*0

    pulse = createPulseSpecNLO(freq,intensity,phase,central_wavelength=800,EPP=0.001)
    return pulse

def _testFiber():
    fiber = createFiberNLO(central_wavelength=800,Length=1.0, beam_dia=1.0, n2=2.6*10**(-20),beta2=0.0,beta3=0,beta4=0)
    return fiber

def _testPyNLO():
    pulse = _testPulse()
    fiber = _testFiber()
    y, AW, AT, pulse_out = runNLOSim(pulse,fiber)
    ax = pulse_out.plot_pulse()
    ax = pulse.plot_pulse(ax=ax,colors=['r','r'])
    ax[0].set_xlim([-100,100])
    ax[1].set_xlim([200,600])
    print('Initial FTL: ' + str(pulse.ftl()))
    print('Final FTL: ' + str(pulse_out.ftl()))
    return pulse_out,ax

def _testLoad():
    print('it worked!9')

def _testTBP(shape='gaussian',width = 15.0,freqLength=10000.0, bins = 2**17,cen_freq=374.0,GDD=0.0,TOD=0.0,FOD=0.0):
    freq = np.linspace(-freqLength*0.5,freqLength*0.5,bins) + cen_freq

    if shape == 'gaussian':
        #f_intensity = 1/(width*np.sqrt(2*np.pi))*np.exp(-0.5*(freq-cen_freq)**2/width**2)
        f_intensity = np.exp(-4*np.log(2)*(freq-cen_freq)**2/(width)**2)

        ang_freq = 2*np.pi*(freq-cen_freq)*10**(-3)
        phase = 0.5*GDD*ang_freq**2 + TOD/6*ang_freq**3 + FOD/24*ang_freq**4
        true_TBP = 0.441

    timeB,freqB = TBP(freq-cen_freq,f_intensity,phase)

    print('Temporal Width: ' + str(2.355*np.sqrt(timeB)))

