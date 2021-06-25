
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

    def compress(self,GDDRange=1000.0,bins=256):
        '''
        Finds and removes the GDD of a pulse by optimizing for highest peak intensity.


        Parameters:
            GDDRange (float): The range of GDD values that the pulse optimization
                                occurs over
            bins (int): The number of GDD values that are tested. Increasing to
                large values can take non-neglible compute time.
        '''
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
            pulse object
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
        Shifts temporal profile so first moment of the temporal profile is at t=0
        '''
        self.time_shift(-1.0*np.sum(self.T_ps*np.abs(self.AT)**2)/np.sum(np.abs(self.AT)**2),units='ps')


    def spectral_std(self,center_freq=None):
        '''
        Calculates spectral intensity's second moment of the pulse.
        Inputs:
            center_freq (float): Calculates the spectral deviation based around this value. If no value is given the default value for hte pulse is used.
        Returns:
            Standard Devitation Width (float)
        '''
        if center_freq == None:
            center_freq = np.sum(self.F_THz*np.abs(self.AW)**2)/np.sum(np.abs(self.AW)**2)
            sigma = np.sqrt(np.abs(np.sum((self.F_THz-center_freq)**2*np.abs(self.AW)**2)/np.sum(np.abs(self.AW)**2)))
        return sigma


    def temporal_std(self,center_time=None):
        '''
        Calculates temporal intensity's second moment of the pulse.
        Inputs:
            center_time (float): Calculates the spectral deviation based around this value. If no value is given the default value for hte pulse is used.
        Returns:
            Standard Devitation Width (float)
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
            dF (float): frequency resolution
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


    def save_fields(self,path,filename = 'pulse.txt',dT=0.003,number_bins=2**14):
        '''
        Saves field containing time axis, temporal intenisty, and temporal fields
        at the desired location
        Inputs:
            path (string): Directory where file will be saved
            filename (string): Name of file to be saved
            dT (float): Resolution of interpolation in temporal domain
            number_bins (int): number of spaces that interpolation is done over
        '''
        #dT in ps
        T = np.linspace(-0.5*(number_bins-1)*dT,0.5*(number_bins-1)*dT,number_bins)

        AT_re = np.interp(T,self.T_ps,np.real(self.AT),left=0.0,right=0.0)
        AT_im = np.interp(T,self.T_ps,np.imag(self.AT),left=0.0,right=0.0)

        phase =  np.interp(T,self.T_ps,np.unwrap(np.angle(self.AT)),left=0.0,right=0.0)
        It = np.abs(AT_re+1j*AT_im)**2

        output = np.array([1000.0*T,It,phase,AT_re,AT_im]).T
        np.savetxt(path + filename,output)


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
        Fit infomation of phase
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
            return z[2:]
        elif coef == 'taylor':
            phi_coef = z*0
            for ii in range(z.shape[0]):
                if ii == 0 or ii == 1:
                    phi_coef[ii] = 0
                else:
                    phi_coef[ii] = z[ii]*factorial(ii)*10**(3*ii)
            return phi_coef[2:]
        else:
            print('Invalid coef style')


    def phase_from_coef(self,phase_coef=[0.0]):
        """
        Calculates the phase across the pulses spectral range from an array
        of the taylor series coef starting at n=2 (GDD)
        Inputs:
            phase_coef (list/array)
                    Contains the taylor series expansion coef of the phase. starting
                    at n=2 (GDD)
        Returns
        -------
        numpy array of floats
            The spectral phase across the frequency axis

        """
        from scipy.special import factorial
        ang_freq = 2*np.pi*(self.F_THz-self.center_frequency_THz)*10**(-3)
        phase = ang_freq*0.0
        for idx,coef in enumerate(phase_coef):
            phase += coef/factorial(idx + 2)*ang_freq**(idx+2)
        return phase

    def save_spectrum(self,path, fname='spectrum.csv',delimiter=','):
        """
        Saves spectral infomation of the pulse in a csv
        Inputs:
            path (string): Directory where file will be saved
            fname (string): name of the saved file
            delimiter (string): delimiter character

        """
        data = np.zeros((3,self.F_THz.shape[0]))
        data[0] = self.F_THz
        data[1] = np.abs(self.AW)**2
        data[2] = np.unwrap(np.angle(self.AW))
        np.savetxt(path + fname,data,delimiter=delimiter)


    def clear_history(self):
        """
        Clears history variables. Reset to default.
        """
        self.fiber_hist = []
        self.AW_hist = []
        self.AT_hist = []
        self.y_hist = []
        self.B_int = [0.0]

    def plot(self,ax=[],cutoff_percent=0.0,phase=0,linestyles=['b-','r--'],normalize=1,labels=['Temporal','Spectrum','Phase'],x_axis_wave=False):
        '''
        Plotting pulses
        '''
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

"""Intensity Generating Functions"""

def phase_from_coef(freq,center_freq,phase_coef=[0.0]):
    from scipy.special import factorial
    ang_freq = 2*np.pi*(freq-center_freq)*10**(-3)
    phase = freq*0.0
    for idx,coef in enumerate(phase_coef):
        phase += coef/factorial(idx + 2)*ang_freq**(idx+2)
    return phase


def generateIntensityGaussianSpectrum(number_bins = 2**14,freq_range = 4000.0 , freq_fwhm = 15.0 ,center_freq = 374.7, peakPower = 1,phase_coef=[0.0,0.0,0.0],EPP=0.001):
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

    return createPulseSpecNLO(freq,intensity,phase,central_wavelength=299700.0/center_freq,EPP=EPP)


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


def createFSFiber(central_wavelength=800,Length=1.0, beam_dia=1.0, n2=2.5*10**(-20),beta2=36.1,beta3=27.49,beta4=-11.4335,gaussian_beam=True,alpha=0):
    """
    Creates a fiber object. Default values are values for 800 nm in Fused Silica
    Inputs:
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




def runNLOSim(pulse,fiber,Steps=100,Raman=False,Steep=False,USE_SIMPLE_RAMAN=True):
    '''
    Function to run simulation
    pulse : object
        The pulse object which contains the pulse that you want to propegate through material

    fiber: object
        Fiber object representing the material the pulse will propegate through

    Raman : bool
        Enables delayed raman effect during the simulations. If false no raman effect occurs

    Steep : bool
        Enables self-steepening during the simulations. If false no self-steepening occurs

    USER_SIMPLE_RAMAN : bool
        Determines if PyNLO uses the simple formulation for delayed raman effect


    Returns
    -------
    y : 1d array
        An array containing the positions inside of the material while propegating.
    AW : 2d array
        The spectral fields at each step of the simulation.
    AT
        The temporal fields at each step of the simulation
    pulse_out
        The output pulse of the simulation after SPM

    '''

    evol = pynlo.interactions.FourWaveMixing.SSFM.SSFM(local_error=0.005, USE_SIMPLE_RAMAN=USE_SIMPLE_RAMAN,disable_Raman = np.logical_not(Raman), disable_self_steepening = np.logical_not(Steep))

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



"""General Funtions"""







def load_frog(fileName,EPP=0.001,cen_wave=800,freq_range=2000.0,number_of_bins=2**11,flip_phase = 0):
    '''
    Load FROG file generated from Trebino's code
    Arguments
    ------
    file name : string
        Path to the file of interest
    EPP : float
        Energy of pulse in milliJoules
    cen_wave: floats
        Central wavelength of the pulse
    freq_range : floats
        Range of frequencies the frog file will be interpolated
    number_of_bins : int
        Number of bins that the pulse is interpolated onto
    flip_phase : bool
        Loads FROG with negative phase

    Returns
    -------

    pulse_out : pulse object
        The output pulse loaded from the file
    '''
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
    '''
    Loads the data contained in the FLAME calibration files

    returns
    --------
    2d array
        Outputs a 2d array containing the wavelength and calibration amounts for the flame
    '''
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
    '''
    Loads a pulse object from FlAME spectrum.

    Inputs
    ------
    file_list : list of strings
        List containing all files you want to average over for the Spectrum
    bkg_list : list of strings
        List containing all background files. If empty then no background
        files are subtracted.
    boxcar : int
        width of the boxcar average that occurs to smooth noise
    wavelength limits : list of floats
        the upper and lower limits of the spectra. anything outside the limits is
        set to zero
    wavelength_shift : floats
        Wavelength offset of spectra

    percent_cutoff : floats
        Percent of max value of spectrum. Anything below this value is set to zero
    freq_range : float
        The range of frequencies the data is interpolated to
    number_of_bins : int
        The number of bins the data is interpolated across
    EPP : floats
        The energy of the pulse in milliJoules
    cen_wave : float
        The central wavelength of the pulse in nanometers
    calibration : bool
        If true the included calibration for the FLAME is included.
        If different calibration is required set to False

    Return
    -------
    pulse object
        Returns the pulse object created from the spectrometer files


    '''

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


def compare_pulse(pulse1,pulse2,phase=1,output=0,normalize=0,wavelength=0,axis = -1,linestyle1='-',linestyle2='--',label1='pulse 1',label2='pulse 2',linewidth=1):
    '''
    Plotting two pulses to compare against each other
    '''

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

def save_pulse(pulse=[],name=['pynlo_pulse']):
    '''
    Saves a set of pulses to individual files
    inputs
    ------
    pulses : list of pulse objects
        The set of pulses you want to save
    name : list of strings
        The list of names to save the pulses as. If lists are different sizes
            Then the name defaults to the first value.
    '''
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
    '''
    Loads a saved set of pulses to a list of pusles
    inputs
    ------
    filename : list of strings
        The list of names to load the pulses from.
    returns
    ------
    pulses : list of pulse objects
        The set of pulses loaded from the files
    '''
    import pickle
    pulses = []

    for name in filename:
        pulse_obj = open(name,'rb')
        pulses.append(pickle.load(pulse_obj))

    return pulses

def save_fiber(fiber=[],name=['pynlo_fiber']):
    '''
    Saves a set of fibers to individual files
    inputs
    ------
    fiber : list of fiber objects
        The set of fiber you want to save
    name : list of strings
        The list of names to save the fiber as. If lists are different sizes
            Then the name defaults to the first value.
    '''
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
    '''
    Loads a saved set of pulses to a list of fibers
    inputs
    ------
    filename : list of strings
        The list of names to load the fibers from.
    returns
    ------
    pulses : list of fiber objects
        The set of fibers loaded from the files
    '''
    import pickle
    fiber = []

    for name in filename:
        fiber_obj = open(name,'rb')
        fiber.append(pickle.load(fiber_obj))

    return fiber
