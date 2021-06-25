import unittest
import spym
import numpy as np


class test_spym(unittest.TestCase):

    @classmethod
    def setUpClass(cls):
        '''
        Defining varaibles which will be used between multiple tests
        '''
        #constants for total simulations
        cls.SOL = 299700.0 #speed of light in nm/fs
        cls.base_epp = 0.001
        cls.center_freq = 374.7
        cls.center_wave = cls.SOL/cls.center_freq
        cls.number_steps = 20

        #constants for pulse
        cls.phase_coef = [550.0,10000.0,-10.0**6]
        cls.number_bins = 2**12
        cls.freq_range = 4000.0
        cls.freq_fwhm = 14.7


        #constants for media
        cls.Length = 1.0
        cls.beam_dia = 3.0
        cls.n2 = 2.5*10**(-20)
        cls.beta2 = 36.1

        #creating base fibers and pulses
        cls.base_pulse = spym.generateIntensityGaussianSpectrum(
                number_bins = cls.number_bins,freq_range = cls.freq_range , freq_fwhm = cls.freq_fwhm , \
                center_freq = cls.center_freq, phase_coef=[0.0] , \
                EPP=cls.base_epp)
        cls.dispersion_fiber = spym.createFSFiber(central_wavelength=cls.center_wave,Length=cls.Length, \
                                beam_dia=cls.beam_dia, n2=cls.n2,beta2=cls.beta2,beta3=0.0,\
                                beta4=0.0,gaussian_beam=True,alpha=0)
        cls.no_dispersion_fiber = spym.createFSFiber(central_wavelength=cls.center_wave,Length=cls.Length, \
                                beam_dia=cls.beam_dia, n2=cls.n2,beta2=0.0,beta3=0.0,\
                                beta4=0.0,gaussian_beam=True,alpha=0)

        _,_,_,cls.pulse_nodisp = spym.runNLOSim(cls.base_pulse, cls.no_dispersion_fiber,Steps=cls.number_steps,Raman=False,Steep=False)
        _,_,_,cls.pulse_disp = spym.runNLOSim(cls.base_pulse, cls.dispersion_fiber,Steps=cls.number_steps,Raman=False,Steep=False)


        #Defining important constants from pulses
        cls.calc_ftl_fwhm = 441/cls.freq_fwhm
        cls.dT = cls.base_pulse.dT()
        cls.dF = cls.base_pulse.dF()

        #calculating analytical values for system
        cls.calc_peak_intensity = 2*0.94*cls.base_epp/(441/cls.freq_fwhm*10**(-15)*np.pi*(cls.beam_dia/2*0.1)**2)
        cls.calc_db20_width = 2.57756*cls.freq_fwhm
        cls.calc_b_integral = 2*np.pi/(cls.center_wave*10**(-9))*(cls.n2)*(cls.Length*10**(-3))*cls.calc_peak_intensity*10**4

    def test_peak_power(self):
        '''
        Ensuring peak power from object and calculates agree withing half a percent
        '''
        pulse_peak_power = np.max(np.abs(self.base_pulse.AT)**2)
        calc_peak_power  = 0.94*self.base_epp/(441/self.freq_fwhm*10**(-15))
        assert(np.abs(pulse_peak_power/calc_peak_power - 1) <= 0.005)

    def test_ftl_time_fwhm(self):
        '''
        ensures the temporal width of the pulse is same as predicted up to grid resolution
        both sides can be off by 1 so total acceptable error is 2*dT
        '''
        pulse_fwhm = self.base_pulse.fwhm_t()
        assert(np.abs(pulse_fwhm-self.calc_ftl_fwhm) <= 2.0*self.dT)

    def test_freq_fwhm(self):
        '''
        ensures the spectral width of the pulse is same as defined up to grid resolution
        both sides can be off by 1 so total acceptable error is 2*dF
        '''
        pulse_fwhm = self.base_pulse.fwhm_f()
        assert(np.abs(pulse_fwhm-self.freq_fwhm) <= 2.0*self.dF)

    def test_GDD_fwhm(self):
        '''
        ensures the temporal width of the pulse is same as predicted up to grid resolution
        after adding known GDD value. Compared to predicted value from APE-Berlin
        '''
        pulse = self.base_pulse.copy()
        pulse.add_phase([1000])
        positive_GDD_fwhm = pulse.fwhm_t()
        pulse.add_phase([-2000])
        negative_GDD_fwhm = pulse.fwhm_t()
        APE_fwhm = 97.17
        assert(np.abs(positive_GDD_fwhm-APE_fwhm) <= 2.0*self.dT and np.abs(negative_GDD_fwhm-APE_fwhm) <= 2.0*self.dT)

    def test_ftl_calculation(self):
        '''
        ensure correct FTL is calculated even with phase added
        '''
        pulse = self.base_pulse.copy()
        pulse.add_phase(self.phase_coef)
        ftl_fwhm = pulse.ftl()
        assert(np.abs(ftl_fwhm - self.calc_ftl_fwhm) <= 2.0*self.dT)

    def test_zero_phase(self):
        '''
        ensuring phase is set to zero where specturm exists
        '''
        pulse = self.base_pulse.copy()
        pulse.add_phase(self.phase_coef)
        pulse.zero_phase()
        phase = pulse.get_phase()
        If = np.abs(pulse.AW)**2
        indices = If/np.max(If) > 0.01
        assert(np.all(np.abs(phase[indices]) - 0.001))

    def test_peak_intesnity(self):
        '''
        ensures the peak value of intenisty is same as predicted for a
        gaussian spatial profile
        '''
        peak_intensity = self.base_pulse.peakI(beam_dia=self.beam_dia)
        assert(np.abs(peak_intensity/self.calc_peak_intensity -1) <= 0.01)

    def test_calculate_db20(self):
        '''
        ensures the calculated db20 witdh matches value calculated from fwhm
        '''
        db20_width = self.base_pulse.db20_f()
        assert(np.abs(db20_width -self.calc_db20_width) <= 2*self.dF)

    def test_compress(self):
        '''
        Ensuring the compress function is able to correctly calculate and remove
        the added phase to get back to the initial value
        '''
        pulse = self.base_pulse.copy()
        initial_fwhm = pulse.fwhm_t()
        initial_power = np.max(np.abs(pulse.AT)**2)
        pulse.add_phase([-283])
        pulse.compress()
        final_fwhm = pulse.fwhm_t()
        final_power = np.max(np.abs(pulse.AT)**2)
        assert(np.abs(initial_fwhm - final_fwhm ) <= 2*self.dT)
        assert(np.abs(initial_power/final_power-1) <= 0.005)

    def test_calc_B(self):
        '''
        ensures the B-intgral calculates agree with analytical equation in
        dispersionless media
        '''
        B_integral = self.pulse_nodisp.calc_B()[0]
        assert(np.abs(B_integral/self.calc_b_integral - 1) <= 0.005)

    def test_TBP(self):
        '''
        tests time, bandwidth product test_ftl_calculation
        value compared to the prediction made from APE-Berlin
        '''
        pulse = self.base_pulse.copy()
        pulse.add_phase([562.23])
        TBP = pulse.TBP(method='fwhm')
        assert(np.abs(TBP/0.88 - 1) <= 0.01)

    def test_zero_If(self):
        pulse = self.base_pulse.copy()
        wavelength_limits = [self.center_freq-3.0,self.center_freq+3.0]

        non_zero_indices = (pulse.F_THz>wavelength_limits[0])*(pulse.F_THz<wavelength_limits[1])
        zero_indices = 1 - non_zero_indices

        pulse.zero_If(wavelength_limits[0],wavelength_limits[1])
        spectrum = np.abs(pulse.AW)**2

        assert(np.all(spectrum[zero_indices] == 0.0))
        assert(np.all(spectrum[non_zero_indices] > 0.0))

    def test_fit_phase(self):
        '''
        testing to ensure that the phase fitting function is correctly predicting
        the predefined phase coefficients
        '''
        pulse = self.base_pulse.copy()

        pulse.add_phase(self.phase_coef)
        fit_coef = pulse.fit_phase(fit_order=3)
        assert(np.all(np.abs(self.phase_coef/fit_coef -1) < 0.005))

if __name__ == "__main__":
    unittest.main()
