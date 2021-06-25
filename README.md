# spym

A wrapper for the code [PyNLO](https://github.com/pyNLO/PyNLO). The code is used to run 1D simulations for the nonlinear effect self-phase modulation using the split-step Fourier Method. In addition to the features normally used in PyNLO this code adds additional functions to help automate some calculations/conversions and abstract the pulse objects a bit more.

Effects that can be modeled include:

* Linear Material Dispersion(beta2,beta3,beta4)
* Kerr Nonlinearity (n2)
* Self-steepening
* Delayed Raman Effect

The package is centered around two main objects, the `pulse` and `fiber` objects.

The pulse object contains infomation tied to the laser light itself, such as intensity and phase infomation after propegation.

The fiber object contains infomation tied to the media that the pulse is interacting with, such as the linear and nonlinear responses.

# Common Functions

A list of important package functions are:

* `spym.generateIntensityGaussianSpectrum()`: Generates a pulse object assuming a gaussian spectral profile.
* `spym.createPulseSpecNLO()`: Generates a pulse object from an arbitrary spectrum and spectral phase.
* `spym.createFiberNLO()`: Generates a fiber object used
* `spym.createFSFiber()`:
* `spym.runNLOSim()`:

A list of common pulse object functions are:
* `pulse.fwhm_t()`: Calculates the pulse's current temporal full-width at half-max value in fs
* `pulse.fwhm_f()`: Calculates the pulse's current spectral full-width at half-max value in THz
* `pulse.ftl()`:    Calculates the pulse's current temporal transform limited FWHM pulse duration
* `pulse.add_phase()`: Adds spectral phase to the pulse in the form of a list of taylor series coefficients starting with GDD, in units of fs^n
* `pulse.zero_phase()`: Removes all spectral phase from the pulse, making it the transform limited pulse
* `pulse.compress()`: Optimizes the GDD value of the pulse to generate the highest intensity.


# Examples
```
import spym
import matplotlib.pyplot as plt

#~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
# Defining simulation parameters
#~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

GDD = 0.0                 #Groupd delay dispersion in fs^2
TOD = -10000.0            #Third order dispersion in fs^3
FOD = 0.0                 #Fourth order dispersion in fs^4
Length = 1.2              #Sets thickness of material in mm
beam_dia = 11.7*4.5/20.0  #Sets the 1/e^2 beam diameter fo the beam
EPP = 0.001               #Energy of the beam in mJ
ftl_duration = 30.0       #Tranform limited pulse duration
Raman = True              # Boolean to include Delayed Raman Effect
Steep = True              # Boolean to include self-steepening
central_wavelength = 800.0
#~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
# Defining simulation objects
#~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
#Create a pulse object and fused silica fiber object based on above parameters.
pulse_0 = spym.generateIntensityGaussianSpectrum(freq_fwhm = 441.0/ftl_duration,center_freq=299700.0/central_wavelength,phase_coef=[GDD,TOD,FOD],EPP=EPP)
fiber = spym.createFSFiber(Length=Length, beam_dia=beam_dia)

#~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
# Running simulation
#~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
y,AW,AT,pulse_out = spym.runNLOSim(pulse_0,fiber,Raman=Raman,Steep=Steep)
```

The output of the `runNLOSim` function are:
* `y` - the 1D array containing the propegation distances for the simulation
* `AW` -  The 2D complex number array containing the spectral electric fields as a function of propegation.
* `AT` -  The 2D complex number array containing the temporal electric fields as a function of propegation.
* `pulse_out` - A new pulse object created from the electric fields after all of the propegation. 


