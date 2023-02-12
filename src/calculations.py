import numpy as np

from numpy import log10, pi
# Prefer numpy over math as handling list/arrays of values works in numpy
# Produces same result for single values

import scipy.constants as constants  # https://docs.scipy.org/doc/scipy/reference/constants.html
"""RF functions and classes in Python."""

### !rm -rf latexify_py
### !git clone https://github.com/odashi/latexify_py -b develop
### !pip install -e latexify_py




power = [10000, 1000, 100, 10, 1, 0.1, 0.01, 0.001]
# print(power.to('dBm')) ?
10 * log10(power) + 30

'''
BS EN 61000-4-6:2014
Conducted Immunity Cal                         150
SigGen - Cable - Amp - Cable - Sampler - Pad - CDN
                                Cable          100
                                Pad            Pad(s)
                                Head           Head
                                Meter          Meter
'''
'''
Conducted Immunity Test                        AE
SigGen - Cable - Amp - Cable - Coupler - Pad - CDN
                                Cable          EUT
                                Pad
                                Head
                                Meter
'''

'''
BS EN 61000-4-3:2006+A2:2010
Radiated Immunity Cal
SigGen - Cable - Amp - Cable - Coupler - Antenna - )) (( - Field Strength Meter
                               Cable
                               Pad
                               Head
                               Meter
'''
'''
Radiated Immunity Test
SigGen - Cable - Amp - Cable - Coupler - Antenna - )) (( - Field Strength Meter
                               Cable
                               Pad
                               Head
                               Meter
'''

# ureg = pint.UnitRegistry()
# ureg.default_format = '~P'
## ureg.default_format = '~W'

# Q_ = ureg.Quantity

# ureg.define('emf = 0.5 * volt')  # half of one volt is one emf
# ureg.define('dB = []')  # Whilst Pint has NotImplemented dB units
# ureg.define('dBm = []')  # Whilst Pint has NotImplemented dBm units
# ureg.define('dBμA = []')  # Whilst Pint has NotImplemented dBμA units
# ureg.define('dBμV = []')  # Whilst Pint has NotImplemented dBμV units


# uref.define('dBu = 0.7746 volt')  # log20
# uref.define('dBV = 1 volt')  # log20

'''
def wattstoemf(watts, *, ohms=50):
    """."""
    return np.sqrt(Q_(watts, 'W')*Q_(ohms, 'ohm')).to('emf')

def emftowatts(emf, *, ohms=50):
    return ((Q_(emf, 'emf').to('V')) ** 2 / Q_(ohms, 'ohms')).to('W')
'''

class dBDeltaFromRatioOf:

    @staticmethod
    def volt(ratio):
        """Calculate the dB equivalent of the voltage ratio."""
        return 20*log10(ratio)

    @staticmethod
    def watt(ratio):
        """Calculate the dB equivalent of the power ratio.

        Useful for correcting readings from rf power heads"""
        return 10*log10(ratio)

    '''
    def correctpowerreadingwithfactor(reading, factor):
    """."""
    return reading + watt(factor)
    '''

    @staticmethod
    def amp(ratio):
        """Calculate the dB equivalent of the current ratio."""
        return 20*log10(ratio)


class NewAmplitudeFromdBRatioOf:

    @staticmethod
    def watt(dBDelta, watt):
        return 10 ** ((dBDelta + WattTo.dBW(watt)) / 10)


    @staticmethod
    def volt(dBDelta, volt):
        return 10 ** ((dBDelta + VoltTo.dBV(volt)) / 20)

    @staticmethod
    def amp(dBDelta, amp):
        return 10 ** ((dBDelta + AmpTo.dBA(amp)) / 20)


class Antenna:
    
    @staticmethod
    def near_field_distance(d, f):
        d = d
        λ = constants.speed_of_light / f
        reactive_near_field = 0.62*(d**3/λ)**0.5
        radiating_near_field_distance = (2*d**2)/λ

        return f'Reactive Near Field Distance: {reactive_near_field:.8f} m > Radiating Near Field Distance (Fresnel region) < {radiating_near_field_distance:.8f} m > Far Field (Greater than this distance)'
        # https://www.everythingrf.com/rf-calculators/antenna-near-field-distance-calculator

    @staticmethod
    def Fresnel_Zone(m, f):
        # https://www.everythingrf.com/rf-calculators/fresnel-zone-calculator
        # m, f = 5000, 1e9

        FresnelZone = 17.31 * ( (m/1000) /4 * (f/1e9) )**0.5
        return round(FresnelZone, 3)  # in meters

    @staticmethod
    def frequency(wavelength):
        # λ (Lambda) = Wavelength in meters
        # c = Speed of Light (299,792,458 m/s)
        # f = Frequency (MHz)
        # wavelength = length  # m

        # f = c/λ
        Hz = constants.speed_of_light / wavelength
        return Hz

    @staticmethod
    def length(frequency):
        m = constants.speed_of_light / frequency  # m
        return m

    @staticmethod
    def farfield_dipole(wavelength):
        """Dipole and log-perodic antenna."""
        farfield = wavelength / (2 * constants.Pi)
        return farfield

    @staticmethod
    def farfield_horn(aperture, wavelength):
        """."""
        farfield = (2 * aperture**2) / wavelength 
        return farfield

    @staticmethod
    def dBfordistancechange(distanceref, distanceto):
        return 20 * log10(distanceto / distanceref)

    @staticmethod
    def dBi___numericgain(dBi):
        return 10.**(dBi/10)

    @staticmethod
    def numericgain___dBi(numericgain):
        return 10*log10(numericgain)



# From http://emc.toprudder.com/formulas2.pdf

class FieldStrength:

    @staticmethod
    def V_m___dBuV_m(v_m):
        return 20 * np.log10(V_m) + 120

    @staticmethod
    def dBuV_m___V_m(dBuV_m):
        return 10**((dBuV_m - 120) / 20)

    @staticmethod
    def dBm_m2___dBuV_m(dBm_m):
        return dBm_m + 115.8

    @staticmethod
    def dBuV_m___dBm_m2(dBuV_m):
        return dBuV_m - 115.8 

    @staticmethod
    def dBuA_m___dBuV_m(dBuA_m):
        return dBuA_m - 51.5

    @staticmethod
    def dBuV_m___dBuA_m(dBuA_m):
        return dBuA_m + 51.5

    @staticmethod
    def dBuA_m___dBpT(dBuA_m):
        return dBuA_m + 2

    @staticmethod
    def dBpT___dBuA_m(dBpT):
        return dBpT - 2 

    @staticmethod
    def V_m___W_m2(V_m):
        return (V_m**2) / 377

    @staticmethod
    def W_m2___V_m(W_m2):
        return (W_m2 * 377) ** 0.5

    '''
    @staticmethod
    def fieldstrength_Watts_for_V_m(V_m, dBi, meters):  # fieldstrength_Watts_for_Vm
        return ((V_m * meters)**2) / (30 * (10.**(dBi/10)))  # Watt
    @staticmethod
    def fieldstrength_power_for_V_m(V_m, gain, meters):
        return ((V_m * meters)**2)/(30 * gain)


    @staticmethod
    def givenWGDist(watts, gain, distance):
        return Q_(np.sqrt(30*watts*gain)/meters, 'V/m')

    @staticmethod
    def givenWdBiDist(watts, dBi, distance):
        return Q_(np.sqrt(30*watts*(10**(dBi/10)))/meters, 'V/m')

    @staticmethod
    def neededpowergforV_m(V_m, gain, meters):
        return ((V_m * meters)**2)/(30 * gain)

watt, gain, meters = 10, 3, 3
V_m = (30 * watt * gain)**0.5 / meters
V_m

watt, dBi, meters = 10, 3, 3
gain = 10 ** (dBi / 10)  # copyof
V_m = (30 * watt * gain)**0.5 / meters
V_m

V_m, gain, meters = 18, 3, 3
watt = (V_m * meters)** 2 / (30 * gain)

V_m, dBi, meters = 18, 3, 3
gain = 10 ** (dBi / 10)  # copyof
watt = (V_m * meters)** 2 / (30 * gain)

watt

    '''


class AFTo:

    @staticmethod
    def dBi(AF, MHz):
        return 20 * np.log10(MHz) - AF - 29.79


    @staticmethod
    def gain(AF, MHz):
        pass


class dBiTo:

    @staticmethod
    def AF(dBi, MHz):
        return 20 * np.log10(MHz) - dBi - 29.79

    @staticmethod
    def gain(dBi):
        return 10 ** (dBi / 10)

    
class GainTo:

    @staticmethod
    def dBi(Gain):
        return 10 * np.log10(Gain)

    @staticmethod
    def AF(Gain, MHz):
        return 20 * np.log10(MHz) - 10 * np.log10(gain) - 29.79 


class Magnetic:

    @staticmethod
    def woundcoilfluxdensity(turns, amps, radiusm):
        return (4*np.pi*turns*amps)/(log20(radiusm))

class Clamp:
    
    @staticmethod
    def Z(height, diamater):  # mm
        z = 60 * np.arccosh((2*height)/diamater)

        return round(z, 2)

# %%
class SourceTests():
    
    def pulse(on, off):
        # Pulse depth (dBc)
        return on - off
    
    def harmonics(a, b, c):
        # a, b, c = -3.55, -63, -61.2 
        return round(a - b, 2), round(a - c, 2)
    
    def SpectralMarkerAM(LSB, CF, USB):
        
        # AM Lin
        # Lower SB-Fc AM % = (LSB / Fc) * 200
        # Upper SB-Fc AM % = (USB / Fc) * 200
        # Avg AM % = (Lower SB AM % + Upper SB AM % ) / 2 
        # LSB, CF, USB = 2.443, 6.166, 2.432
        LSBAM, USBAM = (LSB / CF) * 200, (USB / CF) * 200
        # print(LSBAM, USBAM)
        AM = (LSBAM + USBAM) / 2
        # print(AM)
        return AM
    
    def linearity(ref, point):
        pass # return ref - point

#def AMSidebands(Fc, fm):
#    return {'LSB': (Fc-fm), 'Fc': Fc, 'USB': (Fc+fm)}

# assert am(83.44, 202.8, 83.87) == {'LSB': 82.28796844181458, 'USB': 82.7120315581854, 'Ave': 82.5}
# assert AMSidebands(1e9, 400) == {'Fc': 1000000000.0, 'LSB': 999999600.0, 'USB': 1000000400.0}"""
    

class MagnitudeTests():
    
    def magmag(Ref, UUT, lmt):
        # difference between Ref and UUT less than limit
        # Ref, UUT, lmt = 90.065, 90.07, 0.08400
        # errmag = 
        return abs(Ref - UUT) < lmt

    def targeterr(target, err, lmt):
        # target, err, lmt = 10e6, 0.017, 0.000018
        return (((err-target)/target)*-100)-100 < lmt
    
    def boundreadingbound(lbound, reading, ubound):
        # -1, 0, 1 = True
        # -1, -1, 1 = True
        # -1, 1, 1 = True
        
        # -1, -1.1, 1 = False
        # -1, 1.1, 1 = False
        return lbound <= reading <= ubound
    
    
class Filters():
    
    def bandselectivity(low60, low3, high3, high60):
        # low60, low3, high3, high60 = 885.3E6, 975.3E6, 3.1E9, 3.2E9
        # high3 - low3
        # high60 - low60 
        return (high60 - low60) / (high3 - low3)


class Attenuator():
    def Pi(Z0, attenuation):
        # https://www.qorvo.com/-/media/images/qorvopublic/design-tool/pi-tee-formulas.png?h=227&w=320&la=en&hash=B1702071C86609C889CCF501F987507C34BEE588
        factor = 10**(attenuation/20)
        return {
            'R1': Z0*((factor+1)/(factor-1))**2,
            'R2': (Z0/2)*(1/factor),
        }

    def T(Z0, attenuation):
        return NotImplemented
# Attenuator.Pi(44, 50)


class HarmonicMixer:
    """."""

    def LO(frequency, harmonic, IF=310.7e6):  # LO_given_frequency_harmonic_IF
        """Calculate LO for HarmonicMixer.

        :param frequency: Target center frequency
        :param harmonic: Harmonic of mixer
        :param IF: IF frequency
        :returns: Frequency for LO source
        """
        return (frequency - IF) / harmonic

    def PreselectorVoltage(LO, mode='PSA'):  # frequency_given_LO_harmonic_IF
        """Calculate voltage for Preselector.

        :param frequency: Target center frequency
        :param mode: Config on the Preselector (PSA or 8563E)
        :returns: Frequency for LO source
        """
        calculation = 1.5*(LO/1e9)
        if mode == 'PSA':
            return calculation
        if mode == '8563E':
            return calculation - 0.2054

    def RF(LO, harmonic, IF=310.7e6):
        """Calculate RF frequency.

        :param LO: LO frequency
        :param harmonic: Harmonic of mixer
        :param IF: IF frequency
        :returns: Frequency of RF signal
        """
        return (LO * harmonic) + IF


class SourceMultiplier:
    """."""

    def sourcefrequency_given_frequency_multiplier(frequency, multiplier):
        """."""
        return frequency / multiplier

    def outputfrequency_given_frequency_multiplier(frequency, multiplier):
        """."""
        return frequency * multiplier


class ReflectedPower:
    
    def VSWR2RL(VSWR):
        return(-20 * np.log10((VSWR - 1) / (VSWR + 1)))


    def RL2VSWR(RL):
        return((10 ** (RL / 20) + 1) / (10 ** (RL / 20) - 1))



    def VSWR2Refl(VSWR):  # ectionCoefficient
        return((VSWR - 1)/(VSWR + 1))
    '''
    Γ=10(‐ReturnLoss/20)
    VSWR=(1+|Γ|)/(1‐|Γ|)
    MismatchLoss(dB)=10log(Γ**2)
    ReflectedPower(%)=100*Γ **2
    ReturnLoss(dB)= ‐20log|Γ|
    Γ=(VSWR‐1)/(VSWR+1)
    ThroughPower(%)=100(1‐Γ2)
    '''

    def VSWR_RL(VSWR):
        # VSWR to RL dB
        # i = 1.00001
        # i = 1.4
        return round(20 * np.log10((VSWR+1)/(VSWR-1)), 3)
    
    def VSWR2RL(VSWR):
        return(-20 * np.log10((VSWR - 1) / (VSWR + 1)))


    def RL2VSWR(RL):
        return((10 ** (RL / 20) + 1) / (10 ** (RL / 20) - 1))

    def VSWR2ReflectionCoefficient(VSWR):
        return (VSWR - 1)/(VSWR + 1)


    def test_VSWR2Refl():
        assert VSWR2Refl(1.50) == 0.2



class Trace():

    def lin(start, stop, points):
        return np.linspace(start, stop, points)

    def log(start, stop, points):
        return np.geomspace(start, stop, points)

    def percent(start, stop, percentstep):
        # start, stop = 80e6, 1e9
        # start, stop = 150e3, 230e6
        
        decade = np.log10(stop/start)

        incrementratio = (percentstep / 100) + 1
        points = int(decade / np.log10(incrementratio)) + 1

        # return np.geomspace(start/10, stop/10, points, dtype=int, endpoint=True) * 10
        return np.geomspace(start, stop, points, dtype=int, endpoint=True)
        # decade, len(freq), points, freq / 1e6
    # x = percent(150e3, 230e6, 1)
    #x, len(x)


    def DANL(trace):
        pass
    
    def Signals(trace):
        pass
    
    def envelope(trace):
        mean, maxima, minima, ptp = np.mean(trace), max(trace), min(trace), np.ptp(trace)
        pk_to_mean = pk - mean
        
    def spacing(start, stop, spacing):
        """Number of points to obtain a given spacing."""
        # Need to check if output is a whole number
        # Is the +1 a Keysight thing?
        if (stop-start) % spacing == 0:
            return int(((stop-start)/spacing)+1)
            # return f'Start {start}, Stop {stop}, Step size {spacing}, Points {int(((stop-start)/spacing)+1)}'
        else:
            return ValueError

    def stepsize(start, stop, points):
        # Is the -1 a Keysight thing?
        return (stop-start)/(points-1)
    
    
class Waveguide():
    
    def cutoff(broadwall):
        # Calculate the Waveguide Cut-off Frequency
        # https://www.everythingrf.com/rf-calculators/waveguide-calculator
        broadwall = 5.6896 / 1000  # mm
        fc = constants.speed_of_light / (2 * broadwall)
        f1, f2 = 1.25 * fc, 1.89 * fc
        return f1/1e9, f2/1e9
    
    def what(dunno):
        # radius = 5.69 / 2 / 1000 # mm
        fc = (1.8412 * constants.speed_of_light) / (2*constants.pi * broadwall)
        return fc, fc / 1e9


    def horn_gain(frequency, flaredflangebroad, flaredflangenarrow):
        # frequency = 1e9
        # broad = float(input('Flared flange dimension(broad) in mm : ')) / 1e3
        # narrow = float(input('Flared flange dimension(narrow) in mm : ')) / 1e3
        broad = broad / 1e3
        narrow = narrow / 1e3

        area = narrow * broad
        factor = 3e8 / frequency  # 300000000
        gain = 10 * log10((10 * area) / (factor**2))
        return {
            'Gain [dB]': gain,
            'vertical_beamwidth [deg]': (51 * factor)/narrow,
            'horizontal_beamwidth [deg]': (70 * factor)/broad,
        }

