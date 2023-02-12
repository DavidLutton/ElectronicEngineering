import numpy as np

from numpy import sqrt, log10, pi, arctan2  # arccosh, conj
# Prefer numpy over math as handling list/arrays of values works in numpy
# Produces same result for single values

import scipy.constants as constants
# https://docs.scipy.org/doc/scipy/reference/constants.html


class VoltTo():

    @staticmethod
    def amp(volt):
        Z = 50
        return volt / Z

    @staticmethod
    def watts(volt):
        Z = 50
        return volt**2 / Z

    @staticmethod
    def dBm(volt):
        """Volt to dBm."""
        return 20 * log10(volt) + 13
        # return WattsTo.dBm(voltTo.watts(volt))

    @staticmethod
    def dBuA(volt):
        """Volt to dBuA."""
        return 20 * log10(volt) + 86

    @staticmethod
    def dBuV(volt):
        return 20 * log10(volt) + 120  # checked against R&S unit converter

    @staticmethod
    def dBV(volt):
        return 20 * log10(volt)

    @staticmethod
    def watt(volt):
        Z = 50
        return volt ** 2 / Z


class dBmTo():

    @staticmethod
    def watt(dBm):
        """.

        :param dBm: dBm
        :returns: Watts
        """
        return 10 ** ((dBm - 30) / 10)

    @staticmethod
    def milliwatt(dBm):
        """.

        :param dBm: dBm
        :returns: Milliwatts
        """
        return 10 ** (dBm / 10)

    @staticmethod
    def dBuV(dBm):
        """μ."""
        Z = 50
        return dBm + 10 * log10(Z) + 90

    @staticmethod
    def dBuA(dBm):  # , *, Z=50):
        Z = 50
        return dBm - 10 * log10(Z) + 90

    @staticmethod
    def volt(dBm):  # , *, Z=50):
        # Z = 50
        return 10 ** ((dBm - 13) / 20)

    @staticmethod
    def amp(dBm):
        return 10 ** ((dBm - 47) / 20)


class MilliwattTo:

    @staticmethod
    def dBm(milliwatt):
        """milliwatt to dBm.

        :param watt: Watt
        :returns: dBm
        """
        # return 10 * log10((watt / 0.001))
        return 10 * log10(milliwatt)


class WattTo:

    @staticmethod
    def volt(watt):
        """Watts to volt."""
        Z = 50  # Nominal 50 Ohms
        # watt = 6.5
        return sqrt(watt * Z)
        # >>> 18.027756377319946

    @staticmethod
    def amp(watt):
        Z = 50  # Nominal 50 Ohms
        return sqrt(watt / Z)

    @staticmethod
    def dBm(watt):
        """Watts to dBm.

        :param watt: Watt
        :returns: dBm
        """
        # return 10 * log10((watt / 0.001))
        return 10 * log10(watt) + 30

    @staticmethod
    def dBW(watt):
        """Watts to dBW.
        
        :param watt: Watt
        :returns: dBW
        """
        return 10 * log10(watt)

    @staticmethod
    def dBuV(watt):
        return 10 * log10(watt) + 137

    @staticmethod
    def dBuA(watt):
        return 10 * log10(watt) + 103


class OhmsTo:

    @staticmethod
    def dBOhm(ohm):
        return 20 * log10(ohm)


class dBOhmsTo:

    @staticmethod
    def Ohm(dBOhm):
        return 10 ** (dBOhm / 20)


class dBuVTo:

    @staticmethod
    def watt(dBuV):
        # Z = 50
        return 10 ** ((dBuV - 137) / 10)

    @staticmethod
    def amp(dBuV):
        # Z = 50
        return 10 ** ((dBuV - 154) / 20)

    @staticmethod
    def dBm(dBuV):
        Z = 50
        # return dBuV - 90 + 10*log10(Z)
        return dBuV - 10 * log10(Z) - 90

    @staticmethod
    def dBuA(dBuV):
        Z = 50
        # return dBuV - log20(Z)
        return dBuV - 20 * log10(Z)

    @staticmethod
    def volt(dBuV):
        return 10**((dBuV - 120) / 20)  # checked against R&S unit converter

    @staticmethod
    def dBV(dBuV):
        return dBuV - 120


class dBVTo:

    @staticmethod
    def dBuV(dBV):
        return dBV + 120

    @staticmethod
    def volt(dBV):
        return 10 ** (dBV / 20)


class dBuATo:

    @staticmethod
    def dBuV(dBuA):
        Z = 50
        # return dBuA + log20(Z)
        return dBuA + 20 * log10(Z)

    @staticmethod
    def dBm(dBuA):
        Z = 50
        return dBuA + 10 * log10(Z) - 90

    @staticmethod
    def amp(dBuA):
        return 10 ** ((dBuA - 120) / 20)

    # uA = 10**(dBuA / 20)

    @staticmethod
    def volt(dBuA):
        return 10 ** ((dBuA - 86) / 20)

    @staticmethod
    def watt(dBuA):
        return 10 ** ((dBuA - 103) / 10)

    @staticmethod
    def dBA(dBuA):
        return dBuA - 120


class dBATo:

    @staticmethod
    def dBuA(dBA):
        return dBA + 120


class AmpTo:

    @staticmethod
    def volt(amp):
        Z = 50
        return amp * Z

    @staticmethod
    def watt(amp):
        Z = 50
        return amp**2 * Z

    @staticmethod
    def dBuA(amp):
        return 20 * log10(amp) + 120

    @staticmethod
    def dBm(amp):
        return 20 * log10(amp) + 47

    @staticmethod
    def dBuV(amp):
        return 20 * log10(amp) + 154

    @staticmethod
    def dBA(amp):
        return 20 * log10(amp)

    # 20 * log10(uA)


class uTTo:

    @staticmethod
    def A_m(uT):
        return uT / 1.25


class AmTo:

    @staticmethod
    def uT(A_m):
        return 1.25 * A_m


class IQTo:
    """Inphase and Quadrature."""
    # https://pysdr.org/content/sampling.html
    # https://uk.tek.com/blog/quadrature-iq-signals-explained
    # http://whiteboard.ping.se/SDR/IQ
    # https://www.itu.int/dms_pubrec/itu-r/rec/sm/R-REC-SM.2117-0-201809-I!!PDF-E.pdf

    # https://uk.mathworks.com/help/instrument/reading-inphase-and-quadrature-iq-data-from-a-signal-analyzer-over-tcp-ip.html
    # https://en.wikipedia.org/wiki/Complex_number

    @staticmethod
    def Magnitude(data):
        # https://www.rohde-schwarz.com/tr/faq/how-to-read-iq-data-from-spectrum-analyzer-and-convert-to-dbm-values-faq_78704-781632.html
        # 3.46423039446e-4, 4.35582856881e-4
        return sqrt(data.real**2 + data.imag**2)

    @classmethod
    def Watts(cls, data):
        return cls.Mag(data.real, data.imag)**2 / 50

    @classmethod
    def dBm(cls, data):
        # return 10 * log10(cls.Watts(I, Q) / 0.001)
        return 10 * np.log10(10 * (data.real**2 + data.imag**2))
        # https://www.tek.com/en/blog/calculating-rf-power-iq-samples
        # https://dsp.stackexchange.com/questions/19615/converting-raw-i-q-to-db
        # https://www.pe0sat.vgnet.nl/sdr/iq-data-explained/
        # https://www.rohde-schwarz.com/uk/faq/how-to-read-iq-data-from-spectrum-analyzer-and-convert-to-dbm-values-faq_78704-781632.html

    @staticmethod
    def Phase(data):
        return arctan2(data.imag, data.real)  # Phase Angle Rad

    @classmethod
    def Phase_deg(cls, data):
        return cls.Phase(data) * 180 / pi  # Phase Angle Deg

    @staticmethod
    def Complex(I, Q):
        return I + 1j * Q
    
    @classmethod
    def AM_Demod(cls, data):
        mag = cls.Magnitude(data)
        maxima = mag.max()
        minima = mag.min()
        avermag = np.average(mag)

        peak_pos = ((maxima - avermag) / avermag ) * 100
        peak_neg = (-(minima - avermag) / avermag ) * 100
        peak_aver = (peak_pos - -peak_neg) / 2
        peak_peak = ((maxima - minima) / (maxima + minima)) * 100

        # R&S AM Mod https://youtu.be/I46eP8uZh_Y?t=182 
        # m.round(4)  # AM Modulation depth
        return np.array([peak_peak, peak_pos, peak_neg, peak_aver]).round(2)
        
        # return m.round(6) * 100 # AM Modulation %
        
    @classmethod
    def FM_audio_Demod(cls, data):
        # IQ = cls.Complex(I, Q)

        # https://witestlab.poly.edu/blog/capture-and-decode-fm-radio/

        # http://witestlab.poly.edu/~ffund/el9043/labs/lab1.html
        # We'll use a kind of frequency discriminator called a polar
        # discriminator. A polar discriminator measures the phase
        # difference between consecutive samples of a
        # complex-sampled FM signal.

        # More specifically, it takes successive complex-valued
        # samples and multiplies the new sample by the conjugate
        # of the old sample. Then it takes the angle of this
        # complex value.

        # This turns out to be the instantaneous frequency
        # of the sampled FM signal.

        rad = data[1:] * np.conj(data[:-1])  # radians?
        return np.angle(rad)  # degrees ?
    
        '''
        a = np.arctan2(Q, I)
        b = np.unwrap(2 * a) / 2
        plt.plot(df.index, y)
        '''
        '''
        I = vsl['I'].values
        Q = vsl['Q'].values
        # IQTo.FM_Demod(I, Q)
        IQ = IQTo.Complex(I, Q)
        FM = IQ[1:] * np.conj(IQ[:-1])
        df = pd.DataFrame(np.column_stack((vsl.index[:-1], np.angle(FM))), columns=['Time', 'FM Audio']) 
        df = df.set_index('Time')  
        df[0.00025:0.00025+0.001].plot()
        # IQ
        # IQ
        '''

    '''
    self.i = np.array(self.df['I'])
    self.q = np.array(self.df['Q'])

    def plot_polar(self):
        zoom = np.floor(1 / np.max((self.i, self.q)))

        plt.scatter(self.i*zoom, self.q*zoom, color="red", alpha=0.2)
        plt.title("We can also plot the constellation, which should have the circular pattern typical of an FM signal")
        plt.xlabel("Real")
        plt.xlim(-1.1,1.1)
        plt.ylabel("Imag")
        plt.ylim(-1.1,1.1)

    from handcalcs.decorator import handcalc

    from numpy import arctan2, pi, sin, cos, sqrt, log10, rad2deg, deg2rad


    # negative mag is undefined
    # [trigonometry - Do you ever say that the amplitude is negative? - Mathematics Stack Exchange](https://math.stackexchange.com/questions/804455/do-you-ever-say-that-the-amplitude-is-negative)
    # phase_rad  > +-pi rollover
    # in phase_deg 185 returns 175


    @handcalc(override="long", precision=3, jupyter_display=True)
    def i_from_amplitude_phase(magnitude, phase_rad):
        i = magnitude * cos(phase_rad)  # I
        return i


    @handcalc(override="long", precision=3, jupyter_display=True)
    def q_from_amplitude_phase(magnitude, phase_rad):
        q = magnitude * sin(phase_rad)  # Q
        return q

    '''


def impedance_of_free_space():
    c = constants.speed_of_light  # Speed of Light
    h = constants.h  # Planck constant
    e = constants.elementary_charge  # elementary_charge
    R_inf = constants.Rydberg  # Rydberg constant
    m_e = constants.electron_mass  # electron mass

    α = sqrt(2 * h * R_inf / (m_e * c))
    # = α = alpha = fine_structure_constant

    vacuum_permeability = 2 * α * h / (e ** 2 * c)
    # = µ_0 = mu_0 = mu0 = magnetic_constant

    vacuum_permittivity = e ** 2 / (2 * α * h * c)
    # = ε_0 = epsilon_0 = eps_0 = eps0 = electric_constant

    impedance_of_free_space = np.sqrt(
        vacuum_permeability / vacuum_permittivity
        )
    # impedance_of_free_space = 2 * α * h / e ** 2
    # = Z_0 = characteristic_impedance_of_vacuum

    return round(impedance_of_free_space, 9)
    # uncertainty beyond 9 decimal places, according to NIST publication


Z_0 = impedance_of_free_space()
