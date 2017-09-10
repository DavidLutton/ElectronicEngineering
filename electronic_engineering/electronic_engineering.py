# -*- coding: utf-8 -*-

"""Main module."""

import functools
import pint
import numpy as np
from pprint import pprint

ureg = pint.UnitRegistry()
ureg.default_format = '~P'
Q_ = ureg.Quantity

ureg.define('emf = 0.5 * volt')  # half of one volt is one emf
ureg.define('dB = []')  # Whilst Pint has NotImplemented dB units
ureg.define('dBm = []')  # Whilst Pint has NotImplemented dBm units
ureg.define('dBμA = []')  # Whilst Pint has NotImplemented dBμA units
ureg.define('dBμV = []')  # Whilst Pint has NotImplemented dBμV units

Z = Q_(50, 'ohm')
# print(Z)


def wavelength_to_frequency(length='1m', *, returnunit='Hz'):
    """."""
    with ureg.context('sp'):
        return Q_(length).to(returnunit)


def test_wavelength_to_frequency():
    assert wavelength_to_frequency(length='1m', returnunit='MHz') == Q_(299.792458, 'MHz')
    assert wavelength_to_frequency(length='30cm', returnunit='Hz') == Q_(999308193.3333334, 'Hz')


def wavelength_to_frequency_antenna(length='1m', *, returnunit='Hz'):
    length = Q_(length)
    with ureg.context('sp'):
        return {
            '1/1': length.to(returnunit),
            '1/2': (length/(1/2)).to(returnunit),
            '1/4': (length/(1/4)).to(returnunit),
            '1/8': (length/(1/8)).to(returnunit),
            '3/4': (length/(3/4)).to(returnunit),
            '5/8': (length/(5/8)).to(returnunit),
        }


# pprint(wavelength_to_frequency_antenna(length='1m', returnunit='MHz'))


def frequency_to_wavelength_antenna(frequency='13.56MHz', *, returnunit='mm'):
    length = Q_(frequency)
    with ureg.context('sp'):
        return {
            '1/1': length.to(returnunit),
            '1/2': (length/(1/2)).to(returnunit),
            '1/4': (length/(1/4)).to(returnunit),
            '1/8': (length/(1/8)).to(returnunit),
            '3/4': (length/(3/4)).to(returnunit),
            '5/8': (length/(5/8)).to(returnunit),
        }


# pprint(frequency_to_wavelength_antenna(frequency='13.56MHz', returnunit='mm'))


def frequency_to_wavelength(frequency='100MHz', *, returnunit='m'):
    """."""
    with ureg.context('sp'):
        return Q_(frequency).to(returnunit)


def test_frequency_to_wavelength():
    assert frequency_to_wavelength(frequency='300GHz', returnunit='mm') == Q_(0.9993081933333336, 'mm')
    assert frequency_to_wavelength(frequency='1GHz', returnunit='m') == Q_(0.29979245800000004, 'm')
    assert frequency_to_wavelength(frequency='230MHz', returnunit='m') == Q_(1.3034454695652176, 'm')


# @functools.lru_cache(maxsize=1024, typed=False)
def log(value):
    """.

    .. math::

        \log (x)

     """
    return np.log(value)


def test_log():
    assert log(np.pi) == 1.1447298858494002


# @functools.lru_cache(maxsize=1024, typed=False)
def log10(value):
    """.

    .. math::

        \log_{10} (x)

     """
    return np.log10(value)


def test_log10():
    assert log10(np.pi) == 0.4971498726941338
    assert log10(50) == 1.6989700043360187
    assert log10(10) == 1.0


# @functools.lru_cache(maxsize=1024, typed=False)
def log20(value):
    """.

    .. math::

        \log_{20} (x)

     """
    return np.log(value)/np.log(20)


def test_log20():
    assert log20(np.pi) == 0.38212022347756341
    assert log20(50) == 1.3058653605207224
    assert log20(10) == 0.76862178684024096


def almost_equal(x, y, threshold=0.0000001):
    return abs(x-y) < threshold


def present(func, v):
    ret = func(v)
    print('{}\t{}\t{}'.format(func.__name__, str(v), str(ret)))
    return(ret)


# @functools.lru_cache(maxsize=1024, typed=False)
def watts_to_dBm(watts):
    return Q_(10 * log10((watts / 0.001)), 'dBm')


def volts_to_dBm(V, *, ohms=50):
    return(watts_to_dBm(Q_(V, 'V') ** 2. / ohms))


# @functools.lru_cache(maxsize=1024, typed=False)
def dBm_to_watts(dBm):
    return Q_(10 ** ((dBm - 30)/10), 'W')


# @functools.lru_cache(maxsize=1024, typed=False)
def watts_to_volts(watts, *, ohms=50):
    return np.sqrt(Q_(watts, 'W')*Q_(ohms, 'ohm')).to('V')


def test_watts_to_volts():
    assert watts_to_volts(2) == Q_(9.999999999999998, 'V')


# @functools.lru_cache(maxsize=1024, typed=False)
def watts_to_emf(watts, *, ohms=50):
    """."""
    return watts_to_volts(watts=watts, ohms=ohms).to('emf')


def test_watts_to_emf():
    assert watts_to_emf(2) == Q_(19.999999999999996, 'emf')
    # emf =
    assert watts_to_emf(0.189) * 3 == Q_(18.444511378727274, 'emf')  # For CDNs
    assert watts_to_emf(0.147) == Q_(5.422176684690383, 'emf')  # For Clampss
    # print('{:.3f}'.format(wattstoemf(.74)*3))  # For CDNs
    # print('{:.3f}'.format(wattstoemf(6.66)))  # For Clamps


# @functools.lru_cache(maxsize=1024, typed=False)
def emf_to_watts(emf, *, ohms=50):
    return ((Q_(emf, 'emf').to('V')) ** 2 / Q_(ohms, 'ohms')).to('W')


def test_emf_equilivence():
    assert Q_(10, 'emf') * 1.8 == Q_(18, 'emf')
    assert Q_(10, 'emf') * 1.8 == Q_(18.0, 'emf')
    assert Q_(20, 'emf') * 1.8 == Q_(36, 'emf')
    assert Q_(20.0, 'emf') * 1.8 == Q_(36, 'emf')


def dBm_to_dBuV(dBm, *, Z=50):
    return Q_(90 + 10*log10(Z) + dBm, 'dBμV')


def test_dBm_to_dBuV():
    assert not dBm_to_dBuV(10) == Q_(117, 'dBμV')
    assert dBm_to_dBuV(10) == Q_(116.98970004336019, 'dBμV')


def dBuV_to_dBm(dBuV, *, Z=50):
    return Q_(dBuV - 90 + 10*log10(Z), 'dBm')


def dBuA_to_dBm(dBuA, *, Z=50):
    return Q_(dBuA + 10*log10(Z) - 90, 'dBm')


def dBm_to_dBuA(dBm, *, Z=50):
    return Q_(dBm - 10*log10(Z)+90, 'dBμA')


def dBuA_to_dBuV(dbuA, *, Z=50):
    return Q_(dbuA + log20(Z), 'dBμV')


def dBuV_to_dBuA(dbuV, *, Z=50):
    return Q_(dbuV - log20(Z), 'dBμA')


def dBi_to_AF(MHz, dBi):
    return 20*log10(MHz) - dBi - 29.79


def test_dBi_to_AF():
    assert AF_to_dBi(10, dBi_to_AF(10, 3)) == 3
    assert not AF_to_dBi(30, 38.5) == -38.7
    assert AF_to_dBi(30, 38.5) == -38.747574905606754
    assert not AF_to_dBi(50, 32.5) == -28.3
    assert AF_to_dBi(50, 32.5) == -28.310599913279624
    assert not AF_to_dBi(100, 27.8) == -17.6
    assert AF_to_dBi(100, 27.8) == -17.59
    assert not AF_to_dBi(200, 22.6) == -6.4
    assert AF_to_dBi(200, 22.6) == -6.3694000867203755
    assert not AF_to_dBi(500, 23.3) == 0.9
    assert AF_to_dBi(500, 23.3) == 0.88940008672037507
    assert not AF_to_dBi(1000, 29.5) == 0.7
    assert AF_to_dBi(1000, 29.5) == 0.71000000000000085
    assert not AF_to_dBi(1500, 40.2) == -6.4
    assert AF_to_dBi(1500, 40.2) == -6.468174818886375
    assert not AF_to_dBi(2000, 47.3) == -11.1
    assert AF_to_dBi(2000, 47.3) == -11.069400086720371
    assert not AF_to_dBi(3000, 50.5) == -10.7
    assert AF_to_dBi(3000, 50.5) == -10.747574905606747


def AF_to_dBi(MHz, AF):
    return 20*log10(MHz) - AF - 29.79


def test_AF_to_dBi():
    assert dBi_to_AF(10, AF_to_dBi(10, 3)) == 3

    assert not dBi_to_AF(30, -38.7) == 38.5
    assert dBi_to_AF(30, -38.7) == 38.452425094393256
    assert not dBi_to_AF(50, -28.3) == 32.5
    assert dBi_to_AF(50, -28.3) == 32.489400086720373
    assert not dBi_to_AF(100, -17.6) == 27.8
    assert dBi_to_AF(100, -17.6) == 27.810000000000002
    assert not dBi_to_AF(200, -6.4) == 22.6
    assert dBi_to_AF(200, -6.4) == 22.630599913279625
    assert not dBi_to_AF(500, 0.9) == 23.3
    assert dBi_to_AF(500, 0.9) == 23.289400086720377
    assert not dBi_to_AF(1000, 0.7) == 29.5
    assert dBi_to_AF(1000, 0.7) == 29.509999999999998
    assert not dBi_to_AF(1500, -6.4) == 40.2
    assert dBi_to_AF(1500, -6.4) == 40.131825181113634
    assert not dBi_to_AF(2000, -11.1) == 47.3
    assert dBi_to_AF(2000, -11.1) == 47.33059991327962
    assert not dBi_to_AF(3000, -10.7) == 50.5
    assert dBi_to_AF(3000, -10.7) == 50.452425094393256


def dBi_to_numericgain(dBi):
    return 10.**(dBi/10)


def test_dBi_to_numericgain():
    for number in 1.0, 10.0, np.pi, 5.4321:
        assert almost_equal(numericgain_to_dBi(dBi_to_numericgain(number)), number)


def numericgain_to_dBi(numericgain):
    return 10*log10(numericgain)


def test_numericgain_to_dBi():
    for number in 1.0, 10.0, np.pi, 5.4321:
        assert almost_equal(dBi_to_numericgain(numericgain_to_dBi(number)), number)


def ReturnLoss_to_VSWR(RL):
    return((10 ** (RL / 20) + 1) / (10 ** (RL / 20) - 1))


def test_ReturnLoss_to_VSWR():
    assert ReturnLoss_to_VSWR(30) == 1.0653108640674351
    assert ReturnLoss_to_VSWR(60) == 1.002002002002002
    assert ReturnLoss_to_VSWR(0.5) == 34.75315212699187


def VSWR_to_ReturnLoss(VSWR):
    return(-20 * log10((VSWR - 1) / (VSWR + 1)))


def test_VSWR_to_ReturnLoss():
    assert VSWR_to_ReturnLoss(1.2) == 20.827853703164504
    assert VSWR_to_ReturnLoss(1.002) == 60.00868154958637
    assert VSWR_to_ReturnLoss(100) == 0.17372358370185334


def VSWR_to_ReflectionCoefficient(VSWR):
    return((VSWR - 1)/(VSWR + 1))


def test_VSWR_to_ReflectionCoefficient():
    assert VSWR_to_ReflectionCoefficient(1.50) == 0.2


'''
 Γ=10(‐ReturnLoss/20)
 VSWR=(1+|Γ|)/(1‐|Γ|)
 MismatchLoss(dB)=10log(Γ**2)
 ReflectedPower(%)=100*Γ **2
 ReturnLoss(dB)= ‐20log|Γ|
 Γ=(VSWR‐1)/(VSWR+1)
 ThroughPower(%)=100(1‐Γ2)
'''


def period_to_frequency(time='0.02s', *, returnunit='Hz'):
        return (1 / Q_(time)).to(returnunit)


def test_period_to_frequency():
    assert period_to_frequency(time='0.02s') == Q_(50, 'Hz')
    assert period_to_frequency(time='0.016666666666666666s') == Q_(60, 'Hz')


def frequency_to_period(frequency='60Hz', *, returnunit='s'):
        return (1 / Q_(frequency)).to(returnunit)


def test_frequency_to_period():
    assert frequency_to_period(frequency='60Hz') == Q_('0.016666666666666666s')
    assert frequency_to_period(frequency='50Hz') == Q_('0.02s')


def fieldstrength_dBuVm_to_Vm(dBuVm):
    return Q_(10**(((dBuVm)-120)/20), 'V/m')


def fieldstrength_Vm_to_dBuVm(Vm):
    return Q_(20*log10(Vm)+120, 'dBμV/m')


def fieldstrength_Watts_for_Vm(Vm, dBi, meters):
    return Q_(((Vm * meters)**2) / (30 * (10.**(dBi/10))), 'watt')


'''
print(fieldstrength_Watts_for_Vm(Vm=18.45, dBi=10, meters=2))
print(fieldstrength_Watts_for_Vm(Vm=18.45, dBi=10, meters=2.5))
print(fieldstrength_Watts_for_Vm(Vm=18.45, dBi=10, meters=3))
print(fieldstrength_Watts_for_Vm(Vm=18.45, dBi=10, meters=3.5))
print(fieldstrength_Watts_for_Vm(Vm=18.45, dBi=10, meters=4))
'''

'''
def fieldstrength_power_for_Vm(Vm, gain, meters):
    return ((Vm * meters)**2)/(30 * gain)


def givenWGDist(watts, gain, distance):
    return Q_(np.sqrt(30*watts*gain)/meters, 'V/m')


def givenWdBiDist(watts, dBi, distance):
    return Q_(np.sqrt(30*watts*(10**(dBi/10)))/meters, 'V/m')


def neededpowergforVm(Vm, gain, meters):
    return ((Vm * meters)**2)/(30 * gain)
'''


def uT_to_Am(uT):
    return Q_(uT/1.25, 'A/m')


def Am_to_uT(Amps_per_meter):
    return Q_(1.25*Amps_per_meter, 'μT')

# def woundcoilfluxdensity(turns, amps, radiusm):
#    return Q_((4*np.pi*turns*amps)/(log20(radiusm)), 'μT')


'''def dBm_ResolutionBandwidth_to_dBmPerHz(dBm, ResolutionBandwidth):
    return Q_((Q_(dBm, 'dBm') - (10*log10(ResolutionBandwidth))), 'dBm/Hz')

    # http://www.eevblog.com/forum/rf-microwave/from-dbm-to-dbmhz/
    # https://www.maximintegrated.com/en/app-notes/index.mvp/id/2875
    # First order, rough approximation would be to take your dBm reading and subtract the log
    # value of the RBW used for the measurement.  For example, if you use 1kHz RBW in your
    # analyzer, and you measure -90dBm, then:

    # dBm/Hz = -90dBm - 10LOG(1kHz)

    # -120dBm/Hz (approximate)

print(dBm_ResolutionBandwidth_to_dBmPerHz(-90, 1e3))
'''


def VIR(I, R):  # V=I*R
    return((I * R).to(ureg.volt))


def test_VIR():
    assert VIR(20 * ureg.amp, 8 * ureg.ohm) == 160 * ureg.volt


def VPI(P, I):  # V=P/I
    return((P / I).to(ureg.volt))


def test_VPI():
    assert VPI(20 * ureg.watt, 8 * ureg.amp) == 2.5 * ureg.volt


def VPR(P, R):  # V=SQRT(P*R)
    return((P * R) ** .5).to(ureg.volt)


def test_VPR():
    assert VPR(8 * ureg.watt, 50 * ureg.ohm) == 19.999999999999996 * ureg.volt


def IVR(V, R):  # I=V/R
    return((V / R).to(ureg.amp))


def test_IVR():
    assert IVR(10 * ureg.volt, 50 * ureg.ohm) == 0.2 * ureg.amp


def IPR(P, R):  # I=SQRT(P/R)
    return(((P / R) ** 0.5).to(ureg.amp))


def test_IPR():
    assert IPR(10 * ureg.watt, 50 * ureg.ohm) == 0.44721359549995787 * ureg.amp


def IPV(P, V):  # I=P/V
    return((P / V).to(ureg.amp))


def test_IPV():
    assert IPV(10 * ureg.watt, 10 * ureg.volt) == 1 * ureg.amp


def PVI(V, I):  # P=V*I
    return((V * I).to(ureg.watt))


def test_PVI():
    assert PVI(10 * ureg.volt, 10 * ureg.amp) == 100 * ureg.watt


def PIR(I, R):  # P=I²*R
    return((I ** 2 * R).to(ureg.watt))


def test_PIR():
    assert PIR(1 * ureg.amp, 50 * ureg.ohm) == 50 * ureg.watt


def PVR(V, R):  # P=V²/R
    """.

    .. math::

        Power = Voltage^2 / Ohms

     """
    return((V ** 2 / R).to(ureg.watt))


def test_PVR():
    assert PVR(17.99 * ureg.volt, 50 * ureg.ohm) == 6.472801999999999 * ureg.watt


def RVP(V, P):  # R=V²/P
    """.

    .. math::

        Ohms = Volts^2 / Power

     """
    return((V ** 2 / P).to(ureg.ohm))


def test_RVP():
    assert RVP(17.99 * ureg.volt, 6.472801999999999 * ureg.watt) == 50 * ureg.ohm


def RPI(P, I):  # R=P/I²
    """.

    .. math::

        Ohms = Power / Current^2

     """
    return((P / I ** 2).to(ureg.ohm))


def test_RPI():
    assert RPI(10 * ureg.watt, 1 * ureg.amp) == 10 * ureg.ohm


def RVI(V, I):  # R=V/I
    return((V / I).to(ureg.ohm))


def test_RVI():
    assert RVI(10 * ureg.volt, 1 * ureg.amp) == 10 * ureg.ohm


def ratioV_as_dB(ratio):
    """Calculate the dB equivalent of the voltage ratio."""

    return Q_(20*log10(ratio), 'dB')


def dB_as_ratio_of_V(dB):
    return 10.**(dB/20.)


def ratioA_as_dB(ratio):
    """Calculate the dB equivalent of the current ratio."""

    return Q_(20*log10(ratio), 'dB')


def test_ratioV_as_dB():
    assert ratioV_as_dB(1.8) == 5.105450102066121
    assert ratioV_as_dB(float(18/10)) == 5.105450102066121

    assert ratioV_as_dB(0.5555555555555556) == -5.10545010206612
    assert ratioV_as_dB(float(10/18)) == -5.10545010206612

    assert ratioV_as_dB(float(10/9.4)) == 0.5374429280060267
    # Wanted level is 10V/m, measured level is 9.4V/m
    # What amount of power (dB) change is required to get to wanted level

    assert ratioV_as_dB(float(10/11)) == -0.8278537031645011
    # Wanted level is 10 volts, measured level is 11
    # What amount of power (dB) change is required to get to wanted level

    assert ratioV_as_dB(float(10/0.2)) == 33.979400086720375

    assert ratioV_as_dB(float(10/3)) == 10.457574905606752
    assert ratioV_as_dB(float(10/5.4)) == 5.352124803540629

    assert ratioV_as_dB(float(18/5.4)) == 10.45757490560675


def ratioP_as_dB(ratio):
    """Calculate the dB equivalent of the power ratio.

    Useful for correcting readings from rf power heads"""
    return Q_(10*log10(ratio), 'dB')


def dB_as_ratio_of_P(dB):
    return 10.**(dB/10.)
