# -*- coding: utf-8 -*-

"""Main module."""

import functools
import pint
import numpy as np


ureg = pint.UnitRegistry()
ureg.default_format = '~P'
Q_ = ureg.Quantity

ureg.define('emf = 0.5 * volt')  # half of one volt is one emf
ureg.define('dB = []')  # Whilst Pint has NotImplemented dB units
ureg.define('dBm = []')  # Whilst Pint has NotImplemented dBm units
ureg.define('dBμA = []')  # Whilst Pint has NotImplemented dBμA units
ureg.define('dBμV = []')  # Whilst Pint has NotImplemented dBμV units


def wavelength_2_frequency(length='1m', *, returnunit='Hz'):
    """."""
    with ureg.context('sp'):
        return Q_(length).to(returnunit)


def test_wavelength_2_frequency():
    assert wavelength_2_frequency(length='1m', returnunit='MHz') == Q_(299.792458, 'MHz')
    assert wavelength_2_frequency(length='30cm', returnunit='Hz') == Q_(999308193.3333334, 'Hz')


def frequency_2_wavelength(frequency='100MHz', *, returnunit='m'):
    """."""
    with ureg.context('sp'):
        return Q_(frequency).to(returnunit)


def test_frequency_2_wavelength():
    assert frequency_2_wavelength(frequency='300GHz', returnunit='mm') == Q_(0.9993081933333336, 'mm')
    assert frequency_2_wavelength(frequency='1GHz', returnunit='m') == Q_(0.29979245800000004, 'm')
