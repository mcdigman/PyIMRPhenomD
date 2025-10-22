"""Python implementation of IMRPhenomD by Matthew Digman (C) 2021"""

# Copyright (C) 2015 Michael Puerrer, Sebastian Khan, Frank Ohme, Ofek Birnholtz, Lionel London
#
#  This program is free software; you can redistribute it and/or modify
#  it under the terms of the GNU General Public License as published by
#  the Free Software Foundation; either version 2 of the License, or
#  (at your option) any later version.
#
#  This program is distributed in the hope that it will be useful,
#  but WITHOUT ANY WARRANTY; without even the implied warranty of
#  MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
#  GNU General Public License for more details.
#
#  You should have received a copy of the GNU General Public License
#  along with with program; see the file COPYING. If not, write to the
#  Free Software Foundation, Inc., 59 Temple Place, Suite 330, Boston,
#  MA  02111-1307  USA

# LAL independent code (C) 2017 Michael Puerrer
import numpy as np
from numpy.typing import NDArray

import PyIMRPhenomD.IMRPhenomD_const as imrc
from PyIMRPhenomD.IMRPhenomD_deriv_internals import IMRPhenDAmpPhaseFI
from PyIMRPhenomD.IMRPhenomD_internals import AmpPhaseFDWaveform, COMPLEX16FrequencySeries, FinalSpin0815, IMRPhenDAmplitude, IMRPhenDPhase, NextPow2


def IMRPhenomDGenerateFD_internal(phi0: float, fRef_in: float, deltaF: float, m1_in: float, m2_in: float, chi1_in: float, chi2_in: float, f_min: float, f_max: float, distance: float) -> COMPLEX16FrequencySeries:
    """The following private function generates IMRPhenomD frequency-domain waveforms
    given coefficients
    """
    # LIGOTimeGPS ligotimegps_zero = LIGOTIMEGPSZERO; # = {0, 0}
    ligotimegps_zero = 0.0

    if m1_in > m2_in:
        chi1: float = chi1_in
        chi2: float = chi2_in
        m1: float = m1_in
        m2: float = m2_in
    else:  # swap spins and masses
        chi1 = chi2_in
        chi2 = chi1_in
        m1 = m2_in
        m2 = m1_in

    Mt: float = m1 + m2
    eta: float = m1 * m2 / Mt**2

    if not 0 <= eta <= 0.25:
        msg = f'Unphysical eta {eta}. Must be between 0. and 0.25'
        raise ValueError(msg)

    Mt_sec: float = Mt * imrc.MTSUN_SI

    # Compute the amplitude pre-factor
    amp0: float = float(2 * np.sqrt(5. / (64. * np.pi)) * Mt**2 * imrc.MRSUN_SI * imrc.MTSUN_SI / distance)

    # Coalesce at t=0
    # shift by overall length in time
    ligotimegps_zero += -1. / deltaF

    # Allocate htilde
    nf: int = int(NextPow2(f_max / deltaF) + 1)
    htilde = COMPLEX16FrequencySeries(ligotimegps_zero, 0.0, deltaF, nf)

    # range that will have actual non-zero waveform values generated
    ind_min: int = int(np.int64(f_min / deltaF))
    ind_max: int = int(np.int64(f_max / deltaF))
    if not ind_min <= ind_max <= nf:
        raise ValueError('minimum freq index %5d and maximum freq index %5d do not fulfill 0<=ind_min<=ind_max<=htilde->data>length=%5d.' % (ind_min, ind_max, nf))

    # Calculate phenomenological parameters
    chis: float = (chi1 + chi2) / 2
    chia: float = (chi1 - chi2) / 2
    finspin: float = FinalSpin0815(eta, chis, chia)  # FinalSpin0815 - 0815 is like a version number

    if finspin < imrc.MIN_FINAL_SPIN:
        print('Final spin (Mf=%g) and ISCO frequency of this system are small, the model might misbehave here.' % (finspin))

    # Now generate the waveform
    Mfs: NDArray[np.floating] = Mt_sec * deltaF * np.arange(ind_min, ind_max)  # geometric frequency
    phis, _times, _timeps, _t0, _MfRef, _itrFCut = IMRPhenDPhase(Mfs[ind_min:ind_max], Mt_sec, eta, chis, chia, ind_max - ind_min, fRef_in, phi0)
    amps = IMRPhenDAmplitude(Mfs[ind_min:ind_max], eta, chis, chia, ind_max - ind_min, amp_mult=amp0)
    htilde.data[:ind_max - ind_min] = amps[:ind_max - ind_min] * np.exp(-1j * phis[:ind_max - ind_min])

    for i in range(ind_min, ind_max):
        phi: float = phis[i - ind_min]
        amp: float = amps[i - ind_min]
        htilde.data[i] = amp * np.exp(-1j * phi)
    return htilde


def IMRPhenomDGenerateh22FDAmpPhase_internal(h22: AmpPhaseFDWaveform, freq: NDArray[np.floating], phi0: float, fRef_in: float, m1_in: float, m2_in: float, chi1_in: float, chi2_in: float, distance: float) -> AmpPhaseFDWaveform:
    """SM: similar to IMRPhenomDGenerateFD_internal, but generates h22 FD amplitude and phase on a given set of frequencies"""
    nf: int = freq.size
    if m1_in > m2_in:
        chi1: float = chi1_in
        chi2: float = chi2_in
        m1: float = m1_in
        m2: float = m2_in
    else:  # swap spins and masses
        chi1 = chi2_in
        chi2 = chi1_in
        m1 = m2_in
        m2 = m1_in

    Mt: float = m1 + m2
    eta: float = m1 * m2 / Mt**2

    if not 0. <= eta <= 0.25:
        msg = 'Unphysical eta. Must be between 0. and 0.25'
        raise ValueError(msg)

    Mt_sec: float = Mt * imrc.MTSUN_SI

    # Compute the amplitude pre-factor
    # NOTE: we will output the amplitude of the 22 mode - so we remove the factor 2. * sqrt(5. / (64.*PI)), which is part of the Y22 spherical harmonic factor
    amp0: float = Mt**2 * imrc.MRSUN_SI * imrc.MTSUN_SI / distance

    # Max frequency covered by PhenomD
    # fCut = imrc.f_CUT / Mt_sec  # convert Mf -> Hz

    # Calculate phenomenological parameters
    chis: float = (chi1 + chi2) / 2
    chia: float = (chi1 - chi2) / 2
    finspin: float = FinalSpin0815(eta, chis, chia)  # FinalSpin0815 - 0815 is like a version number

    if finspin < imrc.MIN_FINAL_SPIN:
        print('Final spin (Mf=%g) and ISCO frequency of this system are small, the model might misbehave here.' % (finspin))

    # Now generate the waveform on the frequencies given by freq
    # f = freq

    # Mfs = Mt_sec*f #geometric frequency

    # for frequencies exceeding the maximal frequency covered by PhenomD, put 0 amplitude and phase
    # phase,time,t0,MfRef,itrFCut = IMRPhenDPhase(Mfs,Mt,eta,chis,chia,nf,fRef_in,phi0)
    # amp = IMRPhenDAmplitude(Mfs,eta,chis,chia,nf,amp_mult=amp0)
    h22.phase, h22.time, h22.timep, h22.amp, h22.t0, MfRef, _ = IMRPhenDAmpPhaseFI(h22.phase, h22.time, h22.timep, h22.amp, freq, Mt_sec, eta, chis, chia, nf, fRef_in, phi0, amp0, imr_default_t=True)
    h22.fRef = MfRef / Mt_sec

    # for itrf in range(0,nf):
    #    print("%5d %+.8e %+.8e %+.8e %+.8e"%(itrf,freq[itrf],phase[itrf],amp[itrf],time[itrf]))

    return h22


def IMRPhenomDGenerateh22FDAmpPhase(h22: AmpPhaseFDWaveform, freq: NDArray[np.floating], phi0: float, fRef_in: float, m1_SI: float, m2_SI: float, chi1: float, chi2: float, distance: float) -> AmpPhaseFDWaveform:
    """SM: similar to IMRPhenomDGenerateFD, but generates h22 FD amplitude and phase on a given set of frequencies"""
    m1: float = m1_SI / imrc.MSUN_SI
    m2: float = m2_SI / imrc.MSUN_SI

    f_min: float = freq[0]
    f_max: float = freq[-1]

    # check inputs for sanity
    # if np.all(freq==0.):
    #    raise ValueError("freq is null")
    if fRef_in < 0.0:
        msg = f"fRef_in {fRef_in} must be positive (or 0 for 'ignore')"
        raise ValueError(msg)
    if m1 <= 0.0:
        msg = f'm1 {m1} must be positive'
        raise ValueError(msg)
    if m2 <= 0.0:
        msg = f'm2 {m2} must be positive'
        raise ValueError(msg)
    if f_min <= 0.0:
        msg = f'f_min {f_min} must be positive'
        raise ValueError(msg)
    if f_max < 0.0:
        msg = f'f_max {f_max} must be greater than 0'
        raise ValueError(msg)
    if distance <= 0.0:
        msg = f'distance {distance} must be positive'
        raise ValueError(msg)

    if m1 > m2:
        q: float = m1 / m2
    else:
        q = m2 / m1
    assert q > 1.

# if (q > MAX_ALLOWED_MASS_RATIO) PRINT_WARNING("Warning: The model is not supported for high mass ratio, see MAX_ALLOWED_MASS_RATIO\n");

    if not (-1. < chi1 <= 1. and -1. < chi2 <= 1.):
        msg = f'Spins chi1={chi1} chi2={chi2}outside the range [-1,1] are not supported'
        raise ValueError(msg)

    # NOTE: we changed the prescription, now fRef defaults to fmaxCalc (fpeak in the paper)
    # if no reference frequency given, set it to the starting GW frequency
    # double fRef = (fRef_in == 0.0) ? f_min : fRef_in;

    Mt_sec: float = (m1 + m2) * imrc.MTSUN_SI  # Conversion factor Hz -> dimensionless frequency
    fCut: float = imrc.f_CUT / Mt_sec  # convert Mf -> Hz
    # Somewhat arbitrary end point for the waveform.
    # Chosen so that the end of the waveform is well after the ringdown.
    if fCut <= f_min:
        print('(fCut = %g Hz) <= f_min = %g' % (fCut, f_min))
    # Check that at least the first of the output frequencies is strictly positive - note that we don't check for monotonicity
    if f_min <= 0:
        print('(f_min = %g Hz) <= 0' % (f_min))
    h22 = IMRPhenomDGenerateh22FDAmpPhase_internal(h22, freq, phi0, fRef_in, m1, m2, chi1, chi2, distance)
    return h22
