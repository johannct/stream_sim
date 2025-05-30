'''
A file containing the classes and functions to generate a dark matter subhalo (DMS) distribution with corresponding impact parameters,
to use with galpy.peppered.
'''

import numpy as np
from scipy.integrate import quad
import astropy.units as u
from astropy.constants import G
import galpy.potential as gp
import galpy.actionAngle as ga
from galpy.orbit import Orbit
from galpy.df import streamdf
import stream_galsim.stream_utils as sutils
import pandas as pd
import matplotlib.pyplot as plt


def expected_N_encounters(sigma_h, t_d,  b_max, n_h, r_avg=10.0, delta_omega=1.0, phys_length=10.0, use_phys_length = False):
    '''
    From unperturbed stream data and DMS properties, compute the expected number of encounters between sts and DMS
    Parameters:
     - r_avg: average galactocentric radius of the stream (kpc)
     - sigma_h: velocity dispersion (km/s)
     - t_d: stream age (Gyr)
     - delta_omega: mean-paralel-frequency parameter of the smooth stream (rad/Gyr)
     - b_max: max impact parameter (kpc)
     - n_h: sub_halo density (kpc^-3)
    '''
    #convert to correct quantity
    def _to_quantity(val, unit):
        if isinstance(val, u.Quantity):
            return val.to(unit)
        else:
            return val * unit

    sigma_h = _to_quantity(sigma_h, u.km/u.s).to(u.kpc/u.Gyr)
    t_d = _to_quantity(t_d, u.Gyr)
    b_max = _to_quantity(b_max, u.kpc)
    n_h = _to_quantity(n_h, 1/u.kpc**(3))
    
    if use_phys_length == True:
        phys_length = _to_quantity(phys_length, u.kpc)
        N_enc = (np.sqrt(np.pi) / 2) * sigma_h * t_d * b_max * n_h * phys_length

    else:
        delta_omega =_to_quantity(delta_omega, u.rad/u.Gyr)
        r_avg = _to_quantity(r_avg, u.kpc)
        N_enc = (np.sqrt(np.pi) / 2) * r_avg * sigma_h * t_d**2 \
                * delta_omega * b_max * n_h


    return N_enc


class NonPerturbedStreamModel:
    '''
    Generate a unperturbed stream model and retrieve data to generate impact parameters distribution
    '''
    def __init__(self, sigv, progenitor, pot, aA, tdisrupt, leading=True, ro=8.122, vsun=[-12.9, 245.6, 7.78]):
        self.tdisrupt = tdisrupt.to_value('Gyr')
        self.orbit = progenitor
        self.pot = pot
        # Generate stream
        self.stream = streamdf(sigv = sigv,
            progenitor=progenitor,
            pot=pot,
            aA=aA,
            tdisrupt=tdisrupt,
            leading=leading,
            ro = ro,
            vsun=vsun,
            nTrackChunks=26,
            )
    
    def stream_length(self, threshold=0.2, phys=False):
        return self.stream.length(threshold=threshold, phys=phys)
    
    def streamdf(self):
        return self.stream

    def compute_r_avg_max(self, npts=1000):
        #average and maximum radii from the galactic center of the orbit 
        ts = np.linspace(0., -self.tdisrupt, npts)
        self.orbit.integrate(ts, self.pot)
        r = np.sqrt(self.orbit.x(ts)**2 + self.orbit.y(ts)**2 + self.orbit.z(ts)**2)
        return np.mean(r), np.max(r), np.min(r)

    def compute_average_Omega_parallel(self, npts=1000):
        # Sample angles and frequencies
        aA_data = self.stream.sample(n=npts, returnaAdt=True)  # list of (actions, angles, freqs)
        freqs = aA_data[0].T
        angles = aA_data[1].T

        # Find dominant direction in angle-space
        angles_centered = angles - angles.mean(axis=0)
        _, _, Vt = np.linalg.svd(angles_centered)
        diff_dir = Vt[0]  # principal component

        # Project average frequency onto that direction
        Omega_mean = freqs.mean(axis=0)
        Omega_parallel = np.dot(Omega_mean, diff_dir)
        return abs(Omega_parallel)
    
    def estimate_Omega_parallel(self):
        # estimate stream growth frequency, considering a linear growth over time
        return self.stream_length()/self.tdisrupt

    def get_disruption_time(self):
        return self.tdisrupt


class DMS_Distribution:
    """
    A class to model the distribution of dark matter subhalos (DMS)
    in a galactic halo. Includes:
     - Number of subhalos within a mass range and galactic radius;
     - Subhalo intrinsic spatial profile;
     - Mass and radius configuration.
    """

    def __init__(self,
                 profile='einasto',
                 profile_params=None,
                 gal_rmax=300.0):
        self.profile = profile
        self.profile_params = profile_params or {'alpha': 0.678, 'r_minus2': 199, 'rs': 21}
        alpha = self.profile_params['alpha']
        r_minus2 = self.profile_params['r_minus2']
        rs = self.profile_params['rs']
        self.gal_rmax = gal_rmax
        print('Galactic parameters:', '\n', f'alpha = {alpha} | r_minus2 = {r_minus2} | rs = {rs}',
                '\n', f'galactic radius = {self.gal_rmax}, galactic profile = {self.profile}')

    def _to_quantity(self, val, unit):
        """
        Convert float or int to Quantity with assumed unit,
        or return the Quantity with correct unit.
        """
        if isinstance(val, u.Quantity):
            return val.to(unit)
        else:
            return val * unit

    def mass_function(self, M, a0=3.26e-5, m0=2.52e7, n=-1.9):
        """
        Subhalo mass function dN/dM (cf: The Aquarius Project: the subhalos of galactic halosV. Springel1, J. Wang).
        """
        return a0 * (M / m0)**n

    def integrate_mass_function(self, M_min, M_max):
        """
        Integrate the subhalo mass function over a given mass range.
        """
        return quad(lambda M: self.mass_function(M), M_min, M_max)[0]

    def spatial_profile(self, r):
        """
        Spatial number density profile of the milky way halo.
        Currently supports only 'einasto' and 'NFW'.
        """
        if self.profile == 'einasto':
            alpha = self.profile_params['alpha']
            r_minus2 = self.profile_params['r_minus2']
            x = r / r_minus2
            return np.exp(-2 / alpha * (x**alpha - 1))
        elif self.profile == 'NFW':
            rs = self.profile_params['rs']
            x = r / rs
            return 1.0 / (x * (1 + x)**2)
        else:
            raise NotImplementedError(f"Profile {self.profile} not implemented. Profiles: ['einasto', 'NFW']")

    def integrate_spatial_profile(self, rmax, r_min=0.01):
        """
        Fraction of subhalos within [r_min, gal_rmax] based on spatial profile.
        """

        integrand = lambda r: 4 * np.pi * r**2 * self.spatial_profile(r)

        total = quad(integrand, 0.01, self.gal_rmax)[0]
        partial = quad(integrand, r_min, rmax)[0]

        return partial / total

    def number_involume_inmassrange(self, rmin, rmax, M_min=1e5, M_max=1e9):
        """
        Total number of subhalos within radius R.
        """
        rmin = self._to_quantity(rmin, u.kpc).value
        rmax = self._to_quantity(rmax, u.kpc).value
        M_min = self._to_quantity(M_min, u.Msun).value
        M_max = self._to_quantity(M_max, u.Msun).value

        N_mass = self.integrate_mass_function(M_min, M_max)
        N_radius = self.integrate_spatial_profile(rmax, rmin)
        return N_mass * N_radius

    def density_involume_inmassrange(self, rmin, rmax, M_min=1e5, M_max=1e9):
        """
        density (/kpc3) of subhalos within radius R.
        """
        rmin = self._to_quantity(rmin, u.kpc)
        rmax = self._to_quantity(rmax, u.kpc)
        V = (4/3) * np.pi * rmax**3 - (4/3) * np.pi * rmin**3
        N = self.number_involume_inmassrange(rmin, rmax, M_min, M_max)
        return (N / V).to(1 / u.kpc**3)

    def bmax_inmassrange(self, M_min, M_max, profile='NFW', alpha=5.0, n=1.9):
        """
        Typical bmax in the considered mass range. Depends of the subhalo profile and an adjustable parameter
        """
        M_min = self._to_quantity(M_min, u.Msun).value
        M_max = self._to_quantity(M_max, u.Msun).value
        num = (M_max**(2 - n) - M_min**(2 - n)) / (2 - n)
        den = (M_max**(1 - n) - M_min**(1 - n)) / (1 - n)
        M_avg = num / den #mass function weighted mean mass

        print('subhalo profile:', profile)
        if profile == 'NFW' or 'einasto':
            # c = 15 * (M_avg / 1e8)**-0.1 #concentration profile
            # r_vir = 1.0 * (M_avg / 1e8)**(1/3)  # virial radius
            # rs = r_vir / c
            # rs = (M_avg / 1e9)**0.4 * 1
            rs = (M_avg / 1e8)**0.39 * 1.24
        else:
            raise ValueError("Unrecognized profile. choose NFW or Plummer")

        return (alpha * rs) * u.kpc



class ImpactSampler:
    '''
    Sample impact properties of a DMS distribution with a STS, from:
     - N_enc: number of impacts the STS has undergone
     - sigma_h: velocity dispersion of the subhalos in km/s
     - smooth_stream: unperturbed stream PDF
     - mass_range: DMS mass range
    
    With functions related to:
     - impact time: t
     - stream parallel angle at which the impact occurs at the time of closest approach: theta
     - intrinsic properties of the subhalo (mass, density profile and radius): mi
     - impact parameter: b
     - fly-by velocity of the dark matter halo: w
    
    Returns sampled DMS with impact properties {(ti, thetai, mi, bi, wi)}
    '''
    def __init__(self, N_enc, mass_range, sigma_h, tdisrupt, stream_length, use_phys_length=False, profile='NFW'):
        if use_phys_length:
            self.stream_length = self._to_quantity(stream_length, u.kpc).value
        else:       
            self.stream_length = self._to_quantity(stream_length, u.rad).value
        self.mass_range = self._to_quantity(mass_range, u.Msun).value
        self.sigma_h = self._to_quantity(sigma_h,u.km/u.s).value
        self.tdisrupt = self._to_quantity(tdisrupt, u.Gyr).value
        self.N_enc = np.random.poisson(self._to_quantity(N_enc, u.rad).value)
        self.profile = profile
        print('Generating', int(self.N_enc), 'impacts between the stream and subhalos')

    def _to_quantity(self, val, unit):
        """
        Convert float or int to Quantity with assumed unit,
        or return the Quantity with correct unit.
        """
        if isinstance(val, u.Quantity):
            return val.to(unit)
        else:
            return val * unit

    def impact_time(self):
        '''
        Sample impact times t between [tdisrupt, today]. Constuct a probability law that increases linearly with time (stream length) as d_omega*t.
         - N_enc: number of encounters, (int,float)
         - tdisrupt: float.
        '''
        time_function = lambda x: x
        times = []
        t_vals = np.linspace(0, self.tdisrupt, 1000)
        f_vals = time_function(t_vals)
        f_max = f_vals.max()

        while len(times) < int(self.N_enc):
            t = np.random.uniform(0, np.max(t_vals))
            u_ = np.random.uniform(0, f_max)
            if u_ < time_function(t):
                times.append(t)
        return np.array(times)

    def impact_angle(self, impact_times):
        '''
        Sample impact angles theta at corresponding ti.
         - smooth_stream: stream model from galpy.
         - impact_times: float or list.
        '''
        ti_length = self.stream_length * np.array(impact_times) / self.tdisrupt
        impact_angles = np.random.uniform(low=0.05, high=ti_length)
        return impact_angles
    
    def subhalo_masses(self, a0=3.26e-5, m0=2.52e7, n=-1.9):
        '''
        Sample halo masses m in mass range, following a power function as a probability distribution function.
         - mass_range: tuple (Mmin, Max)
         - a0,m0,n = floats (distribution parameters)
        '''
        mass_function = lambda M: a0 * (M / m0)**n

        M_min, M_max = self.mass_range
        masses = []
        M_vals = np.logspace(np.log10(M_min), np.log10(M_max), 1000)
        f_vals = mass_function(M_vals)
        f_max = f_vals.max()

        while len(masses) < int(self.N_enc):
            M = np.random.uniform(M_min, M_max)
            u_ = np.random.uniform(0, f_max)
            if u_ < mass_function(M):
                masses.append(M)

        return np.array(masses)

    def impact_parameter(self, masses, alpha = 5):
        '''
        Sample impact parameter b from mass distribution. Use the direct relation between the scale radius rs and Mhalo,
        and the relation between rs and b.
        Parameters:
         - mass: of the considered subhalo, float or list, in Msun
         - alpha: factor of the b range to consider b ∈ [-Xrs,Xrs]
        Outputs:
         - impact parameter
         - scale radius, approximated from an empirical relation
        '''
        masses = self._to_quantity(masses, u.Msun)
        rs = 1.24 * u.kpc * (masses / (1e8 * u.Msun))**0.39
        b=np.random.uniform(-alpha * rs, alpha * rs)
        return b, rs

    def subhalo_potential(self, masses, rss, alpha = 0.16):
        '''
        Compute galpy.potential instance for the subhalo.
        Parameters:
         - m, masses of the subhalos (list of Quantity)
         - rs, scale radius (list of Quantity)
        Output:
         - potential instance: [NFW]
        '''
        masses = self._to_quantity(masses, u.Msun)
        rss = self._to_quantity(rss, u.kpc)

        subhalopots = []
        for (m, rs) in zip(masses, rss):
            if self.profile == 'einasto':
                rho = lambda r: np.exp(-2 / alpha * ((r/rs)**alpha - 1))
                subhalopot = gp.AnySphericalPotential(amp=(m/(4*np.pi*rs**3)).value, dens = rho)
            elif self.profile == 'NFW':
                subhalopot = gp.NFWPotential(amp = m*G, a=rs)
            subhalopots.append(subhalopot)
        return subhalopots


    def flyby_velocity(self):
        '''
        From velocity dispersion in the accreting host,
        sample velocity components of the subhalo in the stream fream, because we don't want positive radial velocity.
        return {(w_r, w_phi, w_z)}
        ''' 
        w_r,w_phi,w_z = [],[],[]
        i=0
        while i <int(self.N_enc):
            i+=1
            w_r.append(np.random.rayleigh(scale=self.sigma_h))
            w_phi.append(np.random.normal(scale=self.sigma_h))
            w_z.append(np.random.normal(scale=self.sigma_h))
        return np.array([w_r, w_phi, w_z])

    def impact_list(self):
        t = self.impact_time()
        theta = self.impact_angle(t)
        m = self.subhalo_masses()
        b = self.impact_parameter(m)[0]
        rs = self.impact_parameter(m)[1]
        w = self.flyby_velocity()
        shpot= self.subhalo_potential(m, rs)

        return ImpactList(
            t=t,
            theta=theta,
            m=m,
            b=b,
            rs=rs,
            w_r=w[0],
            w_phi=w[1],
            w_z=w[2],
            mass_range=self.mass_range,
            t_range=(0,self.tdisrupt),
            shpot = shpot,
            profile = self.profile
        )

class ImpactList:
    def __init__(self, t, theta, m, b, rs, w_r, w_phi, w_z, shpot, profile, mass_range=(1e5, 1e9), t_range=(0, 5)):
        self.t = t
        self.theta = theta
        self.m = m
        self.b = b
        self.rs = rs
        self.w_r = w_r
        self.w_phi = w_phi
        self.w_z = w_z
        self.shpot = shpot
        self.profile = profile        
        self.mass_range = mass_range
        self.t_range = t_range
        

    def values(self):
        print('t, theta, m, b, w_r,w_phi,w_z, shpot')
        return np.array([self.t,self.theta,self.m,self.b, self.w_r, self.w_phi, self.w_z, self.shpot]).T

    def plot_distributions(self, bins=20, fontsize=12):
        fig, axs = plt.subplots(2, 3, figsize=(18, 10))
        axs = axs.ravel()

        # 1. Impact times
        axs[0].hist(self.t, bins=bins, color='steelblue', edgecolor='black')
        axs[0].set_title("Impact times",fontsize=fontsize)
        axs[0].set_xlabel("Time [Gyr]",fontsize=fontsize)
        axs[0].set_ylabel("Counts",fontsize=fontsize)

        # 2. Impact angles
        axs[1].hist(self.theta, bins=bins, color='darkorange', edgecolor='black')
        axs[1].set_title("Impact angles",fontsize=fontsize)
        axs[1].set_xlabel("Angle [rad]",fontsize=fontsize)
        # axs[1].set_ylabel("Counts",fontsize=fontsize)

        # 3. Subhalo masses (log bins)
        logbins = np.logspace(np.log10(self.mass_range[0]),
                            np.log10(self.mass_range[1]), int(np.log10(self.mass_range[1])-np.log10(self.mass_range[0])+1))
        axs[2].hist(self.m, bins=logbins, color='seagreen', edgecolor='black')
        axs[2].set_xscale('log')
        axs[2].set_title("Subhalo masses",fontsize=fontsize)
        axs[2].set_xlabel("Mass [M$_\odot$]",fontsize=fontsize)
        # axs[2].set_ylabel("Counts",fontsize=fontsize)

        # 4. Impact parameters
        axs[3].hist(self.b, bins=bins, color='purple', edgecolor='black')
        axs[3].set_title("Impact parameters",fontsize=fontsize)
        axs[3].set_xlabel("b [kpc]",fontsize=fontsize)
        axs[3].set_ylabel("Counts",fontsize=fontsize)

        # 5. Velocity components
        axs[4].hist(self.w_r, bins=bins, alpha=0.5, label='w_r', color='red')
        axs[4].hist(self.w_phi, bins=bins, alpha=0.5, label='w_phi', color='blue')
        axs[4].hist(self.w_z, bins=bins, alpha=0.5, label='w_z', color='green')
        axs[4].set_title("Velocity components",fontsize=fontsize)
        axs[4].set_xlabel("Velocity [km/s]",fontsize=fontsize)
        # axs[4].set_ylabel("Counts",fontsize=fontsize)
        axs[4].legend()

        # 6. Subhalo profile: Einasto + NFW
        r = np.logspace(-2, 2.5, 500)  # kpc

        rs = np.mean(self.rs).value
        if self.profile == 'einasto':
            # Einasto
            alpha_e = 0.05

            r_minus2 = rs
            x_e = r/r_minus2
            rho = np.exp(-2 / alpha_e * (x_e ** alpha_e - 1))
        elif self.profile == 'NFW':
            x_n = r/rs
            rho = 1 / (x_n * (1 + x_n)**2)

        axs[5].loglog(r, rho, label=f"{self.profile}", color='black')

        # Vertical dashed line for rs
        axs[5].axvline(rs, color='gray', linestyle=':', label=f'$r_s$ = {rs:.2f} kpc')

        axs[5].set_title(f"Subhalo density profile",fontsize=fontsize)
        axs[5].set_xlabel("r [kpc]",fontsize=fontsize)
        axs[5].set_ylabel("ρ(r)/<ρ>",fontsize=fontsize)
        axs[5].legend()

        plt.tight_layout()
        plt.show()


from scipy.signal.windows import hann
from scipy.fft import fft, fftfreq #scipy.stats.csd

class PowerSpectrum1D:
    def __init__(self, delta, xi):
        """
        Initializes with a regularly sampled density contrast.

        delta: array-like
        Array of density contrasts.
        xi: array-like
        Corresponding angular positions (degrees or radians).
        """
        self.delta = np.asarray(delta)
        self.xi = np.asarray(xi)
        self.d_xi = self.xi[1] - self.xi[0]
        self.N = len(self.delta)

        self.mods = None
        self.power = None
        self.noise = None

    def compute(self, window=True):
        """Calculates the signal power spectrum."""
        signal = self.delta * hann(self.N) if window else self.delta
        fft_vals = fft(signal)
        power = np.abs(fft_vals)**2# / self.N**2
        mods = fftfreq(self.N, d=self.d_xi)
        mask = mods > 0
        self.mods = mods[mask]
        self.power = power[mask]

    def noise_compute(self, n_iter=100):
        """Computes the average power spectrum of background noise"""
        sigma = np.std(self.delta)
        noise_power = []

        for _ in range(n_iter):
            noise = np.random.normal(0, sigma, self.N)
            fft_vals = fft(noise * hann(self.N))
            power = np.abs(fft_vals)**2 #/ self.N**2
            noise_power.append(power)

        noise_power = np.array(noise_power)
        mods = fftfreq(self.N, d=self.d_xi)
        mask = mods > 0
        self.noise = np.mean(noise_power[:, mask], axis=0)

    def plot(self, loglog=True, plot_noise=True, xscale='mode', sqrt=False):
        """The spectrum has not yet been calculated. Call .compute()"""
        if self.mods is None or self.power is None:
            raise RuntimeError()

        plt.figure(figsize=(6, 4))

        if sqrt ==True:
            power = np.sqrt(self.power)
        else:
            power = self.power

        if xscale == 'mode':
            x = self.mods
            xlabel = r"$k_{\phi}$ (1/deg)"
        elif xscale == 'size':
            x = 1 / self.mods
            xlabel = r"$1/k_{\phi}$ (deg)"
        else:
            raise NotImplementedError(f'{scale} not implemented. Choose size or mode.')

        if loglog:
            plt.loglog(x, power, label="Signal", c='k')
            if plot_noise and self.noise is not None:
                if sqrt ==True:
                    noise = np.sqrt(self.noise)
                else:
                    noise = self.noise
                plt.loglog(x, noise, '--', label="noise")
        else:
            plt.plot(x, power, label="Signal", c='k')
            if plot_noise and self.noise is not None:
                if sqrt ==True:
                    noise = np.sqrt(self.noise)
                else:
                    noise = self.noise
                plt.plot(x, noise, '--', label="noise")

        if sqrt ==True:
            plt.ylabel(r"$\sqrt{\delta\delta}$")
        else:
            plt.ylabel(r"$\delta\delta$")
        
        plt.xlabel(xlabel)
        # plt.title("density contrast power spectrum")
        # plt.legend()
        # plt.grid(True, which='both')
        plt.tight_layout()
        plt.show()
