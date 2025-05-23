'''
A file containing the classes and functions to generate a dark matter subhalo (DMS) distribution with corresponding impact parameters,
to use with galpy.peppered.
'''

import numpy as np
from scipy.integrate import quad
import astropy.units as u
from astropy.units import Quantity
import galpy.potential as gp
import galpy.actionAngle as ga
from galpy.orbit import Orbit
from galpy.df import streamdf
import stream_galsim.stream_utils as sutils


def expected_N_encounters(r_avg, sigma_h, t_d, delta_omega, b_max, n_h):
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
        if isinstance(val, Quantity):
            return val.to(unit)
        else:
            return val * unit

    r_avg = _to_quantity(r_avg, u.kpc)
    sigma_h = _to_quantity(sigma_h, u.km/u.s).to(u.kpc/u.Gyr)
    t_d = _to_quantity(t_d, u.Gyr)
    delta_omega =_to_quantity(delta_omega, u.rad/u.Gyr)
    b_max = _to_quantity(b_max, u.kpc)
    n_h = _to_quantity(n_h, 1/u.kpc**(3))
    
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
    
    def stream_length(self):
        return self.stream.length()
    
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
    
    def estimate_Omega_parallel(self, npts=1000):
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
        self.gal_rmax = gal_rmax

    def _to_quantity(self, val, unit):
        """
        Convert float or int to Quantity with assumed unit,
        or return the Quantity with correct unit.
        """
        if isinstance(val, Quantity):
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
        Spatial number density profile of subhalos.
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
        V = (4/3) * np.pi * rmax**3# - (4/3) * np.pi * rmin *3
        N = self.number_involume_inmassrange(rmin, rmax, M_min, M_max)
        return (N / V).to(1 / u.kpc**3)

    def bmax_inmassrange(self, M_min, M_max, profile='NFW', alpha=5.0):
        """
        Typical bmax in the considered mass range. Depends of the subhalo profile and an adjustable parameter
        """
        M_min = self._to_quantity(M_min, u.Msun).value
        M_max = self._to_quantity(M_max, u.Msun).value
        M_avg = 10**np.mean([np.log10(M_min), np.log10(M_max)])

        if profile == 'NFW':
            c = 15 * (M_avg / 1e8)**-0.1 #concentration profile
            r_vir = 1.0 * (M_avg / 1e8)**(1/3)  # virial radius
            rs = r_vir / c
        elif profile == 'Plummer':
            rs = (M_avg / 1e8)**0.5 * 1.0
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
    def __init__(self, N_enc, mass_range, sigma_h, tdisrupt, stream_length, profile='einasto'):
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
        if isinstance(val, Quantity):
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
    
    def subhalo_masses(self, distribution_profile='Einasto', a0=3.26e-5, m0=2.52e7, n=-1.9):
        '''
        Sample halo masses m in mass range.
         - mass_range: tuple (Mmin, Max)
         - distribution_profile: string (profile)
         - a0,m0,n = floats (distribution parameters)
        '''
        if distribution_profile == 'Einasto':
            mass_function = lambda M: a0 * (M / m0)**n
        else:
            raise NotImplementedError(f"Profile {distribution_profile} not implemented.")

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
        '''
        masses = self._to_quantity(masses, u.Msun)
        rs = 1.05 * u.kpc * (masses / (1e8 * u.Msun))**0.5
        b=np.random.uniform(-alpha * rs, alpha * rs)
        return b, rs

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
            profile = self.profile
        )

class ImpactList:
    def __init__(self, t, theta, m, b, rs, w_r, w_phi, w_z, mass_range=(1e5, 1e9), t_range=(0, 5), profile = 'einasto'):
        self.t = t
        self.theta = theta
        self.m = m
        self.b = b
        self.rs = rs
        self.w_r = w_r
        self.w_phi = w_phi
        self.w_z = w_z
        self.mass_range = mass_range
        self.t_range = t_range
        self.profile = profile

    def values(self):
        return np.array([self.t,self.theta,self.m,self.b, self.w_r, self.w_phi, self.w_z]).T

    def plot_distributions(self, bins=20):
        import matplotlib.pyplot as plt
        import numpy as np
        import astropy.units as u

        fig, axs = plt.subplots(2, 3, figsize=(18, 10))
        axs = axs.ravel()

        # 1. Impact times
        axs[0].hist(self.t, bins=bins, color='steelblue', edgecolor='black')
        axs[0].set_title("Impact times")
        axs[0].set_xlabel("Time [Gyr]")
        axs[0].set_ylabel("Counts")

        # 2. Impact angles
        axs[1].hist(self.theta, bins=bins, color='darkorange', edgecolor='black')
        axs[1].set_title("Impact angles")
        axs[1].set_xlabel("Angle [rad]")
        axs[1].set_ylabel("Counts")

        # 3. Subhalo masses (log bins)
        logbins = np.logspace(np.log10(self.mass_range[0]),
                            np.log10(self.mass_range[1]), int(np.log10(self.mass_range[1])-np.log10(self.mass_range[0])+1))
        axs[2].hist(self.m, bins=logbins, color='seagreen', edgecolor='black')
        axs[2].set_xscale('log')
        axs[2].set_title("Subhalo masses")
        axs[2].set_xlabel("Mass [M$_\odot$]")
        axs[2].set_ylabel("Counts")

        # 4. Impact parameters
        axs[3].hist(self.b, bins=bins, color='purple', edgecolor='black')
        axs[3].set_title("Impact parameters")
        axs[3].set_xlabel("b [kpc]")
        axs[3].set_ylabel("Counts")

        # 5. Velocity components
        axs[4].hist(self.w_r, bins=bins, alpha=0.5, label='w_r', color='red')
        axs[4].hist(self.w_phi, bins=bins, alpha=0.5, label='w_phi', color='blue')
        axs[4].hist(self.w_z, bins=bins, alpha=0.5, label='w_z', color='green')
        axs[4].set_title("Velocity components")
        axs[4].set_xlabel("Velocity [km/s]")
        axs[4].set_ylabel("Counts")
        axs[4].legend()

        # 6. Subhalo profile: Einasto + NFW
        r = np.logspace(-2, 2.5, 500)  # kpc

        rs = np.mean(self.rs).value
        if self.profile == 'einasto':
            # Einasto
            alpha_e = 0.05

            r_minus2 = rs
            x_e = r/r_minus2
            x_n = r/rs
            rho = np.exp(-2 / alpha_e * (x_e ** alpha_e - 1))
        elif self.profile == 'NFW':
            rho = 1 / (x_n * (1 + x_n)**2)

        axs[5].loglog(r, rho, label=f"{self.profile}", color='black')

        # Vertical dashed line for rs
        axs[5].axvline(rs, color='gray', linestyle=':', label=f'$r_s$ = {rs:.2f} kpc')

        axs[5].set_title(f"Subhalo density profile")
        axs[5].set_xlabel("r [kpc]")
        axs[5].set_ylabel("ρ(r)/<ρ>")
        axs[5].legend()

        plt.tight_layout()
        plt.show()


