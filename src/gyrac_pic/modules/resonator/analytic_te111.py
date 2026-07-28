"""Maxwell-consistent rotating TE111 mode for an ideal cylindrical PEC cavity.

The cavity is centred at the origin and its axis is +z.  The mode is built from
the two degenerate m=1 standing-wave solutions (cos(theta), sin(theta)) in time
quadrature. ``electric_field_amplitude_v_per_m`` is the on-axis peak transverse
electric field, rather than an unnormalised modal coefficient.
"""

import math

import torch

from ...constants import TE11_BESSEL_DERIVATIVE_ROOT


def _bessel_j(order: int, x: torch.Tensor, terms: int = 12) -> torch.Tensor:
    """Device-native power series, accurate on 0 <= x <= 1.8412."""
    half_x = x / 2
    result = torch.zeros_like(x)
    for k in range(terms):
        coefficient = (-1.0) ** k / (math.factorial(k) * math.factorial(k + order))
        result = result + coefficient * half_x ** (2 * k + order)
    return result


class AnalyticRotatingTE111:
    """Ideal rotating TE111 fields satisfying Faraday's law analytically."""

    name = "analytic_rotating_te111"

    def __init__(
        self,
        radius_m: float,
        length_m: float,
        frequency_hz: float,
        electric_field_amplitude_v_per_m: float,
        rf_ramp_cycles: int = 10,
    ) -> None:
        self.radius_m = radius_m
        self.length_m = length_m
        self.frequency_hz = frequency_hz
        self.omega = 2 * math.pi * frequency_hz
        self.E0 = electric_field_amplitude_v_per_m
        self.rf_ramp_cycles = rf_ramp_cycles
        self.k_r = TE11_BESSEL_DERIVATIVE_ROOT / radius_m
        self.k_z = math.pi / length_m
        # At r -> 0, both k_r J1'(k_r r) and J1(k_r r)/r tend to k_r/2.
        self.profile_normalization = self.k_r / 2

    def envelope(self, time: float) -> float:
        ramp_time = self.rf_ramp_cycles / self.frequency_hz
        if time <= 0:
            return 0.0
        if time >= ramp_time:
            return 1.0
        return math.sin(math.pi * time / (2 * ramp_time)) ** 2

    def _spatial_profiles(self, positions: torch.Tensor):
        x, y, z = positions.unbind(-1)
        r = torch.sqrt(x * x + y * y)
        safe_r = torch.clamp(r, min=1e-12)
        cos_theta, sin_theta = x / safe_r, y / safe_r
        argument = self.k_r * r
        j0 = _bessel_j(0, argument)
        j1 = _bessel_j(1, argument)
        j2 = _bessel_j(2, argument)
        j1_prime = (j0 - j2) / 2
        j1_over_r = torch.where(
            r < 1e-9,
            torch.full_like(r, self.k_r / 2),
            j1 / safe_r,
        )
        radial_derivative = self.k_r * j1_prime
        axial = torch.cos(self.k_z * z)
        axial_derivative = -self.k_z * torch.sin(self.k_z * z)
        scale = self.E0 / self.profile_normalization

        # E = z_hat x grad_transverse(psi), for psi_a=J1 cos(theta) and
        # psi_b=J1 sin(theta). Convert each orthogonal profile to Cartesian.
        er_a = j1_over_r * sin_theta * axial
        et_a = radial_derivative * cos_theta * axial
        er_b = -j1_over_r * cos_theta * axial
        et_b = radial_derivative * sin_theta * axial
        ea = scale * torch.stack(
            (er_a * cos_theta - et_a * sin_theta,
             er_a * sin_theta + et_a * cos_theta,
             torch.zeros_like(r)), dim=-1
        )
        eb = scale * torch.stack(
            (er_b * cos_theta - et_b * sin_theta,
             er_b * sin_theta + et_b * cos_theta,
             torch.zeros_like(r)), dim=-1
        )
        axis = (r < 1e-9)[:, None]
        ea_axis = scale * torch.stack(
            (torch.zeros_like(r), torch.full_like(r, self.k_r / 2) * axial,
             torch.zeros_like(r)), dim=-1
        )
        eb_axis = scale * torch.stack(
            (-torch.full_like(r, self.k_r / 2) * axial, torch.zeros_like(r),
             torch.zeros_like(r)), dim=-1
        )
        ea = torch.where(axis, ea_axis, ea)
        eb = torch.where(axis, eb_axis, eb)

        # Analytic curls of ea/eb. B follows from dB/dt=-curl(E), preserving
        # the standing-cavity E/B phase relationship rather than assuming E/c.
        cr_a = -radial_derivative * cos_theta * axial_derivative
        ct_a = j1_over_r * sin_theta * axial_derivative
        cr_b = -radial_derivative * sin_theta * axial_derivative
        ct_b = -j1_over_r * cos_theta * axial_derivative
        curl_a = scale * torch.stack(
            (cr_a * cos_theta - ct_a * sin_theta,
             cr_a * sin_theta + ct_a * cos_theta,
             -(self.k_r**2) * j1 * cos_theta * axial), dim=-1
        )
        curl_b = scale * torch.stack(
            (cr_b * cos_theta - ct_b * sin_theta,
             cr_b * sin_theta + ct_b * cos_theta,
             -(self.k_r**2) * j1 * sin_theta * axial), dim=-1
        )
        curl_a_axis = scale * torch.stack(
            (-torch.full_like(r, self.k_r / 2) * axial_derivative,
             torch.zeros_like(r), torch.zeros_like(r)), dim=-1
        )
        curl_b_axis = scale * torch.stack(
            (torch.zeros_like(r),
             -torch.full_like(r, self.k_r / 2) * axial_derivative,
             torch.zeros_like(r)), dim=-1
        )
        curl_a = torch.where(axis, curl_a_axis, curl_a)
        curl_b = torch.where(axis, curl_b_axis, curl_b)
        inside = ((r < self.radius_m) & (z.abs() < self.length_m / 2))[:, None]
        return ea * inside, eb * inside, curl_a * inside, curl_b * inside

    def fields(self, positions: torch.Tensor, time: float):
        ea, eb, curl_a, curl_b = self._spatial_profiles(positions)
        phase = self.omega * time
        cosine, sine = math.cos(phase), math.sin(phase)
        envelope = self.envelope(time)
        electric = envelope * (ea * cosine + eb * sine)
        magnetic = envelope * (-curl_a * sine + curl_b * cosine) / self.omega
        return electric, magnetic

    def electric_field(self, positions, time):
        return self.fields(positions, time)[0]

    def magnetic_field(self, positions, time):
        return self.fields(positions, time)[1]

    def scene_renderers(self):
        return ["resonator"]

    def state_dict(self):
        return {}

    def load_state_dict(self, state):
        return None

    def metadata(self):
        return {
            "mode": "TE111_rotating_analytic",
            "normalization": "on_axis_peak_transverse_electric_field",
            "frequency_hz": self.frequency_hz,
            "rf_ramp_cycles": self.rf_ramp_cycles,
        }
