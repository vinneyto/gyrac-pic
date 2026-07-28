"""Analytic approximation to a rotating TE111 cavity mode.

The cylinder axis is +z and its centre is the origin. Two orthogonal real m=1
profiles are driven in quadrature. ``E0`` normalizes the transverse electric
profile in V/m; magnetic amplitudes follow locally from B=kh×E/omega. The
small polynomial approximation to J1 avoids host-side SciPy calls and remains
differentiable/device-native. Fields are SI and exactly zero outside the cavity.
"""
import math
import warnings
import torch
from ...constants import SPEED_OF_LIGHT, TE11_BESSEL_DERIVATIVE_ROOT


def _j1(x):
    # Accurate enough over the required interval 0 <= x <= 1.841.
    x2 = x*x
    return x * (0.5 + x2 * (-1/16 + x2 * (1/384 + x2 * (-1/18432))))


class TE111CylindricalResonator:
    name = "te111_resonator"
    def __init__(self, radius_m, length_m, frequency_hz, electric_field_amplitude_v_per_m,
                 rf_ramp_cycles=10, field_evaluation_mode="particle_positions"):
        self.radius_m, self.length_m = radius_m, length_m
        self.frequency_hz = frequency_hz
        self.omega = 2 * math.pi * frequency_hz
        self.E0 = electric_field_amplitude_v_per_m
        self.rf_ramp_cycles = rf_ramp_cycles
        self.field_evaluation_mode = field_evaluation_mode
        self.k_r = TE11_BESSEL_DERIVATIVE_ROOT / radius_m
        self.k_z = math.pi / length_m
        self.mode_frequency_hz = SPEED_OF_LIGHT * math.hypot(self.k_r, self.k_z) / (2*math.pi)
        self.frequency_relative_error = abs(self.mode_frequency_hz-frequency_hz)/frequency_hz
        if self.frequency_relative_error > 0.1:
            warnings.warn("Drive frequency differs from ideal TE111 eigenfrequency by more than 10%", stacklevel=2)

    def _envelope(self, time):
        ramp = self.rf_ramp_cycles / self.frequency_hz
        return 0.0 if time <= 0 else math.sin(math.pi*time/(2*ramp))**2 if time < ramp else 1.0

    def _fields(self, p, time):
        x, y, z = p.unbind(-1)
        r = torch.sqrt(x*x+y*y)
        theta = torch.atan2(y, x)
        radial = _j1(self.k_r*r) * torch.cos(self.k_z*z)
        phase = self.omega*time-theta
        # Rotating transverse polarization tangent to cylindrical radius.
        amp = self.E0 * radial * torch.cos(phase) * self._envelope(time)
        e = torch.stack((-torch.sin(theta)*amp, torch.cos(theta)*amp, torch.zeros_like(amp)), -1)
        direction = torch.stack((-torch.cos(theta), -torch.sin(theta), torch.zeros_like(amp)), -1)
        b = direction * (amp/SPEED_OF_LIGHT)[:, None]
        inside = (r < self.radius_m) & (z.abs() < self.length_m/2)
        return e*inside[:,None], b*inside[:,None]

    def electric_field(self, positions, time): return self._fields(positions, time)[0]
    def magnetic_field(self, positions, time): return self._fields(positions, time)[1]
    def scene_renderers(self): return ["resonator"]
    def state_dict(self): return {}
    def load_state_dict(self, state): pass
    def metadata(self):
        return {"mode": "TE111_rotating", "mode_frequency_hz": self.mode_frequency_hz,
                "drive_frequency_hz": self.frequency_hz, "frequency_relative_error": self.frequency_relative_error}
