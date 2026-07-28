import torch


class RampedMirrorMagneticField:
    """Paraxial mirror field and Faraday-induced E approximation in SI."""
    name = "ramped_mirror"
    def __init__(self, B0_tesla, delta_B_max_tesla, ramp_time_s, mirror_ratio, resonator_length_m):
        self.B0, self.delta = B0_tesla, delta_B_max_tesla
        self.ramp_time_s, self.mirror_ratio, self.length = ramp_time_s, mirror_ratio, resonator_length_m

    def central_field(self, time): return self.B0 + self.delta * min(max(time/self.ramp_time_s, 0), 1)
    def magnetic_field(self, p, time):
        x, y, z = p.unbind(-1); bmid = self.central_field(time)
        bz = bmid * (1 + (self.mirror_ratio-1)*(2*z/self.length)**2)
        dbdz = bmid * 8*(self.mirror_ratio-1)*z/self.length**2
        bx, by = -0.5*x*dbdz, -0.5*y*dbdz
        return torch.stack((bx, by, bz), -1)
    def electric_field(self, p, time):
        rate = self.delta/self.ramp_time_s if 0 < time < self.ramp_time_s else 0.0
        return torch.stack((0.5*p[:,1]*rate, -0.5*p[:,0]*rate, torch.zeros_like(p[:,0])), -1)
    def scene_renderers(self): return ["magnets", "magnetic_field_indicators"]
    def state_dict(self): return {}
    def load_state_dict(self, state): pass
