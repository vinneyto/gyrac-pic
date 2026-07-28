import math

from gyrac_pic.visualization.static_scene import (
    _box_edges,
    _cylinder_mesh,
    _parameter_table,
    _sparse_grid,
    log_static_scene,
)
from gyrac_pic import (
    AnalyticRotatingTE111,
    BoxDomain,
    CylindricalPECDomain,
    RampedMirrorMagneticField,
    make_smoke_config,
)
from gyrac_pic.device import dtype_from_name
from gyrac_pic.grid import CartesianGrid
import torch


def test_sparse_grid_uses_real_grid_nodes_and_domain_bounds():
    bounds = ((-1.0, 1.0), (-2.0, 2.0), (-3.0, 3.0))
    strips = _sparse_grid(bounds, (5, 7, 9), maximum_lines_per_axis=3)
    assert len(strips) == 27
    assert strips[0] == [[-1.0, -2.0, -3.0], [-1.0, -2.0, 3.0]]


def test_cylinder_mesh_respects_radius_and_length():
    radius, length, segments = 0.25, 0.8, 16
    vertices, triangles, colors = _cylinder_mesh(radius, length, segments)
    assert len(vertices) == 2 * segments + 2
    assert len(triangles) == 4 * segments
    assert len(colors) == len(vertices)
    for x, y, z in vertices[:-2]:
        assert math.isclose(math.hypot(x, y), radius, rel_tol=1e-12)
        assert math.isclose(abs(z), length / 2, rel_tol=1e-12)


def test_box_has_twelve_edges():
    assert len(_box_edges(((-1, 1), (-1, 1), (-1, 1)))) == 12


def test_parameter_table_includes_values_and_explicit_units():
    config = make_smoke_config()
    domain = CylindricalPECDomain(
        config.resonator.radius_m, config.resonator.length_m, config.grid.shape
    )
    grid = CartesianGrid(
        config.grid.shape, domain.bounds, torch.device("cpu"),
        dtype_from_name(config.grid.dtype_name),
    )
    resonator = AnalyticRotatingTE111(
        config.resonator.radius_m, config.resonator.length_m,
        config.resonator.frequency_hz,
        config.resonator.electric_field_amplitude_v_per_m,
    )
    magnet = RampedMirrorMagneticField(
        config.magnetic_field.B0_tesla,
        config.magnetic_field.delta_B_max_tesla,
        config.magnetic_field.ramp_time_s,
        config.magnetic_field.mirror_ratio,
        config.resonator.length_m,
    )
    table = _parameter_table(domain, grid, [resonator, magnet], config)
    assert "RF frequency" in table and "Hz" in table
    assert "Configured RF E amplitude" in table and "V/m" in table
    assert "Resonator mode" in table and "TE111_rotating_analytic" in table
    assert "Initial central magnetic field" in table and "| T |" in table


def test_text_failure_does_not_suppress_grid_geometry():
    class FakeRerun:
        def __init__(self):
            self.paths = []

        def LineStrips3D(self, *args, **kwargs):
            return (args, kwargs)

        def TextDocument(self, *args, **kwargs):
            raise TypeError("unsupported text document")

        def log(self, path, archetype, static=False):
            assert static
            self.paths.append(path)

    config = make_smoke_config()
    domain = BoxDomain(
        (-0.05, 0.05), (-0.05, 0.05), (-0.05, 0.05), config.grid.shape
    )
    grid = CartesianGrid(
        config.grid.shape, domain.bounds, torch.device("cpu"), torch.float32
    )
    rerun = FakeRerun()
    log_static_scene(
        rerun, domain, grid, modules=[],
        visualization_config=config.visualization,
        experiment_config=config,
    )
    assert "scene/computational_grid" in rerun.paths
    assert "scene/domain_bounds" in rerun.paths
