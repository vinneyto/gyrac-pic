import math

from gyrac_pic.visualization.static_scene import (
    _box_edges,
    _cylinder_mesh,
    _sparse_grid,
)


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
