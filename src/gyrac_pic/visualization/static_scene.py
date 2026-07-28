"""Generate Rerun-native static geometry from the connected domain and modules."""

from __future__ import annotations

import math

import numpy as np


def _circle(radius, z, segments=96):
    return [
        [radius * math.cos(2 * math.pi * i / segments),
         radius * math.sin(2 * math.pi * i / segments), z]
        for i in range(segments + 1)
    ]


def _box_edges(bounds):
    (xmin, xmax), (ymin, ymax), (zmin, zmax) = bounds
    corners = [
        [x, y, z]
        for x in (xmin, xmax)
        for y in (ymin, ymax)
        for z in (zmin, zmax)
    ]
    pairs = []
    for i, first in enumerate(corners):
        for second in corners[i + 1:]:
            if sum(a != b for a, b in zip(first, second)) == 1:
                pairs.append([first, second])
    return pairs


def _sparse_grid(bounds, shape, maximum_lines_per_axis=11):
    axes = []
    for (low, high), count in zip(bounds, shape):
        node_coordinates = np.linspace(low, high, count)
        selected = np.linspace(
            0, count - 1, min(maximum_lines_per_axis, count), dtype=int
        )
        axes.append(node_coordinates[selected].tolist())
    (xmin, xmax), (ymin, ymax), (zmin, zmax) = bounds
    strips = []
    for x in axes[0]:
        for y in axes[1]:
            strips.append([[x, y, zmin], [x, y, zmax]])
    for x in axes[0]:
        for z in axes[2]:
            strips.append([[x, ymin, z], [x, ymax, z]])
    for y in axes[1]:
        for z in axes[2]:
            strips.append([[xmin, y, z], [xmax, y, z]])
    return strips


def _cylinder_mesh(radius, length, segments=96):
    half = length / 2
    vertices = []
    for z in (-half, half):
        vertices.extend(_circle(radius, z, segments)[:-1])
    vertices.extend([[0.0, 0.0, -half], [0.0, 0.0, half]])
    lower_center, upper_center = 2 * segments, 2 * segments + 1
    triangles = []
    for i in range(segments):
        following = (i + 1) % segments
        lower_i, lower_next = i, following
        upper_i, upper_next = segments + i, segments + following
        triangles.extend(
            [[lower_i, lower_next, upper_next], [lower_i, upper_next, upper_i]]
        )
        triangles.append([lower_center, lower_next, lower_i])
        triangles.append([upper_center, upper_i, upper_next])
    colors = [[70, 150, 255, 42]] * (2 * segments) + [
        [70, 150, 255, 25], [70, 150, 255, 25]
    ]
    return vertices, triangles, colors


def log_static_scene(rr, domain, grid, modules, config):
    """Log only geometry implied by the actual domain and attached modules."""
    if config.log_grid:
        rr.log(
            "scene/computational_grid",
            rr.LineStrips3D(
                _sparse_grid(domain.bounds, grid.shape),
                colors=[125, 135, 150, 35],
                radii=1.5e-5,
            ),
            static=True,
        )
    if config.log_domain:
        rr.log(
            "scene/domain_bounds",
            rr.LineStrips3D(
                _box_edges(domain.bounds), colors=[180, 190, 205, 120], radii=4e-5
            ),
            static=True,
        )

    renderer_hints = {
        hint
        for module in modules
        for hint in module.scene_renderers()
    }
    if "resonator" in renderer_hints:
        resonator = next(module for module in modules if "resonator" in module.scene_renderers())
        vertices, triangles, colors = _cylinder_mesh(
            resonator.radius_m, resonator.length_m
        )
        rr.log(
            "scene/resonator/pec_cavity",
            rr.Mesh3D(
                vertex_positions=vertices,
                triangle_indices=triangles,
                vertex_colors=colors,
            ),
            static=True,
        )
        rr.log(
            "scene/resonator/rims",
            rr.LineStrips3D(
                [_circle(resonator.radius_m, -resonator.length_m / 2),
                 _circle(resonator.radius_m, resonator.length_m / 2)],
                colors=[75, 165, 255, 210],
                radii=1.5e-4,
            ),
            static=True,
        )
        rr.log(
            "scene/resonator/axis",
            rr.LineStrips3D(
                [[[0, 0, -resonator.length_m / 2],
                  [0, 0, resonator.length_m / 2]]],
                colors=[225, 225, 235, 150],
                radii=5e-5,
            ),
            static=True,
        )

    if "magnets" in renderer_hints:
        magnetic = next(module for module in modules if "magnets" in module.scene_renderers())
        coil_radius = max(abs(domain.bounds[0][0]), abs(domain.bounds[0][1])) * 1.08
        end = magnetic.length * 0.40
        spacing = magnetic.length * 0.012
        coils = [
            _circle(coil_radius, sign * end + offset * spacing)
            for sign in (-1, 1)
            for offset in range(-2, 3)
        ]
        rr.log(
            "scene/magnets/coils",
            rr.LineStrips3D(
                coils, colors=[235, 125, 35, 255], radii=7.5e-4
            ),
            static=True,
        )
        rr.log(
            "scene/magnets/field_direction",
            rr.Arrows3D(
                origins=[[0, 0, -magnetic.length * 0.3]],
                vectors=[[0, 0, magnetic.length * 0.6]],
                colors=[255, 205, 55, 220],
                radii=2.5e-4,
            ),
            static=True,
        )
