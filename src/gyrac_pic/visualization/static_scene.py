"""Generate Rerun-native static geometry from the connected domain and modules."""

from __future__ import annotations

import math
import logging

import numpy as np


log = logging.getLogger(__name__)


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


def _local_actual_grid(bounds, shape, plasma):
    """Return every real grid line in a patch enclosing the initial plasma."""
    node_axes = [
        np.linspace(low, high, count)
        for (low, high), count in zip(bounds, shape)
    ]
    requested_bounds = (
        (-1.5 * plasma.radius_m, 1.5 * plasma.radius_m),
        (-1.5 * plasma.radius_m, 1.5 * plasma.radius_m),
        (
            plasma.center_z_m - plasma.length_m / 2,
            plasma.center_z_m + plasma.length_m / 2,
        ),
    )
    selected_axes = []
    for nodes, (requested_low, requested_high) in zip(node_axes, requested_bounds):
        inside = np.flatnonzero((nodes >= requested_low) & (nodes <= requested_high))
        if len(inside) == 0:
            inside = np.array([int(np.argmin(np.abs(nodes - (requested_low + requested_high) / 2)))])
        first = max(int(inside[0]) - 1, 0)
        last = min(int(inside[-1]) + 1, len(nodes) - 1)
        selected_axes.append(nodes[first:last + 1].tolist())
    xs, ys, zs = selected_axes
    strips = []
    for x in xs:
        for y in ys:
            strips.append([[x, y, zs[0]], [x, y, zs[-1]]])
    for x in xs:
        for z in zs:
            strips.append([[x, ys[0], z], [x, ys[-1], z]])
    for y in ys:
        for z in zs:
            strips.append([[xs[0], y, z], [xs[-1], y, z]])
    return strips, selected_axes


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


def _parameter_table(domain, grid, modules, config):
    z_nodes = np.linspace(domain.bounds[2][0], domain.bounds[2][1], grid.shape[2])
    potential_plane_z = float(z_nodes[np.argmin(np.abs(z_nodes))])
    rows = [
        ("Recording / experiment ID", config.name, "—"),
        ("Compute device", str(grid.device), "—"),
        ("Floating-point dtype", str(grid.dtype).removeprefix("torch."), "—"),
        ("Timestep", f"{config.dt_s:.6g}", "s"),
        ("Configured steps", f"{config.num_steps}", "steps"),
        ("Grid shape", " × ".join(map(str, grid.shape)), "nodes"),
        ("Grid spacing dx", f"{grid.spacing[0]:.6g}", "m"),
        ("Grid spacing dy", f"{grid.spacing[1]:.6g}", "m"),
        ("Grid spacing dz", f"{grid.spacing[2]:.6g}", "m"),
        ("Domain x bounds", f"{domain.bounds[0][0]:.6g} … {domain.bounds[0][1]:.6g}", "m"),
        ("Domain y bounds", f"{domain.bounds[1][0]:.6g} … {domain.bounds[1][1]:.6g}", "m"),
        ("Domain z bounds", f"{domain.bounds[2][0]:.6g} … {domain.bounds[2][1]:.6g}", "m"),
        ("Poisson tolerance", f"{config.poisson.relative_tolerance:.3g}", "relative [1]"),
        ("Poisson solve stride", f"{config.poisson.solve_stride}", "steps"),
        ("Global grid display", "at most 11 nodes per axis", "visualization only"),
        ("Local plasma grid display", "every computational node", "actual cells"),
        ("Potential image plane", f"z = {potential_plane_z:.6g}", "m"),
        ("Potential image color", "blue / white / red", "negative / zero / positive V"),
        ("Potential image range", f"symmetric {config.visualization.potential_color_percentile:g}th percentile", "V"),
    ]
    for species in config.species:
        label = species.name.capitalize()
        rows.extend(
            [
                (f"{label} macroparticles", f"{species.count}", "count"),
                (f"{label} density", f"{species.density_m3:.6g}", "m⁻³"),
                (f"{label} temperature", f"{species.temperature_ev:.6g}", "eV"),
            ]
        )
    resonators = [m for m in modules if "resonator" in m.scene_renderers()]
    if resonators:
        resonator = resonators[0]
        metadata = resonator.metadata()
        rows.extend(
            [
                ("Resonator mode", str(metadata.get("mode", type(resonator).__name__)), "—"),
                ("RF frequency", f"{resonator.frequency_hz:.9g}", "Hz"),
                ("RF angular frequency", f"{resonator.omega:.9g}", "rad/s"),
                ("Configured RF E amplitude", f"{resonator.E0:.9g}", "V/m"),
                ("RF ramp", f"{resonator.rf_ramp_cycles}", "RF cycles"),
                ("Resonator radius", f"{resonator.radius_m:.9g}", "m"),
                ("Resonator length", f"{resonator.length_m:.9g}", "m"),
            ]
        )
    magnets = [m for m in modules if "magnets" in m.scene_renderers()]
    if magnets:
        magnetic = magnets[0]
        rows.extend(
            [
                ("Initial central magnetic field", f"{magnetic.B0:.9g}", "T"),
                ("Maximum magnetic-field increment", f"{magnetic.delta:.9g}", "T"),
                ("Magnetic ramp time", f"{magnetic.ramp_time_s:.9g}", "s"),
                ("Mirror ratio", f"{magnetic.mirror_ratio:.9g}", "B_end/B_mid [1]"),
            ]
        )
    lines = [
        f"# GYRAC experiment: `{config.name}`",
        "",
        "> This Viewer tab represents one recording. Other runs are separate `.rrd` recordings.",
        "",
        "| Parameter | Value | Unit |",
        "|---|---:|---|",
    ]
    lines.extend(f"| {name} | {value} | {unit} |" for name, value, unit in rows)
    return "\n".join(lines)


def log_static_scene(rr, domain, grid, modules, visualization_config, experiment_config):
    """Log only geometry implied by the actual domain and attached modules."""
    def safe_log(path, archetype_factory):
        # One unsupported Rerun archetype must not prevent all remaining static
        # geometry from reaching the viewer.
        try:
            rr.log(path, archetype_factory(), static=True)
            return True
        except Exception as error:
            log.warning("Could not log static Rerun entity %s: %s", path, error)
            return False

    if visualization_config.log_grid:
        safe_log(
            "scene/grid/global_sparse",
            lambda: rr.LineStrips3D(
                _sparse_grid(domain.bounds, grid.shape),
                colors=[125, 135, 150, 24],
                radii=1.5e-5,
            ),
        )
        local_strips, _ = _local_actual_grid(
            domain.bounds, grid.shape, experiment_config.plasma
        )
        safe_log(
            "scene/grid/local_actual_cells",
            lambda: rr.LineStrips3D(
                local_strips,
                colors=[80, 225, 235, 105],
                radii=2.5e-5,
            ),
        )
    if visualization_config.log_domain:
        safe_log(
            "scene/domain_bounds",
            lambda: rr.LineStrips3D(
                _box_edges(domain.bounds), colors=[180, 190, 205, 120], radii=4e-5
            ),
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
        safe_log(
            "scene/resonator/pec_cavity",
            lambda: rr.Mesh3D(
                vertex_positions=vertices,
                triangle_indices=triangles,
                vertex_colors=colors,
            ),
        )
        safe_log(
            "scene/resonator/rims",
            lambda: rr.LineStrips3D(
                [_circle(resonator.radius_m, -resonator.length_m / 2),
                 _circle(resonator.radius_m, resonator.length_m / 2)],
                colors=[75, 165, 255, 210],
                radii=1.5e-4,
            ),
        )
        safe_log(
            "scene/resonator/axis",
            lambda: rr.LineStrips3D(
                [[[0, 0, -resonator.length_m / 2],
                  [0, 0, resonator.length_m / 2]]],
                colors=[225, 225, 235, 150],
                radii=5e-5,
            ),
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
        safe_log(
            "scene/magnets/coils",
            lambda: rr.LineStrips3D(
                coils, colors=[235, 125, 35, 255], radii=7.5e-4
            ),
        )
        safe_log(
            "scene/magnets/field_direction",
            lambda: rr.Arrows3D(
                origins=[[0, 0, -magnetic.length * 0.3]],
                vectors=[[0, 0, magnetic.length * 0.6]],
                colors=[255, 205, 55, 220],
                radii=2.5e-4,
            ),
        )

    media_type = (
        rr.MediaType.MARKDOWN
        if hasattr(rr, "MediaType") and hasattr(rr.MediaType, "MARKDOWN")
        else "text/markdown"
    )
    safe_log(
        "experiment/parameters",
        lambda: rr.TextDocument(
            _parameter_table(domain, grid, modules, experiment_config),
            media_type=media_type,
        ),
    )
