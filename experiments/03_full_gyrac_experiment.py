"""Run the complete GYRAC profile with resonator and mirror field."""

from pathlib import Path

from gyrac_pic import (
    CylindricalPECDomain,
    Experiment,
    RampedMirrorMagneticField,
    TE111CylindricalResonator,
    make_classic_gyrac_x_smoke_config,
)


def main() -> None:
    config = make_classic_gyrac_x_smoke_config()
    config.name = "full_gyrac"
    config.visualization.mode = "external_and_file"
    config.visualization.spawn_viewer = True
    config.visualization.recording_path = Path("runs/full_gyrac/run.rrd")

    domain = CylindricalPECDomain(
        radius_m=config.resonator.radius_m,
        length_m=config.resonator.length_m,
        grid_shape=config.grid.shape,
        boundary_potential=0.0,
    )
    resonator = TE111CylindricalResonator(
        radius_m=config.resonator.radius_m,
        length_m=config.resonator.length_m,
        frequency_hz=config.resonator.frequency_hz,
        electric_field_amplitude_v_per_m=(
            config.resonator.electric_field_amplitude_v_per_m
        ),
        rf_ramp_cycles=config.resonator.rf_ramp_cycles,
    )
    magnetic_field = RampedMirrorMagneticField(
        B0_tesla=config.magnetic_field.B0_tesla,
        delta_B_max_tesla=config.magnetic_field.delta_B_max_tesla,
        ramp_time_s=config.magnetic_field.ramp_time_s,
        mirror_ratio=config.magnetic_field.mirror_ratio,
        resonator_length_m=config.resonator.length_m,
    )
    experiment = Experiment.create(
        config=config,
        domain=domain,
        modules=[resonator, magnetic_field],
    )
    experiment.print_configuration()
    experiment.print_physical_scales()
    experiment.initialize()
    experiment.visualize_initial_state()
    experiment.run(num_steps=config.num_steps)
    experiment.save_checkpoint(Path("runs/full_gyrac/checkpoints/final.pt"))


if __name__ == "__main__":
    main()
