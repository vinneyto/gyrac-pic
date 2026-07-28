"""Run plasma in the rotating TE111 resonator without external magnets."""

from pathlib import Path

from gyrac_pic import (
    CylindricalPECDomain,
    Experiment,
    TE111CylindricalResonator,
    make_smoke_config,
)


def main() -> None:
    config = make_smoke_config()
    config.name = "plasma_with_resonator"
    config.visualization.mode = "external"
    config.visualization.spawn_viewer = True

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
    experiment = Experiment.create(
        config=config,
        domain=domain,
        modules=[resonator],
    )
    experiment.print_configuration()
    experiment.print_physical_scales()
    experiment.initialize()
    experiment.visualize_initial_state()
    experiment.run(num_steps=config.num_steps)
    experiment.save_checkpoint(
        Path("runs/plasma_with_resonator/checkpoints/final.pt")
    )


if __name__ == "__main__":
    main()
