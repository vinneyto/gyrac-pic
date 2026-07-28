"""Run neutral hydrogen plasma with only its self-consistent electric field."""

from gyrac_pic import BoxDomain, Experiment, make_smoke_config


def main() -> None:
    config = make_smoke_config()
    config.name = "self_field_only"
    config.visualization.mode = "external"
    config.visualization.spawn_viewer = True

    domain = BoxDomain(
        x_bounds=(-0.05, 0.05),
        y_bounds=(-0.05, 0.05),
        z_bounds=(-0.05, 0.05),
        grid_shape=config.grid.shape,
        boundary_potential=0.0,
    )
    experiment = Experiment.create(config=config, domain=domain, modules=[])
    experiment.print_configuration()
    experiment.print_physical_scales()
    experiment.initialize()
    experiment.visualize_initial_state()
    experiment.run(num_steps=config.num_steps)


if __name__ == "__main__":
    main()
