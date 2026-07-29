"""Run TE111 plasma dynamics without external magnets."""

import argparse
from datetime import datetime, timezone
from pathlib import Path

import torch

from gyrac_pic import (
    AdvancedDiagnosticsCollector,
    AnalyticRotatingTE111,
    CylindricalPECDomain,
    Experiment,
    make_classic_gyrac_x_smoke_config,
)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--steps", type=int, default=10_000)
    parser.add_argument("--report-stride", type=int, default=100)
    parser.add_argument("--no-viewer", action="store_true")
    args = parser.parse_args()

    timestamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    run_id = f"te111_without_magnets_{timestamp}"
    run_directory = Path("runs/te111_without_magnets") / timestamp
    config = make_classic_gyrac_x_smoke_config()
    config.name = run_id
    config.num_steps = args.steps
    config.visualization.mode = "file" if args.no_viewer else "external_and_file"
    config.visualization.spawn_viewer = not args.no_viewer
    config.visualization.recording_path = run_directory / "run.rrd"
    config.visualization.visualization_stride = 10

    domain = CylindricalPECDomain(
        config.resonator.radius_m,
        config.resonator.length_m,
        config.grid.shape,
        boundary_potential=0.0,
    )
    resonator = AnalyticRotatingTE111(
        config.resonator.radius_m,
        config.resonator.length_m,
        config.resonator.frequency_hz,
        config.resonator.electric_field_amplitude_v_per_m,
        config.resonator.rf_ramp_cycles,
    )
    experiment = Experiment.create(config, domain, modules=[resonator])
    experiment.print_configuration()
    experiment.print_physical_scales()
    experiment.initialize()
    experiment.visualize_initial_state()

    collector = AdvancedDiagnosticsCollector(
        experiment, resonator, sample_stride=10, report_stride=args.report_stride
    )
    collector.observe(force_report=True)
    for _ in range(args.steps):
        with torch.inference_mode():
            experiment.step()
        if experiment.state.step % config.visualization.visualization_stride == 0:
            experiment.visualizer.log_frame(
                experiment.state,
                experiment.latest_diagnostics,
                config.visualization.max_particles_per_species,
            )
        collector.observe()
    if experiment.state.step % args.report_stride:
        collector.observe(force_report=True)

    checkpoint = experiment.save_checkpoint(run_directory / "final.pt")
    collector.save_summary(
        run_directory / "summary.json",
        metadata={
            "run_id": run_id,
            "scenario": "te111_without_magnets",
            "steps": args.steps,
            "dt_s": config.dt_s,
            "checkpoint": str(checkpoint),
            "recording": str(config.visualization.recording_path),
            "resonator": resonator.metadata(),
            "experiment_config": config.to_dict(),
        },
    )


if __name__ == "__main__":
    main()
