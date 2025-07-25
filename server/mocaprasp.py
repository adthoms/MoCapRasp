#!/usr/bin/env python3
import os
import pickle
from datetime import datetime
import click

from mcr.capture.CEC import CEC
from mcr.capture.GPE import GPE
from mcr.capture.SCR import SCR


@click.group()
def mocaprasp():
    """
    MoCap Rasp - Optical Tracking Arena\n\n
    Server script for the MoCap system at the Erobotica Lab, UFCG.\n
    Please use it together with the corresponding client script.
    """
    pass


@click.command(name="cec")
@click.option(
    "--cameraids",
    "-c",
    default="1,2,3",
    help="List of active camera IDs (Default: 1,2,3)",
)
@click.option(
    "--markers", "-m", default=3, help="Number of expected markers (Default: 3)"
)
@click.option(
    "--trigger", "-t", default=10, help="Trigger time in seconds (Default: 10)"
)
@click.option(
    "--record", "-r", default=360, help="Recording time in seconds (Default: 360)"
)
@click.option("--fps", "-f", default=100, help="Interpolation FPS (Default: 100)")
@click.option(
    "--verbose", "-v", is_flag=True, help="Show ordering and interpolation verbosity"
)
@click.option("--save", "-s", is_flag=True, help="Save received packages to CSV")
@click.option("--dbscan-eps", default=0.01, help="DBSCAN epsilon for clustering")
@click.option(
    "--dbscan-min-samples", default=10, help="DBSCAN minimum samples for clustering"
)
@click.option(
    "--use-clustering",
    is_flag=True,
    help="Enable 3D clustering for consensus filtering",
)
@click.option(
    "--collect", is_flag=True, help="Run in collection mode only (no calibration)"
)
@click.option(
    "--calibrate",
    type=click.Path(exists=True),
    help="Run in calibration-only mode using saved CSV file",
)
def cec(
    cameraids,
    markers,
    trigger,
    record,
    fps,
    verbose,
    save,
    dbscan_eps,
    dbscan_min_samples,
    use_clustering,
    collect,
    calibrate,
):
    """
    Camera Extrinsics Calibration
    Use either --collect or --calibrate:
    --collect    → Collect raw 2D data from cameras and save
    --calibrate  → Load CSV and compute extrinsics (no live capture)
    """

    if collect and calibrate:
        click.echo("⚠️  You cannot specify both --collect and --calibrate.")
        click.echo("Example: python3 mocaprasp.py cec --collect")
        click.echo("         python3 mocaprasp.py cec --calibrate path/to/file.csv")
        return
    elif not collect and not calibrate:
        click.echo("⚠️  You must specify either --collect or --calibrate.")
        click.echo("Example: python3 mocaprasp.py cec --collect")
        click.echo("         python3 mocaprasp.py cec --calibrate path/to/file.csv")
        return

    cecServer = CEC(
        cameraids,
        markers,
        trigger,
        record,
        fps,
        verbose,
        save,
        dbscan_eps,
        dbscan_min_samples,
        use_clustering,
    )

    if collect:
        cecServer.connect()
        cecServer.collect()

        ymd, now = datetime.now().strftime("%y-%m-%d"), datetime.now().strftime(
            "%H-%M-%S"
        )
        out_dir = "debug/dataSaves/" + ymd + "/"
        os.makedirs(out_dir, exist_ok=True)
        out_path = os.path.join(out_dir, f"CEC-{now}.pkl")
        with open(out_path, "wb") as f:
            pickle.dump(cecServer, f)
        click.echo(f"✅ CEC data saved to {out_path}")

    if calibrate:
        if calibrate.endswith(".pkl"):
            with open(calibrate, "rb") as f:
                cecServer = pickle.load(f)
            cecServer.calibrate(datapath=None)
        elif calibrate.endswith(".csv"):
            cecServer = CEC(
                cameraids,
                markers,
                trigger,
                record,
                fps,
                verbose,
                save,
                dbscan_eps,
                dbscan_min_samples,
                use_clustering,
            )
            cecServer.calibrate(datapath=calibrate)
        else:
            click.echo("❌ Unsupported file type. Use .pkl or .csv")
        click.echo("✅ Camera extrinsics calibration completed.")


@click.command(name="gpe")
@click.option(
    "--cameraids",
    "-c",
    default="1,2,3",
    help="List of active camera IDs (Default: 1,2,3)",
)
@click.option(
    "--markers", "-m", default=3, help="Number of expected markers (Default: 3)"
)
@click.option("--trigger", "-t", default=2, help="Trigger time in seconds (Default: 2)")
@click.option(
    "--record", "-r", default=10, help="Recording time in seconds (Default: 10)"
)
@click.option("--fps", "-f", default=100, help="Interpolation FPS (Default: 100)")
@click.option(
    "--verbose", "-v", is_flag=True, help="Show ordering and interpolation verbosity"
)
@click.option("--save", "-s", is_flag=True, help="Save received packages to CSV")
@click.option(
    "--collect", is_flag=True, help="Run in collection mode only (no estimation)"
)
@click.option(
    "--estimate",
    type=click.Path(exists=True),
    help="Run ground plane estimation using a saved CSV file",
)
def gpe(cameraids, markers, trigger, record, fps, verbose, save, collect, estimate):
    """
    Ground Plane Estimation\n\n
    - Place 3 non-collinear markers in the calibration wand;\n
    - Put it at the center of the capture volume;\n
    - Make sure they are levelled with each other.\n\n
    The default options are already adjusted for this process.
    Use either --collect or --estimate:
    --collect    → Collect raw 2D data from cameras and save
    --estimate   → Load CSV and compute ground plane (no live capture)
    """
    if collect and estimate:
        click.echo("❌ Cannot use --collect and --estimate together.")
        click.echo("Example: python3 mocaprasp.py gpe --collect")
        click.echo("         python3 mocaprasp.py gpe --estimate path/to/file.csv")
        return
    elif not collect and not estimate:
        click.echo("❌ You must specify either --collect or --estimate.")
        click.echo("Example: python3 mocaprasp.py gpe --collect")
        click.echo("         python3 mocaprasp.py gpe --estimate path/to/file.csv")
        return

    gpeServer = GPE(cameraids, markers, trigger, record, fps, verbose, save)

    if collect:
        gpeServer.connect()
        gpeServer.collect()

        ymd, now = datetime.now().strftime("%y-%m-%d"), datetime.now().strftime(
            "%H-%M-%S"
        )
        out_dir = "debug/dataSaves/" + ymd + "/"
        os.makedirs(out_dir, exist_ok=True)
        out_path = os.path.join(out_dir, f"GPE-{now}.pkl")
        with open(out_path, "wb") as f:
            pickle.dump(gpeServer, f)
        click.echo(f"✅ GPE data saved to {out_path}")

    if estimate:
        if estimate.endswith(".pkl"):
            with open(estimate, "rb") as f:
                gpeServer = pickle.load(f)
            gpeServer.estimate(datapath=None)
        elif estimate.endswith(".csv"):
            gpeServer = GPE(cameraids, markers, trigger, record, fps, verbose, save)
            gpeServer.estimate(datapath=estimate)
        else:
            click.echo("❌ Unsupported file type. Use .pkl or .csv")
        click.echo("✅ Ground plane estimation completed.")


@click.command(name="scr")
@click.option(
    "--cameraids",
    "-c",
    default="1,2,3",
    help="List of active camera IDs (Default: 1,2,3)",
)
@click.option(
    "--markers", "-m", default=3, help="Number of expected markers (Default: 3)"
)
@click.option("--trigger", "-t", default=5, help="Trigger time in seconds (Default: 5)")
@click.option(
    "--record", "-r", default=30, help="Recording time in seconds (Default: 30)"
)
@click.option("--fps", "-f", default=100, help="Interpolation FPS (Default: 100)")
@click.option(
    "--verbose", "-v", is_flag=True, help="Show ordering and interpolation verbosity"
)
@click.option("--save", "-s", is_flag=True, help="Save received packages to CSV")
def scr(cameraids, markers, trigger, record, fps, verbose, save):
    """
    Standard Capture Routine\n\n
    - Routines required to be done previously at least once:\n
        1. CEC;\n
        2. GPE.\n
    - With CEC and GPE routines done, execute SCR as much as you like;\n\n
    Adjust the options to match your desired capture.
    """
    scrServer = SCR(cameraids, markers, trigger, record, fps, verbose, save)
    scrServer.connect()
    scrServer.collect()


mocaprasp.add_command(cec)
mocaprasp.add_command(scr)
mocaprasp.add_command(gpe)

if __name__ == "__main__":
    mocaprasp()
