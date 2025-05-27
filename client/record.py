# High-speed grayscale video capture on Raspberry Pi using raspividyuv and OpenCV.
# Originally by @CarlosGS (May 2017), adapted by @debOliveira (May 2022),
# modularized and enhanced by @aaronjohnsabu1999 (May 2025).
# License: Public Domain — attribution appreciated.

import numpy as np
import subprocess as sp
import RPi.GPIO as GPIO
import time, cv2, atexit, socket, argparse, matplotlib.pyplot as plt
import logging

logging.basicConfig(
    level=logging.DEBUG,
    format="%(asctime)s [%(levelname)s] %(message)s",
)

log = logging.getLogger(__name__)


def parse_args():
    """
    Define and parse all configurable camera parameters from command-line input.
    Returns:
        argparse.Namespace: Parsed arguments to control raspividyuv behavior.
    """
    parser = argparse.ArgumentParser(
        description="MoCap IR camera capture client for Raspberry Pi at Erobotica Lab (UFCG).",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
        add_help=False,
    )

    # Core camera settings (resolution, frame rate, shutter, gain, etc.)
    parser.add_argument("-w", type=int, default=960, help="Image width")
    parser.add_argument("-h", type=int, default=720, help="Image height")
    parser.add_argument("-fps", type=int, default=40, help="Frames per second")
    parser.add_argument("-md", type=int, default=4, help="Camera mode")
    parser.add_argument(
        "--max_frames",
        type=int,
        default=1000,
        help="Maximum number of frames to capture (0 for infinite)",
    )

    # Exposure/gain control
    parser.add_argument("-ag", type=int, default=2, help="Analog gain")
    parser.add_argument("-dg", type=int, default=4, help="Digital gain")
    parser.add_argument("-ex", type=str, default="off", help="Exposure mode")
    parser.add_argument("-ss", type=int, default=0, help="Shutter speed in µs")

    # White balance and color settings (can be disabled for IR tracking)
    parser.add_argument("-awb", type=str, default="off", help="Auto white balance mode")
    parser.add_argument(
        "--awbgains", type=str, default="1.0,1.0", help="AWB gains (r,b)"
    )
    parser.add_argument("-co", type=int, default=100, help="Contrast (0-100)")
    parser.add_argument("-fli", type=str, default="off", help="Flicker avoidance")
    parser.add_argument("-cfx", type=str, default="128,128", help="Color FX (u,v)")

    return parser.parse_args()


class CaptureSession:
    """
    Handles GPIO setup, raspividyuv process, and high-speed image capture.
    """

    def __init__(self, args):
        self.args = args
        self.frames = []  # Stores timestamps for FPS analysis
        self.n_frames = 0
        self.max_frames = (
            args.max_frames
        )  # Number of frames to capture (can be exposed later)
        self.w, self.h = args.w, args.h
        self.bytes_per_frame = self.w * self.h  # Grayscale frame size
        self.led_pin = 4  # GPIO pin number used to turn on IR LED ring
        self.video_cmd = self.build_command()
        self.camera_proc = None

    def build_command(self):
        """
        Create the raspividyuv command from arguments.
        Returns:
            list: Command-line argument list for subprocess
        """
        return [
            "./raspividyuv",
            "--save-pts",
            "-",  # Output timestamps to stdout
            "-t",
            "0",  # Infinite duration (controlled manually)
            "--output",
            "-",  # Stream to stdout (piped into Python)
            "-w",
            str(self.args.w),
            "-h",
            str(self.args.h),
            "-fps",
            str(self.args.fps),
            "-md",
            str(self.args.md),
            "-ag",
            str(self.args.ag),
            "-dg",
            str(self.args.dg),
            "-ex",
            self.args.ex,
            "-ss",
            str(self.args.ss),
            "-awb",
            self.args.awb,
            "--awbgains",
            self.args.awbgains,
            "-co",
            str(self.args.co),
            "-fli",
            self.args.fli,
            "-cfx",
            self.args.cfx,
            "--luma",  # Only output the luma (grayscale)
        ]

    def setup_gpio(self):
        """
        Enable IR LED via GPIO before starting video capture.
        """
        GPIO.setmode(GPIO.BCM)
        GPIO.setwarnings(False)
        GPIO.setup(self.led_pin, GPIO.OUT)
        GPIO.output(self.led_pin, 1)
        log.info("LED on and parameters configured")

    def shutdown(self):
        """
        Cleanup GPIO and stop camera subprocess when done or interrupted.
        """
        if self.camera_proc:
            self.camera_proc.terminate()
        GPIO.output(self.led_pin, 0)
        log.info("Buffer closed and LED off")

    def start_recording(self):
        """
        Run the main loop: capture N frames and save them to /dev/shm/.
        Also tracks frame timestamps for timing diagnostics.
        """
        log.info("raspividyuv command: %s", " ".join(self.video_cmd))
        log.info("Connecting to server")
        self.setup_gpio()

        # Start raspividyuv streaming
        self.camera_proc = sp.Popen(self.video_cmd, stdout=sp.PIPE)
        atexit.register(self.shutdown)

        log.info("RECORDING ...")
        start_time = time.time()

        while self.n_frames < self.max_frames:
            frame = np.frombuffer(
                self.camera_proc.stdout.read(self.bytes_per_frame), dtype=np.uint8
            )
            if frame.size != self.bytes_per_frame:
                log.error("Camera stream closed unexpectedly")
                break
            frame.shape = (self.h, self.w)

            ts = self.camera_proc.stdout.readline()[-11:-1].decode().strip()
            self.frames.append(ts)
            cv2.imwrite("/dev/shm/" + ts.zfill(10) + ".bmp", frame)
            self.camera_proc.stdout.flush()
            del frame
            self.n_frames += 1

        end_time = time.time()
        self.shutdown()

        elapsed_seconds = float(self.frames[-1]) / 1e6 if self.frames else 1.0
        log.info(
            "[RESULTS] %.2f FPS (PTS) and %.2f FPS (time lib)",
            self.n_frames / elapsed_seconds,
            self.n_frames / (end_time - start_time),
        )

    def plot_fps_timeline(self):
        """
        Generate and save a plot showing actual frame intervals (useful for debugging).
        """
        if len(self.frames) < 2:
            log.warning("Not enough frames for FPS plotting.")
            return

        timestamps = np.array(self.frames).astype(float) / 1e6
        fps_periods = np.diff(timestamps)

        plt.figure(figsize=(4, 3))
        plt.plot(fps_periods, label="Achieved FPS period")
        plt.axhline(
            y=1 / self.args.fps, color="r", linestyle="-", label="Reference FPS period"
        )
        plt.xlim(0, len(fps_periods))
        plt.legend(loc="best")
        plt.grid()
        plt.xlabel("Image number")
        plt.ylabel("Period to previous picture (s)")
        plt.title(f"FPS period at {self.h}p, {self.args.fps}FPS")
        plt.savefig(f"{self.h}p{self.args.fps}FPS.png", dpi=300, bbox_inches="tight")
        log.info("FPS plot saved to %sp%FPS.png", self.h, self.args.fps)


def main():
    """
    Entry point for the recording script.
    """
    args = parse_args()
    session = CaptureSession(args)
    session.start_recording()
    # Uncomment this line if you want a debug plot:
    # session.plot_fps_timeline()


if __name__ == "__main__":
    main()
