# Fast reading from the raspberry camera with Python, Numpy, and OpenCV
# Made by @CarlosGS in May 2017
# Adapted by @debOliveira in May 2022
# Updated by @aaronjohnsabu1999 in May 2025
# License: Public Domain, attribution appreciated

import numpy as np
import subprocess as sp
import RPi.GPIO as GPIO
import time, cv2, atexit, socket, argparse, matplotlib.pyplot as plt


def parse_args():
    # parser for command line
    parser = argparse.ArgumentParser(
        description="""Capture client for the MoCap system at the Erobotica lab of UFCG. 
                       Please use it together with the corresponding server script.""",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
        add_help=False,
    )

    # Camera settings
    parser.add_argument("-w", type=int, default=960, help="Image width")
    parser.add_argument("-h", type=int, default=720, help="Image height")
    parser.add_argument("-fps", type=int, default=40, help="Frames per second")
    parser.add_argument("-md", type=int, default=4, help="Camera mode")
    parser.add_argument("-ag", type=int, default=2, help="Analog gain")
    parser.add_argument("-dg", type=int, default=4, help="Digital gain")
    parser.add_argument("-ex", type=str, default="off", help="Exposure mode")
    parser.add_argument("-ss", type=int, default=0, help="Shutter speed in µs")
    parser.add_argument("-awb", type=str, default="off", help="Auto white balance mode")
    parser.add_argument(
        "--awbgains", type=str, default="1.3,1.8", help="AWB gains (r,g)"
    )
    parser.add_argument("-co", type=int, default=100, help="Contrast (0–100)")
    parser.add_argument("-fli", type=str, default="off", help="Flicker avoidance")
    parser.add_argument("-cfx", type=str, default="128,128", help="Color FX (u,v)")
    return parser.parse_args()


class CaptureSession:
    def __init__(self, args):
        self.args = args
        self.frames = []  # stores timestamps
        self.n_frames = 0
        self.max_frames = 500
        self.w, self.h = args.w, args.h
        self.bytes_per_frame = self.w * self.h
        self.winH = int(self.h * 960 / self.w)
        self.led_pin = 4
        self.video_cmd = self.build_command()
        self.camera_proc = None

    def build_command(self):
        # Construct raspividyuv command from arguments
        return [
            "./raspividyuv",
            "--save-pts",
            "-",
            "-t",
            "0",
            "--output",
            "-",
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
            "--luma",
        ]

    def setup_gpio(self):
        # turning LED on
        GPIO.setmode(GPIO.BCM)
        GPIO.setwarnings(False)
        GPIO.setup(self.led_pin, GPIO.OUT)
        GPIO.output(self.led_pin, 1)
        print("[INFO] LED on and parameters configured")

    def shutdown(self):
        # closing buffer
        if self.camera_proc:
            self.camera_proc.terminate()
        GPIO.output(self.led_pin, 0)
        print("[INFO] buffer closed and LED off")

    def start_recording(self):
        print("[INFO] raspividyuv command:", " ".join(self.video_cmd))
        print("[INFO] connecting to server")
        self.setup_gpio()

        # Start the camera
        self.camera_proc = sp.Popen(self.video_cmd, stdout=sp.PIPE)
        atexit.register(self.shutdown)

        print("[INFO] RECORDING ...")
        start_time = time.time()

        while self.n_frames < self.max_frames:
            frame = np.frombuffer(
                self.camera_proc.stdout.read(self.bytes_per_frame), dtype=np.uint8
            )
            if frame.size != self.bytes_per_frame:
                print("[ERROR] Camera stream closed unexpectedly")
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
        print(
            "[RESULTS] "
            + str(round(self.n_frames / elapsed_seconds, 2))
            + " FPS (PTS) and "
            + str(round(self.n_frames / (end_time - start_time), 2))
            + " FPS (time lib)"
        )

    def plot_fps_timeline(self):
        """Optional FPS timeline plotting"""
        if len(self.frames) < 2:
            print("[WARN] Not enough frames for FPS plotting.")
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
        print(f"[INFO] FPS plot saved to {self.h}p{self.args.fps}FPS.png")


def main():
    args = parse_args()
    session = CaptureSession(args)
    session.start_recording()
    # Optional: call session.plot_fps_timeline() if you want the plot


if __name__ == "__main__":
    main()
