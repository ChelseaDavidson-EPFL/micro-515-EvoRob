import os
import shutil
import subprocess
from os.path import join

import numpy as np

from final_project_train import FinalWorld


def _write_video(out_path: str, frames: list[np.ndarray], fps: int = 20) -> None:
    if len(frames) == 0:
        raise ValueError("No frames to write")

    ffmpeg_exe = shutil.which("ffmpeg") or shutil.which("ffmpeg.exe")
    if ffmpeg_exe is None:
        raise RuntimeError("ffmpeg executable not found in PATH")

    height, width = frames[0].shape[:2]
    cmd = [
        ffmpeg_exe,
        "-y",
        "-f",
        "rawvideo",
        "-vcodec",
        "rawvideo",
        "-pix_fmt",
        "rgb24",
        "-s",
        f"{width}x{height}",
        "-r",
        str(fps),
        "-i",
        "-",
        "-an",
        "-vcodec",
        "libx264",
        "-pix_fmt",
        "yuv420p",
        out_path,
    ]
    proc = subprocess.Popen(
        cmd, stdin=subprocess.PIPE, stdout=subprocess.PIPE, stderr=subprocess.PIPE
    )
    try:
        for frm in frames:
            arr = np.ascontiguousarray(frm)
            if arr.ndim != 3 or arr.shape[2] not in (3, 4):
                raise ValueError(f"Expected RGB/RGBA frame, got shape {arr.shape}")
            if arr.shape[2] == 4:
                arr = arr[:, :, :3]
            if arr.dtype != np.uint8:
                if np.issubdtype(arr.dtype, np.floating) and arr.max() <= 1.0:
                    arr = (255 * np.clip(arr, 0.0, 1.0)).astype(np.uint8)
                else:
                    arr = arr.astype(np.uint8)
            proc.stdin.write(arr.tobytes())
    finally:
        if proc.stdin is not None:
            proc.stdin.close()
        stderr = (
            proc.stderr.read().decode("utf-8", errors="replace") if proc.stderr else ""
        )
        return_code = proc.wait()
        if return_code != 0:
            raise RuntimeError(
                f"ffmpeg failed with code {return_code}: {stderr.strip()}"
            )


def build_hexapod_warmstart() -> np.ndarray:
    """Build the hexapod warm-start from explicit target leg lengths.

    The intended geometry is:
        - front legs: 30 cm / 30 cm
        - middle legs: 35 cm / 35 cm
        - back legs: 42 cm / 42 cm

    This keeps the body layout inspired by the quadruped symmetry while making
    the 6-legged target morphology explicit and inspectable.
    """
    # Ordre FinalWorld pour Hexapod: [fl/fr_up, fl/fr_lo, ml/mr_up, ml/mr_lo, bl/br_up, bl/br_lo]
    hex_body_lengths = np.array([0.30, 0.30, 0.35, 0.35, 0.42, 0.42], dtype=float)
    hex_body_genes = 4.0 * (hex_body_lengths - 0.1) - 1.0

    world = FinalWorld(directional=True, leg_layout="hexapod")
    controller_zeros = np.zeros(world.controller.n_params, dtype=float)
    return np.concatenate([controller_zeros, hex_body_genes])


def render_video(out_path: str) -> None:
    os.makedirs(os.path.dirname(out_path), exist_ok=True)

    x = build_hexapod_warmstart()
    world = FinalWorld(directional=True, leg_layout="hexapod")

    print("Hexapod warm-start lengths (m):", [0.30, 0.30, 0.35, 0.35, 0.42, 0.42])

    world.update_robot_xml(x)
    env = world._make_env("FlatEnv-v0", world.flat_world_file, render_mode="rgb_array")
    obs, _ = env.reset(seed=42)

    frames = []
    print(f"Rendering hexapod warmstart proof to {out_path}...")
    for _ in range(100):  # 5 secondes
        frames.append(env.render())
        # Action nulle pour observer la posture statique
        obs, _, term, trunc, _ = env.step(np.zeros(world.n_actuators))
        if term or trunc:
            break

    env.close()

    os.makedirs(os.path.dirname(out_path), exist_ok=True)

    if len(frames) == 0:
        raise Exception("No frames captured")

    _write_video(out_path, frames, fps=20)

    warmstart_path = "results/warmstarts/hexapod_bio_warmstart.npy"
    os.makedirs(os.path.dirname(warmstart_path), exist_ok=True)
    np.save(warmstart_path, x)
    print(f"Wrote {out_path}")
    print(f"Warmstart saved to {warmstart_path}")


if __name__ == "__main__":
    render_video(join("evaluation_output", "warmstart", "final_test_hexapod_proof.mp4"))
