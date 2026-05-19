"""Build a warm-start genotype for the hexapod from the current checkpoint.

The goal is to keep the controller from the current hexapod run but replace the
body parameters with the biologically inspired 30/35/42 cm front/mid/back
segment lengths. The result is saved as a standalone .npy plus a small text
manifest so it can be reused for training or inspection.
"""

from __future__ import annotations

import argparse
import os
from os.path import join

import numpy as np

from final_project_train import FinalWorld


def lengths_to_genes(lengths: np.ndarray) -> np.ndarray:
    return 4.0 * (lengths - 0.1) - 1.0


def build_warmstart(checkpoint_path: str) -> tuple[np.ndarray, FinalWorld, np.ndarray, np.ndarray]:
    checkpoint = np.load(checkpoint_path, allow_pickle=True)
    world = FinalWorld(directional=True, leg_layout="hexapod")
    if checkpoint.size != world.n_params:
        raise ValueError(
            f"Checkpoint size {checkpoint.size} does not match hexapod directional warm-start size {world.n_params}."
        )

    body_lengths = np.array([0.30, 0.30, 0.35, 0.35, 0.42, 0.42], dtype=float)
    body_genes = lengths_to_genes(body_lengths)
    warmstart = checkpoint.astype(float, copy=True)
    warmstart[-world.n_body_params :] = body_genes
    return warmstart, world, body_lengths, body_genes


def main() -> None:
    parser = argparse.ArgumentParser(description="Create a hexapod warm-start genotype")
    parser.add_argument(
        "--checkpoint",
        type=str,
        default="results/test_hexapod_cmaes_curriculum_h1_d02/110/x_best.npy",
        help="Source hexapod checkpoint to use as the controller seed.",
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        default="results/test_hexapod_cmaes_curriculum_h1_d02",
        help="Directory where the warm-start files will be written.",
    )
    args = parser.parse_args()

    warmstart, world, body_lengths, body_genes = build_warmstart(args.checkpoint)

    os.makedirs(args.output_dir, exist_ok=True)
    npy_path = join(args.output_dir, "warmstart_hexapod.npy")
    txt_path = join(args.output_dir, "warmstart_hexapod.txt")

    np.save(npy_path, warmstart)

    with open(txt_path, "w", encoding="utf-8") as f:
        f.write("Hexapod warm-start preset\n")
        f.write("=" * 32 + "\n")
        f.write(f"source_checkpoint : {args.checkpoint}\n")
        f.write(f"world_layout      : {world.leg_layout}\n")
        f.write(f"directional       : {world.directional}\n")
        f.write(f"n_params          : {world.n_params}\n")
        f.write(f"n_weights         : {world.n_weights}\n")
        f.write(f"n_body_params     : {world.n_body_params}\n")
        f.write("body_lengths_m    : " + np.array2string(body_lengths, precision=3, separator=", ") + "\n")
        f.write("body_genes        : " + np.array2string(body_genes, precision=3, separator=", ") + "\n")

    print(f"Saved warm-start genotype: {npy_path}")
    print(f"Saved warm-start manifest : {txt_path}")
    print("Body lengths (m):", body_lengths.tolist())


if __name__ == "__main__":
    main()