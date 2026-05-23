import argparse
import numpy as np
import os
from pathlib import Path


def mirror_genotype(genotype, invert_direction=False):
    """
    Construit un robot miroir à partir d'un génotype de 17 paramètres.

    Layout du génotype :
    - [0:4]   : Amplitudes (Symmetric Mapping : Front FL/FR, Back BL/BR)
    - [4]     : Fréquence partagée
    - [5:7]   : Phase Front Left (Hip, Knee)
    - [7:9]   : Phase Back Left (Hip, Knee)
    - [9:11]  : Phase Back Right (Hip, Knee)
    - [11:13] : Phase Front Right (Hip, Knee)
    - [13:17] : Morphologie (Symmetric Mapping)
    """
    # On autorise désormais 17 (Oscillatory) ou 18 (Hysteresis) paramètres
    if genotype.size not in [17, 18]:
        raise ValueError(
            f"Taille de génotype incorrecte ({genotype.size}). Attendu: 17 ou 18."
        )

    mirrored = genotype.copy()

    # Les Amplitudes [0:4], la Fréquence [4] et le Corps [13:17] sont déjà
    # gérés de manière symétrique par FinalWorld et OscillatoryController.
    # Il n'y a donc rien à changer pour ces gènes.

    # 1. Permutation des phases (Swap Gauche/Droite)
    # Ordre cible : [FR_H, FR_K, BR_H, BR_K, BL_H, BL_K, FL_H, FL_K]
    mirr_map = [11, 12, 9, 10, 7, 8, 5, 6]
    mirrored[5:13] = genotype[mirr_map]

    # 2. Compensation des axes des Hips (Indices 5, 7, 9, 11)
    # On ajoute pi (1.0). Si invert_direction est True, on ajoute encore pi,
    # ce qui s'annule (1.0 + 1.0 = 2.0 = 0.0 mod 2).
    hip_indices = [5, 7, 9, 11]
    offset = 0.0 if invert_direction else 1.0
    mirrored[hip_indices] += offset

    # 3. Normalisation cyclique vers [-1, 1]
    mirrored[5:13] = (mirrored[5:13] + 1.0) % 2.0 - 1.0

    # 4. Gestion du paramètre Hystérésis (si 18 params)
    # Si on inverse la dérive du gait, on doit aussi inverser la direction du biais de correction
    if genotype.size == 18:
        mirrored[13] = -genotype[13]

    return mirrored


def main():
    parser = argparse.ArgumentParser(
        description="Miroir de dérive pour robot Ant (17 params)"
    )
    parser.add_argument(
        "--source", type=str, required=True, help="Chemin du fichier x_best.npy source"
    )
    parser.add_argument(
        "--output",
        type=str,
        default="x_best_mirrored.npy",
        help="Nom du fichier de sortie",
    )
    parser.add_argument(
        "--invert",
        action="store_true",
        help="Ajoute pi aux phases pour inverser la marche (avant/arrière)",
    )
    args = parser.parse_args()

    if not os.path.exists(args.source):
        print(f"Erreur : Le fichier {args.source} n'existe pas.")
        return

    geno = np.load(args.source)
    mirrored_geno = mirror_genotype(geno, invert_direction=args.invert)

    np.save(args.output, mirrored_geno)
    print(f"Succès ! Génotype miroir sauvegardé dans : {args.output}")


if __name__ == "__main__":
    main()
