import numpy as np
import os
import re

# Importation des classes nécessaires depuis les scripts existants
# Assurez-vous que le répertoire racine du projet est dans le PYTHONPATH
# ou exécutez ce script depuis le répertoire racine du projet.
from final_project_train import FinalWorld
from analyse_training_results_cmaes import record_videos

# Chemin vers le fichier best_individual.txt
best_individual_txt_path = os.path.join(
    "results", "hexapod_8h_run", "best_individual.txt"
)
# Répertoire de sortie pour les vidéos générées
output_video_dir = os.path.join("results", "hexapod_8h_run", "videos_best_individual")

# Créer le répertoire de sortie si nécessaire
os.makedirs(output_video_dir, exist_ok=True)

# Vérification de l'existence du fichier
if not os.path.exists(best_individual_txt_path):
    raise FileNotFoundError(
        f"Erreur : Le fichier est introuvable à l'emplacement : {os.path.abspath(best_individual_txt_path)}"
    )

# Parser le fichier best_individual.txt pour extraire le génotype
genotype_str_raw = ""
with open(best_individual_txt_path, "r") as f:
    content = f.read()
    # Recherche de la chaîne de caractères représentant le tableau du génotype
    match = re.search(r"genotype      : \[(.*?)\]", content, re.DOTALL)
    if match:
        genotype_str_raw = match.group(1)
        # Nettoyage de la chaîne : suppression des retours à la ligne, des espaces multiples, et division par virgule
        genotype_str_cleaned = re.sub(r"\s+", "", genotype_str_raw).strip(",")
        genotype_list = [float(x) for x in genotype_str_cleaned.split(",") if x]
        genotype = np.array(genotype_list)
    else:
        raise ValueError("Le génotype n'a pas été trouvé dans le fichier.")

print(f"Génotype chargé avec la forme : {genotype.shape}")

# Initialiser FinalWorld avec les paramètres corrects utilisés lors de l'entraînement CMA-ES.
# Le nom du répertoire "hexapod_8h_run" suggère que le layout était "hexapod".
# Les paramètres 'directional' et 'target_direction' sont supposés être les valeurs par défaut.
world = FinalWorld(directional=False, target_direction=(1.0, 0.0), leg_layout="hexapod")

# Enregistrer les vidéos
print(f"Enregistrement des vidéos pour le meilleur individu dans {output_video_dir}...")
record_videos(world, genotype, output_video_dir)
print(
    "Enregistrement des vidéos terminé. Vérifiez le répertoire spécifié pour les vidéos générées."
)
