import numpy as np
import os
import re
from final_project_train import run_cmaes_refinement

# Chemin vers le fichier best_individual.txt du run précédent
best_individual_txt_path = os.path.join(
    "results", "hexapod_8h_run", "best_individual.txt"
)

# Nouveau répertoire de résultats pour cet entraînement de 2 heures
new_results_dir = os.path.join("results", "hexapod_2h_warmstart_run")

# Vérification de l'existence du fichier
if not os.path.exists(best_individual_txt_path):
    raise FileNotFoundError(
        f"Erreur : Le fichier est introuvable à l'emplacement : {os.path.abspath(best_individual_txt_path)}"
    )

# --- Extraction du génotype depuis best_individual.txt ---
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
        warmstart_genotype = np.array(genotype_list)
    else:
        raise ValueError("Le génotype n'a pas été trouvé dans le fichier.")

print(f"Génotype de warm-start chargé avec la forme : {warmstart_genotype.shape}")

# --- Lancement de l'entraînement CMA-ES avec warm-start ---
print(f"Lancement de l'entraînement CMA-ES avec warm-start pour 2 heures maximum...")
run_cmaes_refinement(
    seed_genotype=warmstart_genotype,
    num_generations=200,  # Un nombre de générations élevé, la limite sera max_hours
    population_size=32,
    sigma=0.2,  # Un sigma plus petit pour un affinement (warm-start)
    n_repeats=4,
    n_steps=500,
    results_dir=new_results_dir,
    directional=False,  # Basé sur le script de visualisation précédent
    leg_layout="hexapod",
    max_hours=2.0,
)
print("Entraînement CMA-ES terminé.")
