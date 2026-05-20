import numpy as np
import os


class NeuralCartesianController:
    """
    Un contrôleur neuronal (MLP) conçu pour la locomotion.
    Il combine des signaux d'horloge (CPG) et le feedback des capteurs.

    L'architecture permet de transcrire la logique du Cartesian Controller :
    - Combinaisons linéaires (poids)
    - Offsets (biases)
    - ReLU (seuils de phase)
    - Tanh (clamping des actuateurs)
    """

    def __init__(
        self, n_obs: int, n_actuators: int, n_hidden: int = 32, dt: float = 0.02
    ):
        self.n_obs = n_obs
        self.n_actuators = n_actuators
        self.n_hidden = n_hidden
        self.dt = dt
        self.sim_time = 0.0

        # Entrées : Observations (27+ pour Ant) + Horloges (sin/cos pour 4 pattes = 8)
        self.n_clock = 8
        self.n_total_in = n_obs + self.n_clock

        # Calcul du nombre de paramètres (DNA)
        # [speed (1)] + [W1 (in*hid)] + [B1 (hid)] + [W2 (hid*out)] + [B2 (out)]
        self.n_params = (
            1
            + (self.n_total_in * self.n_hidden + self.n_hidden)
            + (self.n_hidden * self.n_actuators + self.n_actuators)
        )

        self.weights = np.zeros(self.n_params)

    def set_weights(self, weights: np.ndarray):
        """Met à jour les poids du réseau via l'algorithme génétique."""
        if weights.size != self.n_params:
            raise ValueError(
                f"Taille des poids incorrecte. Attendu: {self.n_params}, reçu: {weights.size}"
            )
        self.weights = weights

    def reset(self):
        """Réinitialise la phase temporelle."""
        self.sim_time = 0.0

    def __call__(self, obs: np.ndarray) -> np.ndarray:
        """
        Inférence : Observations -> Actions
        """
        # 1. Dépaquetage des paramètres
        ptr = 0
        # On borne la vitesse entre 0.2 et 2.0 Hz pour éviter les hautes fréquences instables
        # même si CMA-ES propose des valeurs extrêmes.
        speed = np.clip(self.weights[ptr], 0.2, 2.0)
        ptr += 1

        # Couche 1
        w1_size = self.n_total_in * self.n_hidden
        w1 = self.weights[ptr : ptr + w1_size].reshape(self.n_total_in, self.n_hidden)
        ptr += w1_size
        b1 = self.weights[ptr : ptr + self.n_hidden]
        ptr += self.n_hidden

        # Couche 2
        w2_size = self.n_hidden * self.n_actuators
        w2 = self.weights[ptr : ptr + w2_size].reshape(self.n_hidden, self.n_actuators)
        ptr += w2_size
        b2 = self.weights[ptr : ptr + self.n_actuators]
        ptr += self.n_actuators

        # 2. Mise à jour de l'horloge interne
        # On utilise le paramètre 'speed' évolué pour piloter la fréquence
        self.sim_time += self.dt * speed * 2 * np.pi

        clock_inputs = []
        for i in range(4):
            # On fournit sin et cos pour chaque patte (crawl phase offsets par défaut)
            phi = self.sim_time + i * (np.pi / 2)
            clock_inputs.append(np.sin(phi))
            clock_inputs.append(np.cos(phi))

        # 3. Préparation du vecteur d'entrée
        # On concatène les capteurs du robot et les oscillateurs temporels
        x = np.concatenate([obs, clock_inputs])

        # 4. Passage dans le réseau (Forward Pass)
        # Couche cachée avec activation ReLU (transcription des seuils de phase)
        h = np.dot(x, w1) + b1
        h = np.maximum(0, h)

        # Couche de sortie avec Tanh (clamping natif à [-1, 1])
        out = np.tanh(np.dot(h, w2) + b2)

        return out

    def save(self, filename: str):
        """Sauvegarde les poids actuels."""
        np.save(filename, self.weights)

    def load(self, filename: str):
        """Charge des poids depuis un fichier .npy."""
        if os.path.exists(filename):
            self.set_weights(np.load(filename))
        else:
            print(f"Fichier {filename} introuvable.")
