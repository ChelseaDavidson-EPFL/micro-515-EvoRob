import numpy as np
from evorob.world.robot.controllers.base import Controller


class HysteresisSinoidController(Controller):
    """
    Contrôleur sinusoïdal en boucle fermée avec basculement par hystérésis.
    Le robot alterne entre sa marche originale et sa marche miroir pour rester
    dans un couloir central sur l'axe Y.

    Layout du génotype (14 paramètres) :
    - [0:4]   : 4 Amplitudes (Mapping symétrique)
    - [4]     : 1 Fréquence partagée
    - [5:13]  : 8 Phases (Une par actuateur, discrétisation pi/4)
    - [13]    : 1 Paramètre de contrôle Y (Signe = Biais, Valeur = Seuil)
    """

    def __init__(self, output_size: int = 8):
        self.output_size = output_size
        self.time_step = 0.0
        self.n_params = 14

        # Paramètres internes
        self.amplitudes = np.full(8, 0.4)
        self.frequencies = np.full(8, 0.5)
        self.phases_orig = np.zeros(8)
        self.phases_mirr = np.zeros(8)
        self.y_threshold = 0.4
        self.bias_dir = 1.0  # +1 si Original dévie vers +Y, -1 sinon

        self.target_mode = 1.0  # 1.0 = Original, -1.0 = Miroir

    def get_action(self, state: np.ndarray) -> np.ndarray:
        """
        Génère les couples moteurs en fonction de la position latérale y.
        """
        # 1. Extraction de la position Y (Input du contrôleur)
        # On supporte les scalaires (via sensor_fn) ou l'obs Ant standard (index 1)
        if np.isscalar(state):
            y = float(state)
        elif isinstance(state, np.ndarray):
            if state.ndim == 1:
                y = state[1] if state.size > 1 else state[0]
            else:
                # Gestion batched (AsyncVectorEnv)
                y = np.mean(state[:, 1]) if state.shape[1] > 1 else np.mean(state[:, 0])
        else:
            y = 0.0

        # 2. Logique d'Hystérésis (Comportement "Rebond")
        # ty_rel > threshold signifie qu'on s'éloigne trop dans le sens du biais
        ty_rel = y * self.bias_dir
        if ty_rel > self.y_threshold:
            self.target_mode = -1.0  # Activation du miroir pour corriger
        elif ty_rel < -self.y_threshold:
            self.target_mode = 1.0  # Retour à l'original

        # 3. Calcul de l'action sinusoïdale
        phases = self.phases_orig if self.target_mode > 0 else self.phases_mirr

        actions = self.amplitudes * np.sin(
            2 * np.pi * self.frequencies * self.time_step + phases
        )

        # Incrément temporel synchronisé sur 20Hz (MuJoCo default)
        self.time_step += 0.05
        actions = np.clip(actions, -1.0, 1.0)

        # Mapping de sortie pour AsyncVectorEnv
        if isinstance(state, np.ndarray) and state.ndim > 1:
            return np.tile(actions, (state.shape[0], 1))

        return actions

    def set_weights(self, weights: np.ndarray):
        """Décode les 14 paramètres du génotype."""
        weights = np.asarray(weights, dtype=float).ravel()
        if weights.size != self.n_params:
            raise ValueError(f"Attendu 14 paramètres, reçu {weights.size}")

        # A. Amplitudes (Faible sensibilité)
        amps_raw = 0.4 + (weights[:4] * 0.1)
        self.amplitudes = np.array(
            [
                amps_raw[0],
                amps_raw[1],  # Front Left
                amps_raw[2],
                amps_raw[3],  # Back Left
                amps_raw[2],
                amps_raw[3],  # Back Right
                amps_raw[0],
                amps_raw[1],  # Front Right
            ]
        )

        # B. Fréquence (0.5 +/- 0.2 Hz)
        self.frequencies = np.full(8, 0.5 + (weights[4] * 0.2))

        # C. Phases Originales (Snapping pi/4)
        self.phases_orig = np.round(weights[5:13] * 4) * (np.pi / 4)

        # D. Phases Miroirs (Involutivité automatique)
        p = self.phases_orig
        p_mirr = p.copy()
        # Swap Gauche/Droite : FL(0,1)<->FR(6,7) et BL(2,3)<->BR(4,5)
        p_mirr[0], p_mirr[1], p_mirr[6], p_mirr[7] = p[6], p[7], p[0], p[1]
        p_mirr[2], p_mirr[3], p_mirr[4], p_mirr[5] = p[4], p[5], p[2], p[3]
        # Inversion des Hips (0, 2, 4, 6) pour conserver la marche avant
        p_mirr[[0, 2, 4, 6]] += np.pi
        self.phases_mirr = (p_mirr + np.pi) % (2 * np.pi) - np.pi

        # E. Paramètre de pilotage (Le 14ème)
        # Signe = direction du biais (+ dévie vers +Y, - dévie vers -Y)
        self.bias_dir = 1.0 if weights[13] >= 0 else -1.0
        # Amplitude = seuil de déclenchement (entre 0.1m et 1.0m)
        self.y_threshold = 0.1 + abs(weights[13]) * 0.9

        self.reset_controller()

    def geno2pheno(self, genotype):
        self.set_weights(genotype)

    def reset_controller(self, batch_size=1):
        self.time_step = 0.0
        self.target_mode = 1.0

    def get_num_params(self):
        return self.n_params
