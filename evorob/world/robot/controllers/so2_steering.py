import numpy as np
from evorob.world.robot.controllers.base import Controller


class SO2SteeringController(Controller):
    """
    Contrôleur SO2 en boucle fermée avec direction par différentiel de phase.
    L'input 'y' (position latérale) modifie la fréquence des oscillateurs des hanches.

    Layout du génotype intermédiaire (41 paramètres) :
    - [0:8]   : 8 amplitudes individuelles (une par joint)
    - [8:24]  : 16 états initiaux (x, y pour chaque oscillateur)
    - [24:40] : 16 poids de couplage sélectionnés
    - [40]    : 1 gain de steering

    Les 16 couplages conservés relient à la fois les oscillateurs locaux
    (stabilisation du gait) et les paires miroir gauche/droite
    (capacité de correction latérale).
    """

    def __init__(self, output_size: int = 8):
        self.output_size = output_size
        self.n_params = 41
        self.dt = 0.05  # Synchronisé sur MuJoCo (20Hz)

        # Paramètres du phénotype
        self.amplitudes = np.zeros(8)
        self.A = np.zeros((16, 16))
        self.init_state = np.zeros(16)
        self.steer_gain = 0.0

        self.state_y = None
        self.n_states = output_size * 2
        self.state_shape = (self.n_states, 1)

        # Couplages sélectionnés: 8 locaux + 8 miroir gauche/droite.
        self.coupling_pairs = np.array(
            [
                [0, 1],
                [2, 3],
                [4, 5],
                [6, 7],
                [8, 9],
                [10, 11],
                [12, 13],
                [14, 15],
                [0, 12],
                [1, 13],
                [2, 14],
                [3, 15],
                [4, 8],
                [5, 9],
                [6, 10],
                [7, 11],
            ],
            dtype=int,
        )

    def set_weights(self, weights: np.ndarray):
        weights = np.asarray(weights, dtype=float).ravel()
        if weights.size != self.n_params:
            raise ValueError(f"Attendu {self.n_params} paramètres, reçu {weights.size}")

        # 1. Amplitudes (Libres)
        self.amplitudes = weights[0:8]

        # 2. État initial (Libre)
        self.init_state = weights[8:24]

        # 3. Matrice de couplage A (16x16 Anti-symétrique)
        self.A = np.zeros((16, 16))
        for value, (row, col) in zip(weights[24:40], self.coupling_pairs):
            self.A[row, col] = value
            self.A[col, row] = -value

        # 4. Gain de Steering
        self.steer_gain = weights[40] * 2.0

        self.reset_controller()

    def reset_controller(self, batch_size: int = 1):
        self.state_y = np.tile(self.init_state.reshape(-1, 1), (1, batch_size))

    def get_action(self, state: np.ndarray) -> np.ndarray:
        if state.ndim == 1:
            y_env = state[1] if state.size > 1 else state[0]
            batch_size = 1
        else:
            batch_size = state.shape[0]
            y_env = state[:, 1] if state.shape[1] > 1 else state[:, 0]

        # Modulation du steering sur les blocs 2x2 des Hanches
        # Hanches : 0(FL), 2(BL), 4(BR), 6(FR) -> Indices d'état 0,1, 4,5, 8,9, 12,13
        steer_mod = self.steer_gain * y_env

        y_in = self.state_y

        def deriv(y):
            dy = self.A @ y
            # Appliquer le différentiel de fréquence sur les hanches
            # Gauche (FL, BL) accélère si y > 0
            dy[0] += steer_mod * y[1]
            dy[1] -= steer_mod * y[0]  # FL
            dy[4] += steer_mod * y[5]
            dy[5] -= steer_mod * y[4]  # BL
            # Droite (BR, FR) ralentit si y > 0
            dy[8] -= steer_mod * y[9]
            dy[9] += steer_mod * y[8]  # BR
            dy[12] -= steer_mod * y[13]
            dy[13] += steer_mod * y[12]  # FR
            return dy

        k1 = deriv(y_in)
        k2 = deriv(y_in + 0.5 * self.dt * k1)
        k3 = deriv(y_in + 0.5 * self.dt * k2)
        k4 = deriv(y_in + self.dt * k3)
        y_out = y_in + (self.dt / 6.0) * (k1 + 2 * k2 + 2 * k3 + k4)
        self.state_y = y_out

        # Output: Amplitude * composante 'sin' de chaque oscillateur (indices impairs)
        actions = self.amplitudes.reshape(-1, 1) * y_out[1::2, :]
        actions = np.clip(actions, -1.0, 1.0)
        return actions.T.squeeze() if batch_size == 1 else actions.T

    def geno2pheno(self, genotype: np.ndarray):
        self.set_weights(genotype)

    def get_num_params(self):
        return self.n_params
