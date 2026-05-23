import numpy as np


class OscillatoryController:
    """Simple oscillatory controller using sine waves for each actuator."""

    def __init__(
        self, input_size: int = 0, output_size: int = None, hidden_size: int = 0
    ):
        assert (
            output_size is not None
        ), "output_size must be specified for OscillatoryController"

        self.output_size = output_size
        self.time_step = 0.0

        # Initialisation avec les valeurs demandées
        self.amplitudes = np.full(self.output_size, 0.4)
        self.frequencies = np.full(self.output_size, 0.3)
        self.phases = np.zeros(self.output_size)

        self.n_params = self.get_num_params()

    def get_action(self, state):
        """Generate oscillatory actions based on time.

        Args:
            state: Observation from environment. Can be shape (obs_size,) or (n_envs, obs_size)
                  for vectorized environments. Not used by this controller but needed for API.

        Returns:
            actions: Shape matches input - (action_size,) or (n_envs, action_size)
        """
        # Determine if we're dealing with batched observations
        is_batched = len(state.shape) > 1 if state is not None else False
        batch_size = state.shape[0] if is_batched else 1

        # Simple sine wave oscillations
        actions = self.amplitudes * np.sin(
            2 * np.pi * self.frequencies * self.time_step + self.phases
        )
        # Incrément de 0.05 pour correspondre au dt standard de MuJoCo (20Hz)
        self.time_step += 0.05

        # Clip to valid action range
        actions = np.clip(actions, -1.0, 1.0)

        # Handle batched observations (for vectorized environments)
        if is_batched:
            # Return same action for all environments in the batch
            actions = np.tile(actions, (batch_size, 1))

        return actions

    def set_weights(self, weights):
        """Set controller parameters from flat array."""
        # Layout: [4 amplitudes, 1 fréquence, 8 phases]
        # Mapping corrigé pour l'axe X (Forward) :
        # Front = FL(0,1) et FR(6,7) | Back = BL(2,3) et BR(4,5)

        # AMPLITUDES : Faible sensibilité (Sigma effectif bas)
        # On centre sur 0.4 avec une variation de +/- 0.1 seulement.
        amps = 0.4 + (weights[:4] * 0.1)
        self.amplitudes = np.array(
            [
                amps[0],
                amps[1],  # Index 0,1: Front Left (+X, +Y)
                amps[2],
                amps[3],  # Index 2,3: Back Left  (-X, +Y)
                amps[2],
                amps[3],  # Index 4,5: Back Right (-X, -Y)
                amps[0],
                amps[1],  # Index 6,7: Front Right (+X, -Y)
            ]
        )

        # FRÉQUENCE : Faible sensibilité (Sigma effectif bas)
        # On restreint la plage autour d'une valeur fonctionnelle (ex: 0.5Hz +/- 0.2)
        self.frequencies = np.full(self.output_size, 0.5 + (weights[4] * 0.2))

        # PHASES : Haute sensibilité (Sigma effectif haut)
        # Discrétisation en multiples de pi/4 : on transforme [-1, 1] en 9 paliers {-pi, ..., 0, ..., pi}
        # Cela aide à stabiliser les gaits (coordination des pattes)
        self.phases = np.round(weights[5:13] * 4) * (np.pi / 4)

        self.time_step = 0  # Reset time

    def geno2pheno(self, genotype):
        """Alias for set_weights."""
        self.set_weights(genotype)
        self.reset_controller()

    def get_num_params(self):
        """Return total number of parameters."""
        # 4 amplitudes symétriques + 1 fréquence partagée + 8 phases
        return 4 + 1 + self.output_size

    def reset_controller(self, batch_size=1):
        self.time_step = 0.0
