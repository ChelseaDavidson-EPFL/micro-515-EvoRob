"""evorob/algorithms/ea_api.py"""
import os

import numpy as np

from evorob.algorithms.base_ea import EA
from evorob.utils.filesys import search_file_list


class CMAESAPI(EA):
    """Wrapper for the pycma library (pip install cma)."""

    def __init__(
        self,
        n_params: int,
        population_size: int,
        num_generations: int = 100,
        sigma: float = 0.3,
        bounds: tuple = (-1, 1),
        output_dir: str = "./results/CMAES",
    ):
        import cma

        self.population_size = population_size
        self.n_gen = num_generations
        self.n_params = n_params

        self.directory_name = output_dir
        self.current_gen = 0
        self.full_x = []
        self.full_f = []
        self.x_best_so_far = None
        self.f_best_so_far = -np.inf
        self.x = None
        self.f = None

        initial_mean = np.random.uniform(bounds[0], bounds[1], n_params)
        opts = {"popsize": population_size, "bounds": list(bounds)}
        self.es = cma.CMAEvolutionStrategy(x0=initial_mean, sigma0=sigma, inopts=opts)

    def ask(self) -> np.ndarray:
        return np.array(self.es.ask())

    def tell(self, population: np.ndarray, fitnesses: np.ndarray,
             save_checkpoint: bool = False) -> None:
        self.es.tell(population.tolist(), (-fitnesses).tolist())

        self.full_f.append(fitnesses)
        self.full_x.append(population)
        self.f = fitnesses
        self.x = population

        best_idx = np.argmax(fitnesses)
        if fitnesses[best_idx] > self.f_best_so_far:
            self.f_best_so_far = fitnesses[best_idx]
            self.x_best_so_far = population[best_idx].copy()

        if save_checkpoint:
            self.save_checkpoint()
        self.current_gen += 1

    def load_checkpoint(self):
        dir_path = search_file_list(self.directory_name, "f_best.npy")
        assert len(dir_path) > 0, "No checkpoint files found — check directory_name."
        self.current_gen = int(dir_path[-1].split("/")[-2])
        curr_gen_path = os.path.join(self.directory_name, str(self.current_gen))
        self.full_f = np.load(os.path.join(self.directory_name, "full_f.npy"))
        self.full_x = np.load(os.path.join(self.directory_name, "full_x.npy"))
        self.f_best_so_far = np.load(os.path.join(curr_gen_path, "f_best.npy"))
        self.x_best_so_far = np.load(os.path.join(curr_gen_path, "x_best.npy"))
        self.x = np.load(os.path.join(curr_gen_path, "x.npy"))
        self.f = np.load(os.path.join(curr_gen_path, "f.npy"))


class EvosaxAPI(EA):
    """Wrapper for the evosax library (pip install evosax)."""

    def __init__(
        self,
        population_size: int,
        n_params: int,
        num_generations: int = 100,
        output_dir: str = "./results/Evosax",
    ):
        import jax
        from evosax.algorithms import CMA_ES

        self.population_size = population_size
        self.n_gen = num_generations
        self.n_params = n_params
        self.rng = jax.random.key(0)

        self.directory_name = output_dir
        self.current_gen = 0
        self.full_x = []
        self.full_f = []
        self.x_best_so_far = None
        self.f_best_so_far = -np.inf
        self.x = None
        self.f = None

        self.strategy = CMA_ES(popsize=population_size, num_dims=n_params)
        self.rng, rng_init = jax.random.split(self.rng)
        self.es_params = self.strategy.default_params
        self.state = self.strategy.initialize(rng_init, self.es_params)

    def ask(self) -> np.ndarray:
        import jax
        self.rng, rng_ask = jax.random.split(self.rng)
        population, self.state = self.strategy.ask(rng_ask, self.state, self.es_params)
        return np.array(population)

    def tell(self, population: np.ndarray, fitnesses: np.ndarray,
             save_checkpoint: bool = True) -> None:
        import jax.numpy as jnp
        self.state = self.strategy.tell(
            population, jnp.array(fitnesses), self.state, self.es_params
        )

        self.full_f.append(fitnesses)
        self.full_x.append(population)
        self.f = fitnesses
        self.x = population

        best_idx = np.argmax(fitnesses)
        if fitnesses[best_idx] > self.f_best_so_far:
            self.f_best_so_far = fitnesses[best_idx]
            self.x_best_so_far = population[best_idx].copy()

        if save_checkpoint:
            self.save_checkpoint()
        self.current_gen += 1


class PyribsAPI(EA):
    """Wrapper for the pyribs library (pip install ribs)."""

    def __init__(
        self,
        population_size: int,
        n_params: int,
        num_generations: int = 100,
        sigma: float = 0.5,
        output_dir: str = "./results/Pyribs",
    ):
        from ribs.archives import GridArchive
        from ribs.emitters import GaussianEmitter
        from ribs.schedulers import Scheduler

        self.n_params = n_params
        self.n_gen = num_generations
        self.population_size = population_size

        self.directory_name = output_dir
        self.current_gen = 0
        self.full_x = []
        self.full_f = []
        self.x_best_so_far = None
        self.f_best_so_far = -np.inf
        self.x = None
        self.f = None

        self.archive = GridArchive(
            solution_dim=n_params, dims=[1], ranges=[(0, 1)]
        )
        initial_solution = np.random.uniform(-1, 1, n_params)
        self.emitter = GaussianEmitter(
            self.archive, sigma=sigma, x0=initial_solution, batch_size=population_size
        )
        self.scheduler = Scheduler(self.archive, [self.emitter])

    def ask(self) -> np.ndarray:
        return self.scheduler.ask()

    def tell(self, population: np.ndarray, fitnesses: np.ndarray,
             save_checkpoint: bool = True) -> None:
        measures = np.zeros((len(fitnesses), 1))
        self.scheduler.tell(fitnesses, measures)

        self.full_f.append(fitnesses)
        self.full_x.append(population)
        self.f = fitnesses
        self.x = population

        best_idx = np.argmax(fitnesses)
        if fitnesses[best_idx] > self.f_best_so_far:
            self.f_best_so_far = fitnesses[best_idx]
            self.x_best_so_far = population[best_idx].copy()

        if save_checkpoint:
            self.save_checkpoint()
        self.current_gen += 1
