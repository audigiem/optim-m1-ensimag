from abc import abstractmethod
import numpy as np

from src.sim import Simulator


class SimulatorProx(Simulator):
    def __init__(
        self,
        n: int,
        title: str
    ):
        self.n = n
        self.title = title

    @abstractmethod
    def f(self, x: np.ndarray) -> float:
        del x
        raise NotImplementedError()

    @abstractmethod
    def f_grad(self, x: np.ndarray) -> np.ndarray:
        del x
        raise NotImplementedError()

    @abstractmethod
    def g(self, x: np.ndarray) -> float:
        del x
        raise NotImplementedError()

    @abstractmethod
    def g_prox(self, x: np.ndarray, gamma: float) -> np.ndarray:
        del x, gamma
        raise NotImplementedError()

    def F(self, x: np.ndarray) -> float:
        return self.f(x) + self.g(x)

    def primal(self, x: np.ndarray) -> float:
        return self.F(x)

    def gradient(self, x: np.ndarray) -> np.ndarray:
        return self.f_grad(x)
