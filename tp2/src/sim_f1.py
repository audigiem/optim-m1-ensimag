import numpy as np
from src.sim import Simulator


class SimF1(Simulator):
    def __init__(self, n: int, npts: int):
        super().__init__(
            n,
            npts,
            [(-5, 5)]*2,
            0, 30,
            [0.25,1,2,5,10,15],
            'f1',
            np.zeros(n)
        )

    def sim(self, x: np.ndarray):
        """
        Simuler f1: f_1(x) =& \sum_{k=1}^n k x_k^2
        :param x: np.ndarray valeur varaible pour évaluer f1
        :return: float f1(x), np.ndarray gradient de f1(x), np.ndarray hessienne de f1(x)
        """
        assert x.shape == (self.n,)
        # f(x) = \sum_{k=1}^n k x_k^2
        f = np.sum([k * x[k-1]**2 for k in range(1, self.n+1)])
        # gradient de f en x
        g = np.array([2 * k * x[k-1] for k in range(1, self.n+1)])
        # hessienne de f en x
        h = np.diag([2 * k for k in range(1, self.n+1)])
    
        return f, g, None

    def primal(self, x: np.ndarray):
        assert x.shape == (self.n,)
        
        return np.sum([k * x[k-1]**2 for k in range(1, self.n+1)])
        
   

    def gradient(self, x: np.ndarray):
        assert x.shape == (self.n,)
        return np.array([2 * k * x[k-1] for k in range(1, self.n+1)])
    

    def hessian(self, x: np.ndarray):
        assert x.shape == (self.n,)
        return np.diag([2 * k for k in range(1, self.n+1)])




## Bonus: Newton

# > Créer un nouveau simulateur retournant également la Hessienne des fonctions respectives.
# > Implémenter la méthode de Newton $$ x_{k+1} = x_k - [\nabla^2f(x_k)]^{-1} \nabla f(x_k)$$ et comparer avec Quasi-Newton.

class SimNewtonF1(Simulator):
    def __init__(self, n: int, npts: int):
        super().__init__(
            n,
            npts,
            [(-5, 5)]*2,
            0, 30,
            [0.25,1,2,5,10,15],
            'f1',
            np.zeros(n)
        )

    def sim(self, x: np.ndarray):
        """
        Simuler f1: f_1(x) =& \sum_{k=1}^n k x_k^2
        :param x: np.ndarray valeur varaible pour évaluer f1
        :return: float f1(x), np.ndarray gradient de f1(x), np.ndarray hessienne de f1(x)
        """
        assert x.shape == (self.n,)
        # f(x) = \sum_{k=1}^n k x_k^2
        f = np.sum([k * x[k-1]**2 for k in range(1, self.n+1)])
        # gradient de f en x
        g = np.array([2 * k * x[k-1] for k in range(1, self.n+1)])
        # hessienne de f en x
        h = np.diag([2 * k for k in range(1, self.n+1)])
    
        return f, g, h

    def primal(self, x: np.ndarray):
        assert x.shape == (self.n,)
        
        return np.sum([k * x[k-1]**2 for k in range(1, self.n+1)])
        
   

    def gradient(self, x: np.ndarray):
        assert x.shape == (self.n,)
        return np.array([2 * k * x[k-1] for k in range(1, self.n+1)])
    

    def hessian(self, x: np.ndarray):
        assert x.shape == (self.n,)
        return np.diag([2 * k for k in range(1, self.n+1)])
    
    def newton(self, x: np.ndarray):
        """
        Applique la méthode de Newton pour minimiser f1
        :param x: np.ndarray valeur initiale
        :return: np.ndarray valeur minimisant f1
        """
        x = np.copy(x)
        f, g, h = self.sim(x)
        x = x - np.linalg.inv(h) @ g
        
        return x
    
    
    

