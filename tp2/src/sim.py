import numpy as np
import matplotlib.pyplot as plt
import matplotlib.cm as cm

from IPython import display
import time

class Simulator:
    def __init__(
        self,
        n: int,
        npts: int,
        bounds: list[tuple[float, float]],
        vmin: float,
        vmax: float,
        levels: list[float],
        title: str,
        true_min: np.ndarray
    ):
        self.n = n
        self.npts = npts
        self.in_bounds = bounds
        self.out_bounds = (vmin, vmax)
        self.levels = levels
        self.title = title
        self.true_min = true_min

    def sim(self, x: np.ndarray):
        del x
        raise NotImplementedError("Implement the sim method for your simulator")

    def __call__(self, x: np.ndarray):
        return self.sim(x)

    def primal(self, x: np.ndarray):
        """
        Calcule f de manière spécialisée
        """
        return self.sim(x)[0]

    def gradient(self, x: np.ndarray):
        """
        Calcule g de manière spécialisée
        """
        return self.sim(x)[1]

    def hessian(self, x: np.ndarray):
        """
        Calcule H de manière spécialisée
        """
        return self.sim(x)[2]

    def f_2d_broadcast(self, x1, x2):
        """
        Fonction pour appliquer la fonction sur une grille.
        TODO: vectoriser
        """
        assert x1.shape == x2.shape
        nl = x1.shape[0]
        nc = x1.shape[1]
        return np.array([[self.primal(np.array([x1[i,j], x2[i,j]])) for j in range(nc)] for i in range(nl)])

    def xyz(self):
        """
        Produit une grille pour les plots
        """
        assert len(self.in_bounds) == 2
        x , y = np.meshgrid(
            np.linspace(*self.in_bounds[0], self.npts),
            np.linspace(*self.in_bounds[1], self.npts)
        )
        z = self.f_2d_broadcast(x,y)
        return x, y, z

    def custom_3dplot(self):
        """
        Plot 3d de la surface de la fonction
        """
        x, y, z = self.xyz()

        fig = plt.figure()
        ax = fig.add_subplot(111, projection='3d')

        vmin, vmax = self.out_bounds
        ax.plot_surface(
            x, y, z,
            cmap=cm.hot,
            vmin=vmin,
            vmax=vmax
        )
        ax.scatter(
            self.true_min[0],
            self.true_min[1],
            self.primal(self.true_min),
            'r+'
        )
        ax.set_zlim(vmin, vmax)
        plt.title(self.title)
        plt.show()
        return fig

    def plot_true_min(self):
        """
        Montrer dans un plot le "vrai" minimum
        """
        plt.plot(
            self.true_min[0],
            self.true_min[1],
            'r+'
        )
        plt.hlines(
            self.true_min[1],
            self.in_bounds[0][0],
            self.in_bounds[0][1],
            linestyles=':',
            color='r'
        )
        plt.vlines(
            self.true_min[0],
            self.in_bounds[1][0],
            self.in_bounds[1][1],
            linestyles=':',
            color='r'
        )


    def level_plot(self):
        """
        Plotter les fonctions de niveaux
        """
        x, y, z = self.xyz()

        fig = plt.figure()
        graph = plt.contour(x, y, z, self.levels)
        plt.clabel(graph, inline=1, fontsize=10, fmt='%3.2f')
        self.plot_true_min()
        plt.title(self.title)
        plt.show()
        return fig

    def level_points_plot(self, trace: np.ndarray):
        """
        Montrer les étapes d'optimisation dans un plot de niveaux
        """
        x, y, z = self.xyz()
        fig = plt.figure()
        graphe = plt.contour(x, y, z, self.levels)
        self.plot_true_min()
        plt.clabel(graphe, inline=1, fontsize=10, fmt='%3.2f')
        plt.title(self.title)

        if trace.shape[0] > 40:
            sub = int(trace.shape[0]/40.0)
            trace = trace[::sub]

        delay = 2.0 / trace.shape[0]
        for xk in trace:
            plt.plot(xk[0], xk[1], '*b', markersize=10)
            self.plot_true_min()
            plt.draw()	
            display.clear_output(wait=True)
            display.display(fig)
            time.sleep(delay)
        display.clear_output()
        plt.show()

    def level_2points_plot(self, trace1: np.ndarray, trace2: np.ndarray):
        """
        Montrer les étapes d'optimisation dans un plot de niveaux
        Adapté pour deux traces à la suite.
        TODO: à tester
        """
        x, y, z = self.xyz()
        fig = plt.figure()
        graphe = plt.contour(x, y, z, self.levels)
        plt.clabel(graphe, inline=1, fontsize=10, fmt='%3.2f')
        plt.xlim(self.in_bounds[0])
        plt.ylim(self.in_bounds[1])
        plt.title(self.title)

        if trace1.shape[0] > 40:
            sub = int(trace1.shape[0]/40.0)
            trace1 = trace1[::sub]

        delay = 4.0 / trace1.shape[0]
        for xk in trace1:
            plt.plot(xk[0], xk[1], '*b', markersize=10)
            self.plot_true_min()
            plt.draw()	
            display.clear_output(wait=True)
            display.display(fig)
            time.sleep(delay)

        delay = 4.0 / trace2.shape[0]
        for xk in trace2:
            plt.plot(xk[0], xk[1], 'dg', markersize=8)
            self.plot_true_min()
            plt.draw()	
            display.clear_output(wait=True)
            display.display(fig)
            time.sleep(delay)

        display.clear_output()
        plt.show()
