import numpy as np
from abc import ABC, abstractmethod

GAME_TYPES = {}
PRP, SOFTMAX, LINEAR = 'PRP', 'Softmax', 'Linear'

# Plotting parameters
COLORS = {PRP: '#1f77b4', SOFTMAX: '#2ca02c', LINEAR: '#ff7f0e'}
COLORS_RANGE = {PRP: ['#1fb4ad', '#1f77b4'], SOFTMAX: ['#2cff2c', '#2c602c'], LINEAR: ['#ffaa00', '#c00000']}

def square_euclidean_norm(x): return np.inner(x, x)  # = np.linalg.norm(x) ** 2

def default_distance_func(x, y): return square_euclidean_norm(x - y)


class PublishersGame(ABC):
    """Abstract class for a publishers' game."""

    def __init__(self, k, n, lam, x_0, x_star, base_distance_func=default_distance_func):
        """
        k: embedding space dimension
        n: number of publishers
        lam: lambda parameter - the weight of the distance from the initial docs
        x_0: initial docs in rows
        x_star: the information need
        distance_func: distance function
        gradient_func: gradient of the distance function (if exists)
        """
        assert x_0.shape == (n, k), f'x_0.shape={x_0.shape}'
        assert len(x_star) == k, f'len(x_star)={len(x_star)}'
        self.k = k
        self.n = n
        self.lam = lam
        self.N = list(range(n))
        self.x_0 = x_0
        self.x_star = x_star
        max_d = base_distance_func(np.zeros(k), np.ones(k))
        # we assume that the range of the base distance function is [0, max_d] and we want the range of the function to be [0, 1].
        self.d = lambda a, b: base_distance_func(a, b) / max_d

        self.x = x_0.copy()
        self.dist_from_opt = [None] * self.n
        self.dist_from_init = [None] * self.n
        self.r = [None] * self.n
        self.r_flag = False
        self.u = [None] * self.n
        self.u_flag = False

        if n == 2:
            self.calc_u = self.calc_u_for_2
            self.get_r = self.get_r_for_2
            self.save_r_all = self.save_r_all_for_2

    def initialize(self):
        """Initialize the game"""
        self.update_x(self.x_0)

    # Get functions

    def get_dist_from_opt(self, i):
        """Gets the distance from the optimal document for publisher i."""
        if self.dist_from_opt[i] is None:
            self.dist_from_opt[i] = self.calc_dist_from_opt(self.x[i])
        return self.dist_from_opt[i]

    def get_dist_from_init(self, i):
        """Gets the distance from the initial document for publisher i."""
        if self.dist_from_init[i] is None:
            self.dist_from_init[i] = self.calc_dist_from_init(self.x[i], i)
        return self.dist_from_init[i]

    def get_r(self, i):
        """Gets the probability for being ranked first for publisher i."""
        if self.r[i] is None:
            self.r[i] = self.calc_r([self.get_dist_from_opt(j) for j in self.N], i)
        return self.r[i]

    def get_r_for_2(self, i):
        """Gets the probability for being ranked first for publisher i in a two players game."""
        if self.r[i] is None:
            self.r[i] = self.calc_r_for_2(self.get_dist_from_opt(i), self.get_dist_from_opt(1 - i))
        return self.r[i]

    def get_u(self, i):
        """Gets the utility for publisher i."""
        if self.u[i] is None:
            self.u[i] = self.get_r(i) - self.lam * self.get_dist_from_init(i)
        return self.u[i]

    def get_publishers_welfare(self):
        """Gets the publishers' welfare."""
        if not self.u_flag:
            self.save_u_all()
            self.u_flag = True
        return sum(self.u)

    def get_users_welfare(self):
        """Gets the users' welfare."""
        if not self.r_flag:
            self.save_r_all()
            self.r_flag = True
        return 1 - sum([self.get_dist_from_opt(i) * self.r[i] for i in self.N])

    # Set functions

    def update_x(self, new_x):
        """Updates the current profile to new_x."""
        assert new_x.shape == (self.n, self.k), f'new_x.shape={new_x.shape}'
        self.x = new_x.copy()
        self.dist_from_opt = [None] * self.n
        self.dist_from_init = [None] * self.n
        self.r = [None] * self.n
        self.r_flag = False
        self.u = [None] * self.n
        self.u_flag = False
        return self

    def update_x_i(self, new_x_i, i):
        """Updates the current profile to new_x_i for publisher i."""
        assert new_x_i.shape[0] == self.k, f'new_x_i.shape={new_x_i.shape}'
        self.x[i] = new_x_i.copy()
        self.dist_from_opt[i] = None
        self.dist_from_init[i] = None
        self.r = [None] * self.n
        self.r_flag = False
        self.u = [None] * self.n
        self.u_flag = False
        return self

    # Calculations functions

    def calc_dist_from_opt(self, xi):
        """Calculates the distance from the optimal document for publisher i."""
        return self.d(xi, self.x_star)

    def calc_dist_from_init(self, xi, i):
        """Calculates the distance from the initial document for publisher i."""
        return self.d(xi, self.x_0[i])

    @abstractmethod
    def calc_r(self, dist_from_opt, i):
        """Calculates the probability that the document of publisher i will be ranked first."""
        pass

    @abstractmethod
    def calc_r_for_2(self, dist_publisher, dist_other):
        """Calculates the probability that publisher's document will be ranked first in a two players game."""
        pass

    def calc_u(self, xi, i):
        """Calculates the utility for publisher i if he playes xi."""
        temp_dist_from_opt = [self.calc_dist_from_opt(xi) if j == i else self.get_dist_from_opt(j) for j in self.N]
        return self.calc_r(temp_dist_from_opt, i) - self.lam * self.calc_dist_from_init(xi, i)

    def calc_u_for_2(self, xi, i):
        """Calculates the utility for publisher i if he playes xi in a two players game."""
        return self.calc_r_for_2(self.calc_dist_from_opt(xi), self.get_dist_from_opt(1 - i)) - \
               self.lam * self.calc_dist_from_init(xi, i)

    # Saving functions

    @abstractmethod
    def save_r_all(self):
        """Saves the probability for being ranked first in the current profile for each publisher's document."""
        pass

    @abstractmethod
    def save_r_all_for_2(self):
        """Saves the probability for being ranked first in the current profile for each publisher's document in a two players game."""
        pass

    def save_u_all(self):
        """Saves the utility in the current profile for each publisher."""
        if not self.r_flag:
            self.save_r_all()
            self.r_flag = True
        self.u = [self.get_u(i) for i in range(self.n)]


class PRPPublishersGame(PublishersGame):
    """PRP Publishers' Game."""

    def calc_r(self, dist_from_opt, i):
        """Calculates the probability that the document of publisher i will be ranked first."""
        dist_from_opt = np.array(dist_from_opt)
        min_dist = np.amin(dist_from_opt)
        mu = np.argwhere(dist_from_opt == min_dist).flatten()
        return 1 / len(mu) if i in mu else 0

    def calc_r_for_2(self, dist_publisher, dist_other):
        """Calculates the probability that publisher's document will be ranked first in a two players game."""
        diff = dist_publisher - dist_other
        return 0 if diff > 0 else (0.5 if diff == 0 else 1)

    def save_r_all(self):
        """Saves the probability for being ranked first in the current profile for each publisher's document."""
        dist_from_opt = np.array([self.get_dist_from_opt(i) for i in self.N])
        min_dist = np.amin(dist_from_opt)
        mu = np.argwhere(dist_from_opt == min_dist).flatten()
        self.r = [1 / len(mu) if i in mu else 0 for i in self.N]

    def save_r_all_for_2(self):
        """Saves the probability for being ranked first in the current profile for each publisher's document in a two players game."""
        dist1 = self.get_dist_from_opt(0)
        dist2 = self.get_dist_from_opt(1)
        diff = dist1 - dist2
        self.r = [0, 1] if diff > 0 else ([0.5, 0.5] if diff == 0 else [1, 0])


class SoftmaxPublishersGame(PublishersGame):
    """Softmax Publishers' Game."""

    def __init__(self, k, n, lam, x_0, x_star, beta=1, distance_func=default_distance_func):
        super().__init__(k, n, lam, x_0, x_star, distance_func)
        assert beta > 0, f'beta must be positive, but beta={beta}'
        self.beta = beta

    def calc_r(self, dist_from_opt, i):
        """Calculates the probability that the document of publisher i will be ranked first."""
        dist_from_opt = np.array(dist_from_opt)
        r = np.exp(self.beta * (- dist_from_opt))
        return r[i] / sum(r)

    def calc_r_for_2(self, dist_publisher, dist_other):
        """Calculates the probability that publisher's document will be ranked first in a two players game."""
        diff = dist_publisher - dist_other
        return 1 / (1 + np.exp(self.beta * diff))

    def save_r_all(self):
        """Saves the probability for being ranked first in the current profile for each publisher's document."""
        dist_from_opt = np.array([self.get_dist_from_opt(i) for i in self.N])
        r = np.exp(self.beta * (- dist_from_opt))  # = (e ^ beta) ^ -dist_from_opt
        r = r / sum(r)
        self.r = r.tolist()

    def save_r_all_for_2(self):
        """Saves the probability for being ranked first in the current profile for each publisher's document in a two players game."""
        dist1 = self.get_dist_from_opt(0)
        dist2 = self.get_dist_from_opt(1)
        diff = dist1 - dist2
        temp = np.exp(self.beta * diff)
        self.r = [1 / (1 + temp), temp / (1 + temp)]


class LinearPublishersGame(PublishersGame):
    """Linear Publishers' Game."""

    def __init__(self, k, n, lam, x_0, x_star, a=None, distance_func=default_distance_func):
        assert a is None or (0 < a and a <= 1 / n), f'a must satisfy 0 < a <= (1 / n), but a={a}'
        super().__init__(k, n, lam, x_0, x_star, distance_func)
        self.a = a if a is not None else 1 / self.n

    def calc_r(self, dist_from_opt, i):
        """Calculates the probability that the document of publisher i will be ranked first."""
        vi = -dist_from_opt[i]
        for j in self.N:
            if j != i:
                vi += dist_from_opt[j] / (self.n - 1)
        ri = self.a * vi + (1 / self.n)
        return ri

    def calc_r_for_2(self, dist_publisher, dist_other):
        """Calculates the probability that publisher's document will be ranked first in a two players game."""
        v_publisher = dist_other - dist_publisher
        return self.a * v_publisher + 0.5  # (1/self.n) = 0.5 because n=2

    def save_r_all(self):
        """Saves the probability for being ranked first in the current profile for each publisher's document."""
        dist_from_opt = np.array([self.get_dist_from_opt(i) for i in self.N])
        v = (np.full(self.n, sum(dist_from_opt)) - self.n * dist_from_opt) / (self.n - 1)
        r = self.a * v + (1 / self.n)
        self.r = r.tolist()

    def save_r_all_for_2(self):
        """Saves the probability for being ranked first in the current profile for each publisher's document in a two players game."""
        dist1 = self.get_dist_from_opt(0)
        dist2 = self.get_dist_from_opt(1)
        v1 = dist2 - dist1
        self.r = [self.a * v1 + 0.5, self.a * (- v1) + 0.5]  # (1/self.n) = 0.5 because n=2


GAME_TYPES[PRP] = PRPPublishersGame
GAME_TYPES[SOFTMAX] = SoftmaxPublishersGame
GAME_TYPES[LINEAR] = LinearPublishersGame
