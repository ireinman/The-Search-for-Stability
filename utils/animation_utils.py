import numpy as np
import matplotlib
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
from abc import ABC, abstractmethod

from utils.simulations_utils import STEPS_, EPS_, find_best_step_strategy_discrete, create_directions_set
from utils.publishers_game import GAME_TYPES, PRP, LINEAR, SOFTMAX

matplotlib.use('TkAgg')

INFINITY = float('inf')

LEGEND_VERTICAL_POSITION = -0.15


def discrete_better_response_dynamics_generator(G, steps, directions, T, eps):
    """Each time yields a tuple of the next state of the dynamics,
    the publishers' welfare, the users' welfare and
    a boolean indicating whether the dynamics converged.
    If k > 2, yields just the two first dimensions of the dynamics."""
    G.initialize()
    convergence_status = False
    t = 0
    u, v = G.get_publishers_welfare(), G.get_users_welfare()
    yield G.x.copy()[:, :2], u, v, convergence_status
    while t < T:  # Not that if T = INFINITY, this loop will never end
        t += 1
        players_permutation = np.random.permutation(G.N)
        found_improving_player = False
        for i in players_permutation:
            x_i_moved, new_u = find_best_step_strategy_discrete(G, i, directions, steps)
            if G.get_u(i) + eps < new_u:
                G.update_x_i(x_i_moved, i)
                u, v = G.get_publishers_welfare(), G.get_users_welfare()
                found_improving_player = True
                yield G.x.copy()[:, :2], u, v, convergence_status  # convergence_status is False
                break
        if not found_improving_player:
            convergence_status = True
            yield G.x.copy()[:, :2], u, v, convergence_status
            return


def enforce_square(ax: matplotlib.axes.Axes):
    # Enforce axis to be displayed as a square, rather than a rectangle
    (x_min, x_max), (y_min, y_max) = ax.get_xlim(), ax.get_ylim()
    ax.set_aspect(abs((x_max - x_min) / (y_max - y_min)))


def assess_minimum_welfare_measures(G):
    """ Calculates an approximation of the minimum users' welfare
    and the minimum publishers' welfare achievable
    in a Discrete Better Response Dynamics Simulation on G.
    We do this for the sake of effective plotting. """
    G.initialize()
    min_user_welfare = G.get_users_welfare()
    # profile where all publishers play x_star:
    x_star_profile = np.tile(G.x_star, (G.n, 1))
    G.update_x(x_star_profile)
    min_publisher_welfare = G.get_publishers_welfare()
    return min_publisher_welfare, min_user_welfare


class DiscreteBetterResponseAnimationBase(ABC):
    def __init__(self, ranking_function=PRP, n=2, k=2, lam=0.5, T=1000, interval=150, x_0=None, x_star=None,
                 additional_params=None):
        # attributes regarding the game
        additional_params = {} if additional_params is None else additional_params
        x_star = np.random.rand(k) if x_star is None else x_star
        x_0 = np.random.rand(n, k) if x_0 is None else x_0
        self.ranking_function = ranking_function
        self.G = GAME_TYPES[ranking_function](k=k, n=n, lam=lam, x_0=x_0, x_star=x_star, **additional_params)

        # attributes regarding the dynamics
        self.T = T

        # attributes regarding the animation
        self.interval = interval
        self.fig, (self.ax_dynamics, self.ax_welfare) = plt.subplots(1, 2)
        self.ani = None

        # attributes for the dynamics plot
        self.timestamp_subtitle, self.profile_scatter, self.info_need_scatter = None, None, None

        # attributes for the welfare plot
        self.publishers_welfare_color = matplotlib.rcParams['axes.prop_cycle'].by_key()['color'][(self.G.n + 1) % 10]
        # next unused default color

    def show(self):
        plt.tight_layout()
        plt.show()

    def save(self, filename):
        if self.ani is not None:
            self.ani.save(filename)
        else:
            raise ValueError("Animation not initialized")

    def plot_init(self):
        self.fig.suptitle(fr"""Discrete Better Response Dynamics Simulation
        with {self.ranking_function} ranking, $n = {self.G.n}$, $k = {self.G.k}$, $\lambda = {self.G.lam}$""",
                          fontsize=18)
        self.fig.subplots_adjust(wspace=0.3)
        return self.init_ax_dynamics() + self.init_ax_welfare()

    def init_ax_dynamics(self):
        self.ax_dynamics.set_title('The Embedding Space', y=1.05, fontsize=15)
        self.timestamp_subtitle = self.ax_dynamics.text(x=0.5, y=1.01, s=r'$t = 0$', horizontalalignment='center',
                                                        fontsize=12)
        self.ax_dynamics.set_xlim(0, 1), self.ax_dynamics.set_ylim(0, 1)
        enforce_square(self.ax_dynamics)
        self.profile_scatter = [self.init_ax_dynamics_player(player) for player in self.G.N]

        # plot the info need
        self.info_need_scatter = self.ax_dynamics.scatter(*self.G.x_star[:2], s=150,
                                                          marker='*', label=r'$x^*$ (info need)')

        self.ax_dynamics.legend(loc='upper center', bbox_to_anchor=(0.45, LEGEND_VERTICAL_POSITION))
        return [self.timestamp_subtitle] + self.profile_scatter

    def init_ax_dynamics_player(self, player: int):
        initial_doc = self.G.x_0[player][:2]
        # a scatter for the current doc:
        scatter = self.ax_dynamics.scatter(*initial_doc, label=f'Publisher {player + 1}')
        # a scatter for the initial doc:
        self.ax_dynamics.scatter(*initial_doc, marker='s', color=scatter.get_facecolor())
        # plot the line connecting the initial doc to the info need:
        self.ax_dynamics.plot([initial_doc[0], self.G.x_star[0]], [initial_doc[1], self.G.x_star[1]],
                              color=scatter.get_facecolor(), linestyle=':', linewidth=1.5)
        return scatter

    @abstractmethod
    def init_ax_welfare(self):
        pass

    def update(self, frame):
        t, (x, u, v, converged) = frame
        artists_to_redraw = self.update_ax_dynamics(t, x) + self.update_ax_welfare(t, u, v)
        if converged:
            self.plot_convergence_message(t)
        if t == self.T:
            self.plot_timeout_message()

        return artists_to_redraw

    def update_ax_dynamics(self, t, x):
        self.timestamp_subtitle.set_text(fr'$t = {t}$')
        # scatter the current strategy profile
        for player in self.G.N:
            self.profile_scatter[player].set_offsets(x[player])

        return [self.timestamp_subtitle] + self.profile_scatter

    @abstractmethod
    def update_ax_welfare(self, t, u, v):
        pass

    def plot_convergence_message(self, t):
        self.fig.text(0.5, 0.1, """The\ndynamics\nconverged!""", ha='center', va='center', fontsize=16,
                      fontweight='bold',
                      bbox=dict(boxstyle='round', facecolor='white', alpha=0.5))

        # make the timestamp subtitle bold
        self.timestamp_subtitle.set_fontweight('bold')

    def plot_timeout_message(self):
        self.fig.text(0.5, 0.1, "The dynamics \ntimed out \nwithout converging!", ha='center', va='center',
                      fontweight='bold',
                      fontsize=10, bbox=dict(facecolor='white', alpha=0.5, boxstyle='round'))


class DiscreteBetterResponseAnimationINFINITY(DiscreteBetterResponseAnimationBase):
    def __init__(self, ranking_function=PRP, n=2, k=2, lam=0.5, T=INFINITY, interval=150, x_0=None, x_star=None,
                 additional_params=None):
        super().__init__(ranking_function, n, k, lam, T, interval, x_0, x_star, additional_params)
        assert self.T == INFINITY

        # attributes for the welfare plot
        self.publishers_welfare_line, self.users_welfare_line = None, None
        self.u_history = []  # history of publishers' welfare
        self.v_history = []  # history of users' welfare
        self.min_u, self.min_v = assess_minimum_welfare_measures(self.G)

        dynamics_generator = discrete_better_response_dynamics_generator(self.G, STEPS_,
                                                                         create_directions_set(self.G.k),
                                                                         self.T, EPS_)

        self.ani = FuncAnimation(fig=self.fig, func=self.update,
                                 frames=enumerate(dynamics_generator),
                                 init_func=self.plot_init, interval=self.interval, repeat=False,
                                 cache_frame_data=False, blit=False)

    def init_ax_welfare(self):
        self.ax_welfare.set_title("Welfare Measures in Time", y=1.05, fontsize=15)
        self.ax_welfare.set_xlabel(r'$t$')

        self.publishers_welfare_line, = self.ax_welfare.plot([], [], label='Publishers\' Welfare',
                                                             color=self.publishers_welfare_color)
        self.users_welfare_line, = self.ax_welfare.plot([], [], label='Users\' Welfare',
                                                        color=self.info_need_scatter.get_facecolor())


        self.ax_welfare.legend(loc='upper center', bbox_to_anchor=(0.55, LEGEND_VERTICAL_POSITION))
        return [self.publishers_welfare_line, self.users_welfare_line]

    def update_ax_welfare(self, t, u, v):
        self.u_history.append(u), self.v_history.append(v)
        self.publishers_welfare_line.set_data(range(t + 1), self.v_history)
        self.users_welfare_line.set_data(range(t + 1), self.u_history)

        self.update_y_axis_ax_welfare(u, v)
        self.ax_welfare.set_xlim(0, t + 1)
        enforce_square(self.ax_welfare)

        return [self.users_welfare_line, self.publishers_welfare_line]

    def update_y_axis_ax_welfare(self, u, v):
        self.min_u = min(self.min_u, u)
        self.min_v = min(self.min_v, v)
        self.ax_welfare.set_ylim(min(self.min_u, self.min_v), 1)


class DiscreteBetterResponseAnimationFinite(DiscreteBetterResponseAnimationBase):

    def __init__(self, ranking_function=PRP, n=2, k=2, lam=0.5, T=1000, interval=150, x_0=None, x_star=None,
                 additional_params=None, history=None):
        assert T < INFINITY
        super().__init__(ranking_function, n, k, lam, T, interval, x_0, x_star, additional_params)

        # attributes for the welfare plot
        self.users_welfare_scatter, self.publishers_welfare_scatter = None, None
        self.publishers_welfare_line, self.users_welfare_line = None, None

        # attributes for the animation
        dynamics_generator = discrete_better_response_dynamics_generator(self.G, STEPS_,
                                                                         create_directions_set(self.G.k),
                                                                         self.T, EPS_)

        frames = list(enumerate(dynamics_generator)) if history is None else history
        # t, (x, u, v, converged) = frame
        self.u_history = [u for _, (_, u, _, _) in frames]
        self.v_history = [v for _, (_, _, v, _) in frames]

        self.ani = FuncAnimation(fig=self.fig, func=self.update,
                                 frames=frames,
                                 init_func=self.plot_init, interval=self.interval, repeat=False,
                                 cache_frame_data=False, blit=False)

    def init_ax_welfare(self):
        self.ax_welfare.set_title("Welfare Measures in Time", y=1.05, fontsize=16)
        self.ax_welfare.set_xlabel(r'$t$')

        self.publishers_welfare_line, = self.ax_welfare.plot(self.u_history, label='Publishers\' Welfare',
                                                             color=self.publishers_welfare_color)
        self.users_welfare_line, = self.ax_welfare.plot(self.v_history, label='Users\' Welfare',
                                                        color=self.info_need_scatter.get_facecolor())

        self.ax_welfare.set_xlim(0, self.T)
        enforce_square(self.ax_welfare)
        self.ax_welfare.legend(loc='upper center', bbox_to_anchor=(0.55, LEGEND_VERTICAL_POSITION))
        self.publishers_welfare_scatter = self.ax_welfare.scatter([], [], color=self.publishers_welfare_color)
        self.users_welfare_scatter = self.ax_welfare.scatter([], [], color=self.info_need_scatter.get_facecolor())
        return [self.publishers_welfare_scatter, self.users_welfare_scatter]

    def update_ax_welfare(self, t, u, v):
        self.publishers_welfare_scatter.set_offsets([t, u])
        self.users_welfare_scatter.set_offsets([t, v])
        return [self.publishers_welfare_scatter, self.users_welfare_scatter]


def animate_the_dynamics(ranking_function=PRP, n=2, k=2, lam=0.5, T=1000, interval=150, x_0=None, x_star=None,
                         a=None, beta=None, path=None, history=None):
    additional_params = {}
    if a is not None and ranking_function == LINEAR:
        additional_params['a'] = a
    if beta is not None and ranking_function == SOFTMAX:
        additional_params['beta'] = beta

    if T is None and history is not None:
        T = len(history)
    T = T if T not in [None, 'inf', 'infinity'] else INFINITY
    if T != INFINITY:
        animation = DiscreteBetterResponseAnimationFinite(
            ranking_function=ranking_function, n=n, k=k, lam=lam, T=T, interval=interval, x_0=x_0, x_star=x_star,
            additional_params=additional_params, history=history
        )
    else:
        animation = DiscreteBetterResponseAnimationINFINITY(
            ranking_function=ranking_function, n=n, k=k, lam=lam, T=T, interval=interval, x_0=x_0, x_star=x_star,
            additional_params=additional_params
        )

    if path is not None:
        animation.save(path)
    else:
        animation.show()