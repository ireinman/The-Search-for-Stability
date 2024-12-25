import numpy as np
import itertools
from tqdm import tqdm

from utils.publishers_game import GAME_TYPES
from utils.general_utils import bootstrap_ci

SEED = 0

# Additional ranking functions parameters
BETA_VALS = np.concatenate([np.arange(0.5, 2.1, 0.5), np.array([3]), np.arange(5, 10.1, 5)])
A_VALS = np.round(np.arange(0.1, 0.51, 0.1), 1)

# Game parameters
K_VALS = range(2, 11)
N_VALS = range(2, 11)
LAM_VALS = np.round(np.arange(0.05, 1.01, 0.05), 2)

# Simulation parameters
B_ = 500
STEPS_ = sorted(np.round(np.arange(1, 0.49, -0.1), 1).tolist() + [0.5 ** j for j in range(2, 7)])
T_ = 1000
M_ = 900
EPS_ = 1e-6


def generate_normal_x0_x_star_for_2(k, ro1, ro2):
    """Generates x_0 and x_star."""
    mean = [0.5, 0.5, 0.5]
    cov = np.array([[1, ro1, ro2], [ro1, 1, ro1 * ro2], [ro2, ro1 * ro2, 1]])
    x_0 = np.zeros((2, k))
    x_star = np.zeros(k)
    for i in range(k):
        res = np.random.multivariate_normal(mean, cov)
        while not (np.all(res >= 0) and np.all(res <= 1)):
            res = np.random.multivariate_normal(mean, cov)
        x_0[0, i] = res[0]
        x_0[1, i] = res[1]
        x_star[i] = res[2]
    return x_0, x_star


def create_directions_set(k):
    """Creates a set of directions for the discrete better response dynamic."""
    directions = itertools.product([-1, 0, 1], repeat=k)
    directions = np.array([np.array(d) / np.linalg.norm(d) for d in directions if np.linalg.norm(d) > 0])
    return directions


def find_best_step_strategy_discrete(G, i, directions, steps):
    """Finds the best improvement step for player i in the discrete better response dynamic."""
    np.random.shuffle(directions)
    max_util = -np.inf
    strategy = None
    for d, step_size in itertools.product(directions, steps):
        new_xi = G.x[i] + step_size * d
        if np.all((new_xi >= 0) & (new_xi <= 1)):
            new_ui = G.calc_u(new_xi, i)
            if new_ui > max_util:
                strategy, max_util = new_xi, new_ui
    return strategy, max_util


def discrete_better_response_dynamic(G, steps, directions, T, M, eps):
    """Runs the discrete better response dynamic."""
    G.initialize()
    convergence_status = False
    history = [G.x.copy()]
    for _ in range(T):
        players_permutation = np.random.permutation(G.N)
        found_improving_player = False
        for i in players_permutation:
            x_i_moved, new_u = find_best_step_strategy_discrete(G, i, directions, steps)
            if G.get_u(i) + eps < new_u:
                G.update_x_i(x_i_moved, i)
                history.append(G.x.copy())
                found_improving_player = True
                break
        if not found_improving_player:
            publishers_welfare = G.get_publishers_welfare()
            users_welfare = G.get_users_welfare()
            convergence_status = True
            return publishers_welfare, users_welfare, convergence_status

    publishers_welfare = sum([G.update_x(x).get_publishers_welfare() for x in history[-M:]]) / M
    users_welfare = sum([G.update_x(x).get_users_welfare() for x in history[-M:]]) / M

    return publishers_welfare, users_welfare, int(convergence_status)


def full_simulation(ranking_function, additional_param, instances, k, n, lam, steps, T, M, eps, tqdm_key=None):
    """Runs a full simulation."""
    B = len(instances)
    # init results
    publishers_welfare_res = np.zeros(B)
    users_welfare_res = np.zeros(B)
    convergence_ratio_res = np.zeros(B)

    loop = range(B) if tqdm_key is None else tqdm(range(B),
                                                  desc=f'{tqdm_key("k", "n", "lambda")}={tqdm_key(k, n, lam)}')
    for i in loop:
        x_0, x_star = instances[i]

        G = GAME_TYPES[ranking_function](k, n, lam, x_0, x_star, **additional_param)

        directions_set = create_directions_set(k)
        publishers_welfare_res[i], users_welfare_res[i], convergence_ratio_res[i] = \
            discrete_better_response_dynamic(G, steps, directions_set, T, M, eps)

    return np.mean(publishers_welfare_res), np.mean(users_welfare_res), \
        np.mean(convergence_ratio_res), bootstrap_ci(publishers_welfare_res, B), \
        bootstrap_ci(users_welfare_res, B), bootstrap_ci(convergence_ratio_res, B)


def comparison(ranking_function_lst, additional_param_lst, k_vals, n_vals, lam_vals, B, steps, T, M, eps,
               tqdm_key=None, generate_x0_x_star=None):
    """Compares ranking functions on a grid of parameters.

    Args:
        ranking_function_lst (list): list of ranking functions.
        additional_param_lst (list): list of additional parameters for the ranking functions
                                        in the form {param_name: param_value}.
        k_vals (iterable): collection of embedding space dimensions.
        n_vals (iterable): collection publishers' amounts.
        lam_vals (iterable): collection of lambda values.
        B (int): number of samples.
        steps (iterable): collection of step sizes.
        T (int): number of rounds.
        M (int): number of rounds to average from the end in case the simulation doesn't converge.
        eps (float): threshold for improvement and convergence.
        tqdm_key (function or None, optional): key function for tqdm or None. Defaults to None.
        generate_x0_x_star (function or None, optional): function to generate x0 and x_star or None. Defaults to None.

    Returns:
        list: list of results for each ranking function.
    """
    assert len(ranking_function_lst) == len(additional_param_lst), \
        "ranking_function_lst and additional_param_lst must be of the same length"
    assert len(
        max(additional_param_lst, key=len)) <= 1, "additional_param_lst must be of the form {param_name: param_value}"
    assert 0 < M <= T, "M must be between 0 and T"
    assert callable(generate_x0_x_star) or generate_x0_x_star is None, "generate_x0_x_star must be a function or None"

    amount = len(ranking_function_lst)
    res = [{} for _ in range(amount)]
    if generate_x0_x_star is not None:
        generate_func = lambda k, n: generate_x0_x_star(k, n)
    else:
        generate_func = lambda k, n: (np.random.rand(n, k), np.random.rand(k))
    for k in k_vals:
        for n in n_vals:
            instances = [generate_func(k, n) for _ in range(B)]
            for lam in lam_vals:
                for i in range(amount):
                    res[i][(k, n, lam)] = full_simulation(ranking_function_lst[i], additional_param_lst[i],
                                                          instances, k, n, lam, steps, T, M, eps, tqdm_key)
    return res
