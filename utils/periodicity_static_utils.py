import numpy as np
from matplotlib import pyplot as plt

from utils.publishers_game import PRP, GAME_TYPES
from utils.simulations_utils import STEPS_, T_, EPS_, SEED, create_directions_set, find_best_step_strategy_discrete
from utils.general_utils import FIGURES_PATH


def get_BR_history(G, steps, directions, T, eps):
    """Runs the discrete better response dynamics simulation but here returns only the history,
    that is, the generated sequence of profiles.
    """
    G.initialize()
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
            return history

    return history


def plot_welfare_measures(ranking_function=PRP, additional_params=None, k=2, n=2, lam=0.5, steps=STEPS_, T=T_, eps=EPS_,
                          x_0=None, x_star=None):
    """Plot welfare measures (publishers' welfare and users' welfare) of the ranking function.

    Args:
        ranking_function (function, optional): ranking function. Defaults to PRP.
        additional_params (dict, optional): additional parameter for the ranking functions in the form {param_name: param_value}.
        k (int, optional): embedding space dimension. Defaults to 2.
        n (int, optional): number of publishers. Defaults to 2.
        lam (float, optional): lambda value. Defaults to 0.5.
        steps (iterable, optional): collection of step sizes. Defaults to STEPS_.
        T (int, optional): number of iterations. Defaults to 1000.
        eps (float, optional): threshold for improvement and convergence. Defaults to EPS_.
        x_0 (np.ndarray, optional): initial publishers' strategies. Defaults to random.
        x_star (np.ndarray, optional): optimal publishers' strategies. Defaults to random.
    """
    np.random.seed(SEED)
    # get the history of the dynamics
    x_0 = np.random.rand(n, k) if x_0 is None else x_0
    x_star = np.random.rand(k) if x_star is None else x_star
    additional_params = {} if additional_params is None else additional_params
    G = GAME_TYPES[ranking_function](k=k, n=n, lam=lam, x_0=x_0, x_star=x_star, **additional_params)
    history = get_BR_history(G, steps=steps, directions=create_directions_set(k), T=T, eps=eps)

    # calculate the welfare measures
    publishers_welfare_history = np.array([G.update_x(x).get_publishers_welfare() for x in history])
    users_welfare_history = np.array([G.update_x(x).get_users_welfare() for x in history])

    # plot the welfare measures in the same plot
    _, ax = plt.subplots()
    ax.plot(publishers_welfare_history, label='Publishers\' welfare')
    ax.plot(users_welfare_history, label='Users\' welfare')
    ax.set_xlabel(r'$t$')
    # ax.set_title('Welfare Measures in Time')
    ax.set_ylabel('Welfare Measures')
    ax.legend()
    
    save_path = f"{FIGURES_PATH}/welfare_measures_prp.pdf" 
    plt.savefig(save_path, bbox_inches='tight', format="pdf", dpi=600)
    plt.show()
