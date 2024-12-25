import numpy as np
from tqdm import tqdm
import matplotlib.pyplot as plt
from scipy import stats

from utils.simulations_utils import SEED
from utils.general_utils import FIGURES_PATH, K_STR

N_MAX = 11
K_MAX = 21
LINES_COLORS = plt.rcParams['axes.prop_cycle'].by_key()['color'][:N_MAX - 1]
CONFIDENCE_LEVEL = 0.95

default_dist_func = lambda x1, x2, k: np.power(np.linalg.norm(x1 - x2), 2) / k


def plot_regularity_probability(n_vals=range(2, N_MAX, 2), k_vals=range(1, K_MAX, 2), B=10000, 
                                    dist_func=default_dist_func, lines_colors=LINES_COLORS):
    """Plot the probability of regularity condition for different n and k values.

    Args:
        n_vals (iterable, optional): collection of number of publishers values. Defaults to range(2, N_MAX + 1).
        k_vals (iterable, optional): collection of embedding space dimension values. Defaults to range(1, K_MAX + 1).
        B (int, optional): number of samples. Defaults to 100000.
        dist_func (function, optional): distance function. Defaults to default_dist_func.
        lines_colors (iterable, optional): collection of colors for the lines. Defaults to LINES_COLORS.
    """
    
    assert len(n_vals) <= len(lines_colors), 'Not enough colors'
    res = np.zeros((len(n_vals), len(k_vals), B))
    
    np.random.seed(SEED)
    for ni, n in enumerate(n_vals):
        for ki, k in enumerate(k_vals):
            for i in range(B):
                x0 = np.random.rand(n, k)
                x_star = np.random.rand(k)
                d = [dist_func(x0[j], x_star, k) for j in range(n)]
                d_avg = sum(d) / n
                d_tag = sum([(sum([d[j] * d[other] for other in range(n) if other != j]) / (n - 1)) - d[j] ** 2 
                             for j in range(n)]) / n
                res[ni][ki][i] = int(0 <= d_avg + 2*d_tag)
    
    sqrt_len = np.sqrt(B)
    for ni, n in enumerate(n_vals):
        avg_res = np.mean(res[ni], 1)
        plt.plot(k_vals, avg_res, marker='o', label=f'n = {n}', color=lines_colors[ni])
        std_res = np.std(res[ni], 1) / sqrt_len
        res_ci = [stats.norm.interval(CONFIDENCE_LEVEL, loc=avg_res[j], scale=std_res[j]) 
                  if std_res[j] > 0 else (avg_res[j], avg_res[j]) for j in range(len(k_vals))]
        for ki in range(len(k_vals)):
            plt.plot([k, k], (res_ci[ki][0], min([res_ci[ki][0], 1])), color=lines_colors[ni])
    plt.xlabel(K_STR)
    plt.xticks(k_vals[::2])
    plt.ylabel('Probability')
    # plt.title('Probability of regularity condition')
    plt.legend(loc='lower right')
    
    plt.savefig(f'{FIGURES_PATH}/regularity_probability.pdf', bbox_inches='tight', format="pdf", dpi=600)
    plt.show()    