import numpy as np
import itertools
from tqdm import tqdm
from pathlib import Path
from matplotlib import pyplot as plt
from os import listdir

from utils.publishers_game import PRP, COLORS, PRPPublishersGame
from utils.simulations_utils import SEED, K_VALS, STEPS_, EPS_, create_directions_set
from utils.general_utils import SAVE_PATH, FIGURES_PATH, K_STR, CONVERGENCE_LIMITS, CONVERGENCE_TICKS, bootstrap_ci


def check_improve(G, i, directions, steps, eps):
    """Find whether player i can move single step to improve his utility by at least epsilon."""
    curr_u = G.get_u(i)
    for d, step_size in itertools.product(directions, steps): 
        addition = step_size * d
        new_xi = G.x[i] + addition
        if np.all((new_xi >= 0) & (new_xi <= 1)):   
            new_ui = G.calc_u(new_xi, i)   
            if new_ui > curr_u + eps:
                return True
    return False


def save_prp_k_converge_comparison(results, desc, k_vals, n_vals, lam_vals, B, steps, eps):
    """Saves the results in csv files and saves the parameters in parameters text file."""
    try:
        Path(SAVE_PATH + '/' + desc).mkdir(parents=True, exist_ok=True) 
        args_file_name = f"{SAVE_PATH}/{desc}/{desc}_args.txt"
        with open(args_file_name, 'w') as f:
            f.write(str(k_vals)+"\n")
            f.write(str(n_vals)+"\n")
            f.write(str(lam_vals)+"\n")
            f.write(str(B)+"\n")
            f.write(str(steps)+"\n")
            f.write(str(eps)+"\n")
        res_file_path = f'{SAVE_PATH}/{desc}/{desc}.csv'
        with open(res_file_path, 'w') as f:
            f.write("k,n,lam,convergence_ratio,convergence_ratio_ci_low,convergence_ratio_ci_high\n")
            for (k, n, lam), (convergence_ratio, convergence_ratio_ci) in results.items():
                f.write(f"{k},{n},{lam},{convergence_ratio},{convergence_ratio_ci[0]},{convergence_ratio_ci[1]}\n")
    except IOError as e:
        print(e)
        print("Couldn't save results")


def run_prp_k_converge_comparison(k_vals=K_VALS, n=2, lam=0.5, B=10000, steps=STEPS_, eps=EPS_,
                                  desc='PRP_k_converge_comparison'):
    """Run PRP ranking function with different k values. The results are saved in a csv file.

    Args:
        k_vals (iterable, optional): collection of embedding space dimension values. Defaults to K_VALS.
        n (int, optional): number of publishers. Defaults to 2.
        lam (float, optional): lambda value. Defaults to 0.5.
        B (int, optional): number of samples. Defaults to 10000.
        steps (iterable, optional): collection of step sizes. Defaults to STEPS_.
        eps (float, optional): threshold for improvement and convergence. Defaults to EPS_.
        desc (str, optional): directory name of the results. Defaults to 'PRP_k_converge_comparison'.
    """
    np.random.seed(SEED)
    conv = np.ones((len(k_vals), B))
    for j, k in enumerate(k_vals):
        directions = create_directions_set(k)
        x_0 = np.random.rand(B, n, k)
        x_star = np.random.rand(B, k)
        for b in tqdm(range(B), desc=f'k={k}'):
            G = PRPPublishersGame(k, n, lam, x_0[b], x_star[b])
            G.save_r_all()
            for i in range(n):
                if G.r[i] == 1: # if r[i] == 1 and x=x_0 then the player has maximum utility of 1
                    continue
                elif check_improve(G, i, directions, steps, eps):
                    conv[j, b] = 0 # the game doesn't converge
                    break
                  
    res = {(k, n, lam): (np.mean(conv[i]), bootstrap_ci(conv[i], B)) for i, k in enumerate(k_vals)}
    save_prp_k_converge_comparison(res, desc, list(k_vals), [n], [lam], B, list(steps), eps)


def read_results_prp_k_converge_comparison(res_file_name):
    """Loads simulations results from csv files."""
    res = dict()
    with open(res_file_name, 'r') as f:
        lines = [line.split(',') for line in f.readlines()[1:]]
        res = {(int(k), int(n), float(lam)): 
               (float(convergence_ratio), [float(convergence_ratio_ci_low), float(convergence_ratio_ci_high)]) 
               for k, n, lam, convergence_ratio, convergence_ratio_ci_low, convergence_ratio_ci_high, in lines}
    return res


def plot_prp_k_converge_comparison(res, k_vals, fig_name=None):
    """Plot the converge ratio of the PRP with different k values."""

    _, ax = plt.subplots(figsize=(7, 5))

    convergence_ratio_plot = [res[k][0] for k in k_vals]
    ax.plot(k_vals, convergence_ratio_plot, marker='o', color=COLORS[PRP])
    convergence_ratio_ci = [res[k][1] for k in k_vals]
    for j, k in enumerate(k_vals):
        ax.plot([k, k], convergence_ratio_ci[j], color=COLORS[PRP])

    # ax.set_title("Convergence Ratio")
    ax.set_ylabel("Convergence Ratio")
    ax.set_xlabel(K_STR)
    ax.set_ylim(CONVERGENCE_LIMITS)
    ax.set_yticks(CONVERGENCE_TICKS)
    ax.set_xticks(k_vals)
    
    if fig_name is not None:
        plt.savefig(fig_name, bbox_inches='tight', format="pdf", dpi=600)
    plt.show()


def load_prp_k_converge_comparison(desc='PRP_k_converge_comparison'):
    """Load results from csv files and plot them.

    Args:
        desc (str, optional): directory name of the results. Defaults to 'PRP_k_converge_comparison'.
    """
    dir_path = Path(SAVE_PATH + '/' + desc)
    try:
        files_names = listdir(dir_path)
    except IOError:
        print("In order to load plots, you must first run the simulations")
        return
    files_full_path = [dir_path.joinpath(file_name) for file_name in files_names if file_name.endswith('.csv')]
    assert len(files_full_path) == 1, "There should be only one csv file in the directory"
    res = read_results_prp_k_converge_comparison(files_full_path[0])
    
    k_vals = [k for k, _, _ in res.keys()]
    res = {k: res[(k, n, lam)] for k, n, lam in res.keys()}  
    
    fig_name = f"{FIGURES_PATH}/{desc}.pdf" 
    plot_prp_k_converge_comparison(res, k_vals, fig_name)
