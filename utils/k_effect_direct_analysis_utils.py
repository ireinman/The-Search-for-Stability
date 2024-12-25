import numpy as np

from utils.publishers_game import PRP, SOFTMAX, LINEAR, COLORS
from utils.simulations_utils import SEED, K_VALS, B_, STEPS_, T_, M_, EPS_, comparison
from utils.general_utils import k_key_function, ranking_function_label_function, ranking_function_sort_key, \
                            save_results, create_plots_from_files


def run_rrp_k_direct_comparison(k_vals=K_VALS, n=2, lam=0.5, B=B_, steps=STEPS_, T=T_, M=M_, eps=EPS_, 
                                desc='RRP_k_comparison'):
    """Run softmax and linear ranking functions with different k values. The results are saved in a csv file.

    Args:
        k_vals (iterable, optional): collection of embedding space dimension values. Defaults to K_VALS.
        n (int, optional): number of publishers. Defaults to 2.
        lam (float, optional): lambda value. Defaults to 0.5.
        B (int, optional): number of samples. Defaults to B_.
        steps (iterable, optional): collection of step sizes. Defaults to STEPS_.
        T (int, optional): number of simulation's rounds. Defaults to T_.
        M (int, optional): number of rounds to average from the end in case the simulation doesn't converge. Defaults to M_.
        eps (float, optional): threshold for improvement and convergence. Defaults to EPS_.
        desc (str, optional): directory name of the results. Defaults to 'RRP_k_comparison'.
    """
    ranking_function_lst = [SOFTMAX, LINEAR]
    additional_param_lst = [{}, {}]

    np.random.seed(SEED)
    res = comparison(ranking_function_lst, additional_param_lst, k_vals, [n], [lam], B, steps, T, M, eps, tqdm_key=k_key_function)
    save_results(res, desc, ranking_function_lst, additional_param_lst, list(k_vals), [n], [lam], B, list(steps), T, M, eps)


def run_prp_k_direct_comparison(k_vals=range(2, 6), n=2, lam=0.5, B=B_, steps=STEPS_, T=5000, M=4500, eps=EPS_,
                                desc='PRP_k_comparison'):
    """Run PRP ranking function with different k values. The results are saved in a csv file.

    Args:
        k_vals (iterable, optional): collection of embedding space dimension values. Defaults to range(2, 6).
        n (int, optional): number of publishers. Defaults to 2.
        lam (float, optional): lambda value. Defaults to 0.5.
        B (int, optional): number of samples. Defaults to B_.
        steps (iterable, optional): collection of step sizes. Defaults to STEPS_.
        T (int, optional): number of simulation's rounds. Defaults to 5000.
        M (int, optional): number of rounds to average from the end in case the simulation doesn't converge. Defaults to 4500.
        eps (float, optional): threshold for improvement and convergence. Defaults to EPS_.
        desc (str, optional): directory name of the results. Defaults to 'PRP_k_comparison'.
    """
    ranking_function_lst = [PRP]
    additional_param_lst = [{}]

    np.random.seed(SEED)
    res = comparison(ranking_function_lst, additional_param_lst, k_vals, [n], [lam], B, steps, T, M, eps, tqdm_key=k_key_function)
    save_results(res, desc, ranking_function_lst, additional_param_lst, list(k_vals), [n], [lam], B, list(steps), T, M, eps)


def load_rrp_k_direct_comparison(desc='RRP_k_comparison'):
    """Load results from csv files and plot them.

    Args:
        desc (str, optional): directory name of the results. Defaults to 'RRP_k_comparison'.
    """
    create_plots_from_files(desc, k_key_function, ranking_function_label_function, COLORS, ranking_function_sort_key)


def load_prp_k_direct_comparison(desc='PRP_k_comparison'):
    """Load results from csv files and plot them.

    Args:
        desc (str, optional): directory name of the results. Defaults to 'PRP_k_comparison'.
    """
    create_plots_from_files(desc, k_key_function, ranking_function_label_function, COLORS, None)
