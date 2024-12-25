import numpy as np
from tqdm import tqdm

from utils.publishers_game import PRP, SOFTMAX, LINEAR, COLORS
from utils.simulations_utils import SEED, BETA_VALS, A_VALS, LAM_VALS, B_, STEPS_, T_, M_, EPS_, comparison
from utils.general_utils import SAVE_PATH, lam_key_function, ranking_function_label_function, \
                                    additional_param_label_function, ranking_function_sort_key, \
                                    save_results, read_results, create_plots_from_files, \
                                    create_interpolated_colors, is_float


def run_prp_rrp_lam_comparison(k=2, n=2, lam_vals=LAM_VALS, B=B_, steps=STEPS_, T=T_, M=M_, eps=EPS_, 
                               desc='PRP_RRP_lam_comparison'):
    """Run PRP, softmax and linear ranking functions with different lambda values. The results are saved in a csv file.

    Args:
        k (int, optional): embedding space dimension. Defaults to 2.
        n (int, optional): number of publishers. Defaults to 2.
        lam_vals (iterable, optional): collection of lambda values. Defaults to LAM_VALS.
        B (int, optional): number of samples. Defaults to B_.
        steps (iterable, optional): collection of step sizes. Defaults to STEPS_.
        T (int, optional): number of simulation's rounds. Defaults to T_.
        M (int, optional): number of rounds to average from the end in case the simulation doesn't converge. Defaults to M_.
        eps (float, optional): threshold for improvement and convergence. Defaults to EPS_.
        desc (str, optional): directory name of the results. Defaults to 'PRP_RRP_lam_comparison'.
    """
    ranking_function_lst = [PRP, SOFTMAX, LINEAR]
    additional_param_lst = [{}, {}, {}]

    np.random.seed(SEED)
    res = comparison(ranking_function_lst, additional_param_lst, [k], [n], tqdm(lam_vals, miniters=1), B, steps, T, M, eps)
    save_results(res, desc, ranking_function_lst, additional_param_lst, [k], [n], list(lam_vals), B, list(steps), T, M, eps)


def run_softmax_beta_comparison(beta_values=BETA_VALS, k=2, n=2, lam_vals=LAM_VALS, B=B_, steps=STEPS_, T=T_, M=M_, eps=EPS_,
                                desc='Softmax_beta_comparison', prp_directory='PRP_RRP_lam_comparison'):
    """Run softmax ranking function with different beta values. The results are saved in a csv file.
    
    Args:
        beta_values (iterable, optional): collection of beta values. Defaults to BETA_VALS.
        k (int, optional): embedding space dimension. Defaults to 2.
        n (int, optional): number of publishers. Defaults to 2.
        lam_vals (iterable, optional): collection of lambda values. Defaults to LAM_VALS.
        B (int, optional): number of samples. Defaults to B_.
        steps (iterable, optional): collection of step sizes. Defaults to STEPS_.
        T (int, optional): number of simulation's rounds. Defaults to T_.
        M (int, optional): number of rounds to average from the end in case the simulation doesn't converge. Defaults to M_.
        eps (float, optional): threshold for improvement and convergence. Defaults to EPS_.
        desc (str, optional): directory name of the results. Defaults to 'Softmax_beta_comparison'.
        prp_directory (str, optional): directory name of the prp simulation results. Defaults to 'PRP_RRP_lam_comparison'.
    """
    amount = len(beta_values)
    ranking_function_lst = [SOFTMAX] * amount
    additional_param_lst = [{'beta': beta} for beta in beta_values]

    prp_path = f'{SAVE_PATH}/{prp_directory}/{PRP}.csv'
    try:
        prp_res = read_results([prp_path])
    except FileNotFoundError:
        print('Simulation prp_rrp_lam_comparison must be run first')
        return

    np.random.seed(SEED)
    res = comparison(ranking_function_lst, additional_param_lst, [k], [n], 
                     tqdm(lam_vals, miniters=1), B, steps, T, M, eps)
    res = res + prp_res
    save_results(res, desc, ranking_function_lst + [PRP], additional_param_lst + [{}], 
                 [k], [n], list(lam_vals), B, list(steps), T, M, eps)


def run_linear_slope_comparison(a_vals=A_VALS, k=2, n=2, lam_vals=LAM_VALS, B=B_, steps=STEPS_, T=T_, eps=EPS_, M=M_, 
                                desc='Linear_slope_comparison'):
    """Run linear ranking function with different slope values. The results are saved in a csv file.
    
    Args:
        a_vals (iterable, optional): collection of slope values. Defaults to A_VALS.
        k (int, optional): embedding space dimension. Defaults to 2.
        n (int, optional): number of publishers. Defaults to 2.
        lam_vals (iterable, optional): collection of lambda values. Defaults to LAM_VALS.
        B (int, optional): number of samples. Defaults to B_.
        steps (iterable, optional): collection of step sizes. Defaults to STEPS_.
        T (int, optional): number of simulation's rounds. Defaults to T_.
        M (int, optional): number of rounds to average from the end in case the simulation doesn't converge. Defaults to M_.
        eps (float, optional): threshold for improvement and convergence. Defaults to EPS_.
        desc (str, optional): directory name of the results. Defaults to 'Linear_slope_comparison'.
    """
    assert np.all(np.array(a_vals) <= 1 / n), 'a must be smaller or equal to 1/n'
    amount = len(a_vals)
    ranking_function_lst = [LINEAR] * amount
    additional_param_lst = [{'a': a} for a in a_vals]

    np.random.seed(SEED)
    res = comparison(ranking_function_lst, additional_param_lst, [k], [n], 
                     tqdm(lam_vals, miniters=1), B, steps, T, M, eps)
    save_results(res, desc, ranking_function_lst, additional_param_lst, [k], [n], 
                 list(lam_vals), B, list(steps), T, M, eps)


def load_prp_rrp_lam_comparison(desc='PRP_RRP_lam_comparison'):
    """Load and plot the results of the prp_rrp_lam_comparison simulation.

    Args:
        desc (str, optional): directory name of the results. Defaults to 'PRP_RRP_lam_comparison'.
    """
    create_plots_from_files(desc, lam_key_function, ranking_function_label_function, 
                            COLORS, ranking_function_sort_key)


def load_softmax_beta_comparison(desc='Softmax_beta_comparison'):
    """Load and plot the results of the softmax_beta_comparison simulation.

    Args:
        desc (str, optional): directory name of the results. Defaults to 'Softmax_beta_comparison'.
    """
    beta_eq = r'$\beta$='
    prp_softmax_labels = [PRP] + [beta_eq + str(beta) for beta in BETA_VALS]
    prp_softmax_colors_lst = [COLORS[PRP]] + create_interpolated_colors(SOFTMAX, len(BETA_VALS))
    prp_softmax_colors_dict = dict(zip(prp_softmax_labels, prp_softmax_colors_lst))
    file_label_function = lambda file_name: ranking_function_label_function(file_name) \
                            if ranking_function_label_function(file_name) == PRP \
                            else beta_eq + additional_param_label_function(file_name)
    labels_sort_key = lambda x: (0, float(x.strip(beta_eq))) if is_float(x.strip(beta_eq)) else (1, x)
    create_plots_from_files(desc, lam_key_function, file_label_function, 
                            prp_softmax_colors_dict, labels_sort_key)


def load_linear_slope_comparison(desc='Linear_slope_comparison'):
    """Load and plot the results of the linear_slope_comparison simulation.

    Args:
        desc (str, optional): directory name of the results. Defaults to 'Linear_slope_comparison'.
    """
    a_eq = 'a='
    lin_colors_lst = create_interpolated_colors(LINEAR, len(A_VALS))
    lin_colors_dict = dict(zip([a_eq + str(a) for a in A_VALS], lin_colors_lst))
    file_label_function = lambda file_name: a_eq + additional_param_label_function(file_name)
    labels_sort_key = lambda x: float(x.strip(a_eq))
    create_plots_from_files(desc, lam_key_function, file_label_function, lin_colors_dict, labels_sort_key)
