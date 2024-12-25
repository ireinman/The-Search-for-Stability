import numpy as np
from tqdm import tqdm

from utils.publishers_game import PRP, SOFTMAX, LINEAR, COLORS
from utils.simulations_utils import SEED, LAM_VALS, N_VALS, B_, STEPS_, T_, M_, EPS_, generate_normal_x0_x_star_for_2, comparison
from utils.general_utils import lam_key_function, n_key_function, ranking_function_label_function, ranking_function_sort_key, \
                                    save_results, create_plots_from_files


def run_prp_rrp_n_comparison(k=2, n_vals=N_VALS, lam=0.5, beta=20, B=B_, steps=STEPS_, T=T_, M=M_, eps=EPS_, 
                               desc='PRP_RRP_n_comparison'):
    ranking_function_lst = [PRP, SOFTMAX, LINEAR]
    additional_param_lst = [{}, {'beta': beta}, {}]

    np.random.seed(SEED)
    res = comparison(ranking_function_lst, additional_param_lst, [k], 
                     tqdm(n_vals, miniters=1), [lam], B, steps, T, M, eps)
    save_results(res, desc, ranking_function_lst, additional_param_lst, 
                 [k], list(n_vals), [lam], B, list(steps), T, M, eps)
    
    
def run_specific_ro_lam_comparison(k=2, lam_vals=LAM_VALS, B=B_, steps=STEPS_, T=T_, M=M_, eps=EPS_, 
                                    base_desc='PRP_RRP_lam_comparison', ro1=0, ro2=0):
    ranking_function_lst = [PRP, SOFTMAX, LINEAR]
    additional_param_lst = [{}, {}, {}]
    desc = base_desc + f'_ro1_{ro1}_ro2_{ro2}'
    np.random.seed(SEED)
    
    generate_x0_x_star = lambda k, n : generate_normal_x0_x_star_for_2(k, ro1, ro2)
    res = comparison(ranking_function_lst, additional_param_lst, [k], [2], 
                     tqdm(lam_vals, miniters=1), B, steps, T, M, eps, generate_x0_x_star=generate_x0_x_star)
    save_results(res, desc, ranking_function_lst, additional_param_lst, [k], [2], 
                 list(lam_vals), B, list(steps), T, M, eps)



def load_n_comparison(desc='PRP_RRP_n_comparison', beta=20, cond_key=lambda k, n, lam: True):
    file_label_function = lambda name: name.strip('.csv').split('_')[0] if len(name.strip('.csv').split('_')) == 1 else \
                                name.strip('.csv').split('_')[0] + '-' + name.strip('.csv').split('_')[1].split('.')[0]
    color_dict = {PRP: COLORS[PRP], SOFTMAX + '-' + str(beta): COLORS[SOFTMAX], LINEAR: COLORS[LINEAR]}
    create_plots_from_files(desc, n_key_function, file_label_function, color_dict, 
                            ranking_function_sort_key, cond_key=cond_key, publishers_auto=True)
    
    
def load_specific_ro_lam_comparison(base_desc='PRP_RRP_lam_comparison', ro1=0, ro2=0):
    desc = base_desc + f'_ro1_{ro1}_ro2_{ro2}'
    create_plots_from_files(desc, lam_key_function, ranking_function_label_function, 
                            COLORS, ranking_function_sort_key)

