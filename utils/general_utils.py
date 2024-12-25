import numpy as np
from pathlib import Path
from os import listdir
import matplotlib.pyplot as plt

from utils.publishers_game import PRP, SOFTMAX, COLORS_RANGE

SAVE_PATH = 'results'
FIGURES_PATH = 'figures'

K_STR = r'$k$'
N_STR = r'$n$'
LAM_STR = r'$\lambda$'

PUBLISHERS_LIMITS = [0.79, 1.01]
PUBLISHERS_TICKS = np.arange(0.8, 1.01, 0.025)
USERS_LIMITS = [0.79, 1.01]
USERS_TICKS = np.arange(0.8, 1.01, 0.025)
CONVERGENCE_LIMITS = [0, 1.1]
CONVERGENCE_TICKS = np.arange(0, 1.1, 0.1)

k_key_function = lambda k, n, lam: k
n_key_function = lambda k, n, lam: n  # we currently don't use this function
lam_key_function = lambda k, n, lam: lam

ranking_function_label_function = lambda name: name.strip('.csv').split('_')[0]
additional_param_label_function = lambda name: name.strip('.csv').split('_')[1]

ranking_function_sort_key = lambda x: 0 if x == PRP else (1 if x == SOFTMAX else 2)


def save_results(results, desc, ranking_function_lst, additional_param_lst,
                 k_vals, n_vals, lam_vals, B, steps, T, M, eps):
    """Saves the results in csv files and saves the parameters in the parameters text file."""
    try:
        Path(SAVE_PATH + '/' + desc).mkdir(parents=True, exist_ok=True)
        args_file_name = f"{SAVE_PATH}/{desc}/{desc}_args.txt"
        with open(args_file_name, 'w') as f:
            f.write(str(k_vals) + "\n")
            f.write(str(n_vals) + "\n")
            f.write(str(lam_vals) + "\n")
            f.write(str(B) + "\n")
            f.write(str(steps) + "\n")
            f.write(str(T) + "\n")
            f.write(str(M) + "\n")
            f.write(str(eps) + "\n")
        for i, r in enumerate(results):
            additional_param = list(additional_param_lst[i].values())
            additional_param_text = '' if len(additional_param) == 0 else '_' + str(
                additional_param[0])  # len(additional_param) <= 1
            res_file_name = f"{SAVE_PATH}/{desc}/{ranking_function_lst[i]}{additional_param_text}.csv"
            with open(res_file_name, 'w') as f:
                f.write("k,n,lam,publishers_welfare,users_welfare,convergence_ratio," +
                        "publishers_welfare_ci_low,publishers_welfare_ci_high," +
                        "users_welfare_ci_low,users_welfare_ci_high," +
                        "convergence_ratio_ci_low,convergence_ratio_ci_high\n")
                for (k, n, lam), (publishers_welfare, users_welfare, convergence_ratio,
                                  publishers_welfare_ci, users_welfare_ci, convergence_ratio_ci) in r.items():
                    f.write(f"{k},{n},{lam},{publishers_welfare},{users_welfare},{convergence_ratio}," +
                            f"{publishers_welfare_ci[0]},{publishers_welfare_ci[1]},"
                            f"{users_welfare_ci[0]},{users_welfare_ci[1]}," +
                            f"{convergence_ratio_ci[0]},{convergence_ratio_ci[1]}\n")
    except IOError as e:
        print(e)
        print("Couldn't save results")


def read_results(res_file_names):
    """Loads simulations results from csv files."""
    res = []
    for res_file_name in res_file_names:
        with open(res_file_name, 'r') as f:
            lines = [line.split(',') for line in f.readlines()[1:]]
            r = {(int(k), int(n), float(lam)): (float(publishers_welfare), float(users_welfare),
                                                float(convergence_ratio),
                                                [float(publishers_welfare_ci_low), float(publishers_welfare_ci_high)],
                                                [float(users_welfare_ci_low), float(users_welfare_ci_high)],
                                                [float(convergence_ratio_ci_low), float(convergence_ratio_ci_high)])
                 for k, n, lam, publishers_welfare, users_welfare, convergence_ratio,
                 publishers_welfare_ci_low, publishers_welfare_ci_high,
                 users_welfare_ci_low, users_welfare_ci_high,
                 convergence_ratio_ci_low, convergence_ratio_ci_high in lines}
            res.append(r)
    return res


def create_plots(res, x_vals, x_title, labels, colors_dict, publishers_auto=False, save_path=None):
    """Creates the plots for the results."""
    _, ax = plt.subplots(1, 3, figsize=(25, 5), sharey=False)

    # Publisher Welfare
    for i, label in enumerate(labels):
        publishers_welfare_plot = [res[i][x][0] for x in x_vals]
        ax[0].plot(x_vals, publishers_welfare_plot, marker='o', label=label, color=colors_dict[label])
        publishers_welfare_ci = [res[i][x][3] for x in x_vals]
        for j, x in enumerate(x_vals):
            ax[0].plot([x, x], publishers_welfare_ci[j], color=colors_dict[label])

    # ax[0].set_title("Publishers' Welfare")
    ax[0].set_ylabel("Publishers' Welfare", fontsize=14)
    ax[0].set_xlabel(x_title, fontsize=14)
    if len(labels) > 1:
        ax[0].legend(loc='lower left', fontsize=12)
    if not publishers_auto:
        ax[0].set_ylim(PUBLISHERS_LIMITS)
        ax[0].set_yticks(PUBLISHERS_TICKS)

    # Users Welfare

    for i, label in enumerate(labels):
        users_welfare_plot = [res[i][x][1] for x in x_vals]
        ax[1].plot(x_vals, users_welfare_plot, marker='o', label=label, color=colors_dict[label])
        users_welfare_ci = [res[i][x][4] for x in x_vals]
        for j, x in enumerate(x_vals):
            ax[1].plot([x, x], users_welfare_ci[j], color=colors_dict[label])

    # ax[1].set_title("Users' Welfare")
    ax[1].set_ylabel("Users' Welfare", fontsize=14)
    ax[1].set_xlabel(x_title, fontsize=14)
    if len(labels) > 1:
        ax[1].legend(loc='lower left', fontsize=12)
    ax[1].set_ylim(USERS_LIMITS)
    ax[1].set_yticks(USERS_TICKS)

    # Convergence Ratio

    for i, label in enumerate(labels):
        convergence_ratio_plot = [res[i][x][2] for x in x_vals]
        ax[2].plot(x_vals, convergence_ratio_plot, marker='o', label=label, color=colors_dict[label])
        convergence_ratio_ci = [res[i][x][5] for x in x_vals]
        for j, x in enumerate(x_vals):
            ax[2].plot([x, x], convergence_ratio_ci[j], color=colors_dict[label])

    # ax[2].set_title("Convergence Ratio")
    ax[2].set_ylabel("Convergence Ratio", fontsize=14)
    ax[2].set_xlabel(x_title, fontsize=14)
    if len(labels) > 1:
        ax[2].legend(loc='center left', fontsize=12)
    ax[2].set_ylim(CONVERGENCE_LIMITS)
    ax[2].set_yticks(CONVERGENCE_TICKS)

    # X ticks

    base_x_ticks = np.array(x_vals)
    for i in range(3):
        xticks = np.sort(np.intersect1d(np.round(ax[i].get_xticks(), 5), base_x_ticks))
        ax[i].set_xticks(xticks)

    if save_path is not None:
        plt.savefig(save_path, bbox_inches='tight', format="pdf", dpi=600)
    plt.show()


def create_plots_from_files(desc, key_x_function, file_label_function, colors_dict, 
                            labels_sort_key=None, cond_key=lambda k, n, lam: True, publishers_auto=False):
    """Creates the plots for the results from csv files."""
    dir_path = Path(SAVE_PATH + '/' + desc)
    try:
        files_names = listdir(dir_path)
    except FileNotFoundError:
        print("In order to load plots, you must first run the simulations")
        return
    files_full_path = [dir_path.joinpath(file_name) for file_name in files_names if file_name.endswith('.csv')]
    res = read_results(files_full_path)
    x_vals = [key_x_function(k, n, lam) for k, n, lam in res[0].keys() if cond_key(k, n, lam)]
    res = [{key_x_function(k, n, lam): res[i][(k, n, lam)] for k, n, lam in res[i].keys() if cond_key(k, n, lam)}
           for i in range(len(res))]
    labels = [file_label_function(file_name) for file_name in files_names if file_name.endswith('.csv')]

    if labels_sort_key is not None:
        combined_list = sorted(zip(res, labels), key=lambda x: labels_sort_key(x[1]))
        res, labels = zip(*combined_list)

    x_title = key_x_function(K_STR, N_STR, LAM_STR)

    fig_name = f"{FIGURES_PATH}/{desc}.pdf" 
    create_plots(res, x_vals, x_title, labels, colors_dict, publishers_auto,
                 save_path=fig_name)


def is_float(value):
    """Check if the value is float."""
    try:
        float(value)
        return True
    except ValueError:
        return False


def bootstrap_ci(z, bootstrap_B, alpha=0.05):
    """Returns the bootstrap confidence interval for the mean of z."""
    n = len(z)
    estimators = list()
    for b in range(bootstrap_B):
        sample = np.random.choice(z, size=n, replace=True)
        estimators.append(np.mean(sample))
    return [np.quantile(estimators, alpha / 2), np.quantile(estimators, 1 - (alpha / 2))]


def hex_to_rgb(hex_color):
    """ Converts a hex color code to an RGB tuple."""
    hex_color = hex_color.lstrip("#")  # Remove the "#" symbol if present

    # Split the hex code into R, G, and B components
    r = int(hex_color[0:2], 16) / 255.0
    g = int(hex_color[2:4], 16) / 255.0
    b = int(hex_color[4:6], 16) / 255.0
    return r, g, b


def create_interpolated_colors(tag, num_steps):
    """Creates a list of interpolated colors for a given tag."""
    color_range = COLORS_RANGE[tag]
    r1, g1, b1 = hex_to_rgb(color_range[0])
    r2, g2, b2 = hex_to_rgb(color_range[1])

    # Calculate the step size for each color component
    step_r = (r2 - r1) / (num_steps - 1) if num_steps > 1 else 0
    step_g = (g2 - g1) / (num_steps - 1) if num_steps > 1 else 0
    step_b = (b2 - b1) / (num_steps - 1) if num_steps > 1 else 0

    # Generate the interpolated colors
    interpolated_colors = []
    for i in range(num_steps):
        r = r1 + i * step_r
        g = g1 + i * step_g
        b = b1 + i * step_b
        interpolated_colors.append((r, g, b))

    return interpolated_colors
