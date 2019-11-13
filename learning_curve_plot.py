import sys, getopt
import numpy as np
import json
import os
import re
from collections import OrderedDict
import matplotlib.pyplot as plt
import matplotlib.animation as animation
from mpl_toolkits.mplot3d import Axes3D
from matplotlib.collections import PolyCollection
from matplotlib import style

style.use('seaborn')

def extract_training_folders(behavior_folder, training_folders):
    sub_dirs = [d[0] for d in os.walk(behavior_folder)]

    behavior = behavior_folder[behavior_folder.find('humanoid3d'):]
    behavior = behavior.replace("humanoid3d/", "")
    behavior = behavior.split('/', 1)[0]
    behavior = behavior.split('_', 1)[0]
    behavior = behavior.replace('ning', '')
    behavior = behavior.replace('ing', '')
    behavior = behavior.capitalize()
    plot_title = "Learning Curve - " + behavior

    for d in sub_dirs:
        if 'd_2d' in d:
            continue
        elif 'mixed_motions' in d:
            continue
        elif 'scaling' in d:
            continue
        elif 'lrn_std' in d:
            continue
        elif 'ModPD' in d:
            continue
        elif 'RGoal' in d:
            continue
        elif 'NoMean' in d:
            continue
        elif 'variablize_std' in d:
            continue
        elif 'Reward_R1-x' in d:
            continue
        elif 'Older' in d:
            continue

        if 'baseline_axis_pei_fix' in d:
            training_folders.append(d)
        elif 'pca_activation_euler' in d:
            training_folders.append(d)
        elif 'ica_activation_euler' in d:
            training_folders.append(d)

    return behavior, training_folders


def extract_plot_info(training_folders, step_size = 400, verbose=False,
                      multi_train=False, first_trained=False):
    plot_dicts_list_all = []
    plot_dicts_list_R0 = []
    plot_dicts_list_R1 = []
    plot_dicts_list_R2 = []
    plot_dicts_list_R_g = []
    plot_dicts_list_R_e = []

    for d in training_folders:
        learning_curve_data = OrderedDict()

        if 'R0' in d:
            learning_curve_data['reward_type'] = 'R0'
            learning_curve_data['reward_no'] = 0
        elif 'R1' in d:
            learning_curve_data['reward_type'] = 'R1'
            learning_curve_data['reward_no'] = 1
        elif 'R2' in d:
            learning_curve_data['reward_type'] = 'R2'
            learning_curve_data['reward_no'] = 2
        elif 'Rg' in d:
            learning_curve_data['reward_type'] = 'R_g'
            learning_curve_data['reward_no'] = 3
        elif 'Reg' in d:
            learning_curve_data['reward_type'] = 'R_e'
            learning_curve_data['reward_no'] = 4
        else:
            learning_curve_data['reward_type'] = 'R0'
            learning_curve_data['reward_no'] = 0

        if 'centered' in d:
            learning_curve_data['centered'] = True
        else:
            learning_curve_data['centered'] = False

        if 'pca_activation_euler' in d:
            learning_curve_data['type'] = 'PCA'
            dims = re.search('centered_(.*)d_', d)
            if dims is None:
                dims = re.search('euler_(.*)d_', d)
            if dims is None:
                dims = re.search('euler_(.*)d', d)
            dims = int(dims.group(1))
            learning_curve_data['dims'] = dims
            learning_curve_data['label'] = str(dims) + 'D'
        elif 'baseline_axis_pei_fix' in d:
            learning_curve_data['type'] = 'Baseline'
            learning_curve_data['dims'] = 0
            learning_curve_data['label'] = 'Baseline'
        elif 'ica_activation_euler' in d:
            learning_curve_data['type'] = 'ICA'
            dims = re.search('euler_(.*)d_', d)
            if dims is None:
                dims = re.search('euler_(.*)d', d)
            dims = int(dims.group(1))
            learning_curve_data['dims'] = dims
            learning_curve_data['label'] = str(dims) + 'D'

        if verbose:
            if learning_curve_data['type'] in ['PCA','ICA']:
                learning_curve_data['label'] = \
                    learning_curve_data['type'] + '-' + str(dims) + 'D'# + '-' + \
                    #learning_curve_data['reward_type']
            else:
                learning_curve_data['label'] = learning_curve_data['type'] + '-' \
                                               + learning_curve_data['reward_type']

        try:
            with open(d+'/agent0_log.txt') as f:
                lines = f.read().splitlines()
        except:
            continue
        del lines[0]

        learning_iterations = []
        line_array = np.zeros(5)
        iter_offset = np.zeros(5)
        for line in lines:
            try:
                line_array = np.array(line.split()[0:5], dtype=np.float32)
                line_array += iter_offset
            except: # Encountered "Iterations...", which indicates a new training run
                if multi_train:
                    if first_trained:
                        break
                    else:
                        iter_offset = np.copy(line_array)
                        iter_offset[3:5] = 0
                else:
                    print("Warning: Multi-Train Encountered in folder:", d)
                    learning_iterations = []
                continue
            if line_array[0] % step_size == 0:
                learning_iterations.append(line_array)

        learning_matrix =  np.array(learning_iterations)

        learning_curve_data['Iterations'] = np.array(learning_matrix[:,0:1],
                                                     dtype=np.float32)
        learning_curve_data['Wall_Time'] = np.array(learning_matrix[:,1:2],
                                                    dtype=np.float32)
        learning_curve_data['Samples'] = np.array(learning_matrix[:,2:3],
                                                  dtype=np.float32)
        learning_curve_data['Train_Return'] = np.array(learning_matrix[:,3:4],
                                                       dtype=np.float32) / 600.0
        learning_curve_data['Test_Return'] = np.array(learning_matrix[:,4:5],
                                                      dtype=np.float32) / 600.0

        plot_dicts_list_all.append(learning_curve_data)

        if learning_curve_data['reward_type'] == 'R0':
            plot_dicts_list_R0.append(learning_curve_data)
        elif learning_curve_data['reward_type'] == 'R1':
            plot_dicts_list_R1.append(learning_curve_data)
        elif learning_curve_data['reward_type'] == 'R2':
            plot_dicts_list_R2.append(learning_curve_data)
        elif learning_curve_data['reward_type'] == 'R_g':
            plot_dicts_list_R_g.append(learning_curve_data)
        elif learning_curve_data['reward_type'] == 'R_e':
            plot_dicts_list_R_e.append(learning_curve_data)

    return plot_dicts_list_all


def select_and_sort_plots_dicts(plot_dicts_list, plot_dims, reward_fn, baseline,
                                pca, ica, centered):
    plots_list = []
    for i, plot in enumerate(plot_dicts_list):
        if plot['dims'] in plot_dims and plot['reward_no'] in reward_fn:
            if baseline and plot['type'] == 'Baseline':
                plots_list.append(plot)
            elif pca and plot['type'] == 'PCA':
                if centered and not plot['centered']:
                    continue
                else:
                    plots_list.append(plot)
            elif ica and plot['type'] == 'ICA':
                plots_list.append(plot)

    sorted_plots_list = \
        sorted(plots_list, key = lambda i: (i['dims'], i['reward_no'], i['type']))

    return sorted_plots_list


def output_returns(sorted_plots_list, sample_threshold):
    for i, plot in enumerate(sorted_plots_list):
        learning_matrix = np.column_stack([plot['Samples'],
                                           plot['Train_Return'],
                                           plot['Test_Return']])

        print("\n", (plot['label'] + '_' + plot['reward_type']) )
        for i, lm in enumerate(learning_matrix):
            if lm[0] > sample_threshold:
                learning_matrix = learning_matrix[0:i]
                break

        print("Max. Test_Return: ", np.max(learning_matrix[:,2]))
    return


def plot_2D(sorted_plots_list, x_axis, plot_title, x_upper, x_axis_label, x_lim,
            y_axis_label, legend):
    lc = ['xkcd:red', 'xkcd:blue', 'xkcd:green', 'xkcd:brown', 'xkcd:pink',
          'xkcd:purple', 'xkcd:orange', 'xkcd:magenta', 'xkcd:tan', 'xkcd:black',
          'xkcd:cyan', 'xkcd:gold', 'xkcd:dark green', 'xkcd:cream',
          'xkcd:lavender', 'xkcd:turquoise', 'xkcd:dark blue', 'xkcd:violet',
          'xkcd:beige', 'xkcd:salmon', 'xkcd:olive', 'xkcd:light brown',
          'xkcd:hot pink', 'xkcd:dark red', 'xkcd:sand', 'xkcd:army green',
          'xkcd:dark grey', 'xkcd:crimson', 'xkcd:eggplant', 'xkcd:coral']
    ls = ['-', '--', '-.', ':']
    lw = 2.0
    fs = 20
    plt.rcParams.update({'font.size': 52})
    plt.rcParams['xtick.labelsize']=20
    plt.rcParams['ytick.labelsize']=20

    fig = plt.figure()
    fig.suptitle(plot_title, fontsize=22)
    ax = fig.add_subplot(1, 1, 1)

    for i, plot in enumerate(sorted_plots_list):
        lab = plot['label']
        ax.plot(plot[x_axis], plot['Test_Return'], linestyle=ls[i%4],
                linewidth=lw, color=lc[i%30], label=lab)

    if x_axis_label:
        ax.set_xlabel(x_axis, fontsize=fs)
    if y_axis_label:
        ax.set_ylabel('Normalized Test Return', fontsize=fs)
    if x_upper is not None:
        _, upper = ax.get_xlim()
        if upper > x_upper:
            ax.set_xlim(0, x_upper)
    ax.set_yticks(np.arange(0, 1.1, step=0.2))
    if x_lim is not None:
        ax.set_xlim(0, x_lim)
    ax.grid(linestyle="--", linewidth=0.5, color='.25', zorder=-10)
    ax.set_facecolor('xkcd:white')
    if legend:
        box = ax.get_position()
        ax.set_position([box.x0, box.y0 + box.height * 0.15, box.width,
                         box.height * 0.9])

        ax.legend(loc='upper center', bbox_to_anchor=(0.5, -0.17), fancybox=True,
                  shadow=True, ncol=4, prop={'size': 15})
    return


def usage():
    print("Usage: learning_curve_plot.py [-a | --x_axis_label] \n"
          "                              [-b | --baseline] \n"
          "                              [-c | --centered] \n"
          "                              "
          "[-d | --dims_list] <list of reward fn: pos-values (include), neg-values (exclude)> \n"
          "                              [-f | --first_trained] \n"
          "                              [-g | --legend] \n"
          "                              [-h | --help] \n"
          "                              [-i | --ica] \n"
          "                              [-l | --location] <input folder location> \n"
          "                              [-m | --multi_train] \n"
          "                              [-n | --iterations] \n"
          "                              [-o | --output_return] \n"
          "                              [-p | --pca] \n"
          "                              "
          "[-r | --reward_fn] <list of reward fn: pos-values (include), neg-values (exclude)> \n"
          "                              [-s | --samples] \n"
          "                              [-S | --step_size] <X-axis step-size> \n"
          "                              [-t | --time] \n"
          "                              [-v | --verbose] \n"
          "                              [-x | --x_range] <Upper X-range of the plot> \n"
          "                              [-y | --y_axis_label] \n"
          )


def main(argv):
    behavior_folders = []
    reward_fn = np.arange(0, 5).tolist()
    sample_threshold = None
    plot_dims = np.arange(0, 29).tolist()
    exclude_dims_list = []
    output_return = False
    baseline = True
    pca = True
    ica = True
    multi_train = False
    x_axis = 'Samples'
    x_upper = None
    x_axis_label = True
    y_axis_label = True
    legend = True
    centered = False
    verbose = False
    first_trained = False
    step_size = 400

    try:
        opts, args = getopt.getopt(argv, "h bpintsoaygcvmfl:r:d:u:x:S:",
                                   ["baseline", "pca", "ica", "iterations", "time",
                                    "samples", "output_return", "x_axis_label",
                                    "y_axis_label", "legend", "centered", "verbose",
                                    "multi_train", "first_trained", "location=",
                                    "reward_fn=", "plot_dims=", "sample_threshold=",
                                    "x_range=", "step_size="])
    except getopt.GetoptError:
        usage()
        sys.exit(2)

    for opt, arg in opts:
        if opt == '-h':
            usage()
            sys.exit()
        elif opt in ("-l", "--location"):
            behavior_folders.append(arg)
        elif opt in ("-r", "--reward_fn"):
            reward_fn_list = list(map(int, arg.strip('[]').split(',')))
            pos = min(reward_fn_list) >= 0
            neg = max(reward_fn_list) < 0

            if not pos and not neg:
                print("ERROR: 'r'/'--reward_fn': ", dims_list, " contains both ",
                       "positive and negative values. \nCan either be a list ",
                       "of positive reward fn. values (indicating inclusion) ",
                       "or negative reward fn. values (indicating exclusion).")
                sys.exit()
            elif neg:
                for r in reward_fn_list:
                    reward_fn.remove(-r)
            elif pos:
                reward_fn = reward_fn_list
        elif opt in ("-d", "--dims_list"):
            dims_list = list(map(int, arg.strip('[]').split(',')))
            pos = min(dims_list) >= 0
            neg = max(dims_list) < 0

            if not pos and not neg:
                print("ERROR: 'd'/'--dims_list': ", dims_list, " contains both ",
                       "positive and negative values. \nCan either be a list ",
                       "of positive dimensions values (indicating inclusion) ",
                       "or negative dimension values (indicating exclusion).")
                sys.exit()
            elif neg:
                for d in dims_list:
                    plot_dims.remove(-d)
            elif pos:
                plot_dims = dims_list
        elif opt in ("-u", "--x_range"):
            x_upper = float(arg)
        elif opt in ("-x", "--sample_threshold"):
            sample_threshold = float(arg) * 1000000.0
        elif opt in ("-b", "--baseline"):
            baseline = False
        elif opt in ("-p", "--pca"):
            pca = False
        elif opt in ("-i", "--ica"):
            ica = False
        elif opt in ("-o", "--output_return"):
            output_return = True
        elif opt in ("-a", "--x_axis_label"):
            x_axis_label = False
        elif opt in ("-y", "--y_axis_label"):
            y_axis_label = False
        elif opt in ("-g", "--legend"):
            legend = False
        elif opt in ("-n", "--iterations"):
            x_axis = 'Iterations'
        elif opt in ("-t", "--time"):
            x_axis = 'Wall_Time'
        elif opt in ("-s", "--samples"):
            x_axis = 'Samples'
        elif opt in ("-c", "--centered"):
            centered = True
        elif opt in ("-m", "--multi_train"):
            multi_train = True
        elif opt in ("-f", "--first_trained"):
            first_trained = True
        elif opt in ("-v", "--verbose"):
            verbose = True
        elif opt in ("-S", "--step_size"):
            step_size = int(arg)

        if first_trained and not multi_train:
            first_trained = False
            print("Warning: 'first_trained' set to True while 'multi_train' is not!")

    training_folders = []
    # Extract training folders from multiple behavior folders
    for behavior_folder in behavior_folders:
        behavior, training_folders = \
            extract_training_folders(behavior_folder, training_folders)
    plot_title = behavior

    # Extract training info from each training folder
    plot_dicts_list = \
        extract_plot_info(training_folders, step_size=step_size, verbose=verbose,
                          multi_train=multi_train, first_trained=first_trained)

    # Select plot dicts and sort them
    sorted_plots_list = \
        select_and_sort_plots_dicts(plot_dicts_list, plot_dims, reward_fn,
                                    baseline, pca, ica, centered)

    # Print training output
    if output_return:
        output_returns(sorted_plots_list, sample_threshold)

    plot_2D(sorted_plots_list, x_axis, plot_title, x_upper, x_axis_label,
            sample_threshold, y_axis_label, legend)

    plt.show()


if __name__ == "__main__":
    main(sys.argv[1:])
