import sys, getopt
import numpy as np
import json
import os
import re
from collections import OrderedDict
import matplotlib.pyplot as plt
#import matplotlib.animation as animation
from matplotlib import style

style.use('seaborn')

def extract_training_folders(behavior_folder):
    sub_dirs = [d[0] for d in os.walk(behavior_folder)]

    behavior = behavior_folder[behavior_folder.find('humanoid3d'):]
    behavior = behavior.replace("humanoid3d/", "")
    behavior = behavior.split('/', 1)[0]
    behavior = behavior.split('_', 1)[0]

    training_folders = []
    for d in sub_dirs:
        if 'd_2' in d:
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


def extract_evaluation_info(training_folders, baseline, pca, ica, max_episodes):
    num_steps_per_min = 30
    eval_dicts_list_all = []

    # Process each training folder
    for d in training_folders:
        evaluation_data = OrderedDict()

        # Extract reward-type of the training case
        if 'R1' in d:
            evaluation_data['reward_type'] = 'R1'
            evaluation_data['reward_no'] = 1
        elif 'R2' in d:
            evaluation_data['reward_type'] = 'R2'
            evaluation_data['reward_no'] = 2
        else:
            evaluation_data['reward_type'] = 'R0'
            evaluation_data['reward_no'] = 0

        # # Invert reward number R1 <--> R2
        # if 'R1' in d:
        #     evaluation_data['reward_type'] = 'R2'
        #     evaluation_data['reward_no'] = 2
        # elif 'R2' in d:
        #     evaluation_data['reward_type'] = 'R1'
        #     evaluation_data['reward_no'] = 1
        # else:
        #     evaluation_data['reward_type'] = 'R0'
        #     evaluation_data['reward_no'] = 0

        # Extract dims and type of the training case
        if 'pca_activation_euler' in d:
            dims = re.search('euler_(.*)d', d)
            dims = int(dims.group(1))
            evaluation_data['dims'] = dims
            evaluation_data['type'] = 'PCA'
            evaluation_data['label'] = 'PCA_' + str(dims) + 'd_'
        elif 'baseline_axis_pei_fix' in d:
            evaluation_data['dims'] = 0
            evaluation_data['type'] = 'Baseline'
            evaluation_data['label'] = 'Baseline_'
        elif 'ica_activation_euler' in d:
            dims = re.search('euler_(.*)d', d)
            dims = int(dims.group(1))
            evaluation_data['dims'] = dims
            evaluation_data['type'] = 'ICA'
            evaluation_data['label'] = 'ICA_' + str(dims) + 'd_'

        evaluation_data['label'] += evaluation_data['reward_type']

        # Try to open the evaluation file in the training folder, if present
        try:
            with open(d+'/evaluation_R0_20_20') as f:
                data = json.load(f)
        except:
            continue

        # Extract evaluation duration and num-episodes, and calculate max-eval-steps
        eval_duration =  float(data['eval_duration'])
        num_episodes = int(data['num_evals'])
        max_eval_steps = int(num_steps_per_min * eval_duration)

        # Iterate through each episode in the evaluation file
        episodes_return_list = []
        for i in range(1, num_episodes+1):
            # Compute and store normalised return per episode, and incompleted episode flag
            episode_dict = OrderedDict()
            episode_rewards = np.array(data[str(i)])
            episode_dict['norm_return'] = np.sum(episode_rewards) / max_eval_steps
            episode_dict['incomp_eval'] = \
                1 if len(episode_rewards) < (max_eval_steps - 3) else 0
            episodes_return_list.append(episode_dict)

        # Consider only the top-n episodes, in terms of norm. return
        if max_episodes is not None:
            sorted_episodes_return_list = \
                sorted(episodes_return_list, key = lambda i: (i['norm_return']))
            episodes_return_list = sorted_episodes_return_list[-max_episodes:]

        # Store avg. and std of norm. return, and no. of incomplete episodes per
        # training case
        norm_episode_return = [e['norm_return'] for e in episodes_return_list]
        num_incomplete_evals = sum([e['incomp_eval'] for e in episodes_return_list])

        evaluation_data['avg_return'] = np.mean(norm_episode_return)
        evaluation_data['std_return'] = np.std(norm_episode_return)
        evaluation_data['incomplete_evals'] = num_incomplete_evals

        eval_dicts_list_all.append(evaluation_data)

    return eval_dicts_list_all


def select_and_sort_plots_dicts(eval_dicts_list, baseline, pca, ica, reward_fn,
                                plot_dims):
    plots_list = []
    # Iterate through list of evaluation dictionaries
    for i, plot in enumerate(eval_dicts_list):
        # Select based on 'dims' and 'reward_fn'
        if plot['dims'] in plot_dims and plot['reward_no'] in reward_fn:
            if baseline and plot['type'] == 'Baseline':
                plots_list.append(plot)
            elif pca and plot['type'] == 'PCA':
                plots_list.append(plot)
            elif ica and plot['type'] == 'ICA':
                plots_list.append(plot)

    # Sort evaluations based on 'dims', 'reward_no' and 'type'
    sorted_plots_list = \
        sorted(plots_list, key = lambda i: (i['dims'], i['type'], i['reward_no']))

    return sorted_plots_list


def calculate_avg_return_ratio(sorted_evals_list):
    baseline_avg_return = None

    # Look for Baseline-R0 evaluation in evaluations list
    for eval in sorted_evals_list:
        if eval['type'] == 'Baseline' and eval['reward_type'] == 'R0':
            baseline_avg_return = eval['avg_return']
            break

    if baseline_avg_return is None:
        return False
    else:
        # For every evaluation in the evaliations list
        for eval in sorted_evals_list:
            # Calculate ratio of avg.-return based on avg.-return of Baseline-R0
            eval['avg_return_ratio'] = eval['avg_return'] / baseline_avg_return
        return True


def print_evals(sorted_evals_list, ratio):
    if ratio:
        print("Training-Case: avg_return_ratio - (avg_return)")
        for eval in sorted_evals_list:
            print("%s: %1.4f (%1.4f)" % (eval['label'], eval['avg_return_ratio'],
                                         eval['avg_return']))
    else:
        print("Training-Case: avg_return")
        for eval in sorted_evals_list:
            print("%s: %1.4f" % (eval['label'], eval['avg_return']))


def plot_evals(sorted_evals_list, behavior, ratio, error_bars, show_failed_evals):
    cl = ['r', 'b', 'g']
    plt.rcParams['ytick.labelsize']=15

    # Get all 'dims' and 'reward-types' from evals-dictionary
    dims_list = []
    training_type_list = []
    reward_type_list = []
    for eval in sorted_evals_list:
        dims_list.append(eval['dims'])
        training_type_list.append(eval['type'])
        reward_type_list.append(eval['reward_no'])

    # Find all unique 'dims' 'training-types' and 'reward-types' and sort them
    unique_dims_list = list(set(dims_list))
    unique_dims_list.sort()

    unique_reward_type_list = list(set(reward_type_list))
    unique_reward_type_list.sort()

    training_type_list = list(set(training_type_list))
    try:
        training_type_list.remove('Baseline')
    except:
        pass
    training_type_list.sort()

    # Check in PCA and/or ICA exists in the final list
    pca = False
    ica = False
    if 'PCA' in training_type_list:
        pca = True
    if 'ICA' in training_type_list:
        ica = True

    # Create a nested-dictionary of dicts (4-levels):
    # Outer-Dict = Dictionary of Dimensions - {'0', '3', '4', '5', '7', ...}
    # Middle-Dict-1 = Dictionary of Training-types - {'Baseline', 'PCA', 'ICA'}
    # Middle-Dict-2 = Dictionary of Reward-types - {'R0', 'R1', 'R2'}
    # Inner-Dict = Dictionary of Reward stats - {'avg', 'std', 'incomp-evals'}
    dims_dict = OrderedDict()
    for dim in unique_dims_list:
        dims_dict[dim] = OrderedDict()

    # Iterating through each Evaluation, collect and store avg. and std. returns
    # in the appropriate inner-inner-dict
    for eval in sorted_evals_list:
        return_stats_dict = OrderedDict()

        if ratio:
            avg_return = eval['avg_return_ratio']
        else:
            avg_return = eval['avg_return']

        return_stats_dict['avg'] = avg_return
        return_stats_dict['std'] = eval['std_return']
        return_stats_dict['incomp_evals'] = eval['incomplete_evals']

        # Check if a 'training-type' dictionary with key ('Baseline'/'PCA'/'ICA')
        # exists at the 'Middle-Dict-1' layer
        try:
            dims_dict[eval['dims']][str(eval['type'])]
        except:
            # If not, create the dictionary
            dims_dict[eval['dims']][str(eval['type'])] = OrderedDict()

        # Insert the reward-stats dictionary at the right 'reward-type' key
        dims_dict[eval['dims']][str(eval['type'])][str(eval['reward_no'])] = \
            return_stats_dict

    # Iterating through each dictionary, gather and store 'return stats' in the
    # appropriate reward-type-list: [R0[], R1[], R2[]]
    r_types = []
    errors = []
    incomplete_evals = []
    for r in unique_reward_type_list:
        avg_list = []
        std_list = []
        incomp_evals_list = []
        for _, outer_v in dims_dict.items():
            for _, inner_v in outer_v.items():
                # Avg. Return
                try:
                    avg_list.append(inner_v[str(r)]['avg'])
                except:
                    avg_list.append(0)
                # STD of the Returns
                try:
                    std_list.append(inner_v[str(r)]['std'])
                except:
                    std_list.append(0)
                # No. of incomplete evaluations
                try:
                    incomp_evals_list.append(inner_v[str(r)]['incomp_evals'])
                except:
                    incomp_evals_list.append(0)

        r_types.append(avg_list)
        errors.append(std_list)
        incomplete_evals.append(incomp_evals_list)

    # Number of catagories (collection of bars)
    try:
        n_groups = len(r_types[0])
    except:
        n_groups = 0
        print("Warning: Empty Plot!")

    # Create plot
    fig, ax = plt.subplots()
    index = np.arange(n_groups) * 1.5
    bar_width = 0.40
    opacity = 0.5

    catagory_list = []
    for i, r in enumerate(r_types):
        # Create catagory labels
        label = 'R' + str(unique_reward_type_list[i])
        if label == 'R0':
            label = r'$r^I$'
        elif label == 'R1':
            label = r'$r^C$'
        elif label == 'R2':
            label = r'$r^E$'

        # Plot catagory bars ('R0', 'R1', 'R2'), with or without errors
        if error_bars:
            catagory_plot = plt.bar(index + (i * bar_width), r, bar_width,
                                    yerr=errors[i], alpha=opacity, color=cl[i],
                                    label=label, error_kw=dict(lw=1, capsize=3,
                                                               capthick=1.5))
        else:
            catagory_plot = plt.bar(index + (i * bar_width), r, bar_width,
                                    alpha=opacity, color=cl[i], label=label)

        catagory_list.append(catagory_plot)

    x_ticks = []
    # Set x-ticks
    # Iterate through Dims-Dict
    for outer_k, outer_v in dims_dict.items():
        if outer_k == 0:
            x_ticks.append('Baseline')
        else:
            # Iterate through Training-Type-Dict
            for inner_k, inner_v in outer_v.items():
                x_tick = str(outer_k) + 'D'
                if pca and ica:
                     x_tick += '-' + str(inner_k)
                x_ticks.append(x_tick)

    # Show failed evaluation count on the respective bar, if any
    if show_failed_evals:
        for i, catagory in enumerate(catagory_list):
            for j, patch in enumerate(catagory.patches):
                incomp_evals = incomplete_evals[i][j]
                if incomp_evals > 0:
                    plt.text((patch.get_x()+0.075), (patch.get_height() / 2.0), \
                             str(incomp_evals), fontsize=18, color='white')

    plt.xlabel('Dimensions', fontsize=20)
    plt.ylabel('Normalised Avg. Return', fontsize=20)
    plt.title('Evaluation - ' + behavior.upper(), fontsize=25)
    plt.xticks(index + bar_width, x_ticks, fontsize=15)
    plt.legend(prop={'size': 15})
    if plt.gca().get_ylim()[1] < 1.0:
        plt.ylim((0, 1.0))

    plt.tight_layout()
    ax.grid(linestyle="--", linewidth=0.5, color='.25', zorder=-10)
    ax.set_facecolor('xkcd:white')
    plt.show()

def usage():
    print("Usage: pca.py [-a | --all] \n"
          "              [-b | --baseline] \n"
          "              [-c | --ratio] \n"
          "              [-d | --dims_list] <list of reward fn: pos-values (include), neg-values (exclude)> \n"
          "              [-e | --error_bars] \n"
          "              [-f | --failer_evals] \n"
          "              [-h | --help] \n"
          "              [-i | --ica] \n"
          "              [-l | --location] <input folder location> \n"
          "              [-m | --max_episodes] <top-m episodes to consider> \n"
          "              [-o | --output_evals] \n"
          "              [-p | --pca] \n"
          "              [-r | --reward_fn] <list of reward fn: pos-values (include), neg-values (exclude)> \n"
          "              [-s | --show_evals] \n"
          "              [-t | --behavior] <target behavior> \n"
          )

def main(argv):
    behavior_folder = 'data/trained_policies/humanoid3d/'
    behavior = None
    reward_fn = np.arange(0, 3).tolist()
    plot_dims = np.arange(0, 29).tolist()
    max_episodes = None
    output_evals = False
    all = True
    baseline = False
    pca = False
    ica = False
    ratio = False
    show_evals = False
    error_bars = False
    show_failed_evals = False

    try:
        opts, args = getopt.getopt(argv, "h abpiocsefl:t:r:d:m:",
                                   ["all", "baseline", "pca", "ica", "output_evals",
                                    "ratio", "show_evals", "error_bars" "failed_evals",
                                    "location=", "behavior=", "reward_fn=", "dims_list",
                                    "max_episodes="])
    except getopt.GetoptError:
        usage()
        sys.exit(2)

    for opt, arg in opts:
        if opt == '-h':
            usage()
            sys.exit()
        elif opt in ("-a", "--all"):
            baseline = True
            pca = True
            ica = True
        elif opt in ("-b", "--baseline"):
            baseline = True
            all = False
        elif opt in ("-p", "--pca"):
            pca = True
            all = False
        elif opt in ("-i", "--ica"):
            ica = True
            all = False
        elif opt in ("-o", "--output_return"):
            output_evals = True
        elif opt in ("-c", "--ratio"):
            ratio = True
        elif opt in ("-s", "--show_evals"):
            show_evals = True
        elif opt in ("-e", "--error_bars"):
            error_bars = True
        elif opt in ("-f", "--failed_evals"):
            show_failed_evals = True
        elif opt in ("-l", "--location"):
            behavior_folder = arg
        elif opt in ("-t", "--behavior"):
            behavior = arg
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
        elif opt in ("-m", "--max_episodes"):
            max_episodes = int(arg)

    if all:
        baseline = pca = ica = True

    if behavior is not None:
        behavior_folder += behavior + '/'

    if not os.path.isdir(behavior_folder):
        print("Directory '%s' does not exist: " % behavior_folder)
        sys.exit()

    # Extract individual training folders
    behavior, training_folders = extract_training_folders(behavior_folder)
    plot_title = "Cross Evaluation - " + behavior.upper()

    # [print(d) for d in training_folders]

    # Extract evaluation info from individual training folders
    eval_dicts_list = \
        extract_evaluation_info(training_folders, baseline, pca, ica, max_episodes)

    if ratio:
        # Calculate ratio of the normalised average return, w.r.t. baseline
        succ = calculate_avg_return_ratio(eval_dicts_list_all)
        if not succ:
            print("ERROR: Baseline version not found for calculating return ratio:")
            print("Either omit the ratio flag '-c' from the command line, or \
                   include the baseline version in the behavior folder.")
            sys.exit()

    # select and sort evaluations based on 'dim', 'type' and 'reward_fn'
    final_evals_list = select_and_sort_plots_dicts(eval_dicts_list, baseline, pca,
                                                   ica, reward_fn, plot_dims)

    if output_evals:
        print_evals(final_evals_list, ratio)

    plot_evals(final_evals_list, behavior, ratio, error_bars,
               show_failed_evals)


if __name__ == "__main__":
    main(sys.argv[1:])
