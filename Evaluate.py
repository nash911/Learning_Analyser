import sys, getopt
import numpy as np
import json
import os
import re
from collections import OrderedDict
import subprocess

from analysis_arg_parser import AnlyzArgParser

run_args_dict = OrderedDict()
run_args_dict['backflip'] = '/home/nash/DeepMimic/args/run_humanoid3d_backflip_args.txt'
run_args_dict['crawling'] = '/home/nash/DeepMimic/args/run_humanoid3d_crawl_args.txt'
run_args_dict['dance_a'] = '/home/nash/DeepMimic/args/run_humanoid3d_dance_a_args.txt'
run_args_dict['running'] = '/home/nash/DeepMimic/args/run_humanoid3d_run_args.txt'
run_args_dict['walking'] = '/home/nash/DeepMimic/args/run_humanoid3d_walk_args.txt'

def extract_training_folders(behavior_folder, training_folders):
    sub_dirs = [d[0] for d in os.walk(behavior_folder)]

    # Extract behavior name from the path string
    behavior = behavior_folder[behavior_folder.find('humanoid3d'):]
    behavior = behavior.replace("humanoid3d/", "")
    behavior = behavior.split('/', 1)[0]

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


def extract_training_info(training_folders, behavior, baseline, pca, ica, eval,
                          num_evals, eval_duration, eval_reward_fn, force_eval):
    training_dict_list = []
    # Process each training folder
    for d in training_folders:
        training_dict = OrderedDict()

        # Extract training folder location, and get the original run-args-file
        training_dict['location'] = d.rstrip('/')
        training_dict['run_file'] = run_args_dict[behavior]

        # If 'eval' arg is set, then check if a previous evaluation file exists
        if eval:
            eval_file = training_dict['location'] + '/evaluation' + '_R' \
                        + str(eval_reward_fn) + '_' + str(num_evals) + '_' \
                        + str(eval_duration)

            # If so, and NOT 'force_eval', then exclude the current folder from evaluation
            if os.path.isfile(eval_file) and not force_eval:
                continue

        # Extract reward-type of the training case
        if 'R1' in d:
            training_dict['reward_type'] = 'R1'
            training_dict['reward_no'] = 1
        elif 'R2' in d:
            training_dict['reward_type'] = 'R2'
            training_dict['reward_no'] = 2
        else:
            training_dict['reward_type'] = 'R0'
            training_dict['reward_no'] = 0

        # Extract dims,type and reduced_motion_file of the training case
        files = [f[2] for f in os.walk(d)]
        if 'pca_activation_euler' in d:
            dims = re.search('centered_(.*)d_', d)
            if dims is None:
                dims = re.search('euler_(.*)d_', d)
            if dims is None:
                dims = re.search('euler_(.*)d', d)
            dims = int(dims.group(1))
            training_dict['dims'] = dims
            training_dict['label'] = 'pca'
            for f in files[0]:
                if 'pca_euler_humanoid3d' in f:
                    training_dict['red_file'] = f
                    break
                elif 'ica_euler_humanoid3d' in f:
                    training_dict['red_file'] = f
                    break
                # # TODO: To be removed
                # if 'pca_quat_humanoid3d' in f:
                #     training_dict['red_file'] = f
                #     break
        elif 'baseline_axis_pei_fix' in d:
            training_dict['dims'] = 0
            training_dict['label'] = 'baseline'
            training_dict['red_file'] = None
        elif 'ica_activation_euler' in d:
            dims = re.search('euler_(.*)d_', d)
            if dims is None:
                dims = re.search('euler_(.*)d', d)
            dims = int(dims.group(1))
            training_dict['dims'] = dims
            training_dict['label'] = 'ica'
            for f in files[0]:
                if 'ica_euler_humanoid3d' in f:
                    training_dict['red_file'] = f
                    break

        if baseline and training_dict['label'] == 'baseline':
            training_dict_list.append(training_dict)
        elif pca and training_dict['label'] == 'pca':
            training_dict_list.append(training_dict)
        elif ica and training_dict['label'] == 'ica':
            training_dict_list.append(training_dict)

    return training_dict_list


def create_run_file(sorted_training_list, eval, eval_reward_fn, num_evals,
                    eval_duration, reduced_motion_file, imitate_exctn,
                    log_excitations, log_actions, log_pose):
    # Iterate through a list of trained-model folders dictionary
    for train_dict in sorted_training_list:
        # Create an args-parser object and load the args-file
        arg_parser = AnlyzArgParser()
        succ = arg_parser.load_file(train_dict['run_file'])

        # Add trained model-file
        arg_parser._table['--model_files'] = \
            '--model_files ' + train_dict['location'] + '/agent0_model.ckpt'

        # Add reduced-motion-file
        if train_dict['label'] == 'baseline':
            try:
                del arg_parser._table['--reduced_motion_file']
            except:
                pass
        else:
            if reduced_motion_file is None:
                arg_parser._table['--reduced_motion_file'] = \
                    '--reduced_motion_file ' + train_dict['location'] + '/' + \
                    train_dict['red_file']
            else:
                arg_parser._table['--reduced_motion_file'] = \
                    '--reduced_motion_file ' + reduced_motion_file

        # Add args related to evaluation
        if eval:
            eval_file = train_dict['location'] + '/evaluation' + '_R' \
                        + str(eval_reward_fn) + '_' + str(num_evals) + '_' \
                        + str(eval_duration)
            arg_parser._table['--evaluate'] = '--evaluate ' + 'True'
            arg_parser._table['--num_evals'] = '--num_evals ' + str(num_evals)
            arg_parser._table['--evaluation_time'] = '--evaluation_time ' + str(eval_duration)
            arg_parser._table['--evaluation_out_file'] = '--evaluation_out_file ' + eval_file
        else:
            try:
                del arg_parser._table['--evaluate']
            except:
                pass

        # Add 'imitate_excitation' argument
        if imitate_exctn:
            arg_parser._table['--imitate_excitation'] = \
                '--imitate_excitation ' + 'True'
        else:
            try:
                del arg_parser._table['--imitate_excitation']
            except:
                pass

        # Add 'log_excitations' argument
        if log_excitations:
            arg_parser._table['--log_excitations'] = '--log_excitations ' + 'True'
        else:
            try:
                del arg_parser._table['--log_excitations']
            except:
                pass

        # Add 'log_actions' argument
        if log_actions:
            arg_parser._table['--log_actions'] = '--log_actions ' + 'True'
        else:
            try:
                del arg_parser._table['--log_actions']
            except:
                pass

        # Add 'log_pose' argument
        if log_pose:
            arg_parser._table['--log_pose'] = '--log_pose ' + 'True'
        else:
            try:
                del arg_parser._table['--log_pose']
            except:
                pass

        # Add run-args-file to training dictionary
        train_dict['run_file_parser'] = arg_parser

    return


def usage():
    print("Usage: Evaluate.py [-a | --log_actions] <input to PD controller> \n"
          "                   [-b | --baseline] \n"
          "                   [-d | --duration] <evaluation duration> \n"
          "                   [-e | --eval]  \n"
          "                   [-f | --force_eval] \n"
          "                   [-h | --help] \n"
          "                   [-i | --imitate_excitations] \n"
          "                   [-l | --location] <input folder location> \n"
          "                   [-m | --reduced_motion_file] <reduced motion file> \n"
          "                   [-n | --num_evals] <no. of evaluations> \n"
          "                   [-o | --log_pose] \n"
          "                   [-p | --pca] \n"
          "                   [-r | --reward_fn] <evaluation reward function> \n"
          "                   [-x | --log_excitations] <output of policy network> \n"
          )


def main(argv):
    run_file = 'run_args_file.txt'
    reduced_motion_file = None
    behavior_folder = None
    dims = np.arange(0, 29).tolist()
    all = True
    log_excitations = False
    log_actions = False
    log_pose = False
    baseline = False
    pca = False
    ica = False
    eval = False
    num_evals = 20
    eval_duration = 20
    eval_reward_fn = 0
    force_eval = False
    imitate_exctn = False

    try:
        opts, args = getopt.getopt(argv, "h abpiefxol:m:n:d:r:",
                                   ["log_excitations", "baseline", "pca", "eval",
                                    "imitate_excitations", "force_eval", "log_actions",
                                    "log_pose", "location=", "reduced_motion_file=",
                                    "num_evals=", "duration=", "reward_fn="])
    except getopt.GetoptError:
        usage()
        sys.exit(2)

    for opt, arg in opts:
        if opt == '-h':
            usage()
            sys.exit()
        elif opt in ("-x", "--log_excitations"):
            log_excitations = True
        elif opt in ("-a", "--log_actions"):
            log_actions = True
        elif opt in ("-o", "--log_pose"):
            log_pose = True
        elif opt in ("-b", "--baseline"):
            baseline = True
            all = False
        elif opt in ("-p", "--pca"):
            pca = True
            all = False
        elif opt in ("-i", "--imitate_excitations"):
            imitate_exctn = True
        elif opt in ("-e", "--eval"):
            eval = True
        elif opt in ("-f", "--force_eval"):
            force_eval = True
        elif opt in ("-l", "--location"):
            behavior_folder = arg
        elif opt in ("-t", "--behavior"):
            behavior = arg
        elif opt in ("-r", "--reward_fn"):
            reward_fn = int(arg)
        elif opt in ("-n", "--num_evals"):
            num_evals = int(arg)
        elif opt in ("-d", "--duration"):
            eval_duration = float(arg)
        elif opt in ("-m", "--reduced_motion_file"):
            reduced_motion_file = arg

    if all:
        baseline = pca = ica = True

    if behavior_folder is None or not os.path.isdir(behavior_folder):
        print("Directory '%s' does not exist: " % behavior_folder)
        sys.exit()

    if log_excitations and log_actions and not force_eval:
        print("Warning: Both Actions and Activations are set to be logged!")
        sys.exit()

    # Extract individual training folders
    training_folders = []
    behavior, training_folders = \
        extract_training_folders(behavior_folder, training_folders)

    # Extract info from individual training folders
    training_dict_list = \
        extract_training_info(training_folders, behavior, baseline, pca, ica,
                              eval, num_evals, eval_duration, eval_reward_fn,
                              force_eval)

    # Sort training dictionaries
    sorted_training_list = \
        sorted(training_dict_list, key = lambda i: (i['dims'], i['reward_no'],
                                                    i['label']))

    #[print(d['location']) for d in sorted_training_list]

    # Create an appropriate run-args-file for each trained case
    create_run_file(sorted_training_list, eval, eval_reward_fn, num_evals,
                    eval_duration, reduced_motion_file, imitate_exctn,
                    log_excitations, log_actions, log_pose)

    # Evaluate each trailed case in the list
    for train_dict in sorted_training_list:
        # Write run-args-file on to a file
        with open(run_file, 'w') as fp:
            for k, v in train_dict['run_file_parser']._table.items():
                fp.write(v + '\n')

        # Execute command to run the trained case
        cmd = 'python DeepMimic.py --arg_file ' + run_file
        subprocess.call(cmd, shell=True)


if __name__ == "__main__":
    main(sys.argv[1:])
