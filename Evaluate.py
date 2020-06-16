import sys
import getopt
import os
import re
from collections import OrderedDict
import subprocess

from analysis_arg_parser import AnlyzArgParser

run_args_dict = OrderedDict()
run_args_dict['backflip'] = '/home/nash/DeepMimic/args/run_humanoid3d_backflip_args.txt'
run_args_dict['cartwheel'] = '/home/nash/DeepMimic/args/run_humanoid3d_cartwheel_args.txt'
run_args_dict['crawling'] = '/home/nash/DeepMimic/args/run_humanoid3d_crawl_args.txt'
run_args_dict['dance_a'] = '/home/nash/DeepMimic/args/run_humanoid3d_dance_a_args.txt'
run_args_dict['punch'] = '/home/nash/DeepMimic/args/run_humanoid3d_punch_args.txt'
run_args_dict['running'] = '/home/nash/DeepMimic/args/run_humanoid3d_run_args.txt'
run_args_dict['walking'] = '/home/nash/DeepMimic/args/run_humanoid3d_walk_args.txt'

run_args_dict['walk'] = '/home/nash/DeepMimic/args/run_salamander_walk_args.txt'
run_args_dict['run'] = '/home/nash/DeepMimic/args/run_cheetah_args.txt'
run_args_dict['slither'] = '/home/nash/DeepMimic/args/run_snake_slither_args.txt'
run_args_dict['caterpillar'] = '/home/nash/DeepMimic/args/run_snake_caterpillar_args.txt'


def extract_training_folders(behavior_folder, training_folders):
    sub_dirs = [d[0] for d in os.walk(behavior_folder)]

    # Extract behavior name from the path string
    if 'MIG2019' in behavior_folder:
        behavior = behavior_folder[behavior_folder.find('MIG2019'):]
        behavior = behavior.replace("MIG2019/", "")
    elif 'humanoid' in behavior_folder:
        behavior = behavior_folder[behavior_folder.find('humanoid3d'):]
        behavior = behavior.replace("humanoid3d/", "")
    elif 'salamander' in behavior_folder:
        behavior = behavior_folder[behavior_folder.find('salamander'):]
        behavior = behavior.replace("salamander/", "")
    elif 'cheetah' in behavior_folder:
        behavior = behavior_folder[behavior_folder.find('cheetah'):]
        behavior = behavior.replace("cheetah/", "")
    elif 'snake' in behavior_folder:
        behavior = behavior_folder[behavior_folder.find('snake'):]
        behavior = behavior.replace("snake/", "")
    else:
        print("Error: Either missing or unknown character in folder path: ", behavior_folder)
        sys.exit()
    behavior = behavior.split('/', 1)[0]

    for d in sub_dirs:
        if 'd_2' in d and not ('centered_' or '_learn' in d):
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

        if 'baseline' in d:
            training_folders.append(d)
        elif 'pca_activation_euler' in d:
            training_folders.append(d)
        elif 'ica_activation_euler' in d:
            training_folders.append(d)
        elif 'pca' in d:
            training_folders.append(d)

    return behavior, training_folders


def extract_training_info(training_folders, behavior, baseline, pca, ica, eval, num_evals,
                          eval_duration, eval_reward_fn, force_eval):
    training_dict_list = []
    # Process each training folder
    for d in training_folders:
        # Ignore sub-folders such as 'int_output' folder within a training folder
        if 'int_output' in d:
            continue

        training_dict = OrderedDict()

        # Extract training folder location, and get the original run-args-file
        training_dict['location'] = d.rstrip('/')
        training_dict['run_file'] = run_args_dict[behavior]

        # If 'eval' arg is set, then check if a previous evaluation file exists
        if eval:
            eval_file = training_dict['location'] + '/evaluation' + '_R' + str(eval_reward_fn) \
                        + '_' + str(num_evals) + '_' + str(eval_duration)

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
        if 'pca_activation_euler' in d or 'pca' in d:
            dims = re.search('centered_(.*)d_', d)
            # dims = re.search('centered_Run_(.*)d_', d)
            if dims is None:
                dims = re.search('euler_(.*)d_', d)
            if dims is None:
                dims = re.search('euler_(.*)d', d)
            try:
                dims = int(dims.group(1))
            except AttributeError:
                dims = 0
            training_dict['dims'] = dims
            training_dict['label'] = 'pca'
            for f in files[0]:
                if 'pca_euler' in f:
                    training_dict['red_file'] = f
                    break
                elif 'ica_euler' in f:
                    training_dict['red_file'] = f
                    break
                # # TODO: To be removed
                # if 'pca_quat_humanoid3d' in f:
                #     training_dict['red_file'] = f
                #     break
        elif 'baseline' in d:
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
                if 'ica_euler' in f:
                    training_dict['red_file'] = f
                    break

        if baseline and training_dict['label'] == 'baseline':
            training_dict_list.append(training_dict)
        elif pca and training_dict['label'] == 'pca':
            training_dict_list.append(training_dict)
        elif ica and training_dict['label'] == 'ica':
            training_dict_list.append(training_dict)

    return training_dict_list


def create_run_file(sorted_training_list, eval, eval_reward_fn, num_evals, eval_duration,
                    reduced_motion_file, imitate_exctn, intermediate_model, log_excitations,
                    log_actions, log_pose, select_k=None, com_threshold=-1.0, check_collision=False,
                    allow_parent_collision=False):
    # Iterate through a list of trained-model folders dictionary
    for train_dict in sorted_training_list:
        # Create an args-parser object and load the args-file
        arg_parser = AnlyzArgParser()
        arg_parser.load_file(train_dict['run_file'])

        if intermediate_model is not None:
            # Add intermediate model file
            int_itr = '0000000000' + str(intermediate_model)
            int_model_file = "agent0_int_model_%s.ckpt" % int_itr[-10:]
            arg_parser._table['--model_files'] = '--model_files ' + train_dict['location'] \
                + '/int_output/agent0_models/' + int_model_file
        else:
            # Add trained model file
            arg_parser._table['--model_files'] = \
                '--model_files ' + train_dict['location'] + '/agent0_model.ckpt'

        # Add reduced-motion-file
        if train_dict['label'] == 'baseline':
            try:
                del arg_parser._table['--reduced_motion_file']
            except KeyError:
                pass
        else:
            if reduced_motion_file is None:
                arg_parser._table['--reduced_motion_file'] = \
                    '--reduced_motion_file ' + train_dict['location'] + '/' + train_dict['red_file']
            else:
                arg_parser._table['--reduced_motion_file'] = \
                    '--reduced_motion_file ' + reduced_motion_file

        # Add select_k arg
        if select_k is not None:
            arg_parser._table['--select_k'] = '--select_k ' + str(select_k)

        # Add com_height_threshold arg
        if com_threshold > 0.0:
            arg_parser._table['--com_height_threshold'] = '--com_height_threshold ' + \
                str(com_threshold)

        # Add check_collision arg
        if check_collision:
            arg_parser._table['--enable_collision_check'] = '--enable_collision_check ' + 'True'

        # Add parent_collision arg
        if allow_parent_collision:
            arg_parser._table['--allow_parent_collision'] = '--allow_parent_collision ' + 'True'

        # Add args related to evaluation
        if eval:
            eval_file = train_dict['location'] + '/evaluation' + '_R' + str(eval_reward_fn) + '_' \
                        + str(num_evals) + '_' + str(eval_duration)
            arg_parser._table['--evaluate'] = '--evaluate ' + 'True'
            arg_parser._table['--num_evals'] = '--num_evals ' + str(num_evals)
            arg_parser._table['--evaluation_time'] = '--evaluation_time ' + str(eval_duration)
            arg_parser._table['--evaluation_out_file'] = '--evaluation_out_file ' + eval_file
        else:
            try:
                del arg_parser._table['--evaluate']
            except KeyError:
                pass

        # Add 'imitate_excitation' argument
        if imitate_exctn:
            arg_parser._table['--imitate_excitation'] = '--imitate_excitation ' + 'True'
        else:
            try:
                del arg_parser._table['--imitate_excitation']
            except KeyError:
                pass

        # Add 'log_excitations' argument
        if log_excitations:
            arg_parser._table['--log_excitations'] = '--log_excitations ' + 'True'
        else:
            try:
                del arg_parser._table['--log_excitations']
            except KeyError:
                pass

        # Add 'log_actions' argument
        if log_actions:
            arg_parser._table['--log_actions'] = '--log_actions ' + 'True'
        else:
            try:
                del arg_parser._table['--log_actions']
            except KeyError:
                pass

        # Add 'log_pose' argument
        if log_pose:
            arg_parser._table['--log_pose'] = '--log_pose ' + 'true'
        else:
            try:
                del arg_parser._table['--log_pose']
            except KeyError:
                pass

        # Add run-args-file to training dictionary
        train_dict['run_file_parser'] = arg_parser

    return


def run_playback(playback_file=None, character=None, single=False, reduced_motion=None,
                 dimension=None, behavior=None):
    playback_dict = OrderedDict()
    playback_dict['scene'] = "kin_char"

    character_file = "data/characters/{}.txt".format(character)
    if not os.path.exists(character_file):
        print("Error: Incorrect character file/path:", character_file)
        sys.exit()
    else:
        playback_dict['character_file'] = character_file

    if playback_file is not None:
        playback_dict['motion_file'] = playback_file
    elif reduced_motion:
        motion_file = "/home/nash/Dropbox/Clemson/Projects/quat_conversions/pca/Output/"
        if dimension is None:
            playback_dict['motion_file'] = motion_file + 'pca_traj.txt'
        elif single:
            playback_dict['motion_file'] = motion_file + 'single_coactivations/' + \
                ('pca_single_traj_%d.txt' % dimension)
        else:
            print("Error: For reduced-motion playback, set single flag along with dimension using:"
                  "[-S | --single]")
            sys.exit()
    else:
        if behavior is None:
            print("Error: For independent-joint-action motion-playback, specify the behavior: "
                  "[-B | --behavior]")
            sys.exit()

        motion_file = "/home/nash/DeepMimic/data/motions/"
        playback_dict['motion_file'] = motion_file + ('humanoid3d_%s.txt' % behavior)

    if not os.path.exists(playback_dict['motion_file']):
        print("Error: Incorrect motion file/path:", playback_dict['motion_file'])
        sys.exit()
    else:
        # Write playback-file on to a file
        playback_file = "/home/nash/Dropbox/Clemson/Projects/Learning_Analyser/playback_file.txt"
        with open(playback_file, 'w') as fp:
            for k, v in playback_dict.items():
                fp.write('--' + k + ' ' + v + '\n')

        # Execute command to run the trained case
        cmd = 'python DeepMimic.py --arg_file ' + playback_file
        subprocess.call(cmd, shell=True)

    return


def usage():
    print("Usage: Evaluate.py [-a | --log_actions] <input to PD controller> \n"
          "                   [-A | --allow_parent_collision]  \n"
          "                   [-b | --baseline] \n"
          "                   [-B | --behavior] <behavior name (run/walk/...)> \n"
          "                   [-c | --check_collision]  \n"
          "                   [-C | --character]  \n"
          "                   [-d | --duration] <evaluation duration> \n"
          "                   [-D | --dimension] <reduced dimension> \n"
          "                   [-e | --eval]  \n"
          "                   [-f | --force_eval] \n"
          "                   [-h | --help] \n"
          "                   [-i | --imitate_excitations] \n"
          "                   [-k | --select_k] \n"
          "                   [-I | --intermediate_model] \n"
          "                   [-l | --location] <input folder location> \n"
          "                   [-L | --low_com] <COM height threshold> \n"
          "                   [-m | --reduced_motion_file] <reduced motion file> \n"
          "                   [-n | --num_evals] <no. of evaluations> \n"
          "                   [-o | --log_pose] \n"
          "                   [-p | --pca] \n"
          "                   [-P | --playback] \n"
          "                   [-r | --reward_fn] <evaluation reward function (0/1/2/...)> \n"
          "                   [-R | --reduced] \n"
          "                   [-S | --single] \n"
          "                   [-x | --log_excitations] <output of policy network> \n"
          )


def main(argv):
    run_file = 'run_args_file.txt'
    reduced_motion_file = None
    behavior_folder = None
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
    intermediate_model = None
    select_k = None
    check_collision = False
    allow_parent_collision = False
    com_threshold = -1.0

    playback = False
    character = None
    reduced_motion = False
    single = False
    dimension = None
    behavior = None

    try:
        opts, args = getopt.getopt(argv, "h aAbpiefxoPRScC:l:m:n:d:r:I:D:B:k:L:",
                                   ["log_actions", "allow_parent_collision", "baseline", "pca",
                                    "imitate_excitations", "eval", "force_eval", "log_excitations",
                                    "log_pose", "playback", "reduced", "single", "check_collision",
                                    "character=", "location=", "reduced_motion_file=", "num_evals=",
                                    "duration=", "reward_fn=", "intermediate_model=", "dimension=",
                                    "behavior=", "select_k=", "low_com="])
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
            eval_reward_fn = int(arg)
        elif opt in ("-n", "--num_evals"):
            num_evals = int(arg)
        elif opt in ("-d", "--duration"):
            eval_duration = float(arg)
        elif opt in ("-m", "--reduced_motion_file"):
            reduced_motion_file = arg
        elif opt in ("-I", "--intermediate_model"):
            intermediate_model = int(arg)
        elif opt in ("-P", "--playback"):
            playback = True
        elif opt in ("-R", "--reduced"):
            reduced_motion = True
        elif opt in ("-S", "--single"):
            single = True
        elif opt in ("-D", "--dimension"):
            dimension = int(arg)
        elif opt in ("-B", "--behavior"):
            behavior = arg
        elif opt in ("-k", "--select_k"):
            select_k = int(arg)
        elif opt in ("-L", "--low_com"):
            com_threshold = float(arg)
        elif opt in ("-c", "--check_collision"):
            check_collision = True
        elif opt in ("-A", "--allow_parent_collision"):
            allow_parent_collision = True
        elif opt in ("-C", "--character"):
            character = arg

    if playback:
        run_playback(playback_file=reduced_motion_file, character=character, single=single,
                     reduced_motion=reduced_motion, dimension=dimension, behavior=behavior)
        return

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
        extract_training_info(training_folders, behavior, baseline, pca, ica, eval, num_evals,
                              eval_duration, eval_reward_fn, force_eval)

    # Sort training dictionaries
    sorted_training_list = \
        sorted(training_dict_list, key=lambda i: (i['dims'], i['reward_no'], i['label']))

    # Create an appropriate run-args-file for each trained case
    create_run_file(sorted_training_list, eval, eval_reward_fn, num_evals, eval_duration,
                    reduced_motion_file, imitate_exctn, intermediate_model, log_excitations,
                    log_actions, log_pose, select_k, com_threshold, check_collision,
                    allow_parent_collision)

    # Evaluate each trailed case in the list
    for train_dict in sorted_training_list:
        # Write run-args-file on to a file
        with open(run_file, 'w') as fp:
            for k, v in train_dict['run_file_parser']._table.items():
                fp.write(v + '\n')

        # Print Iteration if intermediate model
        if intermediate_model is not None:
            print("\nIteration:", intermediate_model)

        # Execute command to run the trained case
        cmd = 'python DeepMimic.py --arg_file ' + run_file
        subprocess.call(cmd, shell=True)


if __name__ == "__main__":
    main(sys.argv[1:])
