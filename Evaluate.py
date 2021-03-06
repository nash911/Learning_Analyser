import sys
import getopt
import os
import re
from collections import OrderedDict
import subprocess
import multiprocessing

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
run_args_dict['walker'] = '/home/nash/DeepMimic/args/run_biped_args.txt'
run_args_dict['run'] = '/home/nash/DeepMimic/args/run_cheetah_args.txt'
run_args_dict['slither'] = '/home/nash/DeepMimic/args/run_snake_slither_args.txt'
run_args_dict['caterpillar'] = '/home/nash/DeepMimic/args/run_snake_caterpillar_args.txt'


def extract_training_folders(behavior_folder, training_folders):
    sub_dirs = [d[0] for d in os.walk(behavior_folder)]

    # Extract behavior name from the path string
    if 'MIG2019' in behavior_folder:
        behavior = behavior_folder[behavior_folder.find('MIG2019'):]
        behavior = behavior.replace("MIG2019/", "")
        character = None
    elif 'humanoid' in behavior_folder:
        behavior = behavior_folder[behavior_folder.find('humanoid3d'):]
        behavior = behavior.replace("humanoid3d/", "")
        character = 'humanoid3d'
    elif 'biped' in behavior_folder:
        behavior = behavior_folder[behavior_folder.find('biped'):]
        behavior = behavior.replace("biped/", "")
        character = 'biped'
    elif 'salamander' in behavior_folder:
        behavior = behavior_folder[behavior_folder.find('salamander'):]
        behavior = behavior.replace("salamander/", "")
        character = 'salamander'
    elif 'cheetah' in behavior_folder:
        behavior = behavior_folder[behavior_folder.find('cheetah'):]
        behavior = behavior.replace("cheetah/", "")
        character = 'cheetah'
    elif 'snake' in behavior_folder:
        behavior = behavior_folder[behavior_folder.find('snake'):]
        behavior = behavior.replace("snake/", "")
        character = 'snake'
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

    return character, behavior, training_folders


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
                    reduced_motion_file, character_file, imitate_exctn, intermediate_model,
                    log_excitations, log_actions, log_pose, record_video=False, select_k=None,
                    com_threshold=-1.0, check_collision=False, allow_parent_collision=False,
                    no_flight_phase=False, root_rot_thrshld_x=None, root_rot_thrshld_y=None,
                    self=False, character=None):
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

        # Add character file
        if character_file is not None:
            arg_parser._table['--character_files'] = '--character_files ' + character_file

        # Add character file
        if self:
            files = [f for f in os.listdir(train_dict['location'])
                     if os.path.isfile(os.path.join(train_dict['location'], f))]

            motion_file = arg_parser._table['--motion_file']

            for f in files:
                if character in f:
                    if not ('babble' in f or 'ctrl' in f or 'job' in f or 'train' in f):
                        char_file = os.path.join(train_dict['location'], f)
                    if 'ctrl' in f:
                        ctrl_file = os.path.join(train_dict['location'], f)
                    if 'run' in f and 'babble' not in f:
                        motion_file = os.path.join(train_dict['location'], f)

            arg_parser._table['--character_files'] = '--character_files ' + char_file
            arg_parser._table['--char_ctrl_files'] = '--char_ctrl_files ' + ctrl_file
            arg_parser._table['--motion_file'] = '--motion_file ' + motion_file

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

        # Add flight_phase arg
        if no_flight_phase:
            arg_parser._table['--no_flight_phase'] = '--no_flight_phase ' + 'True'

        # Add root_rot-x_threshold arg
        if root_rot_thrshld_x is not None:
            arg_parser._table['--root_rotation_threshold_x'] = '--root_rotation_threshold_x ' + \
                str(root_rot_thrshld_x)

        # Add root_rot-y_threshold arg
        if root_rot_thrshld_y is not None:
            arg_parser._table['--root_rotation_threshold_y'] = '--root_rotation_threshold_y ' + \
                str(root_rot_thrshld_y)

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

        # Add 'record_video' argument
        if record_video:
            arg_parser._table['--record_video'] = '--record_video ' + 'true'
        else:
            try:
                del arg_parser._table['--record_video']
            except KeyError:
                pass

        # Add run-args-file to training dictionary
        train_dict['run_file_parser'] = arg_parser

    return


def work(cmd):
    return subprocess.call(cmd, shell=True)


def run_playback(playback_file=None, character=None, single=False, reduced_motion=False,
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
        title = 'Playback'
    elif reduced_motion:
        motion_file = "/home/nash/Dropbox/Clemson/Projects/quat_conversions/pca/Output/"
        if dimension is None:
            playback_dict['motion_file'] = motion_file + 'pca_traj.txt'
            title = 'Playback-Reduced Motion'
        elif single:
            playback_dict['motion_file'] = motion_file + 'single_coactivations/' + \
                ('pca_single_traj_%d.txt' % dimension)
            title = 'Coactivation-%d' % dimension
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
        if dimension is None:
            playback_file = "/home/nash/Dropbox/Clemson/Projects/Learning_Analyser/Output/" + \
                            "pb_file.txt"
        else:
            playback_file = "/home/nash/Dropbox/Clemson/Projects/Learning_Analyser/Output/" + \
                            "pb_file-%d.txt" % dimension
        with open(playback_file, 'w') as fp:
            for k, v in playback_dict.items():
                fp.write('--' + k + ' ' + v + '\n')

        # Execute command to run the trained case
        cmd = 'python DeepMimic.py --arg_file ' + playback_file + ' --title ' + title

    return cmd


def usage():
    print("Usage: Evaluate.py [-a | --log_actions] <input to PD controller> \n"
          "                   [-A | --allow_parent_collision]  \n"
          "                   [-b | --baseline] \n"
          "                   [-B | --behavior] <behavior name (run/walk/...)> \n"
          "                   [-c | --check_collision]  \n"
          "                   [-C | --character]  \n"
          "                   [-d | --duration] <evaluation duration> \n"
          "                   [-D | --dimension] <reduced dimension>/<list of reduced dims> \n"
          "                   [-e | --eval]  \n"
          "                   [-f | --force_eval] \n"
          "                   [-F | --no_flight_phase] \n"
          "                   [-h | --help] \n"
          "                   [-i | --imitate_excitations] \n"
          "                   [-k | --select_k] \n"
          "                   [-I | --intermediate_model] \n"
          "                   [-l | --location] <input folder location> \n"
          "                   [-L | --low_com] <COM height threshold> \n"
          "                   [-m | --reduced_motion_file] <reduced motion file> \n"
          "                   [-M | --character_model_file] <character model file> \n"
          "                   [-n | --num_evals] <no. of evaluations> \n"
          "                   [-o | --log_pose] \n"
          "                   [-p | --pca] \n"
          "                   [-P | --playback] \n"
          "                   [-r | --reward_fn] <evaluation reward function (0/1/2/...)> \n"
          "                   [-R | --reduced] \n"
          "                   [-S | --single] \n"
          "                   [-v | --record_video] \n"
          "                   [-x | --log_excitations] <output of policy network> \n"
          "                   [-X | --root_rot_threshold_x] <rot. threshold in radians> \n"
          "                   [-Y | --root_rot_threshold_y] <rot. threshold in radians> \n"
          )


def main(argv):
    run_file = 'run_args_file.txt'
    reduced_motion_file = None
    character_file = None
    behavior_folder = None
    all = True
    log_excitations = False
    log_actions = False
    log_pose = False
    record_video = False
    baseline = False
    pca = False
    ica = False
    self = False

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
    no_flight_phase = False
    root_rot_thrshld_x = None
    root_rot_thrshld_y = None

    playback = False
    character = None
    reduced_motion = False
    single = False
    dimension = None
    dim_max = None
    dim_min = None
    behavior = None

    try:
        opts, args = getopt.getopt(argv, "h aAbpiefxoPRScFvsC:l:m:M:n:d:r:I:D:B:k:L:X:Y:",
                                   ["log_actions", "allow_parent_collision", "baseline", "pca",
                                    "imitate_excitations", "eval", "force_eval", "log_excitations",
                                    "log_pose", "playback", "reduced", "single", "check_collision",
                                    "no_flight_phase", "record_video", "self", "character=",
                                    "location=", "reduced_motion_file=", "character_model_file=",
                                    "num_evals=", "duration=", "reward_fn=", "intermediate_model=",
                                    "dimension=", "behavior=", "select_k=", "low_com=",
                                    "root_rot_threshold_x=", "root_rot_threshold_y="])
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
        elif opt in ("-M", "--character_model_file"):
            character_file = arg
        elif opt in ("-I", "--intermediate_model"):
            intermediate_model = int(arg)
        elif opt in ("-P", "--playback"):
            playback = True
        elif opt in ("-R", "--reduced"):
            reduced_motion = True
        elif opt in ("-S", "--single"):
            single = True
        elif opt in ("-D", "--dimension"):
            try:
                dimension = int(arg)
            except ValueError:
                if ':' in arg:
                    dim_min = list(map(int, arg.strip('[]').split(':')))[0]
                    dim_max = list(map(int, arg.strip('[]').split(':')))[1]
                else:
                    dimension = list(map(int, arg.strip('[]').split(',')))
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
        elif opt in ("-F", "--no_flight_phase"):
            no_flight_phase = True
        elif opt in ("-X", "--root_rot_threshold_x"):
            root_rot_thrshld_x = float(arg)
        elif opt in ("-Y", "--root_rot_threshold_y"):
            root_rot_thrshld_y = float(arg)
        elif opt in ("-v", "--record_video"):
            record_video = True
        elif opt in ("-s", "--self"):
            self = True

    if playback:
        cmd = list()
        if dimension is None:
            if dim_max > dim_min:
                dimension = list(range(dim_min, dim_max+1))
            else:
                print("Error: Invalid dimension range for playback")
                sys.exit()

        if type(dimension) == list:
            if len(dimension) == 1:
                dimension = list(range(1, dimension[0]+1))

            count = len(dimension)
            pool = multiprocessing.Pool(processes=count)
            for d in dimension:
                cmd.append(run_playback(playback_file=reduced_motion_file, character=character,
                                        single=single, reduced_motion=reduced_motion, dimension=d,
                                        behavior=behavior))
        else:
            pool = multiprocessing.Pool(processes=1)
            cmd.append(run_playback(playback_file=reduced_motion_file, character=character,
                                    single=single, reduced_motion=reduced_motion,
                                    dimension=dimension, behavior=behavior))
        pool.map(work, cmd)
        return

    if all:
        baseline = pca = ica = True

    if behavior_folder is None or not os.path.isdir(behavior_folder):
        print("Directory '%s' does not exist: " % behavior_folder)
        sys.exit()

    if log_excitations and log_actions and not force_eval:
        print("Warning: Both Actions and Activations are set to be logged!")
        sys.exit()

    if character_file == 'self':
        character_file = behavior_folder + 'salamanderV0.txt'

    # Extract individual training folders
    training_folders = []
    character, behavior, training_folders = \
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
                    reduced_motion_file, character_file, imitate_exctn, intermediate_model,
                    log_excitations, log_actions, log_pose, record_video, select_k, com_threshold,
                    check_collision, allow_parent_collision, no_flight_phase, root_rot_thrshld_x,
                    root_rot_thrshld_y, self, character)

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
