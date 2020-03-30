import sys
import getopt
import numpy as np
import json
import re
import math
from collections import OrderedDict
import matplotlib.pyplot as plt
import matplotlib.animation as animation
from matplotlib import style

sys.path.insert(0, '/home/nash/Dropbox/Clemson/Projects/quat_conversions/pca')
from Quaternion import Quat, normalize

style.use('seaborn')


def decompose_learned_actions(learned_actions):
    # Decomposes trajectories into individual DOFs by joint-names
    quat_dict = OrderedDict()

    quat_dict['root_position'] = np.array(learned_actions[:, 0:3])  # 3D Position
    quat_dict['root_rotation'] = np.array(learned_actions[:, 3:7])  # 4D Joints

    quat_dict['chest_rotation'] = np.array(learned_actions[:, 7:11])  # 4D Joints
    quat_dict['neck_rotation'] = np.array(learned_actions[:, 11:15])  # 4D Joints

    quat_dict['right_hip_rotation'] = np.array(learned_actions[:, 15:19])  # 4D Joints
    quat_dict['right_knee_rotation'] = np.array(learned_actions[:, 19:20])  # 1D Joint
    quat_dict['right_ankle_rotation'] = np.array(learned_actions[:, 20:24])  # 4D Joints
    quat_dict['right_shoulder_rotation'] = np.array(learned_actions[:, 24:28])  # 4D Joints
    quat_dict['right_elbow_rotation'] = np.array(learned_actions[:, 28:29])  # 1D Joint

    quat_dict['left_hip_rotation'] = np.array(learned_actions[:, 29:33])  # 4D Joints
    quat_dict['left_knee_rotation'] = np.array(learned_actions[:, 33:34])  # 1D Joint
    quat_dict['left_ankle_rotation'] = np.array(learned_actions[:, 34:38])  # 4D Joints
    quat_dict['left_shoulder_rotation'] = np.array(learned_actions[:, 38:42])  # 4D Joints
    quat_dict['left_elbow_rotation'] = np.array(learned_actions[:, 42:43])  # 1D Joint

    return quat_dict


def convert_axisangle_to_quaternion(axis_angle_dict):
    quat_dict = OrderedDict()

    for k, v in axis_angle_dict.items():
        if v.shape[1] == 4:
            quaternions = []
            for r in v:
                x = r[1] * math.sin(r[0]/2.0)
                y = r[2] * math.sin(r[0]/2.0)
                z = r[3] * math.sin(r[0]/2.0)
                w = math.cos(r[0]/2.0)

                q = np.array([w, x, y, z])
                quaternions.append(q)
            quat_dict[k] = np.array(quaternions)
        else:
            quat_dict[k] = v

    return quat_dict


def convert_quat_to_euler(quat_dict):
        euler_dict = OrderedDict()

        for k, v in quat_dict.items():
            if v.shape[1] == 4:
                euler_angles = []
                for r in v:
                    q = np.array([r[1], r[2], r[3], r[0]])  # Format: [x, y, z, w]
                    nq = Quat(normalize(q))
                    nq_v = nq._get_q()
                    w = nq_v[3]
                    x = nq_v[0]
                    y = nq_v[1]
                    z = nq_v[2]

                    # roll (x-axis rotation)
                    t0 = +2.0 * (w * x + y * z)
                    t1 = +1.0 - 2.0 * (x * x + y * y)
                    roll = math.atan2(t0, t1)

                    # pitch (y-axis rotation)
                    t2 = +2.0 * (w * y - z * x)
                    t2 = +1.0 if t2 > +1.0 else t2
                    t2 = -1.0 if t2 < -1.0 else t2
                    pitch = math.asin(t2)

                    # yaw (z-axis rotation)
                    t3 = +2.0 * (w * z + x * y)
                    t4 = +1.0 - 2.0 * (y * y + z * z)
                    yaw = math.atan2(t3, t4)

                    euler_angles.append([roll, pitch, yaw])
                euler_dict[k] = np.array(euler_angles)
            else:
                euler_dict[k] = v

        return euler_dict


def concatenate_trajectories(trajs_dict, key_list=[]):
    trajs_data = []

    for k, v in trajs_dict.items():
        if k not in key_list:
            trajs_data.append(v)

    return np.column_stack(trajs_data)


def create_sub_plots(axs, excitations, resolution, cycle_duration, lower_dim, upper_dim, title):
    lc = ['xkcd:red', 'xkcd:blue', 'xkcd:green', 'xkcd:brown', 'xkcd:pink', 'xkcd:purple',
          'xkcd:orange', 'xkcd:magenta', 'xkcd:tan', 'xkcd:black', 'xkcd:cyan', 'xkcd:gold',
          'xkcd:dark green', 'xkcd:cream', 'xkcd:lavender', 'xkcd:turquoise', 'xkcd:dark blue',
          'xkcd:violet', 'xkcd:beige', 'xkcd:salmon', 'xkcd:olive', 'xkcd:light brown',
          'xkcd:hot pink', 'xkcd:dark red', 'xkcd:sand', 'xkcd:army green', 'xkcd:dark grey',
          'xkcd:crimson', 'xkcd:eggplant', 'xkcd:coral']

    time = np.full((excitations[0].shape[0]), resolution)
    time = np.cumsum(time) / cycle_duration

    axs.clear()
    lines = []
    for i, act in enumerate(excitations):
        if i >= lower_dim and i < upper_dim:
            lines.append(axs.plot(time, act, color=lc[i], label=('Comp-%s' % (i+1))))

    axs.set_ylabel('PCA Values', fontsize=14)
    axs.set_title(title, fontsize=15)

    axs.grid(linestyle="--", linewidth=0.5, color='.25', zorder=-10)
    axs.set_facecolor('xkcd:white')

    return lines


def calculate_excitation_violations(excitations, excite_stats, singular_values, num_frames,
                                    ignore_cycles=0):
    # Split across individual excitation-dimensions
    excitations_list = np.hsplit(excitations, excitations.shape[1])

    violations_list = []
    start_idx = num_frames * ignore_cycles

    if singular_values is None:
        singular_values = np.ones(excitations.shape[1])

    for i, (extn, s_val) in enumerate(zip(excitations_list, singular_values)):
        lower_bound = extn[start_idx:, :] - excite_stats[str(i+1)]['min']
        upper_bound = extn[start_idx:, :] - excite_stats[str(i+1)]['max']
        violation_min = np.minimum(lower_bound, 0.0) * s_val
        violation_max = np.maximum(upper_bound, 0.0) * s_val
        violation = np.square(violation_min) + np.square(violation_min)
        violations_list.append(violation)

    violations_matrix = np.squeeze(np.array(violations_list), axis=-1).T
    violations_mean = np.mean(violations_matrix)
    violations_mean_individual = np.mean(violations_matrix, axis=0)

    return violations_mean, violations_mean_individual


def plot_kin_char_excitations(ax, data_file, num_frames, resolution, animate=False, lower_dim=0,
                              cycle_duration=1, upper_dim=28, offset=0, dtype=np.float64):
    # Extract recorded kin-char excitations from data file
    with open(data_file) as f:
        lines = f.read().splitlines()

    kin_char_exctns = []
    sub_title = ''
    for line in lines[-num_frames-offset:]:
        words = line.split()
        if words[0] == '--title':
            for word in words[1:]:
                sub_title += word + ' '
        else:
            kin_char_exctns.append(words)

    if len(kin_char_exctns) == 0:
        return [], None

    if not animate:
        print("Kinematic Character Excitations:")
        print("No. of cycles: ", len(kin_char_exctns)/num_frames, '\n')

    # Store Input-Actions/Output-Pose as a numpy array
    kin_char_exctns_matrix = np.array(kin_char_exctns, dtype=dtype)

    # Consider only the last-cycle of excitations for plotting
    if offset > 0:
        last_cycle = kin_char_exctns_matrix[-num_frames-offset:-offset, :]
    else:
        last_cycle = kin_char_exctns_matrix[-num_frames:, :]

    # Split across individual excitation-dimensions for plotting
    kin_extns = np.hsplit(last_cycle, kin_char_exctns_matrix.shape[1])

    # Plot Actions/Pose
    lines = create_sub_plots(ax, kin_extns, resolution, cycle_duration, lower_dim, upper_dim,
                             ("Kin-Char Excitations" if not sub_title else sub_title))
    return lines, last_cycle


def plot_extracted_excitations(ax, data_file, basis_inv, num_frames, resolution, cycle_duration,
                               mean=None, lower_dim=0, upper_dim=28, kin_extn_mat=None,
                               diff_imitation=False, root_pos=False, root_rot=False,
                               axis_angle=False, excite_stats=None, animate=False, ignore_cycles=0,
                               offset=0, dtype=np.float64):
    # Extract Learned-Actions of Baseline, or recorded Pose from data file
    with open(data_file) as f:
        lines = f.read().splitlines()

    learned_actions = []
    sub_title = ''

    # Look for the plot title
    try:
        words = lines[0].split()
        if words[0] == '--title':
            for word in words[1:]:
                sub_title += word + ' '
            del lines[0]
    except ValueError:
        return []

    # Extract the last cycle from the file
    for line in lines[-num_frames-offset:]:
        words = line.split()
        if words[0] == '--title':
            for word in words[1:]:
                sub_title += word + ' '
        else:
            learned_actions.append(words)

    if len(learned_actions) == 0:
        return []

    if not animate:
        print("Extracted Excitations:")
        print("No. of cycles: ", len(learned_actions)/num_frames, '\n')

    # Store Input-Actions/Output-Pose as a numpy array
    learned_actions_matrix = np.array(learned_actions, dtype=dtype)

    # Decompose Input-Actions/Output-Pose
    learned_actions_dict = decompose_learned_actions(learned_actions_matrix)

    if axis_angle:
        # Convert Axis-Angle-Actions/Pose to Quaternion-Actions/Pose
        quat_actions_dict = convert_axisangle_to_quaternion(learned_actions_dict)
    else:
        quat_actions_dict = learned_actions_dict

    # Convert Quaternion-Actions/Pose to Euler-Actions/Pose
    euler_actions_dict = convert_quat_to_euler(quat_actions_dict)

    key_list = []
    if not root_pos:
        key_list.append('root_position')
    if not root_rot:
        key_list.append('root_rotation')

    # Concatinate Euler-Action/Pose into a single numpy array
    euler_actions_matrix = concatenate_trajectories(euler_actions_dict, key_list)

    # Mean-normalise the Euler-Action/Pose, if mean exists
    if mean is not None:
        euler_actions_matrix -= mean

    # Project Actions/Pose on to Excitation space: U = X (∑V^T)^-1
    euler_excitations = np.matmul(euler_actions_matrix, basis_inv)

    # Consider only the last-cycle of excitations for plotting
    if offset > 0:
        last_cycle = euler_excitations[-num_frames-offset:-offset, :]
    else:
        last_cycle = euler_excitations[-num_frames:, :]

    # Split across individual excitation-dimensions for plotting
    extracted_euler_excits = np.hsplit(last_cycle, euler_excitations.shape[1])

    # Plot Actions/Pose
    lines = create_sub_plots(ax, extracted_euler_excits, resolution, cycle_duration, lower_dim,
                             upper_dim, ("Baseline Actions" if not sub_title else sub_title))

    if diff_imitation and kin_extn_mat is not None:
        if last_cycle.shape == kin_extn_mat.shape:
            diff = (kin_extn_mat - last_cycle) * 1
            sum_sqrd = np.sqrt(np.sum(np.square(diff), axis=-1))
            mean_diff = np.mean(sum_sqrd)
            neg_exp = np.mean(np.exp(-sum_sqrd))
            print("Mean Excitation Imitation Diff: ", mean_diff, " 1/e(-diff): ", neg_exp)
    return lines


def plot_learned_excitations(ax, learned_excitation_file, num_frames, resolution, cycle_duration,
                             lower_dim=0, upper_dim=28, excite_stats=None, animate=False,
                             ignore_cycles=0, offset=0, singular_values=None, dtype=np.float64):
    # Extract Learned-Excitations from PCA/ICA trained evaluation file
    with open(learned_excitation_file) as f:
        lines = f.read().splitlines()
    try:
        del lines[0]
    except:
        pass

    excitation = []
    for line in lines[-num_frames-offset:]:
        try:
            if line[0] == '[':
                line = re.sub('[[]', '', line)
                line = re.sub('[]]', '', line)
                excitation.append(line.split())
        except:
            return []

    if len(excitation) == 0:
        return []

    if not animate:
        print("Learned Excitations:")
        print("No. of cycles: ", len(excitation)/num_frames, '\n')

    # Store Learned-Actions as a numpy array
    learned_excitations = np.array(excitation, dtype=dtype)

    # Consider only the last-cycle of excitations for plotting
    if offset > 0:
        last_cycle = learned_excitations[-num_frames-offset:-offset, :]
    else:
        last_cycle = learned_excitations[-num_frames:, :]

    # Split across individual excitation-dimensions for plotting
    lrnd_excits = np.hsplit(last_cycle, learned_excitations.shape[1])

    # Plot Learned Excitations
    lines = create_sub_plots(ax, lrnd_excits, resolution, cycle_duration, lower_dim, upper_dim,
                             "Lower Dimension (ours) Actions")

    if excite_stats is not None:
        scale_factor = 1000.0
        violations_mean, violations_mean_individual = \
            calculate_excitation_violations(learned_excitations, excite_stats, singular_values,
                                            num_frames, ignore_cycles)

        print("Learned-Excitations violation mean: ", violations_mean*scale_factor, "\n")
        for i, mean in enumerate(violations_mean_individual):
            print("Learned-Excitation-%d violation mean: %f" % (i+1, mean*scale_factor))

    return lines


def plot_reference_excitations(ax, ref_file, resolution, num_frames, cycle_duration, lower_dim,
                               upper_dim, normal_basis=False, sub_title="Reference Motion"):
    # Open reduced reference file
    with open(ref_file) as f:
        data = json.load(f)

    # Extract and split reference excitations data
    try:
        redu_ref_extn = np.array(data['U'])
    except KeyError:
        redu_ref_extn = np.array(data['Signals'])

    if normal_basis:
        sigma = np.array(data['Sigma'])
        redu_ref_extn = np.matmul(redu_ref_extn, sigma)

    ref_extns = np.hsplit(redu_ref_extn[0:num_frames, :], redu_ref_extn.shape[1])

    # Plot reference excitations
    lns = create_sub_plots(ax, ref_extns, resolution, cycle_duration, lower_dim, upper_dim,
                           sub_title)
    return lns


def usage():
    print("Usage: excitations_plot.py [-a | --animate] \n"
          "                           [-b | --bound_violation] \n"
          "                           [-c | --num_cycles] <no. of cycles to animate> \n"
          "                           [-d | --diff_imitation] \n"
          "                           [-e | --excitation] <input excitation file> \n"
          "                           [-i | --ignore_cycles] <no. of init-cycles to be ignored> \n"
          "                           [-k | --kin_excitation] <kin character excitation file> \n"
          "                           [-l | --lower] <lower dim limit for plot> \n"
          "                           [-m | --mot_ref] <motion reference file> \n"
          "                           [-n | --action] <baseline action file> \n"
          "                           [-o | --offsets] <list of sub-plot's x-axis offset values> \n"
          "                           [-p | --pose] <input pose file> \n"
          "                           [-q | --quaternion] \n"
          "                           [-r | --red_ref] <reduced reference file> \n"
          "                           [-u | --upper] <upper dim limit for plot> \n"
          )


def main(argv):
    reduced_reference_files = []
    motion_reference_file = None
    learned_actions_file = None
    learned_excitation_file = None
    kin_char_extn_file = None
    offsets = None
    pose_files = []
    axis_angle = True
    lower_dim = 0
    upper_dim = 28
    bound_violation = False
    excite_stats = None
    ignore_cycles = 0
    num_cycles = 1
    animate = False
    diff_imitation = False
    root_pos = False
    root_rot = False

    plt.rcParams.update({'font.size': 20})
    plt.rcParams['xtick.labelsize'] = 15
    plt.rcParams['ytick.labelsize'] = 15

    try:
        opts, args = getopt.getopt(argv, "h abdqn:c:e:i:k:l:m:o:p:r:u:",
                                   ["animate", "bound_violation", "diff_imitation", "quaternion",
                                    "action", "num_cycles", "excitation", "ignore_cycles",
                                    "kin_excitation", "lower", "mot_ref", "offsets", "pose",
                                    "red_ref", "upper"])
    except getopt.GetoptError:
        usage()
        sys.exit(2)

    for opt, arg in opts:
        if opt == '-h':
            usage()
            sys.exit()
        elif opt in ("-a", "--animate"):
            animate = True
        elif opt in ("-b", "--bound_violation"):
            bound_violation = True
        elif opt in ("-c", "--num_cycles"):
            num_cycles = int(arg)
        elif opt in ("-d", "--diff_imitation"):
            diff_imitation = True
        elif opt in ("-l", "--lower"):
            lower_dim = int(arg)
        elif opt in ("-e", "--excitation"):
            learned_excitation_file = arg
        elif opt in ("-i", "--ignore_cycles"):
            ignore_cycles = int(arg)
        elif opt in ("-k", "--kin_excitation"):
            kin_char_extn_file = arg
        elif opt in ("-l", "--lower"):
            lower_dim = int(arg)
        elif opt in ("-m", "--mot_ref"):
            motion_reference_file = arg
        elif opt in ("-n", "--action"):
            learned_actions_file = arg
        elif opt in ("-o", "--offsets"):
            offsets = list(map(int, arg.strip('[]').split(',')))
        elif opt in ("-p", "--pose"):
            pose_files.append(arg)
        elif opt in ("-q", "--quaternion"):
            axis_angle = False
        elif opt in ("-r", "--red_ref"):
            reduced_reference_files.append(arg)
        elif opt in ("-u", "--upper"):
            upper_dim = int(arg)

    if len(reduced_reference_files) == 0:
        print("Error: Should specify an appropriate Reduced Motion file with ",
              "arg '-r/--ref'")
        sys.exit(2)

    if not animate:
        num_cycles = 1

    # Determine number of sub-plots
    num_plots = len(reduced_reference_files)
    if learned_actions_file is not None:
        num_plots += 1
    if learned_excitation_file is not None:
        num_plots += 1
    if kin_char_extn_file is not None:
        num_plots += 1
    num_plots += len(pose_files)

    if offsets is None or animate:
        offsets = np.zeros(num_plots, dtype=np.int)
    else:
        offsets.insert(0, 0)

    if len(offsets) != num_plots:
        print("Error: No. of sub-plots: %d != no. of offsets passed: %d" %
              (num_plots-1, len(offsets)-1))
        sys.exit(2)

    # Create fig with sub-plots
    fig, axs = \
        plt.subplots(num_plots, sharey=True, sharex=(False if animate and num_cycles > 1 else True))

    # Set main title of the plot
    fig.suptitle('Excitations Plots')

    def plot(i):
        plot_pos = 0
        num_legend = 0
        excite_stats = None
        kin_extn_mat = None

        # Open primary reduced reference file
        with open(reduced_reference_files[0]) as rf:
            data = json.load(rf)

        # Open motion reference file, if provided
        try:
            with open(motion_reference_file) as mf:
                motion_data = json.load(mf)
        except TypeError:
            motion_data = None

        # Extract resolution and cycle-duration from reference file
        if motion_data is not None:
            # From Motion Reference Data
            reduced_ref_frames = np.array(motion_data['Frames'])
        else:
            # From Reduced Reference Data
            reduced_ref_frames = np.array(data['Frames'])

        resolution = reduced_ref_frames[0][0]
        num_frames = reduced_ref_frames.shape[0]
        cycle_duration = resolution * num_frames
        num_frames_to_plot = num_frames * num_cycles

        # Extract domain of the reduced reference motion
        reduced_ref_domain = np.array(data['Domain'])

        if not reduced_ref_domain == "Eulerangle":
            print("Error: Domain of reference motion is:", reduced_ref_domain,
                  " expected: Eulerangle")
            sys.exit(2)

        # Check if Basis is Orthonormal
        try:
            normal_basis = data['normal_basis']
        except KeyError:
            normal_basis = "False"
        normal_basis = True if normal_basis == "True" else False

        # Extract and split reference excitations data from file
        try:
            reduced_ref_extn = np.array(data['U'])
        except KeyError:
            reduced_ref_extn = np.array(data['Signals'])

        if normal_basis:
            sigma = np.array(data['Sigma'])
            reduced_ref_extn = np.matmul(reduced_ref_extn, sigma)

        redu_refs = np.hsplit(reduced_ref_extn, reduced_ref_extn.shape[1])

        # Extract basis-inverse (∑V^T)^-1 from file
        basis_inv = np.array(data['Basis_Inv'])

        # Extract reference mean
        try:
            mean = np.array(data['Reference_Mean'])
        except KeyError:
            mean = None

        # Extract singular-values
        try:
            singular_values = np.array(data['Singular_Values'])
        except KeyError:
            singular_values = None

        # Extract root-pos/rot
        try:
            root_pos = np.array(data['Root_Pos'])
        except KeyError:
            root_pos = False
        try:
            root_rot = np.array(data['Root_Rot'])
        except KeyError:
            root_rot = False

        # Extract excitation stats
        if bound_violation:
            try:
                excite_min = np.array(data['Excite_Min'])
            except KeyError:
                print("ERROR: Flag '-b'|'--bound_violation' set, but reference motion file does ",
                      "not contain Excite_Min")
                sys.exit(2)

            try:
                excite_max = np.array(data['Excite_Max'])
            except KeyError:
                print("ERROR: Flag '-b'|'--bound_violation' set, but reference motion file does ",
                      "not contain Excite_Max")
                sys.exit(2)

            excite_stats = OrderedDict()
            for i, (e_min, e_max) in enumerate(zip(excite_min, excite_max)):
                stat = OrderedDict()
                stat['min'] = e_min
                stat['max'] = e_max
                excite_stats[str(i+1)] = stat

        # Plot reference excitations
        for i, ref_file in enumerate(reduced_reference_files):
            lns = plot_reference_excitations((axs[i] if num_plots > 1 else axs), ref_file,
                                             resolution, num_frames, cycle_duration, lower_dim,
                                             upper_dim, normal_basis, ("Reference Motion-%d" %
                                             (i+1)))
            num_legend = max(num_legend, len(lns))
        plot_pos += len(reduced_reference_files) - 1

        if learned_actions_file is not None:
            plot_pos += 1
            # Plot Excitations extracted from Baseline Actions
            lns = plot_extracted_excitations(axs[plot_pos], learned_actions_file, basis_inv,
                                             num_frames_to_plot, resolution, cycle_duration, mean,
                                             lower_dim, upper_dim, root_pos, root_rot, axis_angle,
                                             excite_stats, offset=offsets[plot_pos],
                                             ignore_cycles=ignore_cycles, animate=animate,
                                             dtype=reduced_ref_extn.dtype)
            num_legend = max(num_legend, len(lns))

        if learned_excitation_file is not None:
            plot_pos += 1
            # Plot Learned Excitations
            lns = plot_learned_excitations(axs[plot_pos], learned_excitation_file,
                                           num_frames_to_plot, resolution, cycle_duration,
                                           lower_dim, upper_dim, excite_stats, animate=animate,
                                           singular_values=singular_values,
                                           ignore_cycles=ignore_cycles, offset=offsets[plot_pos],
                                           dtype=reduced_ref_extn.dtype)
            num_legend = max(num_legend, len(lns))

        if kin_char_extn_file is not None:
            plot_pos += 1
            # Plot Recorded Kinematic Character Excitations
            lns, kin_extn_mat = \
                plot_kin_char_excitations(axs[plot_pos], kin_char_extn_file, num_frames_to_plot,
                                          resolution, animate, lower_dim, cycle_duration, upper_dim,
                                          offset=offsets[plot_pos], dtype=reduced_ref_extn.dtype)
            num_legend = max(num_legend, len(lns))

        for pose_file in pose_files:
            plot_pos += 1
            # Plot Excitations extracted from Recorded Poses
            lns = plot_extracted_excitations(axs[plot_pos], pose_file, basis_inv,
                                             num_frames_to_plot, resolution, cycle_duration, mean,
                                             lower_dim, upper_dim,  kin_extn_mat, diff_imitation,
                                             root_pos, root_rot, axis_angle=False, animate=animate,
                                             excite_stats=None, ignore_cycles=ignore_cycles,
                                             offset=offsets[plot_pos], dtype=reduced_ref_extn.dtype)
            num_legend = max(num_legend, len(lns))

        # Add a common legend for the plot
        line_labels = []
        for i in range(num_legend):
            line_labels.append('PCA-' + str(i+lower_dim+1))

        if num_plots == 1:
            axs.set_xlabel('Gait Cycle', fontsize=14)
        else:
            axs[plot_pos].set_xlabel('Gait Cycle', fontsize=14)

        # Create the legend
        fig.legend(lns,     # The line objects
                   labels=line_labels,   # The labels for each line
                   loc="upper right",   # Position of legend
                   #borderaxespad=0.0,    # Small spacing around legend box
                   #bbox_to_anchor=(0.5, 0., 0.5, 0.5)
                   #title="Top 5 Co-activations"  # Title for the legend
                   )

    if animate:
        ani = animation.FuncAnimation(fig, plot, interval=10)
    else:
        plot(1)

    plt.show()


if __name__ == "__main__":
    main(sys.argv[1:])
