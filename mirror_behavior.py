import numpy as np
from Quaternion import Quat, normalize
import math
from collections import OrderedDict
import sys
import getopt
import json
import os
import glob


def concatenate_trajectories(trajs_dict, key_list=[], include=False):
    trajs_data = []
    for k, v in trajs_dict.items():
        if include:
            if k in key_list:
                trajs_data.append(v)
        if not include:
            if k not in key_list:
                trajs_data.append(v)
    return np.column_stack(trajs_data)


def decompose_quat_trajectories(motion_data):
    # Decomposes trajectories into individual joints (by name)
    quat_trajs = OrderedDict()

    quat_trajs['frame_duration'] = np.array(motion_data[:, 0:1])  # Time
    quat_trajs['root_position'] = np.array(motion_data[:, 1:4])  # Position
    quat_trajs['root_rotation'] = np.array(motion_data[:, 4:8])  # Quaternion

    quat_trajs['chest_rotation'] = np.array(motion_data[:, 8:12])  # Quaternion
    quat_trajs['neck_rotation'] = np.array(motion_data[:, 12:16])  # Quaternion

    quat_trajs['right_hip_rotation'] = np.array(motion_data[:, 16:20])  # Quaternion
    quat_trajs['right_knee_rotation'] = np.array(motion_data[:, 20:21])  # 1D Joint
    quat_trajs['right_ankle_rotation'] = np.array(motion_data[:, 21:25])  # Quaternion
    quat_trajs['right_shoulder_rotation'] = np.array(motion_data[:, 25:29])  # Quaternion
    quat_trajs['right_elbow_rotation'] = np.array(motion_data[:, 29:30])  # 1D Joint

    quat_trajs['left_hip_rotation'] = np.array(motion_data[:, 30:34])  # Quaternion
    quat_trajs['left_knee_rotation'] = np.array(motion_data[:, 34:35])  # 1D Joint
    quat_trajs['left_ankle_rotation'] = np.array(motion_data[:, 35:39])  # Quaternion
    quat_trajs['left_shoulder_rotation'] = np.array(motion_data[:, 39:43])  # Quaternion
    quat_trajs['left_elbow_rotation'] = np.array(motion_data[:, 43:44])  # 1D Joint

    return quat_trajs


def decompose_euler_trajectories(motion_data):
    # Decomposes trajectories into individual joints (by name)
    euler_trajs = OrderedDict()

    euler_trajs['frame_duration'] = np.array(motion_data[:, 0:1])  # Time
    euler_trajs['root_position'] = np.array(motion_data[:, 1:4])  # Position
    euler_trajs['root_rotation'] = np.array(motion_data[:, 4:8])  # Quaternion

    euler_trajs['chest_rotation'] = np.array(motion_data[:, 8:11])  # EulerAngle
    euler_trajs['neck_rotation'] = np.array(motion_data[:, 11:14])  # EulerAngle

    euler_trajs['right_hip_rotation'] = np.array(motion_data[:,  14:17])  # EulerAngle
    euler_trajs['right_knee_rotation'] = np.array(motion_data[:, 17:18])  # 1D Joint
    euler_trajs['right_ankle_rotation'] = np.array(motion_data[:, 18:21])  # EulerAngle
    euler_trajs['right_shoulder_rotation'] = np.array(motion_data[:, 21:24])  # EulerAngle
    euler_trajs['right_elbow_rotation'] = np.array(motion_data[:, 24:25])  # 1D Joint

    euler_trajs['left_hip_rotation'] = np.array(motion_data[:, 25:28])  # EulerAngle
    euler_trajs['left_knee_rotation'] = np.array(motion_data[:, 28:29])  # 1D Joint
    euler_trajs['left_ankle_rotation'] = np.array(motion_data[:, 29:32])  # EulerAngle
    euler_trajs['left_shoulder_rotation'] = np.array(motion_data[:, 32:35])  # EulerAngle
    euler_trajs['left_elbow_rotation'] = np.array(motion_data[:, 35:36])  # 1D Joint

    return euler_trajs


def left_right_traj_swap(trajectory_dict):
    swapped_dict = trajectory_dict.copy()

    root_pos = trajectory_dict['root_position']
    root_pos_mean = np.mean(root_pos, axis=0)
    root_pos_mean_cntrd = root_pos - root_pos_mean
    root_pos_flip_z = (root_pos_mean_cntrd * np.array([1, 1, -1])) + root_pos_mean

    swapped_dict['root_position'] = root_pos_flip_z
    swapped_dict['root_rotation'] = trajectory_dict['root_rotation'] * np.array([-1, -1, 1])

    swapped_dict['chest_rotation'] = trajectory_dict['chest_rotation'] * np.array([-1, -1, 1])
    swapped_dict['neck_rotation'] = trajectory_dict['neck_rotation'] * np.array([-1, -1, 1])

    swapped_dict['right_hip_rotation'] = trajectory_dict['left_hip_rotation'] * \
        np.array([-1, -1, 1])
    swapped_dict['right_knee_rotation'] = trajectory_dict['left_knee_rotation']
    swapped_dict['right_ankle_rotation'] = trajectory_dict['left_ankle_rotation'] * \
        np.array([-1, -1, 1])
    swapped_dict['right_shoulder_rotation'] = trajectory_dict['left_shoulder_rotation'] * \
        np.array([-1, -1, 1])
    swapped_dict['right_elbow_rotation'] = trajectory_dict['left_elbow_rotation']

    swapped_dict['left_hip_rotation'] = trajectory_dict['right_hip_rotation'] * \
        np.array([-1, -1, 1])
    swapped_dict['left_knee_rotation'] = trajectory_dict['right_knee_rotation']
    swapped_dict['left_ankle_rotation'] = trajectory_dict['right_ankle_rotation'] * \
        np.array([-1, -1, 1])
    swapped_dict['left_shoulder_rotation'] = trajectory_dict['right_shoulder_rotation'] * \
        np.array([-1, -1, 1])
    swapped_dict['left_elbow_rotation'] = trajectory_dict['right_elbow_rotation']

    return swapped_dict


def convert_to_quaternion(axis_angle_dict, k_list):
    quat_dict = OrderedDict()

    for k, v in axis_angle_dict.items():
        if v.shape[1] == 4 and k not in k_list:
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


def convert_quat_to_euler(quat_dict, k_list):
    euler_dict = OrderedDict()

    for k, v in quat_dict.items():
        if v.shape[1] == 4 and k not in k_list:
            euler_angles = []
            for r in v:
                q = np.array([r[1], r[2], r[3], r[0]])  # [x, y, z, w]
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


def convert_euler_to_quat(euler_dict, key_list):
    quat_dict = OrderedDict()

    for k, v in euler_dict.items():
        if v.shape[1] == 3 and k not in key_list:
            quats = []
            for r in v:
                roll = r[0]
                pitch = r[1]
                yaw = r[2]

                qx = np.sin(roll/2) * np.cos(pitch/2) * np.cos(yaw/2) - np.cos(roll/2) * \
                    np.sin(pitch/2) * np.sin(yaw/2)
                qy = np.cos(roll/2) * np.sin(pitch/2) * np.cos(yaw/2) + np.sin(roll/2) * \
                    np.cos(pitch/2) * np.sin(yaw/2)
                qz = np.cos(roll/2) * np.cos(pitch/2) * np.sin(yaw/2) - np.sin(roll/2) * \
                    np.sin(pitch/2) * np.cos(yaw/2)
                qw = np.cos(roll/2) * np.cos(pitch/2) * np.cos(yaw/2) + np.sin(roll/2) * \
                    np.sin(pitch/2) * np.sin(yaw/2)

                quats.append([qw, qx, qy, qz])
            quat_dict[k] = np.array(quats)
        else:
            quat_dict[k] = v

    return quat_dict


def convert_to_json(pose_file):
    with open(pose_file, 'r') as pf:
        pose_arr = np.loadtxt(pf)

    motion_dict = OrderedDict()
    motion_dict['Loop'] = 'wrap'
    motion_dict['Frames'] = pose_arr.tolist()

    with open(pose_file, 'w') as jf:
        json.dump(motion_dict, jf, indent=4)

    return motion_dict


def mirror_trajectories(mirrored_traj_dict, trajectory_dict, key_list, eulerangle=False):
    mirrored_trajs = left_right_traj_swap(trajectory_dict)

    if eulerangle:
        mirrored_quat_trajs = convert_euler_to_quat(mirrored_trajs, key_list)
    else:
        mirrored_quat_trajs = mirrored_trajs

    mirrored_traj_dict['Frames'] = \
        concatenate_trajectories(mirrored_quat_trajs).tolist()

    return mirrored_traj_dict


def usage():
    print("Usage: pca.py [-e | --eulerangle] \n"
          "              [-h | --help] \n"
          "              [-m | --mfile] <input motion file(s) or directory> \n"
          )


def main(argv):
    motion_files = list()
    eulerangle = False

    try:
        opts, args = getopt.getopt(argv, "hem:", ["help", "eulerangle", "mfile="])
    except getopt.GetoptError:
        usage()
        sys.exit(2)

    for opt, arg in opts:
        if opt in ("-h", "--help"):
            usage()
            sys.exit()
        elif opt in ("-e", "--eulerangle"):
            eulerangle = True
        elif opt in ("-m", "--mfile"):
            motion_files.append(arg)

    if os.path.isdir(motion_files[0]):
        motion_files = glob.glob(motion_files[0] + "humanoid3d_*.txt")
        motion_files.sort()

    motion_data = list()
    motion_dict_list = list()
    for m_file in motion_files:
        try:
            with open(m_file) as mf:
                motion_dict = json.load(mf)
        except:
            motion_dict = convert_to_json(m_file)
        motion_data.append(np.array(motion_dict['Frames']))
        motion_dict_list.append(motion_dict)
    motion_data = np.vstack(motion_data)

    print("Frames count: ", motion_data.shape[0])

    key_list = ['frame_duration', 'root_position']

    quat_trajectory_dict = decompose_quat_trajectories(motion_data)

    if eulerangle:
        trajectory_dict = convert_quat_to_euler(quat_trajectory_dict, key_list)
    else:
        trajectory_dict = quat_trajectory_dict

    # Create a copy of the original dictionary
    mirrored_traj_dict = motion_dict_list[0].copy()

    mirrored_traj_dict = mirror_trajectories(mirrored_traj_dict, trajectory_dict, key_list,
                                             eulerangle)

    # Create output path and file
    output_file_path = "/home/nash/Dropbox/Clemson/Projects/Learning_Analyser/Output"
    output_file = 'humanoid3d_mirrored_'
    for m_file in motion_files:
        motion_name = m_file.split("/")[-1]
        motion_name = motion_name.split(".")[0]
        motion_name = motion_name.split("humanoid3d_")[-1]
        output_file = output_file_path + output_file + motion_name + ".txt"

    # Save pca trajectories and basis dictionary on to the created output file
    with open(output_file, 'w') as fp:
        json.dump(mirrored_traj_dict, fp, indent=4)

    with open('/home/nash/Dropbox/Clemson/Projects/Learning_Analyser/Output/mirrored_traj.txt',
              'w') as fp:
        json.dump(mirrored_traj_dict, fp, indent=4)

    if not eulerangle:
        print("WARNING: PCA Not in Euler Angle! Use flag: [-e | --eulerangle]")


if __name__ == "__main__":
    main(sys.argv[1:])
