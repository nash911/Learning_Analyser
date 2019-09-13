import sys, getopt
import numpy as np
import json
import os
import re
from collections import OrderedDict
import matplotlib.pyplot as plt
import matplotlib.animation as animation
from matplotlib import style

style.use('seaborn')


def decompose_euler_coactivations(coactivations_mat):
    # Decomposes trajectories into indificual DOFs by joint name
    coactivations_dict = OrderedDict()

    coactivations_dict['chest_rotation'] = np.array(coactivations_mat[:,0:3]) # 3D Joints
    coactivations_dict['neck_rotation'] = np.array(coactivations_mat[:,3:6]) # 3D Joints

    coactivations_dict['right_hip_rotation'] = np.array(coactivations_mat[:,6:9]) # 3D Joints
    coactivations_dict['right_knee_rotation'] = np.array(coactivations_mat[:,9:10]) # 1D Joint
    coactivations_dict['right_ankle_rotation'] = np.array(coactivations_mat[:,10:13]) # 3D Joints
    coactivations_dict['right_shoulder_rotation'] = np.array(coactivations_mat[:,13:16]) # 3D Joints
    coactivations_dict['right_elbow_rotation'] = np.array(coactivations_mat[:,16:17]) # 1D Joint

    coactivations_dict['left_hip_rotation'] = np.array(coactivations_mat[:,17:20]) # 3D Joints
    coactivations_dict['left_knee_rotation'] = np.array(coactivations_mat[:,20:21]) # 1D Joint
    coactivations_dict['left_ankle_rotation'] = np.array(coactivations_mat[:,21:24]) # 3D Joints
    coactivations_dict['left_shoulder_rotation'] = np.array(coactivations_mat[:,24:27]) # 3D Joints
    coactivations_dict['left_elbow_rotation'] = np.array(coactivations_mat[:,27:28]) # 1D Joint

    return coactivations_dict


def main(argv):
    reduced_reference_file = '../data/reduced_motion/'
    learned_activation_file = 'activation.dat'

    lc = ['xkcd:red', 'xkcd:blue', 'xkcd:green', 'xkcd:brown', 'xkcd:pink',
          'xkcd:purple', 'xkcd:orange', 'xkcd:magenta', 'xkcd:tan', 'xkcd:black',
          'xkcd:cyan', 'xkcd:gold', 'xkcd:dark green', 'xkcd:cream',
          'xkcd:lavender', 'xkcd:turquoise', 'xkcd:dark blue', 'xkcd:violet',
          'xkcd:beige', 'xkcd:salmon', 'xkcd:olive', 'xkcd:light brown',
          'xkcd:hot pink', 'xkcd:dark red', 'xkcd:sand', 'xkcd:army green',
          'xkcd:dark grey', 'xkcd:crimson', 'xkcd:eggplant', 'xkcd:coral']

    try:
        opts, args = getopt.getopt(argv, "h r:a:",
                                   ["ref", "act"])
    except getopt.GetoptError:
        print("activation_plot.py -r <reference_file> -a <activation_file>")
        sys.exit(2)

    for opt, arg in opts:
        if opt == '-h':
            print("activation_plot.py -r <reference_file> -a <activation_file>")
            sys.exit()
        elif opt in ("-r", "--ref"):
            reduced_reference_file = arg
        elif opt in ("-a", "--act"):
            learned_activation_file = arg

    def create_plot(title, activations, resolution):
        fig = plt.figure()
        fig.suptitle(title, fontsize=15)
        ax = fig.add_subplot(1, 1, 1)

        #time = np.full((activations[0].shape[0]), resolution)
        time = np.arange(28)

        labels = []
        for i, act in enumerate(activations):
            ax.plot(time, act, color=lc[i], label=('Co-Act-%s' % (i+1)))

        #labels = [str((t+1) * 3) for t in time]
        labels = ['Chest-X', 'Y', 'Z',
                  'Neck-X', 'Y', 'Z',
                  'R.Hip-X', 'Y', 'Z',
                  'R.Knee',
                  'R.Ankle-X', 'Y', 'Z',
                  'R.Shoulder-X', 'Y', 'Z',
                  'R-Elbow',
                  'L.Hip-X', 'Y', 'Z',
                  'L.Knee',
                  'L.Ankle-X', 'Y', 'Z',
                  'L.Shoulder-X', 'Y', 'Z',
                  'L-Elbow']

        ax.set(xlabel='Joints', ylabel='Level')
        plt.xticks(time, labels)
        plt.xticks(rotation=90)
        ax.legend()


    with open(reduced_reference_file) as f:
        data = json.load(f)

    coactivations_mat =  np.array(data['Basis'])
    coactivations_dict = decompose_euler_coactivations(coactivations_mat)
    coactivations_list = np.hsplit(coactivations_mat.T, coactivations_mat.T.shape[1])

    resolution = np.array(data['Frames'])[0][0]

    create_plot("Co-Activations - RUN", coactivations_list, resolution)

    plt.show()

if __name__ == "__main__":
    main(sys.argv[1:])
