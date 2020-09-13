import sys
import getopt
import numpy as np
import os
import matplotlib.pyplot as plt
import matplotlib.animation as animation
from matplotlib import style

style.use('seaborn')

prev_file_mod_time = 0
file_time_counter = 0


def main(argv):
    X = False
    Y = False
    Z = False
    indiv_pos = False
    window_seconds = 5
    window_size = 30 * 20 * window_seconds
    time_step = 0.033332/20.0

    file_name = '/home/nash/DeepMimic/output/pose_sim.dat'

    try:
        opts, args = getopt.getopt(argv, "hxyziw:", ["X", "Y", "Z", "indiv_pos", "window_size"])
    except getopt.GetoptError:
        print("pose_plot.py -i -w [--indiv_pos --window_size]")
        sys.exit(2)

    for opt, arg in opts:
        if opt == '-h':
            print("pose_plot.py -x -y -z -i -w [--X --Y --Z --indiv_pos --window_size]")
            sys.exit()
        elif opt in ("-x", "--X"):
            X = True
        elif opt in ("-y", "--Y"):
            Y = True
        elif opt in ("-z", "--Z"):
            Z = True
        elif opt in ("-i", "--indiv_pos"):
            indiv_pos = True
        elif opt in ("-w", "--window_size"):
            window_seconds = int(arg)

    if not (X or Y or Z):
        print("Error: Specfiy atleast one joint axis (x/y/z) to plot.\n"
              "Use flags [-x | --X] and/or [-y | --Y] and/or [-z | --Z]")
        sys.exit()

    graph_data = open(file_name, 'r').read()
    lns = graph_data.split('\n')
    data_dim = len(lns[1].split(' ')) - 1

    joint_inds = list()
    joint_names = [list(), list(), list(), list()]
    colors = [list(), list(), list(), list()]
    if X:
        joint_inds.append(((np.array([16, 15, 14, 13]) - 1) * 3) + 0)

        joint_names[0].append('Hind.L-X')
        joint_names[1].append('Hind.R-X')
        joint_names[2].append('Front.L-X')
        joint_names[3].append('Front.R-X')

        colors[0].append('yellow')
        colors[1].append('tomato')
        colors[2].append('slateblue')
        colors[3].append('lime')
    if Y:
        joint_inds.append(((np.array([16, 15, 14, 13]) - 1) * 3) + 1)

        joint_names[0].append('Hind.L-Y')
        joint_names[1].append('Hind.R-Y')
        joint_names[2].append('Front.L-Y')
        joint_names[3].append('Front.R-Y')

        colors[0].append('gold')
        colors[1].append('red')
        colors[2].append('blue')
        colors[3].append('green')
    if Z:
        joint_inds.append(((np.array([16, 15, 14, 13]) - 1) * 3) + 2)

        joint_names[0].append('Hind.L-Z')
        joint_names[1].append('Hind.R-Z')
        joint_names[2].append('Front.L-Z')
        joint_names[3].append('Front.R-Z')

        colors[0].append('orange')
        colors[1].append('firebrick')
        colors[2].append('navy')
        colors[3].append('mediumseagreen')

    joint_inds = np.transpose(np.array(joint_inds))

    if indiv_pos:
        num_plots = joint_inds.shape[0]
    else:
        num_plots = 1

    fig, axs = plt.subplots(num_plots, sharey=True, sharex=True)
    fig.suptitle('Pose Plot', fontsize=20)

    def plot_axs(sub_plot):
        return axs[sub_plot] if num_plots > 1 else axs

    def animate(i):
        global prev_file_mod_time
        global file_time_counter

        file_mod_time = os.stat(file_name)[8]
        if file_mod_time == prev_file_mod_time:
            file_time_counter += 1
        else:
            file_time_counter = 0
        prev_file_mod_time = file_mod_time

        if file_time_counter > 10:
            return

        graph_data = open(file_name, 'r').read()
        lines = graph_data.split('\n')
        t = list()

        plot_data = np.empty((window_size, data_dim))

        for i, line in enumerate(lines[-window_size:]):
            if len(line) > 1 and not line[0] == '#':
                data_point = line[1:].split(' ')
                try:
                    plot_data[i] = np.array([float(dp) for dp in data_point])
                    t.append(float(i) * time_step)
                except ValueError:
                    pass

        if num_plots > 1:
            for ax in axs:
                ax.clear()
        else:
            axs.clear()

        if indiv_pos:
            for i, joint in enumerate(joint_inds):
                for axis, n, c in zip(joint, joint_names[i], colors[i]):
                    plot_axs(i).plot(t[-window_size:], plot_data[:len(t), axis].tolist(),
                                     color=c, label=n)
                if i == np.ceil(num_plots/2):
                    plot_axs(i).set(ylabel='Joint Ang.')
                plot_axs(i).legend(loc='center left', bbox_to_anchor=(1, 0.5),
                                   prop={'size': 12})
            plot_axs(num_plots-1).set(xlabel='Time(s)')
            fig.subplots_adjust(right=0.85)
        else:
            for i, joint in enumerate(joint_inds):
                for axis, n, c in zip(joint, joint_names[i], colors[i]):
                    plot_axs(0).plot(t[-window_size:], plot_data[:len(t), axis].tolist(), color=c,
                                     label=n)
            plot_axs(0).set(xlabel='Time(s)', ylabel='Joint Ang.')
            plot_axs(0).legend(bbox_to_anchor=(0.5, -0.08*num_plots), loc='upper center', ncol=5,
                               prop={'size': 12})
            fig.subplots_adjust(bottom=0.15)

    _ = animation.FuncAnimation(fig, animate, interval=100)
    plt.show()


if __name__ == "__main__":
    main(sys.argv[1:])
