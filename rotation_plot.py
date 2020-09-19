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
    avg_rot = False
    x = False
    y = False
    z = False
    window_seconds = 5

    colors = ['red', 'blue', 'green', 'orange', 'black', 'purple', 'cyan', 'grey', 'violet', 'gold',
              'indigo', 'brown', 'pink', 'magenta', 'tan', 'darkgreen', 'lavender',
              'turquoise', 'darkblue', 'beige', 'salmon', 'olive', 'hotpink',
              'darkred', 'sand', 'armygreen', 'darkgrey', 'crimson', 'eggplant', 'coral']

    try:
        opts, args = getopt.getopt(argv, "haxyzw:", ["avg", "X", "Y", "Z", "window_size"])
    except getopt.GetoptError:
        print("rotation_plot.py -a -x -y -z -w [--avg --X --Y --Z --window_size]")
        sys.exit(2)

    for opt, arg in opts:
        if opt == '-h':
            print("rotation_plot.py -a -x -y -z -w [--avg --X --Y --Z --window_size]")
            sys.exit()
        elif opt in ("-a", "--avg"):
            avg_rot = True
        elif opt in ("-x", "--X"):
            x = True
        elif opt in ("-y", "--Y"):
            y = True
        elif opt in ("-z", "--Z"):
            z = True
        elif opt in ("-w", "--window_size"):
            window_seconds = int(arg)

    if not (x or y or z):
        print("Error: Specfiy atleast one rotation axis (x/y/z) to plot.\n"
              "Use flags [-x | --X] and/or [-y | --Y] and/or [-z | --Z]")
        sys.exit()

    if avg_rot:
        file_name = '/home/nash/DeepMimic/output/avg_part_rot.dat'
        window_size = 30 * window_seconds
        time_step = 0.033332
    else:
        file_name = '/home/nash/DeepMimic/output/part_rot.dat'
        window_size = 30 * 20 * window_seconds
        time_step = 0.033332/20

    graph_data = open(file_name, 'r').read()
    lns = graph_data.split('\n')
    data_dim = len(lns[1].split(' '))

    num_plots = int(data_dim/3)

    fig, axs = plt.subplots(num_plots, sharey=True, sharex=True)
    fig.suptitle('Part Rotation Plot', fontsize=20)

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

        plot_data = np.empty((num_plots, window_size, 3))
        for i, line in enumerate(lines[-window_size:]):
            if len(line) > 1 and not line[0] == '#':
                data_point = line.split(' ')
                plot_data[:, i, :] = np.array([float(dp) for dp in data_point]).reshape(-1, 3)
                t.append(float(i) * time_step)

        if num_plots > 1:
            for ax in axs:
                ax.clear()
        else:
            axs.clear()

        for i in range(num_plots):
            if x:
                plot_axs(i).plot(t[-window_size:], plot_data[i, :len(t), 0].tolist(),
                                 color=colors[i], label=('Joint-' + str(i+1)))

            if y:
                plot_axs(i).plot(t[-window_size:], plot_data[i, :len(t), 1].tolist(),
                                 color=colors[i], label=('Joint-' + str(i+1)))

            if z:
                plot_axs(i).plot(t[-window_size:], plot_data[i, :len(t), 2].tolist(),
                                 color=colors[i], label=('Joint-' + str(i+1)))

            if i == np.ceil(num_plots/2):
                plot_axs(i).set(ylabel='Rotation (rad.)')

            # plot_axs(i).legend(loc='center left', bbox_to_anchor=(1, 0.5), prop={'size': 12})
            plot_axs(i).legend(loc='upper right')
            plot_axs(num_plots-1).set(xlabel='Time(s)')
            # fig.subplots_adjust(right=0.85)

    _ = animation.FuncAnimation(fig, animate, interval=100)
    plt.show()


if __name__ == "__main__":
    main(sys.argv[1:])
