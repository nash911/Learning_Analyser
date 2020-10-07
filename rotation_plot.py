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
    rot = False
    x = False
    y = False
    z = False
    window_seconds = 5

    colors = ['red', 'blue', 'green', 'orange', 'black', 'purple', 'cyan', 'grey', 'violet', 'gold',
              'indigo', 'brown', 'pink', 'magenta', 'tan', 'darkgreen', 'turquoise',
              'darkblue', 'beige', 'olive', 'hotpink', 'darkred', 'darkgrey', 'crimson',
              'coral', 'lavender', 'salmon', 'eggplant']

    try:
        opts, args = getopt.getopt(argv, "harxyzw:", ["avg", "rot", "X", "Y", "Z", "window_size"])
    except getopt.GetoptError:
        print("rotation_plot.py -a -x -y -z -w [--avg --X --Y --Z --window_size]")
        sys.exit(2)

    for opt, arg in opts:
        if opt == '-h':
            print("rotation_plot.py -a -x -y -z -w [--avg --X --Y --Z --window_size]")
            sys.exit()
        elif opt in ("-a", "--avg"):
            avg_rot = True
        elif opt in ("-r", "--rot"):
            rot = True
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

    file_names = list()
    window_size = list()
    time_step = list()

    if not (rot or avg_rot):
        print("Error: Specfiy atleast one rotation stat. to plot.\n"
              "Use flags [-r | --rot] and/or [-a | --avg_rot]")
        sys.exit()

    if rot:
        file_names.append('/home/nash/DeepMimic/output/part_rot.dat')
        window_size.append(30 * 20 * window_seconds)
        time_step.append(0.033332/20)

    if avg_rot:
        file_names.append('/home/nash/DeepMimic/output/avg_part_rot.dat')
        window_size.append(30 * window_seconds)
        time_step.append(0.033332)

    graph_data = open(file_names[0], 'r').read()
    lns = graph_data.split('\n')
    data_dim = len(lns[1].split(' '))

    num_plots = int(data_dim/3)

    fig, axs = plt.subplots(num_plots, sharey=True, sharex=True)
    fig.suptitle('Part Rotation Plot', fontsize=20)

    def plot_axs(sub_plot):
        return axs[sub_plot] if num_plots > 1 else axs

    def extract_plot_data(file, win_size, ts, check_file_update=False):
        global prev_file_mod_time
        global file_time_counter

        if check_file_update:
            file_mod_time = os.stat(file)[8]
            if file_mod_time == prev_file_mod_time:
                file_time_counter += 1
            else:
                file_time_counter = 0
            prev_file_mod_time = file_mod_time

            if file_time_counter > 10:
                return [], []

        graph_data = open(file, 'r').read()
        lines = graph_data.split('\n')
        t = list()

        plot_data = np.empty((num_plots, win_size, 3))
        for i, line in enumerate(lines[-win_size:]):
            if len(line) > 1 and not line[0] == '#':
                data_point = line.split(' ')
                plot_data[:, i, :] = np.array([float(dp) for dp in data_point]).reshape(-1, 3)
                t.append(float(i) * ts)

        return plot_data, t

    def animate(i):
        plot_data = list()
        time_data = list()
        for i, (fn, ws, ts) in enumerate(zip(file_names, window_size, time_step)):
            data, time = extract_plot_data(fn, ws, ts, check_file_update=(i == 0))
            if len(data) > 0:
                plot_data.append(data)
                time_data.append(time)
            else:
                return

        if num_plots > 1:
            for ax in axs:
                ax.clear()
        else:
            axs.clear()

        for n, (data, t, ws) in enumerate(zip(plot_data, time_data, window_size)):
            for i in range(num_plots):
                ind = (n * num_plots) + i
                if x:
                    plot_axs(i).plot(t[-ws:], data[i, :len(t), 0].tolist(), color=colors[ind],
                                     label=(('Joint-' + str(i+1)) if n == 0 else None))
                if y:
                    plot_axs(i).plot(t[-ws:], data[i, :len(t), 1].tolist(), color=colors[ind],
                                     label=(('Joint-' + str(i+1)) if n == 0 else None))
                if z:
                    plot_axs(i).plot(t[-ws:], data[i, :len(t), 2].tolist(), color=colors[ind],
                                     label=(('Joint-' + str(i+1)) if n == 0 else None))

                if i == np.ceil(num_plots/2):
                    plot_axs(i).set(ylabel='Rotation (rad.)')

                plot_axs(i).legend(loc='upper right')
                plot_axs(num_plots-1).set(xlabel='Time(s)')

    _ = animation.FuncAnimation(fig, animate, interval=100)
    plt.show()


if __name__ == "__main__":
    main(sys.argv[1:])
