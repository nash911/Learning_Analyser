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
    obj = 'root'
    var = 'pos'
    char = None
    window_seconds = 10
    window_size = 30 * window_seconds
    time_step = 0.033332

    try:
        opts, args = getopt.getopt(argv, "hcrpvaogksw:", ["com", "root", "pos", "vel", "acc",
                                                          "rot", "ang_vel" "kin", "sim",
                                                          "window_size"])
    except getopt.GetoptError:
        print("plot.py -c/r -p/v/a/o/g -k/s")
        sys.exit(2)

    for opt, arg in opts:
        if opt == '-h':
            print("plot.py -c/r -p/v/a/o/g -k/s")
            sys.exit()
        elif opt in ("-c", "--com"):
            obj = 'com'
        elif opt in ("-r", "--root"):
            obj = 'root'
        elif opt in ("-p", "--pos"):
            var = 'pos'
        elif opt in ("-v", "--vel"):
            var = 'vel'
        elif opt in ("-a", "--acc"):
            var = 'acc'
        elif opt in ("-o", "--rot"):
            var = 'rot'
        elif opt in ("-g", "--ang_vel"):
            var = 'ang_vel'
        elif opt in ("-k", "--kin"):
            char = 'kin'
        elif opt in ("-s", "--sim"):
            char = 'sim'
        elif opt in ("-w", "--window_size"):
            window_seconds = int(arg)

    if char == 'kin':
        title_1 = 'Kinematic-Character '
    elif char == 'sim':
        title_1 = 'Simulated-Character '

    if obj == 'root':
        title_2 = 'Root '
    elif obj == 'com':
        title_2 = 'COM '

    if var == 'pos':
        file_name = '/home/nash/DeepMimic/output/' + obj + '_pos.dat'
        y_label = ['Position (m)']
        title_3 = 'Position Plot'
        legend_x = 'Pos-X'
        legend_y = 'Pos-Y'
        legend_z = 'Pos-Z'
    elif var == 'vel':
        file_name = '/home/nash/DeepMimic/output/' + obj + '_vel.dat'
        y_label = ['Velocity (m/s)', 'Speed (m/s)']
        title_3 = 'Velocity Plot'
        legend_x = 'Vel-X'
        legend_y = 'Vel-Y'
        legend_z = 'Vel-Z'
    elif var == 'acc':
        file_name = '/home/nash/DeepMimic/output/' + obj + '_vel.dat'
        y_label = ['Acceleration (m/s^2)']
        title_3 = 'Acceleration Plot'
        legend_x = 'Acc-X'
        legend_y = 'Acc-Y'
        legend_z = 'Acc-Z'
    elif var == 'rot':
        file_name = '/home/nash/DeepMimic/output/' + obj + '_rot.dat'
        y_label = ['Angle (rad)']
        title_3 = 'Rotation Plot'
        legend_x = 'Rot-X'
        legend_y = 'Rot-Y'
        legend_z = 'Rot-Z'
    elif var == 'ang_vel':
        file_name = '/home/nash/DeepMimic/output/' + obj + '_ang_vel.dat'
        y_label = ['Angular Velocity (rad/s)']
        title_3 = 'Angular Velocity Plot'
        legend_x = 'AngVel-X'
        legend_y = 'AngVel-Y'
        legend_z = 'AngVel-Z'
    legend_speed = 'Speed-XZ'

    graph_data = open(file_name, 'r').read()
    lns = graph_data.split('\n')
    num_data = len(lns[1].split(' '))

    if num_data == 6 or var == 'acc':
        num_plots = 1
    elif num_data == 8:
        num_plots = 2
    else:
        print("Error")

    fig, axs = plt.subplots(num_plots, sharey=False, sharex=True)
    fig.suptitle((title_1 + title_2 + title_3), fontsize=20)

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
        s_x = list()
        s_y = list()
        s_z = list()
        s_speed = list()

        k_x = list()
        k_y = list()
        k_z = list()
        k_speed = list()

        for i, line in enumerate(lines[-window_size:]):
            if len(line) > 1 and not line[0] == '#':
                try:
                    sim_x, sim_y, sim_z, kin_x, kin_y, kin_z, sim_speed, kin_speed = line.split(' ')
                except ValueError:
                    sim_x, sim_y, sim_z, kin_x, kin_y, kin_z, = line.split(' ')

                t.append(float(i) * time_step)
                if char == 'sim':
                    s_x.append(float(sim_x))
                    s_y.append(float(sim_y))
                    s_z.append(float(sim_z))
                    if var != 'acc':
                        try:
                            s_speed.append(float(sim_speed))
                        except UnboundLocalError:
                            pass
                elif char == 'kin':
                    k_x.append(float(kin_x))
                    k_y.append(float(kin_y))
                    k_z.append(float(kin_z))
                    if var != 'acc':
                        try:
                            k_speed.append(float(kin_speed))
                        except UnboundLocalError:
                            pass

        if var == 'acc':
            if char == 'sim':
                s_x = np.diff(s_x) / time_step
                s_y = np.diff(s_y) / time_step
                s_z = np.diff(s_z) / time_step
            else:
                k_x = np.diff(k_x) / time_step
                k_y = np.diff(k_y) / time_step
                k_z = np.diff(k_z) / time_step

            t = np.array(t)
            t = ((t[:-1] + t[1:]) / 2).tolist()

        if num_plots > 1:
            for ax in axs:
                ax.clear()
        else:
            axs.clear()

        if char == 'sim':
            (axs[0] if num_plots > 1 else axs).plot(t[-window_size:], s_x[-window_size:],
                                                    label=legend_x)
            (axs[0] if num_plots > 1 else axs).plot(t[-window_size:], s_y[-window_size:],
                                                    label=legend_y)
            (axs[0] if num_plots > 1 else axs).plot(t[-window_size:], s_z[-window_size:],
                                                    label=legend_z)
            if len(s_speed) > 0:
                (axs[1] if num_plots > 1 else axs).plot(t[-window_size:], s_speed[-window_size:],
                                                        label=legend_speed, color='purple')
        elif char == 'kin':
            (axs[0] if num_plots > 1 else axs).plot(t[-window_size:], k_x[-window_size:],
                                                    label=legend_x)
            (axs[0] if num_plots > 1 else axs).plot(t[-window_size:], k_y[-window_size:],
                                                    label=legend_y)
            (axs[0] if num_plots > 1 else axs).plot(t[-window_size:], k_z[-window_size:],
                                                    label=legend_z)
            if len(k_speed) > 0:
                (axs[1] if num_plots > 1 else axs).plot(t[-window_size:], k_speed[-window_size:],
                                                        label=legend_speed, color='purple')

        if num_plots > 1:
            for ax, y_lab in zip(axs, y_label):
                ax.set(xlabel='Time(s)', ylabel=y_lab)
                ax.legend(loc='upper right')
        else:
            axs.set(xlabel='Time(s)', ylabel=y_label[0])
            axs.legend(loc='upper right')

    _ = animation.FuncAnimation(fig, animate, interval=100)
    plt.show()


if __name__ == "__main__":
    main(sys.argv[1:])
