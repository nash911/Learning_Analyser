import sys
import getopt
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation as animation
from matplotlib import style

style.use('seaborn')


def main(argv):
    obj = 'root'
    var = 'pos'
    char = 'sim'

    try:
        opts, args = getopt.getopt(argv, "hcrpvaogks", ["com", "root", "pos", "vel", "acc", "rot",
                                                        "ang_vel" "kin", "sim"])
    except getopt.GetoptError:
        print("plot.py -c/r -p/v/a/r/g -k/s")
        sys.exit(2)

    for opt, arg in opts:
        if opt == '-h':
            print("plot.py -c/r -p/v/a/r/g -k/s")
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
        y_label = 'Position (m)'
        title_3 = 'Position Plot'
        legend_x = 'Pos-X'
        legend_y = 'Pos-Y'
        legend_z = 'Pos-Z'
    elif var == 'vel':
        file_name = '/home/nash/DeepMimic/output/' + obj + '_vel.dat'
        y_label = 'Velocity (m/s)'
        title_3 = 'Velocity Plot'
        legend_x = 'Vel-X'
        legend_y = 'Vel-Y'
        legend_z = 'Vel-Z'
    elif var == 'acc':
        file_name = '/home/nash/DeepMimic/output/' + obj + '_vel.dat'
        y_label = 'Acceleration (m/s^2)'
        title_3 = 'Acceleration Plot'
        legend_x = 'Acc-X'
        legend_y = 'Acc-Y'
        legend_z = 'Acc-Z'
    elif var == 'rot':
        file_name = '/home/nash/DeepMimic/output/' + obj + '_rot.dat'
        y_label = 'Angle (rad)'
        title_3 = 'Rotation Plot'
        legend_x = 'Rot-X'
        legend_y = 'Rot-Y'
        legend_z = 'Rot-Z'
    elif var == 'ang_vel':
        file_name = '/home/nash/DeepMimic/output/' + obj + '_ang_vel.dat'
        y_label = 'Angular Velocity (rad/s)'
        title_3 = 'Angular Velocity Plot'
        legend_x = 'AngVel-X'
        legend_y = 'AngVel-Y'
        legend_z = 'AngVel-Z'
    legend_speed = 'Speed-XZ'

    fig = plt.figure()
    fig.suptitle((title_1 + title_2 + title_3), fontsize=20)
    ax1 = fig.add_subplot(1, 1, 1)

    def animate(i):
        window_size = 300
        time_step = 0.033332
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
                    sim_x, sim_y, sim_z, kin_x, kin_y, kin_z, = line.split(' ')
                except ValueError:
                    sim_x, sim_y, sim_z, kin_x, kin_y, kin_z, sim_speed, kin_speed = line.split(' ')

                t.append(float(i) * time_step)
                if char == 'sim':
                    s_x.append(float(sim_x))
                    s_y.append(float(sim_y))
                    s_z.append(float(sim_z))
                    try:
                        s_speed.append(float(sim_speed))
                    except UnboundLocalError:
                        pass
                else:
                    k_x.append(float(kin_x))
                    k_y.append(float(kin_y))
                    k_z.append(float(kin_z))
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

        ax1.clear()
        if char == 'sim':
            ax1.plot(t[-window_size:], s_x[-window_size:], label=legend_x)
            ax1.plot(t[-window_size:], s_y[-window_size:], label=legend_y)
            ax1.plot(t[-window_size:], s_z[-window_size:], label=legend_z)
            ax1.plot(t[-window_size:], s_speed[-window_size:], label=legend_speed)
        else:
            ax1.plot(t[-window_size:], k_x[-window_size:], label=legend_x)
            ax1.plot(t[-window_size:], k_y[-window_size:], label=legend_y)
            ax1.plot(t[-window_size:], k_z[-window_size:], label=legend_z)
            ax1.plot(t[-window_size:], k_speed[-window_size:], label=legend_speed)

        ax1.set(xlabel='Time(s)', ylabel=y_label)
        ax1.legend()

    ani = animation.FuncAnimation(fig, animate, interval=100)
    plt.show()


if __name__ == "__main__":
    main(sys.argv[1:])
