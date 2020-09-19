import sys
import getopt
import numpy as np
import os
from itertools import product
from collections import OrderedDict
import matplotlib.pyplot as plt
import matplotlib.animation as animation
from matplotlib import style

style.use('seaborn')

prev_file_mod_time = 0
file_time_counter = 0


def usage():
    print("Usage: com_root_plot.py [-s | --sim]\n"
          "                        [-k | --kin] \n"
          "                        [-c | --com] \n"
          "                        [-r | --root] \n"
          "                        [-p | --pos] \n"
          "                        [-v | --vel] \n"
          "                        [-a | --acc] \n"
          "                        [-e | --speed] \n"
          "                        [-o | --rot] \n"
          "                        [-g | --ang_vel] \n"
          "                        [-h | --help] \n"
          "                        [-w | --window_size] <plot window size in seconds> \n"
          )


def main(argv):
    com = False
    root = False
    kin = False
    sim = False
    window_seconds = 10

    char = list()
    obj = list()
    var = list()
    plot_combos = list()

    try:
        opts, args = getopt.getopt(argv, "hcrpvaeogksw:", ["com", "root", "pos", "vel", "acc",
                                                           "speed", "rot", "ang_vel" "kin", "sim",
                                                           "window_size"])
    except getopt.GetoptError:
        usage()
        sys.exit(2)

    for opt, arg in opts:
        if opt == '-h':
            usage()
            sys.exit()
        elif opt in ("-c", "--com"):
            com = True
            obj.append('com')
        elif opt in ("-r", "--root"):
            root = True
            obj.append('root')
        elif opt in ("-p", "--pos"):
            var.append('pos')
        elif opt in ("-v", "--vel"):
            var.append('vel')
        elif opt in ("-a", "--acc"):
            var.append('acc')
        elif opt in ("-e", "--speed"):
            var.append('speed')
        elif opt in ("-o", "--rot"):
            var.append('rot')
        elif opt in ("-g", "--ang_vel"):
            var.append('ang_vel')
        elif opt in ("-k", "--kin"):
            kin = True
            char.append('kin')
        elif opt in ("-s", "--sim"):
            sim = True
            char.append('sim')
        elif opt in ("-w", "--window_size"):
            window_seconds = int(arg)

    if kin and sim:
        title_1 = 'Simulated/Kinematic-Character '
    elif kin:
        title_1 = 'Kinematic-Character '
    elif sim:
        title_1 = 'Simulated-Character '
    else:
        print("Error: Choose either Simulated or Kinematic character with flags:",
              " [-s | --sim] [-k | --kin]")
        sys.exit(2)

    if com and root:
        title_2 = 'COM/Root '
    elif com:
        title_2 = 'COM '
    elif root:
        title_2 = 'Root '
    else:
        print("Error: Choose either COM or Root with flags: [-c | --com] [-r | --root]")
        sys.exit(2)

    if len(var) == 0:
        print("Error: Choose at least one plot variable with flags: [-p | --pos] \n"
              "                                                     [-v | --vel] \n"
              "                                                     [-a | --acc] \n"
              "                                                     [-e | --speed] \n"
              "                                                     [-o | --rot] \n"
              "                                                     [-g | --ang_vel]")
        sys.exit(2)

    var_dict = OrderedDict()
    var_dict['pos'] = {'file_name': '/home/nash/DeepMimic/output/%s_pos.dat',
                       'y_label': 'Pos. (m)',
                       'subtitle': '%s-%s-Position Plot',
                       'legend_x': 'Pos-X',
                       'legend_y': 'Pos-Y',
                       'legend_z': 'Pos-Z'}

    var_dict['vel'] = {'file_name': '/home/nash/DeepMimic/output/%s_vel.dat',
                       'y_label': 'Vel. (m/s)',
                       'subtitle': '%s-%s-Velocity Plot',
                       'legend_x': 'Vel-X',
                       'legend_y': 'Vel-Y',
                       'legend_z': 'Vel-Z'}

    var_dict['acc'] = {'file_name': '/home/nash/DeepMimic/output/%s_vel.dat',
                       'y_label': 'Acc. (m/s^2)',
                       'subtitle': '%s-%s-Acceleration Plot',
                       'legend_x': 'Acc-X',
                       'legend_y': 'Acc-Y',
                       'legend_z': 'Acc-Z'}

    var_dict['speed'] = {'file_name': '/home/nash/DeepMimic/output/%s_vel.dat',
                         'y_label': 'Speed (m/s)',
                         'subtitle': '%s-%s-Speed Plot',
                         'legend_speed': 'Speed-XZ'}

    var_dict['rot'] = {'file_name': '/home/nash/DeepMimic/output/%s_rot.dat',
                       'y_label': 'Ang. (rad)',
                       'subtitle': '%s-%s-Rotation Plot',
                       'legend_x': 'Rot-X',
                       'legend_y': 'Rot-Y',
                       'legend_z': 'Rot-Z'}

    var_dict['ang_vel'] = {'file_name': '/home/nash/DeepMimic/output/%s_ang_vel.dat',
                           'y_label': 'Ang. Vel. (rad/s)',
                           'subtitle': '%s-%s-Angular Velocity Plot',
                           'legend_x': 'AngVel-X',
                           'legend_y': 'AngVel-Y',
                           'legend_z': 'AngVel-Z'}

    for c, v, o in product(char, var, obj):
        if o == 'com' and v in ['rot', 'ang_vel']:
            continue
        else:
            plot_combos.append({'char': c, 'var': v, 'obj': o, 'plot_vars': var_dict[v]})

    window_size = 30 * window_seconds
    time_step = 0.033332

    num_plots = len(plot_combos)
    if num_plots > 0:
        fig, axs = plt.subplots(num_plots, sharey=False, sharex=True)
        fig.suptitle((title_1 + title_2 + 'Plot(s)'), fontsize=20)
    else:
        print("No Graphs to Plot!")
        return

    def extract_plot_data(plot_dict, check_file_update=False):
        global prev_file_mod_time
        global file_time_counter

        file_name = plot_dict['plot_vars']['file_name'] % plot_dict['obj']

        if check_file_update:
            file_mod_time = os.stat(file_name)[8]
            if file_mod_time == prev_file_mod_time:
                file_time_counter += 1
            else:
                file_time_counter = 0
            prev_file_mod_time = file_mod_time

            if file_time_counter > 10:
                return dict()

        graph_data = open(file_name, 'r').read()
        lines = graph_data.split('\n')
        t = list()
        x = list()
        y = list()
        z = list()
        speed = list()

        for i, line in enumerate(lines[-window_size:]):
            if len(line) > 1 and not line[0] == '#':
                try:
                    sim_x, sim_y, sim_z, kin_x, kin_y, kin_z, sim_speed, kin_speed = line.split(' ')
                except ValueError:
                    sim_x, sim_y, sim_z, kin_x, kin_y, kin_z, = line.split(' ')

                t.append(float(i) * time_step)
                if plot_dict['char'] == 'sim':
                    if plot_dict['var'] == 'speed':
                        try:
                            speed.append(float(sim_speed))
                        except UnboundLocalError:
                            pass
                    else:
                        x.append(float(sim_x))
                        y.append(float(sim_y))
                        z.append(float(sim_z))
                elif plot_dict['char'] == 'kin':
                    if plot_dict['var'] == 'speed':
                        try:
                            speed.append(float(kin_speed))
                        except UnboundLocalError:
                            pass
                    else:
                        x.append(float(kin_x))
                        y.append(float(kin_y))
                        z.append(float(kin_z))

        if plot_dict['var'] == 'acc':
            x = np.diff(x) / time_step
            y = np.diff(y) / time_step
            z = np.diff(z) / time_step

            t = np.array(t)
            t = ((t[:-1] + t[1:]) / 2).tolist()

        return {'t': t, 'x': x, 'y': y, 'z': z, 'speed': speed}

    def plot_axs(sub_plot):
        return axs[sub_plot] if num_plots > 1 else axs

    def animate(i):
        plot_data = list()
        for i, p_d in enumerate(plot_combos):
            data = extract_plot_data(p_d, check_file_update=(i == 0))
            if data:
                plot_data.append(data)
            else:
                return

        if num_plots > 1:
            for ax in axs:
                ax.clear()
        else:
            axs.clear()

        for i, (p_c, p_d) in enumerate(zip(plot_combos, plot_data)):
            if p_c['var'] == 'speed':
                plot_axs(i).plot(p_d['t'][-window_size:], p_d['speed'][-window_size:],
                                 label=p_c['plot_vars']['legend_speed'])
            else:
                plot_axs(i).plot(p_d['t'][-window_size:], p_d['x'][-window_size:],
                                 label=p_c['plot_vars']['legend_x'])
                plot_axs(i).plot(p_d['t'][-window_size:], p_d['y'][-window_size:],
                                 label=p_c['plot_vars']['legend_y'])
                plot_axs(i).plot(p_d['t'][-window_size:], p_d['z'][-window_size:],
                                 label=p_c['plot_vars']['legend_z'])

            plot_axs(i).title.set_text(p_c['plot_vars']['subtitle'] % (p_c['char'].upper(),
                                                                       p_c['obj'].upper()))
            plot_axs(i).set(ylabel=p_c['plot_vars']['y_label'])
            plot_axs(i).legend(loc='upper right')

            if i == num_plots-1:
                plot_axs(i).set(xlabel='Time(s)')

    _ = animation.FuncAnimation(fig, animate, interval=100)
    plt.show()


if __name__ == "__main__":
    main(sys.argv[1:])
