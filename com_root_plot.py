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
          "                        [-m | --mean_vel] \n"
          "                        [-a | --acc] \n"
          "                        [-e | --speed] \n"
          "                        [-o | --rot] \n"
          "                        [-g | --ang_vel] \n"
          "                        [-h | --help] \n"
          "                        [-x | --X] \n"
          "                        [-y | --Y] \n"
          "                        [-z | --Z] \n"
          "                        [-w | --window_size] <plot window size in seconds> \n"
          )


def main(argv):
    com = False
    root = False
    kin = False
    sim = False
    vel = False
    mean = False
    X = True
    Y = True
    Z = True
    window_seconds = 10

    char = list()
    obj = list()
    var = list()
    plot_combos = list()

    try:
        opts, args = getopt.getopt(argv, "hcrpvaeogksmxyzw:", ["com", "root", "pos", "vel", "acc",
                                                               "speed", "rot", "ang_vel" "kin",
                                                               "sim", "mean", "X", "Y",
                                                               "Z", "window_size"])
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
            vel = True
        elif opt in ("-a", "--acc"):
            var.append('acc')
        elif opt in ("-e", "--speed"):
            var.append('speed')
        elif opt in ("-o", "--rot"):
            var.append('rot')
        elif opt in ("-g", "--ang_vel"):
            var.append('ang_vel')
        elif opt in ("-m", "--mean_vel"):
            var.append('mean_vel')
            mean = True
        elif opt in ("-k", "--kin"):
            kin = True
            char.append('kin')
        elif opt in ("-s", "--sim"):
            sim = True
            char.append('sim')
        elif opt in ("-x", "--X"):
            X = False
        elif opt in ("-y", "--Y"):
            Y = False
        elif opt in ("-z", "--Z"):
            Z = False
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
                       'legend_x': '$\mathit{p}_x$',
                       'legend_y': '$\mathit{p}_y$',
                       'legend_z': '$\mathit{p}_z$'}

    var_dict['vel'] = {'file_name': '/home/nash/DeepMimic/output/%s_vel.dat',
                       'y_label': 'Vel. (m/s)',
                       'subtitle': '%s-%s-Velocity Plot',
                       'legend_x': '$\mathit{v}_x$',
                       'legend_y': '$\mathit{v}_y$',
                       'legend_z': '$\mathit{v}_z$'}

    var_dict['mean_vel'] = {'file_name': '/home/nash/DeepMimic/output/avg_%s_vel.dat',
                            'y_label': 'Vel. (m/s)',
                            'subtitle': '%s-%s-Velocity Plot',
                            'legend_x': '$\mathit{\overline{v}}_x$',
                            'legend_y': '$\mathit{\overline{v}}_y$',
                            'legend_z': '$\mathit{\overline{v}}_z$'}

    var_dict['acc'] = {'file_name': '/home/nash/DeepMimic/output/%s_vel.dat',
                       'y_label': 'Acc. (m/s^2)',
                       'subtitle': '%s-%s-Acceleration Plot',
                       'legend_x': '$\mathit{a}_x$',
                       'legend_y': '$\mathit{a}_y$',
                       'legend_z': '$\mathit{a}_z$'}

    var_dict['speed'] = {'file_name': '/home/nash/DeepMimic/output/%s_vel.dat',
                         'y_label': 'Speed (m/s)',
                         'subtitle': '%s-%s-Speed Plot',
                         'legend_speed': '$\mathbf{v}$'}

    var_dict['rot'] = {'file_name': '/home/nash/DeepMimic/output/%s_rot.dat',
                       'y_label': 'Ang. (rad)',
                       'subtitle': '%s-%s-Rotation Plot',
                       'legend_x': '$\mathit{\Theta}_x$',
                       'legend_y': '$\mathit{\Theta}_y$',
                       'legend_z': '$\mathit{\Theta}_z$'}

    var_dict['ang_vel'] = {'file_name': '/home/nash/DeepMimic/output/%s_ang_vel.dat',
                           'y_label': 'Ang. Vel. (rad/s)',
                           'subtitle': '%s-%s-Angular Velocity Plot',
                           'legend_x': '$\mathit{\omega}_x$',
                           'legend_y': '$\mathit{\omega}_y$',
                           'legend_z': '$\mathit{\omega}_z$'}

    for c, v, o in product(char, var, obj):
        if o == 'com' and v in ['rot', 'ang_vel']:
            continue
        else:
            plot_combos.append({'char': c, 'var': v, 'obj': o, 'plot_vars': var_dict[v]})

    window_size = 30 * window_seconds
    time_step = 0.033332

    num_plots = len(plot_combos)
    if mean and vel:
        num_plots -= 1

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
                    try:
                        sim_x, sim_y, sim_z, kin_x, kin_y, kin_z, = line.split(' ')
                    except ValueError:
                        sim_x, sim_y, sim_z = line.split(' ')
                        kin_x = kin_y = kin_z = 0

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

        common_plot_indx = None
        for i, (p_c, p_d) in enumerate(zip(plot_combos, plot_data)):
            if p_c['var'] == 'speed':
                plot_axs(i).plot(p_d['t'][-window_size:], p_d['speed'][-window_size:],
                                 label=p_c['plot_vars']['legend_speed'])
            elif p_c['var'] == 'mean_vel':
                if common_plot_indx is None:
                    common_plot_indx = indx = i
                else:
                    indx = common_plot_indx

                if X:
                    plot_axs(indx).plot(p_d['t'][-window_size:], p_d['x'][-window_size:],
                                        label=p_c['plot_vars']['legend_x'], color='purple')
                if Y:
                    plot_axs(indx).plot(p_d['t'][-window_size:], p_d['y'][-window_size:],
                                        label=p_c['plot_vars']['legend_y'], color='orange')
                if Z:
                    plot_axs(indx).plot(p_d['t'][-window_size:], p_d['z'][-window_size:],
                                        label=p_c['plot_vars']['legend_z'], color='black')
            else:
                if p_c['var'] == 'vel':
                    if common_plot_indx is None:
                        common_plot_indx = indx = i
                    else:
                        indx = common_plot_indx
                else:
                    indx = i

                if X:
                    plot_axs(indx).plot(p_d['t'][-window_size:], p_d['x'][-window_size:],
                                        label=p_c['plot_vars']['legend_x'])
                    # x_mean = [np.mean(p_d['x'][-window_size:])] * (window_size-1)
                    # plot_axs(indx).plot(p_d['t'][-window_size:], x_mean, label='X-Mean',
                    #                     color='pink')
                if Y:
                    plot_axs(indx).plot(p_d['t'][-window_size:], p_d['y'][-window_size:],
                                        label=p_c['plot_vars']['legend_y'])
                if Z:
                    plot_axs(indx).plot(p_d['t'][-window_size:], p_d['z'][-window_size:],
                                        label=p_c['plot_vars']['legend_z'])
                    # z_mean = [np.mean(p_d['z'][-window_size:])] * (window_size-1)
                    # plot_axs(indx).plot(p_d['t'][-window_size:], z_mean, label='Z-Mean',
                    #                     color='yellow')

            plot_axs(i).title.set_text(p_c['plot_vars']['subtitle'] % (p_c['char'].upper(),
                                                                       p_c['obj'].upper()))
            plot_axs(i).set(ylabel=p_c['plot_vars']['y_label'])
            # plot_axs(i).legend(loc='upper right')
            plot_axs(i).legend(loc='center left', bbox_to_anchor=(1, 0.5), prop={'size': 12})

            if i == num_plots-1:
                plot_axs(i).set(xlabel='Time(s)')

    _ = animation.FuncAnimation(fig, animate, interval=100)
    plt.show()


if __name__ == "__main__":
    main(sys.argv[1:])
