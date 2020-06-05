import sys
import getopt
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation as animation
from matplotlib import style

style.use('seaborn')


def main(argv):
    plot_type = None
    file_name = None
    title = None

    try:
        opts, args = getopt.getopt(argv, "hcfr", ["contact", "force", "ratio"])
    except getopt.GetoptError:
        print("plot.py -c/f")
        sys.exit(2)

    for opt, arg in opts:
        if opt == '-h':
            print("plot.py -c/f")
            sys.exit()
        elif opt in ("-c", "--contact"):
            plot_type = 'contact'
        elif opt in ("-f", "--force"):
            plot_type = 'force'
        elif opt in ("-r", "--ratio"):
            plot_type = 'ratio'

    if plot_type == 'contact':
        file_name = '/home/nash/DeepMimic/output/feet_contact_force.dat'
        title = 'Feet Contact Plot'
    elif plot_type == 'force':
        file_name = '/home/nash/DeepMimic/output/avg_feet_contact_force.dat'
        title = 'Feet Avg. Force Plot'
    elif plot_type == 'ratio':
        file_name = '/home/nash/DeepMimic/output/avg_feet_contact_force.dat'
        title = 'Feet Force Ratio Plot'

    fig = plt.figure()
    fig.suptitle(title, fontsize=20)
    ax1 = fig.add_subplot(1, 1, 1)

    def contact_plot_animate(i):
        seconds = 3
        window_size = seconds * 30 * 20
        time_step = 0.033332 / 20.0
        graph_data = open(file_name, 'r').read()
        lines = graph_data.split('\n')

        t = list()
        plot_data = np.zeros((window_size, 4))

        for i, line in enumerate(lines[-window_size:]):
            if len(line) > 1 and not line[0] == '#':
                h_l, h_r, f_l, f_r = line.split(' ')
                plot_data[i] = np.array([float(h_l), float(h_r), float(f_l), float(f_r)],
                                        dtype=bool)
                t.append(float(i) * time_step)
        plot_data *= np.array([4, 3, 2, 1], dtype=np.int)

        ax1.clear()
        feet = ('Front-Right', 'Front-Left', 'Hinde-Right', 'Hinde-Left')
        y_pos = np.arange(1, len(feet) + 1)
        ax1.set_yticks(y_pos)
        ax1.set_yticklabels(feet)

        ax1.plot(t[-window_size:], plot_data[:len(t), 0].tolist(), '.', color='gold',
                 label='Hinde-Left')
        ax1.plot(t[-window_size:], plot_data[:len(t), 1].tolist(), '.', color='red',
                 label='Hinde-Right')
        ax1.plot(t[-window_size:], plot_data[:len(t), 2].tolist(), '.', color='blue',
                 label='Front-Left')
        ax1.plot(t[-window_size:], plot_data[:len(t), 3].tolist(), '.', color='green',
                 label='Front-Right')

        ax1.set(xlabel='Time(s)', ylabel='Feet')
        ax1.set_ylim(0.5, 5)
        ax1.legend(loc='upper right')

    def force_plot_animate(i):
        seconds = 5
        window_size = seconds * 30
        time_step = 0.033332
        graph_data = open(file_name, 'r').read()
        lines = graph_data.split('\n')

        t = list()
        plot_data = np.empty((window_size, 4))

        for i, line in enumerate(lines[-window_size:]):
            if len(line) > 1 and not line[0] == '#':
                h_l, h_r, f_l, f_r = line.split(' ')
                plot_data[i] = np.array([float(h_l), float(h_r), float(f_l), float(f_r)])
                t.append(float(i) * time_step)

        ax1.clear()

        ax1.plot(t[-window_size:], plot_data[:len(t), 0].tolist(), color='gold',
                 label='Hinde-Left')
        ax1.plot(t[-window_size:], plot_data[:len(t), 1].tolist(), color='red',
                 label='Hinde-Right')
        ax1.plot(t[-window_size:], plot_data[:len(t), 2].tolist(), color='blue',
                 label='Front-Left')
        ax1.plot(t[-window_size:], plot_data[:len(t), 3].tolist(), color='green',
                 label='Front-Right')

        ax1.set(xlabel='Time(s)', ylabel='Avg. Force')
        ax1.legend(bbox_to_anchor=(0.5, -0.125), loc='upper center', prop={'size': 12}, ncol=4)
        fig.subplots_adjust(bottom=0.2)

    def force_ratio_plot_animate(i):
        seconds = 5
        window_size = seconds * 30
        time_step = 0.033332
        graph_data = open(file_name, 'r').read()
        lines = graph_data.split('\n')

        t = list()
        plot_data = np.empty((window_size, 6))

        for i, line in enumerate(lines[-window_size:]):
            if len(line) > 1 and not line[0] == '#':
                h_l, h_r, f_l, f_r = line.split(' ')
                feet_forces = [float(h_l), float(h_r), float(f_l), float(f_r)]
                f_ratios = list()

                for j, f1 in enumerate(feet_forces):
                    for f2 in feet_forces[j+1:]:
                        if f1 == 0. or f2 == 0.:
                            f_ratios.append(1.0)
                        else:
                            if f1 >= f2:
                                f_ratios.append(f2/f1)
                            else:
                                f_ratios.append(f1/f2)

                plot_data[i] = np.array(f_ratios)
                t.append(float(i) * time_step)

        ax1.clear()

        ax1.plot(t[-window_size:], plot_data[:len(t), 0].tolist(), color='orange',
                 label='H-Left/H-Right')
        ax1.plot(t[-window_size:], plot_data[:len(t), 1].tolist(), color='green',
                 label='H-Left/F-Left')
        ax1.plot(t[-window_size:], plot_data[:len(t), 2].tolist(), color='chartreuse',
                 label='H-Left/F-Right')
        ax1.plot(t[-window_size:], plot_data[:len(t), 3].tolist(), color='purple',
                 label='H-Right/F-Left')
        ax1.plot(t[-window_size:], plot_data[:len(t), 4].tolist(), color='brown',
                 label='H-Right/F-Right')
        ax1.plot(t[-window_size:], plot_data[:len(t), 5].tolist(), color='cyan',
                 label='F-Left/F-Right')

        ax1.set(xlabel='Time(s)', ylabel='Force Ratio')
        ax1.set_yticks(np.arange(0, 1.1, step=0.2))
        ax1.legend(bbox_to_anchor=(0.5, -0.125), loc='upper center', prop={'size': 12}, ncol=3)
        fig.subplots_adjust(bottom=0.2)

    if plot_type == 'contact':
        _ = animation.FuncAnimation(fig, contact_plot_animate, interval=100)
    elif plot_type == 'force':
        _ = animation.FuncAnimation(fig, force_plot_animate, interval=100)
    elif plot_type == 'ratio':
        _ = animation.FuncAnimation(fig, force_ratio_plot_animate, interval=100)

    plt.show()


if __name__ == "__main__":
    main(sys.argv[1:])
