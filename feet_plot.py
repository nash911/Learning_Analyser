import sys
import getopt
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation as animation
from matplotlib import style

style.use('seaborn')


def main(argv):
    character = 'salamander'
    plot_type = None
    file_name = None
    title = None
    window_seconds = 3

    try:
        opts, args = getopt.getopt(argv, "hcfrC:w:", ["contact", "force", "ratio", "character",
                                                      "window_size"])
    except getopt.GetoptError:
        print("plot.py -c/f")
        sys.exit(2)

    for opt, arg in opts:
        if opt == '-h':
            print("plot.py -c/f")
            sys.exit()
        elif opt in ("-c", "--contact"):
            plot_type = 'contact'
        elif opt in ("-C", "--character"):
            character = arg
        elif opt in ("-f", "--force"):
            plot_type = 'force'
        elif opt in ("-r", "--ratio"):
            plot_type = 'ratio'
        elif opt in ("-w", "--window_size"):
            window_seconds = int(arg)

    if plot_type == 'contact':
        file_name = '/home/nash/DeepMimic/output/feet_contact_force.dat'
        if character in ['salamander', 'cheetah']:
            title = 'Feet Contact Plot'
        elif character == 'snake':
            title = 'Spine Contact Plot'
    elif plot_type == 'force':
        file_name = '/home/nash/DeepMimic/output/avg_feet_contact_force.dat'
        title = 'Feet Avg. Force Plot'
    elif plot_type == 'ratio':
        file_name = '/home/nash/DeepMimic/output/avg_feet_contact_force.dat'
        title = 'Feet Force Ratio Plot'

    fig = plt.figure()
    fig.suptitle(title, fontsize=20)
    axs = fig.add_subplot(1, 1, 1)

    def contact_plot_salamander_animate(i):
        window_size = window_seconds * 30 * 20
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

        axs.clear()
        feet = ('Front-Right', 'Front-Left', 'Hinde-Right', 'Hinde-Left')
        y_pos = np.arange(1, len(feet) + 1)
        axs.set_yticks(y_pos)
        axs.set_yticklabels(feet)

        axs.plot(t[-window_size:], plot_data[:len(t), 0].tolist(), '.', color='gold',
                 label='Hinde-Left')
        axs.plot(t[-window_size:], plot_data[:len(t), 1].tolist(), '.', color='red',
                 label='Hinde-Right')
        axs.plot(t[-window_size:], plot_data[:len(t), 2].tolist(), '.', color='blue',
                 label='Front-Left')
        axs.plot(t[-window_size:], plot_data[:len(t), 3].tolist(), '.', color='green',
                 label='Front-Right')

        axs.set(xlabel='Time(s)', ylabel='Feet')
        axs.set_ylim(0.5, 5)
        axs.legend(loc='upper right')

    def contact_plot_snake_animate(i):
        colors = ['gold', 'red', 'blue', 'green', 'orange', 'black', 'purple', 'cyan', 'grey',
                  'violet', 'indigo']
        seconds = 3
        window_size = seconds * 30 * 20
        time_step = 0.033332 / 20.0
        graph_data = open(file_name, 'r').read()
        lines = graph_data.split('\n')

        t = list()
        num_plots = len(lines[1].split(' '))
        plot_data = np.zeros((window_size, num_plots))

        for i, line in enumerate(lines[-window_size:]):
            if len(line) > 1 and not line[0] == '#':
                ln = line.split(' ')
                plot_data[i] = np.array([[float(l) for l in ln]], dtype=bool)
                t.append(float(i) * time_step)
        plot_data *= np.array(list(reversed(range(1, num_plots+1))), dtype=np.int)

        axs.clear()
        feet = (['S-%d' % i for i in list(reversed(range(1, num_plots+1)))])
        y_pos = np.arange(1, len(feet) + 1)
        axs.set_yticks(y_pos)
        axs.set_yticklabels(feet)

        for i, lab in enumerate(reversed(feet)):
            axs.plot(t[-window_size:], plot_data[:len(t), i].tolist(), '.', color=colors[i],
                     label=lab)

        axs.set(xlabel='Time(s)', ylabel='Spine')
        axs.set_ylim(0.5, num_plots+1)
        axs.legend(bbox_to_anchor=(0.5, -0.125), loc='upper center', prop={'size': 12}, ncol=5)
        fig.subplots_adjust(bottom=0.2)

    def contact_plot_cheetah_animate(i):
        window_size = window_seconds * 30 * 20
        time_step = 0.033332 / 20.0
        graph_data = open(file_name, 'r').read()
        lines = graph_data.split('\n')

        t = list()
        plot_data = np.zeros((window_size, 4))

        for i, line in enumerate(lines[-window_size:]):
            if len(line) > 1 and not line[0] == '#':
                f_l, f_r, h_l, h_r = line.split(' ')
                plot_data[i] = np.array([float(f_l), float(f_r), float(h_l), float(h_r)],
                                        dtype=bool)
                t.append(float(i) * time_step)
        plot_data *= np.array([4, 3, 2, 1], dtype=np.int)

        axs.clear()
        feet = ('Hinde-Right', 'Hinde-Left', 'Front-Right', 'Front-Left')
        y_pos = np.arange(1, len(feet) + 1)
        axs.set_yticks(y_pos)
        axs.set_yticklabels(feet)

        axs.plot(t[-window_size:], plot_data[:len(t), 0].tolist(), '.', color='brown',
                 label='Front-Left')
        axs.plot(t[-window_size:], plot_data[:len(t), 1].tolist(), '.', color='salmon',
                 label='Front-Right')
        axs.plot(t[-window_size:], plot_data[:len(t), 2].tolist(), '.', color='grey',
                 label='Hinde-Left')
        axs.plot(t[-window_size:], plot_data[:len(t), 3].tolist(), '.', color='black',
                 label='Hinde-Right')

        axs.set(xlabel='Time(s)', ylabel='Feet')
        axs.set_ylim(0.5, 5)
        axs.legend(loc='upper right')

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

        axs.clear()

        axs.plot(t[-window_size:], plot_data[:len(t), 0].tolist(), color='gold',
                 label='Hinde-Left')
        axs.plot(t[-window_size:], plot_data[:len(t), 1].tolist(), color='red',
                 label='Hinde-Right')
        axs.plot(t[-window_size:], plot_data[:len(t), 2].tolist(), color='blue',
                 label='Front-Left')
        axs.plot(t[-window_size:], plot_data[:len(t), 3].tolist(), color='green',
                 label='Front-Right')

        axs.set(xlabel='Time(s)', ylabel='Avg. Force')
        axs.legend(bbox_to_anchor=(0.5, -0.125), loc='upper center', prop={'size': 12}, ncol=4)
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

        axs.clear()

        axs.plot(t[-window_size:], plot_data[:len(t), 0].tolist(), color='orange',
                 label='H-Left/H-Right')
        axs.plot(t[-window_size:], plot_data[:len(t), 1].tolist(), color='green',
                 label='H-Left/F-Left')
        axs.plot(t[-window_size:], plot_data[:len(t), 2].tolist(), color='chartreuse',
                 label='H-Left/F-Right')
        axs.plot(t[-window_size:], plot_data[:len(t), 3].tolist(), color='purple',
                 label='H-Right/F-Left')
        axs.plot(t[-window_size:], plot_data[:len(t), 4].tolist(), color='brown',
                 label='H-Right/F-Right')
        axs.plot(t[-window_size:], plot_data[:len(t), 5].tolist(), color='cyan',
                 label='F-Left/F-Right')

        axs.set(xlabel='Time(s)', ylabel='Force Ratio')
        axs.set_yticks(np.arange(0, 1.1, step=0.2))
        axs.legend(bbox_to_anchor=(0.5, -0.125), loc='upper center', prop={'size': 12}, ncol=3)
        fig.subplots_adjust(bottom=0.2)

    if plot_type == 'contact':
        if character == 'salamander':
            _ = animation.FuncAnimation(fig, contact_plot_salamander_animate, interval=100)
        elif character == 'snake':
            _ = animation.FuncAnimation(fig, contact_plot_snake_animate, interval=100)
        elif character == 'cheetah':
            _ = animation.FuncAnimation(fig, contact_plot_cheetah_animate, interval=100)
    elif plot_type == 'force':
        _ = animation.FuncAnimation(fig, force_plot_animate, interval=100)
    elif plot_type == 'ratio':
        _ = animation.FuncAnimation(fig, force_ratio_plot_animate, interval=100)

    plt.show()


if __name__ == "__main__":
    main(sys.argv[1:])
