import sys
import getopt
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation as animation
from matplotlib import style

style.use('seaborn')


def main(argv):
    file_name = '/home/nash/DeepMimic/output/feet_contact_force.dat'

    fig = plt.figure()
    fig.suptitle('Feet Contact Plot', fontsize=20)
    ax1 = fig.add_subplot(1, 1, 1)

    def animate(i):
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

                plot_data[i] = np.array([float(h_r), float(h_l), float(f_r), float(f_l)],
                                        dtype=bool)
                t.append(float(i) * time_step)
        plot_data *= np.array([1, 2, 3, 4], dtype=np.int)

        ax1.clear()
        feet = ('Hinde-Right', 'Hinde-Left', 'Front-Right', 'Front-Left')
        y_pos = np.arange(1, len(feet) + 1)
        ax1.set_yticks(y_pos)
        ax1.set_yticklabels(feet)

        ax1.plot(t[-window_size:], plot_data[:len(t), 0].tolist(), '.', color='red',
                 label='Hinde-Right')
        ax1.plot(t[-window_size:], plot_data[:len(t), 1].tolist(), '.', color='gold',
                 label='Hinde-Left')
        ax1.plot(t[-window_size:], plot_data[:len(t), 2].tolist(), '.', color='green',
                 label='Front-Right')
        ax1.plot(t[-window_size:], plot_data[:len(t), 3].tolist(), '.', color='blue',
                 label='Front-Left')

        ax1.set(xlabel='Time(s)', ylabel='Feet')
        ax1.set_ylim(0.5, 5)
        ax1.legend(loc='upper right')

    ani = animation.FuncAnimation(fig, animate, interval=100)
    plt.show()


if __name__ == "__main__":
    main(sys.argv[1:])
