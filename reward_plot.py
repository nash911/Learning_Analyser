import sys
import getopt
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation as animation
from matplotlib import style

style.use('seaborn')


def main(argv):
    reward = True
    root_f_vel = True
    root_l_vel = True
    tau = True
    part_rot = True
    window_seconds = 5

    try:
        opts, args = getopt.getopt(argv, "hflprtw:", ["root_f_vel", "root_l_vel", "part_rot",
                                                      "reward", "tau", "window_size"])
    except getopt.GetoptError:
        print("Usage: reward_plot.py [-f | --root_f_vel] \n"
              "                      [-l | --root_l_vel] \n"
              "                      [-p | --part_rot] \n"
              "                      [-r | --reward] \n"
              "                      [-t | --tau] \n"
              "                      [-w | --window_size] \n"
              )
        sys.exit()

    for opt, arg in opts:
        if opt == '-h':
            print("Usage: reward_plot.py [-f | --root_f_vel] \n"
                  "                      [-l | --root_l_vel] \n"
                  "                      [-p | --part_rot] \n"
                  "                      [-r | --reward] \n"
                  "                      [-t | --tau] \n"
                  "                      [-w | --window_size] \n"
                  )
            sys.exit()
        elif opt in ("-f", "--root_f_vel"):
            root_f_vel = False
        elif opt in ("-l", "--root_l_vel"):
            root_l_vel = False
        elif opt in ("-p", "---part_rot"):
            part_rot = False
        elif opt in ("-r", "--reward"):
            reward = False
        elif opt in ("-t", "--tau"):
            tau = False
        elif opt in ("-w", "--window_size"):
            window_seconds = int(arg)

    file_name = '/home/nash/DeepMimic/output/reward_terms.dat'
    title = 'Reward Terms Plot'

    fig = plt.figure()
    fig.suptitle(title, fontsize=20)
    ax1 = fig.add_subplot(1, 1, 1)

    def reward_values(i):
        window_size = window_seconds * 30
        time_step = 0.033332
        graph_data = open(file_name, 'r').read()
        lines = graph_data.split('\n')

        t = list()
        plot_data = np.empty((window_size, 7))

        for i, line in enumerate(lines[-window_size:]):
            if len(line) > 1 and not line[0] == '#':
                rew, f_vel, tau_r, p_rot, l_vel, x, rew_2 = line.split(' ')
                plot_data[i] = np.array([float(rew), float(f_vel), float(tau_r), float(p_rot),
                                         float(l_vel), float(x), float(rew_2)])
                t.append(float(i) * time_step)

        ax1.clear()
        num_cols = 0

        if reward:
            ax1.plot(t[-window_size:], plot_data[:len(t), 0].tolist(), color='red', label='Reward')
            num_cols += 1
        if root_f_vel:
            ax1.plot(t[-window_size:], plot_data[:len(t), 1].tolist(), color='blue',
                     label='Root-F-Vel')
            num_cols += 1
        if tau:
            ax1.plot(t[-window_size:], plot_data[:len(t), 2].tolist(), color='green', label='Tau')
            num_cols += 1
        if part_rot:
            ax1.plot(t[-window_size:], plot_data[:len(t), 3].tolist(), color='gold',
                     label='Part-Rot')
            num_cols += 1
        if root_l_vel:
            ax1.plot(t[-window_size:], plot_data[:len(t), 4].tolist(), color='brown',
                     label='Root-L-Vel')
            num_cols += 1

        ax1.plot(t[-window_size:], plot_data[:len(t), 5].tolist(), color='purple', label='Fwd-Lat')
        ax1.plot(t[-window_size:], plot_data[:len(t), 6].tolist(), color='black', label='Reward-2')
        num_cols = 4

        ax1.set(xlabel='Time(s)', ylabel='Reward Values')
        ax1.legend(bbox_to_anchor=(0.5, -0.125), loc='upper center', prop={'size': 12},
                   ncol=num_cols)
        fig.subplots_adjust(bottom=0.2)

    _ = animation.FuncAnimation(fig, reward_values, interval=100)
    plt.show()


if __name__ == "__main__":
    main(sys.argv[1:])
