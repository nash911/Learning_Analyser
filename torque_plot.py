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
    avg_trq = False
    indiv_trq = False
    trq_rew = False
    joint_rew = False
    window_seconds = 5
    window_size = 30 * window_seconds
    time_step = 0.033332

    indiv_trq_only = False
    joint_rew_only = False

    # colors = ['gold', 'violet', 'blue', 'green', 'salmon', 'black', 'purple', 'cyan', 'red',
    #           'grey', 'indigo', 'lavender']
    colors = ['red', 'blue', 'green', 'orange', 'black', 'purple', 'cyan', 'grey', 'violet', 'gold',
              'indigo', 'brown', 'pink', 'magenta', 'tan', 'darkgreen', 'lavender',
              'turquoise', 'darkblue', 'beige', 'salmon', 'olive', 'hotpink',
              'darkred', 'sand', 'armygreen', 'darkgrey', 'crimson', 'eggplant', 'coral']

    file_name = '/home/nash/DeepMimic/output/avg_torque.dat'

    try:
        opts, args = getopt.getopt(argv, "haijrw:", ["avg_torque", "individual_torque",
                                                     "joint_reward", "torque_reward",
                                                     "window_size"])
    except getopt.GetoptError:
        print("torque_plot.py -a -i -j -r -w [--avg_torque --individual_torque --joint_reward "
              "--torque_reward --window_size]")
        sys.exit(2)

    for opt, arg in opts:
        if opt == '-h':
            print("torque_plot.py -a -i -j -r -w [--avg_torque --individual_torque --joint_reward "
                  "--torque_reward --window_size]")
            sys.exit()
        elif opt in ("-a", "--avg_torque"):
            avg_trq = True
        elif opt in ("-i", "--individual_torque"):
            indiv_trq = True
        elif opt in ("-j", "--joint_reward"):
            joint_rew = True
        elif opt in ("-r", "--torque_reward"):
            trq_rew = True
        elif opt in ("-w", "--window_size"):
            window_seconds = int(arg)

    if not (avg_trq or trq_rew or joint_rew) and indiv_trq:
        indiv_trq_only = True

    if not (avg_trq or trq_rew or indiv_trq) and joint_rew:
        joint_rew_only = True

    graph_data = open(file_name, 'r').read()
    lns = graph_data.split('\n')
    data_dim = len(lns[1].split(' '))

    if indiv_trq_only or joint_rew_only:
        num_plots = int((data_dim - 2) / 2)
    else:
        num_plots = 0
        if avg_trq:
            num_plots += 1
        if indiv_trq:
            num_plots += 1
        if trq_rew:
            num_plots += 1
        if joint_rew:
            num_plots += 1

    fig, axs = plt.subplots(num_plots, sharey=(True if indiv_trq_only or joint_rew_only else False),
                            sharex=True)
    fig.suptitle('Torque Plot', fontsize=20)

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

        plot_data = np.empty((window_size, data_dim))

        for i, line in enumerate(lines[-window_size:]):
            if len(line) > 1 and not line[0] == '#':
                data_point = line.split(' ')
                plot_data[i] = np.array([float(dp) for dp in data_point])
                t.append(float(i) * time_step)

        if num_plots > 1:
            for ax in axs:
                ax.clear()
        else:
            axs.clear()

        sub_plot = 0

        if avg_trq:
            plot_axs(sub_plot).plot(t[-window_size:], plot_data[:len(t), 0].tolist(), color='red',
                                    label='Avg. Torque')
            plot_axs(sub_plot).set(xlabel='Time(s)', ylabel='Avg. Torque')
            plot_axs(sub_plot).legend(loc='upper right')
            sub_plot += 1

        if trq_rew:
            plot_axs(sub_plot).plot(t[-window_size:], plot_data[:len(t), 1].tolist(), color='green',
                                    label='Torque Reward')
            plot_axs(sub_plot).set(xlabel='Time(s)', ylabel='Avg. Reward')
            plot_axs(sub_plot).set_yticks(np.arange(0, 1.1, step=0.2))
            plot_axs(sub_plot).legend(loc='upper right')
            sub_plot += 1

        if indiv_trq:
            if indiv_trq_only:
                sub_plot = 0
                for i, j in enumerate(range(2, data_dim, 2)):
                    plot_axs(i).plot(t[-window_size:], plot_data[:len(t), j].tolist(),
                                     color=colors[i], label=('Joint-' + str(i+1)))
                    if i == np.ceil(num_plots/2):
                        plot_axs(i).set(ylabel='Torque Norm.')
                    # plot_axs(i).legend(bbox_to_anchor=(0.5, -0.08*num_plots), loc='upper center',
                    #                    ncol=5, prop={'size': 12})
                    plot_axs(i).legend(loc='center left', bbox_to_anchor=(1, 0.5),
                                       prop={'size': 12})
                plot_axs(num_plots-1).set(xlabel='Time(s)')
                fig.subplots_adjust(right=0.85)
            else:
                for i, j in enumerate(range(2, data_dim, 2)):
                    plot_axs(sub_plot).plot(t[-window_size:], plot_data[:len(t), j].tolist(),
                                            color=colors[i], label=('Joint-' + str(i+1)))
                plot_axs(sub_plot).set(xlabel='Time(s)', ylabel='Torque Norm.')
                plot_axs(sub_plot).legend(bbox_to_anchor=(0.5, -0.08*num_plots), loc='upper center',
                                          ncol=5, prop={'size': 12})
                # axs[sub_plot].legend(loc='center left', bbox_to_anchor=(1, 0.5),
                #                      prop={'size': 12})
                fig.subplots_adjust(bottom=0.15)
                sub_plot += 1

        if joint_rew:
            for i, j in enumerate(range(3, data_dim, 2)):
                plot_axs(sub_plot).plot(t[-window_size:], plot_data[:len(t), j].tolist(),
                                        color=colors[i], label=('Joint-' + str(i+1)))
            plot_axs(sub_plot).set(xlabel='Time(s)', ylabel='Torque Inv')
            plot_axs(sub_plot).set_yticks(np.arange(0, 1.1, step=0.2))
            plot_axs(sub_plot).legend(bbox_to_anchor=(0.5, -0.08*num_plots), loc='upper center',
                                      ncol=5, prop={'size': 12})
            fig.subplots_adjust(bottom=0.15)
            sub_plot += 1

    _ = animation.FuncAnimation(fig, animate, interval=100)
    plt.show()


if __name__ == "__main__":
    main(sys.argv[1:])
