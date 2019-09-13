import sys, getopt
import numpy as np
import json
import os
import re
import matplotlib.pyplot as plt
import matplotlib.animation as animation
from matplotlib import style

style.use('seaborn')


def main(argv):
    reduced_reference_file = '../data/reduced_motion/'
    learned_activation_file = 'activation.dat'

    lc = ['xkcd:red', 'xkcd:blue', 'xkcd:green', 'xkcd:brown', 'xkcd:pink',
          'xkcd:purple', 'xkcd:orange', 'xkcd:magenta', 'xkcd:tan', 'xkcd:black',
          'xkcd:cyan', 'xkcd:gold', 'xkcd:dark green', 'xkcd:cream',
          'xkcd:lavender', 'xkcd:turquoise', 'xkcd:dark blue', 'xkcd:violet',
          'xkcd:beige', 'xkcd:salmon', 'xkcd:olive', 'xkcd:light brown',
          'xkcd:hot pink', 'xkcd:dark red', 'xkcd:sand', 'xkcd:army green',
          'xkcd:dark grey', 'xkcd:crimson', 'xkcd:eggplant', 'xkcd:coral']

    try:
        opts, args = getopt.getopt(argv, "h r:a:",
                                   ["ref", "act"])
    except getopt.GetoptError:
        print("activation_plot.py -r <reference_file> -a <activation_file>")
        sys.exit(2)

    for opt, arg in opts:
        if opt == '-h':
            print("activation_plot.py -r <reference_file> -a <activation_file>")
            sys.exit()
        elif opt in ("-r", "--ref"):
            reduced_reference_file = arg
        elif opt in ("-a", "--act"):
            learned_activation_file = arg

    def create_plot(title, activations, resolution):
        fig = plt.figure()
        fig.suptitle(title, fontsize=15)
        ax = fig.add_subplot(1, 1, 1)

        time = np.full((activations[0].shape[0]), resolution)
        time = np.cumsum(time)

        for i, act in enumerate(activations):
            ax.plot(time, act, color=lc[i], label=('Comp-%s' % (i+1)))

        ax.set(xlabel='Time(s)', ylabel='Activation')
        ax.legend()


    with open(reduced_reference_file) as f:
        data = json.load(f)

    reduced_ref_act =  np.array(data['U'])
    redu_refs = np.hsplit(reduced_ref_act, reduced_ref_act.shape[1])

    resolution = np.array(data['Frames'])[0][0]

    create_plot("Reference Activations", redu_refs, resolution)

    with open(learned_activation_file) as f:
        lines = f.read().splitlines()
    del lines[0]

    activation = []
    for line in lines:
        line = re.sub('[[]', '', line)
        line = re.sub('[]]', '', line)
        activation.append(line.split())

    learned_ref_act =  np.array(activation, dtype=reduced_ref_act.dtype)
    lrnd_refs = np.hsplit(learned_ref_act, learned_ref_act.shape[1])

    create_plot("Learned Lower-Dim Activations", lrnd_refs, resolution)

    plt.show()

if __name__ == "__main__":
    main(sys.argv[1:])
