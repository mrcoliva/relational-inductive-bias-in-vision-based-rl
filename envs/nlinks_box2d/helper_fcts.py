import numpy as np


def normalize(foo):
    foo -= foo.mean()
    foo /= (foo.std() + 1e-8)
    return foo


def reset_log_file(hyper_dict):
    import os
    import pprint
    try:
        if os.path.exists("log.txt"):
            os.remove("log.txt")
    finally:
        f = open("log.txt", "w+")
        f.close()

    write_str_to_file('***************************')
    write_str_to_file(pprint.pformat(hyper_dict))
    write_str_to_file('***************************\n')

def get_file_name(hyper_dict):
    import os

    env_key = hyper_dict['env_key']
    env_para = hyper_dict['env_para']

    idx = 0
    while os.path.exists("./Model/model_" + str(env_key) + "-" + str(env_para) + "_%s.pkl" % idx):
        idx += 1

    file_name =  "./Model/model_" + str(env_key) + "-" + str(env_para) + "_%s.pkl" % (idx)

    return file_name

def write_str_to_file(msg):
    f = open("log.txt", 'a')
    f.write(msg)
    f.write("\n")
    f.close()


def duration_string(duration):
    min = int(duration / 60)
    sec = int(duration % 60)
    return "{0:2d}m:{1:2d}s".format(min, sec)


def angle_between(v1, v2):
    def unit_vector(vector):
        return vector / np.linalg.norm(vector)

    v1_u = unit_vector(v1)
    v2_u = unit_vector(v2)
    return np.arccos(np.clip(np.dot(v1_u, v2_u), -1.0, 1.0))


def plt_hpo_results(results):
    import time
    import matplotlib.pyplot as plt
    import os
    import skopt
    folder_str = time.asctime().replace(" ", "_").replace(":", "_")
    os.mkdir('./Log/HPO/' + folder_str)

    f = open('./Log/HPO/' + folder_str + '/results.txt', 'w')
    f.write(str(results))

    plt.close()

    _ = skopt.plots.plot_objective(results, size=4) # hast no attribute plots
    plt.savefig('./Log/HPO/' + folder_str + '/objective_plot.pdf')
    plt.close()

    skopt.plots.plot_convergence(results, size=2)
    plt.savefig('./Log/HPO/' + folder_str + '/convergence_plot.pdf')
    plt.close()

    f.close()

