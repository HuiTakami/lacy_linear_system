import numpy as np
import matplotlib.pyplot as plt


# Visualized generated dynamical system and do time sampling on it.
# input
# (dim, ) initial point
def generate_ds(A_matrix, initial_point, dt, max_iter, do_plot, A_rep, rep_plot):
    dim = len(initial_point)
    eig_A = np.linalg.eig(A_matrix)[0]
    if not np.all(eig_A < 0):
        print('This dynamical system is not stable')
    cur_step = 1
    cur_position = initial_point.copy()
    epi = 10 ** (-6)
    samples = np.zeros((max_iter, dim))
    samples[0] = initial_point
    while cur_step < max_iter and not np.all(cur_position < epi):
        cur_velo = A_matrix @ cur_position
        cur_position += cur_velo * dt
        samples[cur_step] = cur_position
        cur_step += 1

    print('iteration terminated, final step is {}'.format(cur_step))
    samples = samples[:cur_step].T
    if do_plot:
        if dim == 2:
            plt.plot(samples[0], samples[1], label='generated trajectory')
            plt.scatter(initial_point[0], initial_point[1], c='r', label='Initial Point')

    if rep_plot == 1:
        cur_position = initial_point.copy()
        cur_step_1 = 1
        reproduced_traj = np.zeros((samples.shape[1], dim))
        reproduced_traj[0] = initial_point
        while cur_step_1 < cur_step and not np.all(cur_position < epi):
            cur_velo = A_rep @ cur_position
            cur_position += cur_velo * dt
            reproduced_traj[cur_step_1] = cur_position
            cur_step_1 += 1
        if dim == 2:
            reproduced_traj = reproduced_traj.T
            plt.plot(reproduced_traj[0], reproduced_traj[1], label='reproduced trajectory')
    plt.legend()
    plt.show()
    return samples


def generate_nl_ds(A_1, A_2, initial_point, dt, max_iter, do_plot):
    dim = len(initial_point)
    eig_A_1 = np.linalg.eig(A_1)[0]
    eig_A_2 = np.linalg.eig(A_2)[0]
    if not np.all(eig_A_1 < 0) and not np.all(eig_A_2 < 0):
        print('This DS is not stable')

    # initialize the iteration
    cur_step = 1
    cur_position = initial_point.copy()
    epi = 10 ** (-6)
    samples = np.zeros((max_iter, dim))
    samples[0] = initial_point

    # Start iteration
    while cur_step < max_iter and not np.all(cur_position < epi):
        gamma_1 = activate_func(cur_position)
        cur_velo = gamma_1 * (A_1 @ cur_position) + (1 - gamma_1) * (A_2 @ cur_position)
        print(gamma_1)
        cur_position += cur_velo * dt
        samples[cur_step] = cur_position
        cur_step += 1

    print('iteration terminated, final step is {}'.format(cur_step))
    samples = samples[:cur_step].T
    if do_plot:
        if dim == 2:
            plt.plot(samples[0], samples[1])
            plt.scatter(initial_point[0], initial_point[1], c='r')

    plt.show()

    return samples


def activate_func(x):
    return 1 / (1 + np.exp(3 - 10 * np.linalg.norm(x)))


if __name__ == '__main__':
    A_1 = np.array([[10, 1],[-1, -3]])
    A_2 = np.array([[-2, 1], [-1, -10]])
    initial_point = np.array([0.1, 0.1])
    dt = 0.01
    generate_nl_ds(A_1, A_2, initial_point, dt, 1000, 1)
