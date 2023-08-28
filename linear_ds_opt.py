import cvxpy as cp
import numpy as np
from generate_stable_ds import generate_ds


def linear_ds_opt(Data, delta, dt):
    dim, nb_data = Data.shape
    x_k = Data[:, :nb_data-1]
    x_k_1 = Data[:, 1:]

    # define the variable mentioned in paper
    Z_1 = cp.Variable(1)
    Z_2 = cp.Variable((dim, dim))
    Z_3 = cp.Variable((2*dim, 2*dim), symmetric=True)

    # define numeral component
    L_x = np.eye(dim)
    R_x_1 = np.eye(dim)
    X_1 = x_k_1 @ calculate_right_inv(x_k)
    constr_1 = cp.hstack([-L_x, L_x @ X_1]) @ Z_3 @ cp.vstack([np.zeros((dim, dim)), R_x_1])
    constr_2 = cp.hstack([np.zeros((dim ,dim)), np.eye(dim)]) @ Z_3 @ cp.vstack([np.zeros((dim, dim)), np.eye(dim)])
    constr_3 = cp.hstack([np.eye(dim), np.zeros((dim ,dim))]) @ Z_3 @ cp.vstack([np.eye(dim), np.zeros((dim, dim))])

    # implement every constrain
    constrains = []
    constrains += [constr_1 == Z_2]
    constrains += [constr_2 - delta * np.eye(dim) == constr_3]
    constrains += [Z_1 >= cp.norm(Z_2,'fro')]
    constrains += [Z_3 >> 0]

    # define object and constrains
    c_x = np.zeros(4*(dim**2) + dim**2 + 1)
    c_x[0] = 1
    Objective = cp.hstack([cp.vec(Z_1), cp.vec(Z_2), cp.vec(Z_3)]).T @ c_x
    # Objective = Z_1
    prob = cp.Problem(cp.Minimize(Objective), constrains)

    prob.solve(solver=cp.MOSEK, verbose=True)

    Z_3_value = Z_3.value
    Q = Z_3.value[:dim, dim:]
    P = Z_3.value[dim:, dim:]

    A = (Q @ np.linalg.inv(P) - np.eye(dim)) / dt # dt is 0.01
    print(A)
    return A, P


def calculate_right_inv(V):
    return V.T @ np.linalg.pinv(V @ V.T)


if __name__ == '__main__':
    A_matrix = np.array([[-1, 3], [0, -0.5]])
    Data = np.load('problematic_data.npy')[0:23]
    init_point = np.array([0.1, 0.1], dtype=np.float64)
    dt = 0.01
    samples = generate_ds(A_matrix, init_point, dt, 51, 0, [], 0)
    delta = 10 ** (-3)
    A_opt, P = linear_ds_opt(samples, delta, dt)
    print('A matrix we set is {}'.format(A_matrix))
    print('the learning A result is ', A_opt)
    print('the learning P result is', P)
    print(np.linalg.eig(A_matrix)[0])
    print(np.linalg.eig(A_opt)[0])
    samples = generate_ds(A_matrix, init_point, dt, 1000, 1, A_opt, 1)
