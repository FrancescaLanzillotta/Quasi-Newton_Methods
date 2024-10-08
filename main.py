import pprint

import pycutest
import numpy as np
import time
import datetime

import scipy.optimize


def armijo_step(problem, x, d, gamma, delta, g):
    """
    This function determines the length of the step as to satisfy the Armijo rule.

    :param problem: current pycutest problem
    :param x: starting point
    :param d: descent direction
    :param gamma: parameter used in the Armijo condition, with 0 < gamma < 1
    :param delta: reduction factor of the step at each iteration, with 0 < delta < 1
    :param g: problem's gradient computed in x

    :return: optimal step that satisfies the Armijo condition using specified parameters
    """

    f = problem.obj(x)
    alpha = 1
    while problem.obj(x + alpha * d) > f + gamma * alpha * np.dot(g, d) and alpha > 1e-6:
        alpha = alpha * delta
    return alpha


def armijo_grad_descent(problem, x, eps, gamma, delta, evol=False):
    """
    This function implements the gradient descent method with Armijo line search. When the norm of the gradient falls
    below a certain threshold eps, the algorithm stops and returns the stationary point found.

    :param problem: current pycutest problem
    :param x: starting point
    :param eps: stopping threshold
    :param gamma: parameter used in the armijo_step function
    :param delta: reduction factor used in the armijo_step function

    :return: stationary point, number of iterations
    """
    it = 0
    satisfied = False
    evolution = []
    start = time.time()
    while not satisfied:

        grad = problem.grad(x)
        norm = np.linalg.norm(grad)
        if evol:
            evolution.append(p.obj(x))
        if norm <= eps:
            satisfied = True
        else:

            alpha = armijo_step(problem, x, -grad, gamma, delta, grad)
            # print("Armijo step found: ", alpha)
            x = x - alpha * grad
            end = time.time()

            # if evol:
            #     evolution.append(end - start)
            it += 1

    if evol:
        return x, it, evolution
    else:
        return x, it


def strong_wolfe_step(problem, x, d, gamma, sigma, phi0, dot_phi0):
    """
    This function determines the length of the step as to satisfy the strong Wolfe conditions.

    :param problem: current pycutest problem
    :param x: starting point
    :param d: descent direction
    :param gamma: parameter used in the first Wolfe condition (Armijo rule), with 0 < gamma < 1/2
    :param sigma: parameter used in the second Wolfe condition (curvature condition), with gamma < sigma < 1
    :param phi0: problem's objective function computed in x
    :param dot_phi0: scalar product between the problem's gradient and the descent direction

    :return: optimal step that satisfies the strong Wolfe conditions using specified parameters
    """
    lb = 0
    ub = np.inf
    alpha = 1
    satisfied = False

    while (not satisfied) and (alpha > 1e-6) and (ub - lb) > 1e-6:

        if problem.obj(x + alpha * d) > phi0 + gamma * alpha * dot_phi0:
            ub = alpha
            alpha = (lb + ub) / 2

        else:  # Armijo condition is satisfied
            dot_phi_alpha = np.dot(problem.grad(x + alpha * d), d)

            if dot_phi_alpha < sigma * dot_phi0:
                lb = alpha
                alpha = min(((lb + ub) / 2), 2 * alpha)
            elif dot_phi_alpha > sigma * abs(dot_phi0):  # weak Wolfe condition is satisfied
                ub = alpha
                alpha = (lb + ub) / 2
            elif abs(dot_phi_alpha) <= sigma * abs(dot_phi0):  # strong Wolfe condition is satisfied
                satisfied = True

    return alpha


def strong_wolfe_grad_descent(problem, x, eps, gamma, sigma):
    """
    This function implements the gradient descent method with step length satisfying the Strong Wolfe conditions. When
    the norm of the gradient falls below a certain threshold eps, the algorithm stops and returns the stationary point
    found.

    :param problem: current pycutest problem
    :param x: starting point
    :param eps: stopping threshold
    :param gamma: parameter used in the strong_wolfe_step function
    :param sigma: parameter used in the strong_wolfe_step function

    :return: stationary point, number of iterations
    """
    satisfied = False

    it = 0
    while not satisfied:
        grad = problem.grad(x)
        if np.linalg.norm(grad) <= eps:
            satisfied = True

        else:
            w_step = strong_wolfe_step(problem, x, -grad, gamma, sigma, problem.obj(x), np.dot(grad, -grad))
            # print("Wolfe step found: ", w_step)
            x = x - w_step * grad
            it += 1

    return x, it


def qn_method(problem, x, eps, hess, gamma, sigma, update_rule, direct=True, evol=False):
    """
    This function implements the quasi-Newton method using a specified update rule and a step size satisfying the strong
    Wolfe conditions. When the norm of the gradient falls below a certain threshold eps, the algorithm stops and returns
    the stationary point found.

    :param problem: current pycutest problem
    :param x: starting point
    :param eps: stopping threshold
    :param hess: hessian approximation
    :param gamma: parameter used in the strong_wolfe_step function
    :param sigma: parameter used in the strong_wolfe_step function
    :param update_rule: method to use to update the hessian matrix approximation
    :param direct: whether we're approximating the hessian matrix or its inverse

    :return: stationary point, number of iterations
    """

    it = 0
    start = time.time()
    satisfied = False
    grad = problem.grad(x)
    evolution = []
    d_time = []
    w_time = []
    h_time = []
    while not satisfied:
        obj = problem.obj(x)
        if evol:
            evolution.append(obj)
        if np.linalg.norm(grad) <= eps:
            satisfied = True
        else:

            if direct:
                # d_start = time.time()
                d = np.linalg.solve(hess, -grad)
                # d_end = time.time()

            else:
                # d_start = time.time()
                d = - np.dot(hess, grad)
                # d_end = time.time()


            # w_start = time.time()
            opt_step = strong_wolfe_step(problem, x, d, gamma, sigma, obj, np.dot(grad, d))
            # print(f"It-{it} obj eval: {round(obj, 9)} \t grad norm: {round(np.linalg.norm(grad), 9)} \t opt_step: {round(opt_step, 7)}")
            # w_end = time.time()

            new_x = x + opt_step * d
            new_grad = problem.grad(new_x)

            y = new_grad - grad
            s = new_x - x

            # h_start = time.time()
            hess = update_rule(hess, y, s, direct)
            # h_end = time.time()

            # eigs = list(np.linalg.eig(hess)[0])
            # eigs.sort()
            # print(f"It-{it} condition number: {abs(eigs[-1]) / abs(eigs[0])}")

            x = new_x
            grad = new_grad
            # end = time.time()

            # if evol:
            #     evolution.append(end - start)
            #     d_time.append(d_end - d_start)
            #     w_time.append(w_end - w_start)
            #     h_time.append(h_end - h_start)

            # if len(evolution) > 1:
            #     print(f"It-{it}: {round((h_end - h_start), 3)} s spent in computing approx")
            # else:
            #     print(
            #         f"It-{it}: {round((h_end - h_start), 3)} s spent in computing approx")
            it += 1

    if evol:
        # print(f"\n\nComputing d: {np.average(d_time)}\n"
        #       f"Choose optimal step: {np.average(w_time)}\n"
        #       f"Approximating h: {np.average(h_time)}")
        return x, it, evolution
    else:
        return x, it


def Broyden(hess, y, s, direct):
    """
    This function updates the hessian matrix approximation using Broyden's formula (rank-1 update rule).

    :param hess: current hessian approximation to update, size n x n
    :param y: vector computed as problem.grad(new_x) - problem.grad(x)
    :param s: vector computed as new_x - x
    :param direct: whether we're approximating the hessian matrix or its inverse

    :return: updated matrix given specified parameters
    """
    if direct:
        return hess + (np.dot(np.array([y - np.dot(hess, s)]).transpose(), np.array([s])) / np.inner(s, s))
    else:

        hy = np.dot(hess, y)
        return hess + (np.dot(np.array([s - hy]).transpose(), np.array([np.dot(s, hess)]))) / np.dot(s, hy)


def DFP(hess, y, s, direct):
    """
    This function updates the hessian matrix approximation using the DFP formula (rank-2 update rule).

    :param hess: current hessian approximation to update, size n x n
    :param y: vector computed as problem.grad(new_x) - problem.grad(x)
    :param s: vector computed as new_x - x
    :param direct: whether we're approximating the hessian matrix or its inverse

    :return: updated matrix given specified parameters
    """
    if direct:

        sy = np.dot(s, y)  # scalar value
        yhs = y - np.dot(hess, s)  # row vector

        first = (np.dot(np.array([yhs]).transpose(), np.array([y])) + np.dot(np.array([y]).transpose(),
                                                                             np.array([yhs]))) / sy
        second = (np.dot(s, yhs) * np.dot(np.array([y]).transpose(), np.array([y]))) / sy ** 2
        return hess + first - second

    else:

        first = np.dot(np.array([s]).transpose(), np.array([s])) / np.dot(s, y)
        second = (np.linalg.multi_dot([hess, np.array([y]).transpose(), np.array([y]), hess])
                  / np.linalg.multi_dot([np.array([y]), hess, np.array([y]).transpose()]))
        return hess + first - second


def BFGS(hess, y, s, direct):
    """
    This function updates the hessian matrix approximation using the BFGS formula (rank-2 update rule).

    :param hess: current hessian approximation to update, size n x n
    :param y: vector computed as problem.grad(new_x) - problem.grad(x)
    :param s: vector computed as new_x - x
    :param direct: whether we're approximating the hessian matrix or its inverse

    :return: updated matrix given specified parameters
    """

    if direct:
        first = np.dot(np.array([y]).transpose(), np.array([y])) / np.dot(s, y)
        second = np.linalg.multi_dot([hess, np.array([s]).transpose(), np.array([s]), hess]) / np.linalg.multi_dot(
            [np.array([s]), hess, np.array([s]).transpose()])

        return hess + first - second
    else:
        sy = np.dot(s, y)  # scalar value
        first = ((1 + (np.linalg.multi_dot([np.array([y]), hess, np.array([y]).transpose()]) / sy))
                 * np.dot(np.array([s]).transpose(), np.array([s]) / sy))
        second = (np.linalg.multi_dot([np.array([s]).transpose(), np.array([y]), hess]) +
                  np.linalg.multi_dot([hess, np.array([y]).transpose(), np.array([s])])) / sy

        return hess + first - second


def get_problem(prob_name):
    params = None
    if prob_name in ["BOX", "NONDQUAR", "POWER"]:
        params = {'N': 100}
    elif prob_name == "CRAGGLVY":
        params = {'M': 499}
    elif prob_name == "FMINSRF2":
        params = {'P': 11}

    return pycutest.import_problem(p_name, sifParams=params)


if __name__ == '__main__':

    # probs = pycutest.find_problems(objective="other", constraints="unconstrained", regular=True)

    armijo_parameters = {"ARWHEAD": (0.5, 0.9),         # prob_name: (gamma, delta)
                         "BOX": (0.5, 0.4),
                         "COSINE": (0.5, 0.6),
                         "CRAGGLVY": (0.6, 0.9),
                         "DIXMAANA1": (0.1, 0.5),
                         "EG2": (0.1, 0.1),
                         "FMINSRF2": (0.1, 0.3),
                         "NONDQUAR": (0.6, 0.9),
                         "POWER": (0.1, 0.2),
                         "TOINTGOR": (0.5, 0.9)}

    qn_parameters = {"ARWHEAD": (0.1, 0.7),         # prob_name: (gamma, sigma)
                     "BOX": (0.2, 0.8),
                     "COSINE": (0.3, 0.6),
                     "CRAGGLVY": (0.2, 0.7),
                     "DIXMAANA1": (0.1, 0.8),
                     "EG2": (0.4, 0.6),
                     "FMINSRF2": (0.3, 0.7),
                     "NONDQUAR": (0.1, 0.7),
                     "POWER": (0.05, 0.7),
                     "TOINTGOR": (0.3, 0.6)}

    problems = ["ARWHEAD",      # f^* ==
                "BOX",          # f^* ==
                "COSINE",       # f^* ==
                "CRAGGLVY",     # f^* ==
                "DIXMAANA1",    # f^* ==
                "EG2",          # f^* ==
                "FMINSRF2",     # f^* ==
                "NONDQUAR",     # f^* ==
                "POWER",        # f^* ==
                "TOINTGOR"]     # f^* ==

    # p_name = "ARWHEAD"
    # fileName = "res/{}-evaluations-{:%Y%m%d-%H%M%S}.txt".format(p_name, datetime.datetime.now())
    # save = False
    # eps = 0.01
    #
    # p = get_problem(p_name)
    #
    # print(f"{p_name} - {{N: {p.n}, M: {p.m} }}\n"
    #       f"Gradient tolerance: {eps}\n")
    #
    # if save:
    #     f = open(fileName, 'w')
    #     f.write(f"{p_name} - {{N: {p.n}, M: {p.m} }}\n"
    #             f"Gradient tolerance: {eps}\n")
    #
    #                               ##################### Armijo #####################
    # gamma = armijo_parameters[p_name][0]
    # delta = armijo_parameters[p_name][1]
    #
    # start = time.time()
    # stat_point, it = armijo_grad_descent(p, p.x0, eps, gamma, delta)
    # end = time.time()
    # stats = p.report()
    # print(f"############## Armijo ##############\n"
    #       f"Gamma: {gamma}\tDelta: {delta}\n\n"
    #       f"Objective Value: {p.obj(stat_point)}\n"
    #       f"Gradient norm: {np.linalg.norm(p.grad(stat_point))}\n"
    #       f"Iterations: {it}\n"
    #       f"Total obj evaluations: {stats['f']}\n"
    #       f"Total grad evaluations: {stats['g']}\n"
    #       f"Runtime: {end - start} s\n")
    # #
    # if save:
    #     f.write(f"\n############## Armijo ##############\n"
    #       f"Gamma: {gamma}\tDelta: {delta}\n\n"
    #       f"Objective Value: {p.obj(stat_point)}\n"
    #       f"Gradient norm: {np.linalg.norm(p.grad(stat_point))}\n"
    #       f"Iterations: {it}\n"
    #       f"Total obj evaluations: {stats['f']}\n"
    #       f"Total grad evaluations: {stats['g']}\n"
    #       f"Runtime: {end - start} s\n")

                              ##################### Quasi-Newton #####################
    # gamma = qn_parameters[p_name][0]
    # sigma = qn_parameters[p_name][1]
    # updates = [Broyden, DFP, BFGS]
    # updates = [BFGS]
    #
    # for update, dir in ((update, dir) for update in updates for dir in [True]):
    #     p = get_problem(p_name)
    #     start = time.time()
    #     stat_point, it = qn_method(p, p.x0, eps, np.identity(p.n), gamma, sigma,
    #                                update, dir)
    #     end = time.time()
    #     stats = p.report()
    #     if dir:
    #         spec = "direct"
    #     else:
    #         spec = "indirect"
    #
    #     print(f"############## Quasi-Newton ({spec} {update.__name__}) ##############\n"
    #           f"Gamma: {gamma}\t Sigma: {sigma}\n\n"
    #           f"Objective Value: {p.obj(stat_point)}\n"
    #           f"Gradient norm: {np.linalg.norm(p.grad(stat_point))}\n"
    #           f"Iterations: {it}\n"
    #           f"Total obj evaluations: {stats['f']}\n"
    #           f"Total grad evaluations: {stats['g']}\n"
    #           f"Runtime: {end - start} s\n")
    #
    #     if save:
    #         f.write(f"\n############## Quasi-Newton ({spec} {update.__name__}) ##############\n"
    #           f"Gamma: {gamma}\t Sigma: {sigma}\n\n"
    #           f"Objective Value: {p.obj(stat_point)}\n"
    #           f"Gradient norm: {np.linalg.norm(p.grad(stat_point))}\n"
    #           f"Iterations: {it}\n"
    #           f"Total obj evaluations: {stats['f']}\n"
    #           f"Total grad evaluations: {stats['g']}\n"
    #           f"Runtime: {end - start} s\n")

                              ##################### Scipy #####################

    # p = get_problem(p_name)
    # start = time.time()
    # res = scipy.optimize.minimize(p.obj, p.x0, method='BFGS', jac=p.grad,
    #                               options={'gtol': eps, 'c1': gamma, 'c2': sigma, 'norm': 2})
    # end = time.time()
    # stats = p.report()
    #
    # print(f"############## Quasi-Newton (Scipy BFGS) ##############\n"
    #       f"c1: {gamma}\t c2: {sigma}\n\n"
    #       f"Objective Value: {p.obj(res.x)}\n"
    #       f"Gradient norm: {np.linalg.norm(res.jac)}\n"
    #       f"Iterations: {res.nit}\n"
    #       f"Total obj evaluations: {res.nfev}\n"
    #       f"Total grad evaluations: {res.njev}\n"
    #       f"Runtime: {end - start} s\n")
    #
    # if save:
    #     f.write(f"\n############## Quasi-Newton (Scipy BFGS) ##############\n"
    #       f"c1: {gamma}\t c2: {sigma}\n\n"
    #       f"Objective Value: {p.obj(res.x)}\n"
    #       f"Gradient norm: {np.linalg.norm(res.jac)}\n"
    #       f"Iterations: {res.nit}\n"
    #       f"Total obj evaluations: {res.nfev}\n"
    #       f"Total grad evaluations: {res.njev}\n"
    #       f"Runtime: {end - start} s\n")

    ######################################### Objective function evolution #########################################

    p_name = "DIXMAANA1"
    fileName = "res/{}-evolution.txt".format(p_name, datetime.datetime.now())
    save = True
    eps = 0.01
    evolution = []

    # p = pycutest.import_problem(p_name, sifParams={'N': 10})
    p = get_problem(p_name)

    print(f"{p_name} - {{N: {p.n}, M: {p.m} }}\n"
          f"Gradient tolerance: {eps}\n")

    if save:
        f = open(fileName, 'w')
        f.write(f"{p_name} - {{N: {p.n}, M: {p.m} }}\n"
                f"Gradient tolerance: {eps}\n")

                                #################### Armijo #####################

    gamma = armijo_parameters[p_name][0]
    delta = armijo_parameters[p_name][1]

    stat_point, it, evolution = armijo_grad_descent(p, p.x0, eps, gamma, delta, evol=True)

    stats = p.report()
    print(f"############## Armijo (Gamma: {gamma}, Delta: {delta}) ##############\n\n"
          f"{evolution} \t {len(evolution)} elements")

    if save:
        f.write(f"\nArmijo: {evolution}\n")

                            #################### Quasi-Newton #####################

    gamma = qn_parameters[p_name][0]
    sigma = qn_parameters[p_name][1]
    updates = [Broyden, DFP, BFGS]

    for update, dir in ((update, dir) for update in [Broyden, DFP, BFGS] for dir in [True, False]):
        p = get_problem(p_name)
        stat_point, it, evolution = qn_method(p, p.x0, eps, np.identity(p.n), gamma, sigma,
                                              update, dir, evol=True)
        stats = p.report()
        if dir:
            spec = "direct"
        else:
            spec = "indirect"

        print(f"\n############## {spec} {update.__name__} (Gamma: {gamma}, Sigma: {sigma}) ##############\n\n"
              f"{evolution} \t {len(evolution)} elements")

        if save:
            f.write(f"\n{spec} {update.__name__}: {evolution}\n")

                                #################### Scipy #####################

    def callback(intermediate_result):
        global evolution
        evolution.append(intermediate_result.fun)

    def time_callback(intermediate_result):
        end = time.time()
        global evolution
        global start
        evolution.append(end - start)

    p = get_problem(p_name)
    evolution.clear()
    start = time.time()
    res = scipy.optimize.minimize(p.obj, p.x0, method='BFGS', jac=p.grad, callback=callback, tol=0.001,
                                  options={'gtol': eps, 'c1': gamma, 'c2': sigma, 'norm': 2})

    print(f"\n############## Scipy BFGS (Gamma: {gamma}, Sigma: {sigma}) ##############\n\n"
          f"{evolution} \t {len(evolution)} elements")

    if save:
        f.write(f"\nScipy: {evolution}\n")


# for p in sorted(probs):
    #     print("######################################")
    #     print(f"{p}: {list(pycutest.problem_properties(p).items())[3:]} \n ")
    #     pycutest.print_available_sif_params(p)
    #     print("######################################\n")
    #
    # print(len(probs))

    # pycutest.print_available_sif_params('STRATEC')

    # f = open(fileName, "w")
    # gammas = [float(gamma) for gamma in np.arange(0.1, 1, 0.1)]
    # deltas = [float(delta) for delta in np.arange(0.1, 1, 0.1)]
    # eps = 0.01
    # f.write(f"Parameters:\n\tEps: {eps}\n\tGamma: {gammas}\n\tDelta: {deltas}\n")
    # f.write("############################################################\n")
    # for prob_name in [p_name]:
    #     iters = {}
    #     objs = {}
    #     grads = {}
    #     times = {}
    #
    #     for gamma in gammas:
    #         its = []
    #         fs = []
    #         gs = []
    #         ts = []
    #         for sigma in deltas:
    #             if prob_name in ["BOX", "NONDQUAR"]:
    #                 p = pycutest.import_problem(prob_name, sifParams={'N': 100})
    #             else:
    #                 p = pycutest.import_problem(prob_name)
    #             print(f"{p.name} ({p.n}): GAMMA == {gamma} AND SIGMA == {sigma}\n")
    #             start = time.time()
    #             stat_point, it = armijo_grad_descent(p, p.x0, eps, gamma, sigma)
    #             end = time.time()
    #             stats = p.report()
    #             its.append(it)
    #             fs.append(stats['f'])
    #             gs.append(stats['g'])
    #             ts.append(end - start)
    #
    #             print(f"Obj value: {p.obj(stat_point)}\n"
    #                   f"Grad norm: {np.linalg.norm(p.grad(stat_point))}\n\n"
    #                   f"###############################\n")
    #
    #         iters[gamma] = its
    #         objs[gamma] = fs
    #         grads[gamma] = gs
    #         times[gamma] = ts
    #
    #     f.write(f"{prob_name} - {{N: {p.n}, M: {p.m} }}\n")
    #
    #     min_iter = min(list(map(lambda x: tuple((x[0], np.argmin(x[1]), min(x[1]))), iters.items())), key=lambda t: t[2])
    #     min_objs = min(list(map(lambda x: tuple((x[0], np.argmin(x[1]), min(x[1]))), objs.items())), key=lambda t: t[2])
    #     min_grads = min(list(map(lambda x: tuple((x[0], np.argmin(x[1]), min(x[1]))), grads.items())), key=lambda t: t[2])
    #     min_times = min(list(map(lambda x: tuple((x[0], np.argmin(x[1]), min(x[1]))), times.items())), key=lambda t: t[2])
    #
    #     f.write(f"\nMin Iterations: Gamma == {min_iter[0]} and Delta == {deltas[min_iter[1]]}: {min_iter[2]} iterations\n"
    #           f"Min obj evals: Gamma == {min_objs[0]} and Delta == {deltas[min_objs[1]]}: {min_objs[2]} evaluations\n"
    #           f"Min grad evals: Gamma == {min_grads[0]} and Delta == {deltas[min_grads[1]]}: {min_grads[2]} evaluations\n"
    #           f"Min runtime: Gamma == {min_times[0]} and Delta == {deltas[min_times[1]]}: {min_times[2]} runtime\n")
    #
    #     f.write(f"Iterations: {iters}\n Obj evals: {objs}\n Grad evals: {grads}\n Runtimes: {times}\n")
    #     f.write("############################################################\n")
    #
    #     print(f"\nMin Iterations: Gamma == {min_iter[0]} and Delta == {deltas[min_iter[1]]}: {min_iter[2]} iterations\n"
    #           f"Min obj evals: Gamma == {min_objs[0]} and Delta == {deltas[min_objs[1]]}: {min_objs[2]} evaluations\n"
    #           f"Min grad evals: Gamma == {min_grads[0]} and Delta == {deltas[min_grads[1]]}: {min_grads[2]} evaluations\n"
    #           f"Min runtime: Gamma == {min_times[0]} and Delta == {deltas[min_times[1]]}: {min_times[2]} runtime\n")
    #
    #     pprint.pp(f"Iterations: {iters}\n Obj evals: {objs}\n Grad evals: {grads}\n Runtimes: {times}\n")
    #

    #
    # p = pycutest.import_problem(p_name, sifParams={'N': 100})
    # print(p)
    # start = time.time()
    # stat_point, it = armijo_grad_descent(p, p.x0, eps, 0.5, 0.5)
    # end = time.time()
    # stats = p.report()
    #
    # print(f"Obj value: {p.obj(stat_point)}\n"
    #       f"Grad norm: {np.linalg.norm(p.grad(stat_point))}\n"
    #       f"Time: {end - start}\n"
    #       f"###############################\n")
    #
    # print(stats)
    # print(scipy.optimize.minimize(p.obj, p.x0, method='BFGS', options={'disp': True, 'gtol': eps, 'c1': gamma, 'c2': sigma}))
    #
    # for update, dir in ((update, dir) for update in updates for dir in [True, False]):
    #
    #     p = pycutest.import_problem(p_name, sifParams={'N': 100})
    #     print(p)
    #     print("Starting...")
    #     start = time.time()
    #     stat_point, it = qn_method(p, p.x0, eps, np.identity(p.n), gamma, sigma,
    #                                update, dir)
    #     end = time.time()
    #     stats = p.report()
    #     if dir:
    #         spec = "direct"
    #     else:
    #         spec = "indirect"
    #
    #     print(f"{p_name} - {{N: {p.n}, M: {p.m} }}\n"
    #           f"############## Quasi-Newton evaluation ({spec} {update.__name__})##############\n"
    #           f"Eps: {eps}\tGamma: {gamma}\t Sigma: {sigma}\n\n"
    #           f"Objective Value: {p.obj(stat_point)}\n"
    #           f"Gradient norm: {np.linalg.norm(p.grad(stat_point))}\n"
    #           f"Iterations: {it}\n"
    #           f"Total obj evaluations: {stats['f']}\n"
    #           f"Total grad evaluations: {stats['g']}\n"
    #           f"Runtime: {end - start} s\n")

    # eps = 0.01
    # gamma = 0.1
    # sigma = 0.7
    # updates = [Broyden, DFP, BFGS]
    # save = False
    #
    # if save:
    #     f = open(fileName, "w")
    #     f.write(f"############## Quasi-Newton evaluation ({p_name})##############\n"
    #             f"Eps: {eps}\n")
    #
    # for update, dir in ((update, dir) for update in updates for dir in [True, False]):
    #
    #     p = pycutest.import_problem(p_name, sifParams={'N': 100})
    #     print(p)
    #     print("Starting...")
    #     start = time.time()
    #     stat_point, it = qn_method(p, p.x0, eps, np.identity(p.n), gamma, sigma,
    #                                update, dir)
    #     end = time.time()
    #     stats = p.report()
    #     if dir:
    #         spec = "direct"
    #     else:
    #         spec = "indirect"
    #
    #     print(f"{p_name} - {{N: {p.n}, M: {p.m} }}\n"
    #           f"############## Quasi-Newton evaluation ({spec} {update.__name__})##############\n"
    #           f"Eps: {eps}\tGamma: {gamma}\t Sigma: {sigma}\n\n"
    #           f"Objective Value: {p.obj(stat_point)}\n"
    #           f"Gradient norm: {np.linalg.norm(p.grad(stat_point))}\n"
    #           f"Iterations: {it}\n"
    #           f"Total obj evaluations: {stats['f']}\n"
    #           f"Total grad evaluations: {stats['g']}\n"
    #           f"Runtime: {end - start} s\n")
    #
    #     if save:
    #         f.write(f"\n- {spec} {update.__name__}\n"
    #               f"\tGamma: {gamma}\t Sigma: {sigma}\n\n"
    #               f"\tObjective Value: {p.obj(stat_point)}\n"
    #               f"\tGradient norm: {np.linalg.norm(p.grad(stat_point))}\n"
    #               f"\tIterations: {it}\n"
    #               f"\tTotal obj evaluations: {stats['f']}\n"
    #               f"\tTotal grad evaluations: {stats['g']}\n"
    #               f"\tRuntime: {end - start} s\n")






