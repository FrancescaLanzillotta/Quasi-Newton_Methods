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
    while problem.obj(x + alpha * d) > f + gamma * alpha * np.dot(g, d):
        alpha = alpha * delta
    return alpha


def armijo_grad_descent(problem, x, eps, gamma, delta):
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
    while not satisfied:
        grad = problem.grad(x)
        norm = np.linalg.norm(grad)

        if norm <= eps:
            satisfied = True
        else:
            alpha = armijo_step(problem, x, -grad, gamma, delta, grad)
            # print("Armijo step found: ", alpha)
            x = x - alpha * grad

            it += 1

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

    while not satisfied:

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
            w_step = strong_wolfe_step(problem, x, -grad, gamma, sigma, problem.obj(x),
                                       np.dot(grad, -grad))
            # print("Wolfe step found: ", w_step)
            x = x - w_step * grad
            it += 1

    return x, it


def qn_method(problem, x, eps, hess, gamma, sigma, update_rule, direct=True):
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
    satisfied = False
    grad = problem.grad(x)

    while not satisfied:
        obj = problem.obj(x)

        if np.linalg.norm(grad) <= eps:
            satisfied = True
        else:
            if direct:
                d = np.linalg.solve(hess, -grad)
            else:
                d = - np.dot(hess, grad)

            opt_step = strong_wolfe_step(problem, x, d, gamma, sigma, obj, np.dot(grad, d))

            new_x = x + opt_step * d
            new_grad = problem.grad(new_x)

            y = new_grad - grad
            s = new_x - x

            hess = update_rule(hess, y, s, direct)

            x = new_x
            grad = new_grad

            it += 1

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

    armijo_parameters = {"ARWHEAD": (0.5, 0.9),  # prob_name: (gamma, delta)
                         "BOX": (0.5, 0.4),
                         "COSINE": (0.5, 0.6),
                         "CRAGGLVY": (0.6, 0.9),
                         "DIXMAANA1": (0.1, 0.5),
                         "EG2": (0.1, 0.1),
                         "FMINSRF2": (0.1, 0.3),
                         "NONDQUAR": (0.6, 0.9),
                         "POWER": (0.1, 0.2),
                         "TOINTGOR": (0.5, 0.9)}

    qn_parameters = {"ARWHEAD": (0.1, 0.7),  # prob_name: (gamma, sigma)
                     "BOX": (0.2, 0.8),
                     "COSINE": (0.3, 0.6),
                     "CRAGGLVY": (0.2, 0.7),
                     "DIXMAANA1": (0.1, 0.8),
                     "EG2": (0.4, 0.6),
                     "FMINSRF2": (0.3, 0.7),
                     "NONDQUAR": (0.1, 0.7),
                     "POWER": (0.05, 0.7),
                     "TOINTGOR": (0.3, 0.6)}

    problems = ["ARWHEAD",
                "BOX",
                "COSINE",
                "CRAGGLVY",
                "DIXMAANA1",
                "EG2",
                "FMINSRF2",
                "NONDQUAR",
                "POWER",
                "TOINTGOR"]

    p_name = "ARWHEAD"
    fileName = "res/{}-evaluations-{:%Y%m%d-%H%M%S}.txt".format(p_name, datetime.datetime.now())
    save = False
    eps = 0.01

    p = get_problem(p_name)

    print(f"{p_name} - {{N: {p.n}, M: {p.m} }}\n"
          f"Gradient tolerance: {eps}\n")

    if save:
        f = open(fileName, 'w')
        f.write(f"{p_name} - {{N: {p.n}, M: {p.m} }}\n"
                f"Gradient tolerance: {eps}\n")

        ##################### Armijo #####################

    gamma = armijo_parameters[p_name][0]
    delta = armijo_parameters[p_name][1]

    start = time.time()
    stat_point, it = armijo_grad_descent(p, p.x0, eps, gamma, delta)
    end = time.time()
    stats = p.report()
    print(f"############## Armijo ##############\n"
          f"Gamma: {gamma}\tDelta: {delta}\n\n"
          f"Objective Value: {p.obj(stat_point)}\n"
          f"Gradient norm: {np.linalg.norm(p.grad(stat_point))}\n"
          f"Iterations: {it}\n"
          f"Total obj evaluations: {stats['f']}\n"
          f"Total grad evaluations: {stats['g']}\n"
          f"Runtime: {end - start} s\n")

    if save:
        f.write(f"\n############## Armijo ##############\n"
                f"Gamma: {gamma}\tDelta: {delta}\n\n"
                f"Objective Value: {p.obj(stat_point)}\n"
                f"Gradient norm: {np.linalg.norm(p.grad(stat_point))}\n"
                f"Iterations: {it}\n"
                f"Total obj evaluations: {stats['f']}\n"
                f"Total grad evaluations: {stats['g']}\n"
                f"Runtime: {end - start} s\n")

        #################### Quasi-Newton #####################

    gamma = qn_parameters[p_name][0]
    sigma = qn_parameters[p_name][1]
    updates = [Broyden, DFP, BFGS]

    for update, dir in ((update, dir) for update in updates for dir in [True, False]):
        p = get_problem(p_name)
        start = time.time()
        stat_point, it = qn_method(p, p.x0, eps, np.identity(p.n), gamma, sigma,
                                   update, dir)
        end = time.time()
        stats = p.report()
        if dir:
            spec = "direct"
        else:
            spec = "indirect"

        print(f"############## Quasi-Newton ({spec} {update.__name__}) ##############\n"
              f"Gamma: {gamma}\t Sigma: {sigma}\n\n"
              f"Objective Value: {p.obj(stat_point)}\n"
              f"Gradient norm: {np.linalg.norm(p.grad(stat_point))}\n"
              f"Iterations: {it}\n"
              f"Total obj evaluations: {stats['f']}\n"
              f"Total grad evaluations: {stats['g']}\n"
              f"Runtime: {end - start} s\n")

        if save:
            f.write(f"\n############## Quasi-Newton ({spec} {update.__name__}) ##############\n"
                    f"Gamma: {gamma}\t Sigma: {sigma}\n\n"
                    f"Objective Value: {p.obj(stat_point)}\n"
                    f"Gradient norm: {np.linalg.norm(p.grad(stat_point))}\n"
                    f"Iterations: {it}\n"
                    f"Total obj evaluations: {stats['f']}\n"
                    f"Total grad evaluations: {stats['g']}\n"
                    f"Runtime: {end - start} s\n")

            #################### Scipy #####################

    p = get_problem(p_name)
    start = time.time()
    res = scipy.optimize.minimize(p.obj, p.x0, method='BFGS', jac=p.grad,
                                  options={'gtol': eps, 'c1': gamma, 'c2': sigma, 'norm': 2})
    end = time.time()
    stats = p.report()

    print(f"############## Quasi-Newton (Scipy BFGS) ##############\n"
          f"c1: {gamma}\t c2: {sigma}\n\n"
          f"Objective Value: {p.obj(res.x)}\n"
          f"Gradient norm: {np.linalg.norm(res.jac)}\n"
          f"Iterations: {res.nit}\n"
          f"Total obj evaluations: {res.nfev}\n"
          f"Total grad evaluations: {res.njev}\n"
          f"Runtime: {end - start} s\n")

    if save:
        f.write(f"\n############## Quasi-Newton (Scipy BFGS) ##############\n"
                f"c1: {gamma}\t c2: {sigma}\n\n"
                f"Objective Value: {p.obj(res.x)}\n"
                f"Gradient norm: {np.linalg.norm(res.jac)}\n"
                f"Iterations: {res.nit}\n"
                f"Total obj evaluations: {res.nfev}\n"
                f"Total grad evaluations: {res.njev}\n"
                f"Runtime: {end - start} s\n")
