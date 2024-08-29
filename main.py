import pycutest
import numpy as np


def armijo_step(problem, x, d, step, gamma, delta):
    """
    This function determines the length of the step as to satisfy the Armijo rule.

    :param problem: current pycutest problem
    :param x: starting point
    :param d: descent direction
    :param step: initial length of the step, with step > 0
    :param gamma: parameter used in the Armijo condition, with 0 < gamma < 1
    :param delta: reduction factor of the step at each iteration, with 0 < delta < 1

    :return: optimal step that satisfies the Armijo condition using specified parameters
    """
    f, g = problem.obj(x, gradient=True)
    while problem.obj(x + step * d) > f + gamma * step * np.dot(g, d):
        step = step * delta

    return step


def armijo_grad_descent(problem, x, eps, step, gamma, delta):
    """
    This function implements the gradient descent method with Armijo line search. When the norm of the gradient falls
    below a certain threshold eps, the algorithm stops and returns the stationary point found.

    :param problem: current pycutest problem
    :param x: starting point
    :param eps: stopping threshold
    :param step: initial length of the step, with step > 0
    :param gamma: parameter used in the armijo_step function
    :param delta: reduction factor used in the armijo_step function

    :return: stationary point
    """

    while np.linalg.norm(problem.grad(x)) > eps:
        grad = problem.grad(x)
        a_step = armijo_step(problem, x, -grad, step, gamma, delta)
        print("Armijo step found: ", a_step)
        x = x - a_step * grad
    return x


def strong_wolfe_step(problem, x, d, gamma, sigma):
    """
    This function determines the length of the step as to satisfy the strong Wolfe conditions.

    :param problem: current pycutest problem
    :param x: starting point
    :param d: descent direction
    :param gamma: parameter used in the first Wolfe condition (Armijo rule), with 0 < gamma < 1/2
    :param sigma: parameter used in the second Wolfe condition (curvature condition), with gamma < sigma < 1

    :return: optimal step that satisfies the strong Wolfe conditions using specified parameters
    """
    lb = 0
    ub = 100
    alpha = 0
    satisfied = False

    while not satisfied:
        alpha = (lb + ub) / 2
        phi_zero = problem.obj(x)
        dot_phi_zero = np.dot(problem.grad(x), d)

        if problem.obj(x + alpha * d) > phi_zero + gamma * alpha * dot_phi_zero:
            ub = alpha

        else:   # Armijo condition is satisfied
            dot_phi_alpha = np.dot(problem.grad(x + alpha * d), d)

            if dot_phi_alpha < sigma * dot_phi_zero:
                lb = alpha
            elif dot_phi_alpha > sigma * abs(dot_phi_zero):          # weak Wolfe condition is satisfied
                ub = alpha
            elif abs(dot_phi_alpha) <= sigma * abs(dot_phi_zero):   # strong Wolfe condition is satisfied
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

    :return: stationary point
    """
    while np.linalg.norm(problem.grad(x)) > eps:
        grad = problem.grad(x)
        w_step = strong_wolfe_step(problem, x, -grad, gamma, sigma)
        print("Wolfe step found: ", w_step)
        x = x - w_step * grad

    return x


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

    :return: stationary point
    """
    i = 0
    while np.linalg.norm(problem.grad(x)) > eps:
        i += 1
        obj, grad = problem.obj(x, True)
        if direct:
            d = - np.dot(np.linalg.inv(hess), grad)
        else:
            d = - np.dot(hess, grad)

        opt_step = strong_wolfe_step(problem, x, -grad, gamma, sigma)

        new_x = x + opt_step * d

        s = new_x - x
        y = problem.grad(new_x) - grad

        hess = update_rule(hess, y, s, direct)
        x = new_x

    print(f"Found min in {i} iterations:\n x: {x}\n Gradient norm: {np.linalg.norm(problem.grad(x))}")
    return x


def Broyden_formula(hess, y, s, direct):
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


def DFP_formula(hess, y, s, direct):
    """
    This function updates the hessian matrix approximation using the DFP formula (rank-2 update rule).

    :param hess: current hessian approximation to update, size n x n
    :param y: vector computed as problem.grad(new_x) - problem.grad(x)
    :param s: vector computed as new_x - x
    :param direct: whether we're approximating the hessian matrix or its inverse

    :return: updated matrix given specified parameters
    """
    if direct:

        sy = np.dot(s, y)           # scalar value
        yhs = y - np.dot(hess, s)   # row vector

        first = (np.dot(np.array([yhs]).transpose(), np.array([y])) + np.dot(np.array([y]).transpose(), np.array([yhs]))) / sy
        second = (np.dot(s, yhs) * np.dot(np.array([y]).transpose(), np.array([y]))) / sy ** 2
        return hess + first - second

    else:

        first = np.dot(np.array([s]).transpose(), np.array([s])) / np.dot(s, y)
        second = (np.linalg.multi_dot([hess, np.array([y]).transpose(), np.array([y]), hess])
                  / np.linalg.multi_dot([np.array([y]), hess, np.array([y]).transpose()]))
        return hess + first - second


def BFGS_formula(hess, y, s, direct):
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
        second = np.linalg.multi_dot([hess, np.array([s]).transpose(), np.array([s]), hess]) / np.linalg.multi_dot([np.array([s]), hess, np.array([s]).transpose()])

        return hess + first - second
    else:
        sy = np.dot(s, y)       # scalar value
        first = ((1 + (np.linalg.multi_dot([np.array([y]), hess, np.array([y]).transpose()]) / sy))
                 * np.dot(np.array([s]).transpose(), np.array([s]) / sy))
        second = (np.linalg.multi_dot([np.array([s]).transpose(), np.array([y]), hess]) +
                  np.linalg.multi_dot([hess, np.array([y]).transpose(), np.array([s])])) / sy

        return hess + first + second


if __name__ == '__main__':


    # # Find unconstrained, variable-dimension problems
    # probs = pycutest.find_problems(objective='other', constraints='unconstrained', regular=True, origin='academic')
    #
    # # print("\n".join(sorted(probs)))
    # print(len(probs))


    box = pycutest.import_problem('BOX', sifParams={'N':10})

    x = np.ones(10)
    f, g = box.obj(x, True)
    id = np.identity(10)

    problem = pycutest.import_problem('BOX', sifParams={'N':10})
    alpha = np.inf
    gamma = 0.3
    sigma = 0.7
    eps = 0.1
    d = -g
    # strong_wolfe_grad_descent(box, x, eps, gamma, sigma)
    # print(np.dot(np.array([x]), np.array([x]).transpose()))
    # print(np.inner(x, x))
    # print(np.identity(10) @ np.identity(10))

    # qn_method(box, x, eps, np.identity(10), gamma, sigma, Broyden_formula, False)
    BFGS_formula(np.identity(10), np.random.rand(10), np.random.rand(10), False)
    hess = np.identity(10)
    grad = problem.grad(x)
    d = - np.dot(hess, grad)
    # print(d)

    d = - hess.dot(np.array([grad]).transpose())
    # print(d)
    #
    # print(np.linalg.norm(box.grad(start)))
    # min_point = armijo_grad_descent(box, start, 0.01, 1, 0.5, 0.2)
    #
    # print("Stationary point found at ", min_point)






