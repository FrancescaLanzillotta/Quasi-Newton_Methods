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

    :return: optimal step that satisfies the Armijo condition using specified parameters, number of objective function
    evaluations and number of gradient evaluations
    """

    f, g = problem.obj(x, gradient=True)
    obj_count = grad_count = 1
    while problem.obj(x + step * d) > f + gamma * step * np.dot(g, d):
        step = step * delta
        obj_count += 1
    return step, obj_count, grad_count


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

    :return: stationary point, number of iterations, number of objective function evaluations and number of gradient
    evaluations
    """
    it = obj_count = grad_count = 0
    satisfied = False
    while not satisfied:
        grad = problem.grad(x)
        grad_count += 1
        if np.linalg.norm(grad) <= eps:
            satisfied = True
        else:
            a_step, a_obj_count, a_grad_count = armijo_step(problem, x, -grad, step, gamma, delta)
            obj_count += a_obj_count
            grad_count += a_grad_count
            print("Armijo step found: ", a_step)
            x = x - a_step * grad
            it += 1
    return x, it, obj_count, grad_count


def strong_wolfe_step(problem, x, d, gamma, sigma):
    """
    This function determines the length of the step as to satisfy the strong Wolfe conditions.

    :param problem: current pycutest problem
    :param x: starting point
    :param d: descent direction
    :param gamma: parameter used in the first Wolfe condition (Armijo rule), with 0 < gamma < 1/2
    :param sigma: parameter used in the second Wolfe condition (curvature condition), with gamma < sigma < 1

    :return: optimal step that satisfies the strong Wolfe conditions using specified parameters, number of objective
    function evaluations and number of gradient evaluations
    """
    lb = 0
    ub = np.inf
    alpha = 1
    satisfied = False

    obj_count = grad_count = 0

    while not satisfied:

        phi_zero = problem.obj(x)
        obj_count += 1
        dot_phi_zero = np.dot(problem.grad(x), d)
        grad_count += 1

        if problem.obj(x + alpha * d) > phi_zero + gamma * alpha * dot_phi_zero:
            obj_count += 1
            ub = alpha
            alpha = (lb + ub) / 2

        else:   # Armijo condition is satisfied
            dot_phi_alpha = np.dot(problem.grad(x + alpha * d), d)
            grad_count += 1

            if dot_phi_alpha < sigma * dot_phi_zero:
                lb = alpha
                alpha = min(((lb + ub) / 2), 2*alpha)
            elif dot_phi_alpha > sigma * abs(dot_phi_zero):          # weak Wolfe condition is satisfied
                ub = alpha
                alpha = (lb + ub) / 2
            elif abs(dot_phi_alpha) <= sigma * abs(dot_phi_zero):   # strong Wolfe condition is satisfied
                satisfied = True

    return alpha, obj_count, grad_count


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

    :return: stationary point, number of iterations, number of objective function evaluations and number of gradient
    evaluations
    """
    satisfied = False

    it = obj_count = grad_count = 0
    while not satisfied:
        grad = problem.grad(x)
        grad_count += 1

        if np.linalg.norm(grad) <= eps:
            satisfied = True

        else:
            w_step, w_obj_count, w_grad_count = strong_wolfe_step(problem, x, -grad, gamma, sigma)
            obj_count += w_obj_count
            grad_count += w_grad_count
            print("Wolfe step found: ", w_step)
            x = x - w_step * grad
            it += 1

    return x, it, obj_count, grad_count


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

    :return: stationary point, number of iterations, number of objective function evaluations and number of gradient
    evaluations
    """

    it = obj_count = grad_count = 0

    satisfied = False

    while not satisfied:

        obj, grad = problem.obj(x, True)
        obj_count += 1
        grad_count += 1

        if np.linalg.norm(grad) <= eps:
            satisfied = True
        else:

            if direct:
                d = - np.dot(np.linalg.inv(hess), grad)
            else:
                d = - np.dot(hess, grad)

            opt_step, w_obj_count, w_grad_count = strong_wolfe_step(problem, x, -grad, gamma, sigma)
            obj_count += w_obj_count
            grad_count += w_grad_count

            new_x = x + opt_step * d

            s = new_x - x
            y = problem.grad(new_x) - grad
            grad_count += 1
            hess = update_rule(hess, y, s, direct)
            x = new_x
            it += 1

    return x, it, obj_count, grad_count


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

        return hess + first - second


if __name__ == '__main__':

    # probs = pycutest.find_problems(objective="other", constraints="unconstrained", regular=True)
    #
    # print(*[f"{x}: {list(pycutest.problem_properties(x).items())[3:]} \n " for x in sorted(probs)], sep="\n")
    #
    # print(len(probs))
    
    prob = pycutest.import_problem("BOX", sifParams={'N': 10})

    print(prob)

    min, iters, objs, grads = strong_wolfe_grad_descent(prob, np.ones(10), 0.01, 0.1, 0.3)

    print(f"Stationary Point: {min}\nObjective function value: {prob.obj(min)}\nIterations: {iters}"
          f"\nObjective function evaluations: {objs}\nGradient evaluations {grads}\n")




