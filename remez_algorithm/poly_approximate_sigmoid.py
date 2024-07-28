from mpmath import mp
import numpy as np
import matplotlib.pyplot as plt

def bisection_search(f, low:float, high:float):
    """
    A root finding method that does not rely on derivatives

    :param f: a function f: X -> R
    :param low: the lower bracket
    :param high: the upper limit bracket
    :return: the location of the root, e.g. f(mid) ~ 0
    """
    if f(high) < f(low):
        low, high = high, low

    mid = .5 * (low + high)

    while True:
        if f(mid) < 0:
            low = mid
        else:
            high = mid

        mid = .5 * (high + low)

        if abs(high - low) < 10 ** (-(mp.dps / 2)):
            break

    return mid


def concave_max(f, low:float, high:float):
    """
    Forms a lambda for the approximate derivative and finds the root

    :param f: a function f: X -> R
    :param low: the lower bracket
    :param high: the upper limit bracket
    :return: the location of the root f'(mid) ~ 0
    """
    scale = high - low

    h = mp.mpf('0.' + ''.join(['0' for i in range(int(mp.dps / 1.5))]) + '1') * scale
    df = lambda x: (f(x + h) - f(x - h)) / (2.0 * h)

    return bisection_search(df, low, high)

def chev_points(n:int, lower:float = -1, upper:float = 1):
    """
    Generates a set of chebychev points spaced in the range [lower, upper]
    :param n: number of points
    :param lower: lower limit
    :param upper: upper limit
    :return: a list of multipressison chebychev points that are in the range [lower, upper]
    """
    index = np.arange(1, n+1)
    range_ = abs(upper - lower)
    return [(.5*(mp.cos((2*i-1)/(2*n)*mp.pi)+1))*range_ + lower for i in index]


def remez(func, n_degree, interval, max_iter = 100):
    """
    :param func: a function (or lambda) f: X -> R
    :param n_degree: the degree of the polynomial to approximate the function f
    :param lower: lower range of the approximation
    :param upper: upper range of the approximation
    :return: the polynomial coefficients, and an approximate maximum error associated with this approximation
    """
    lower, upper = interval
    x_points = chev_points(n_degree + 2, lower, upper)

    A = mp.matrix(n_degree + 2)
    coeffs = np.zeros(n_degree + 2)

    mean_error = float('inf')

    for i in range(n_degree + 2):
        A[i, n_degree + 1] = (-1) ** (i + 1)

    for i in range(max_iter):
        vander = np.polynomial.chebyshev.chebvander(x_points, n_degree)

        for i in range(n_degree + 2):
            for j in range(n_degree + 1):
                A[i, j] = vander[i, j]

        b = mp.matrix([func(x) for x in x_points])
        l = mp.lu_solve(A, b)

        coeffs = l[:-1]

        r_i = lambda x: (func(x) - np.polynomial.chebyshev.chebval(x, coeffs))

        interval_list = list(zip(x_points, x_points[1:]))

        intervals = [upper]
        intervals.extend([bisection_search(r_i, *i) for i in interval_list])
        intervals.append(lower)

        extermum_interval = [[intervals[i], intervals[i + 1]] for i in range(len(intervals) - 1)]

        extremums = [concave_max(r_i, *i) for i in extermum_interval]

        extremums[0] = mp.mpf(upper)
        extremums[-1] = mp.mpf(lower)

        errors = [abs(r_i(i)) for i in extremums]
        mean_error = np.mean(errors)

        if np.max([abs(error - mean_error) for error in errors]) < 0.000001 * mean_error:
            break

        x_points = extremums

    return [float(i) for i in np.polynomial.chebyshev.cheb2poly(coeffs)], float(mean_error)

def poly_eval(p, x):
    
    return np.polyval(p, x)

def draw_sigmoid_graph(poly_coeffs, degree, interval, function):
    
    _, axs = plt.subplots(1, 2, figsize = (20,5))
    for i in range(len(degree)):
        x = np.linspace(interval[0], interval[1], 400)
        y_exact = np.array([function(x_i) for x_i in x])
        y_poly = poly_eval(poly_coeffs[i][::-1], x)

        axs[i].set_title('Approximate Sigmoid Function')
        axs[i].plot(x, y_exact, label=f"Sigmoid")  
        axs[i].plot(x, y_poly, label=f"Polynomial Approximation (degree={degree[i]})")

        axs[i].legend()
        axs[i].set_xlabel("x")
        axs[i].set_ylabel("y")

    plt.show()
    
def eval_poly_approximate_sigmoid(x):
    return 0.5 + 0.197*x - 0.004*(x**3)
