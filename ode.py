import autograd.numpy as np
from autograd import grad
import numpy.random as random
from matplotlib import pyplot as plt


# условия задачи Коши
# x0 = 0
#x1 = np.pi/2
# y0 = 1
x0 = 0
x1 = 1
y0 = 1

# количество точек, в которых вычисляем значения
nx = 10


def rhs(x, y):
    """
    правая часть уравнения y' = rhs(x, y)
    """
    # return np.cos(x) * y
    return x**3 + 2*x + x**2 * ((1 + 3*x**2) / (1 + x + x**3)) - y * (x + (1 + 3*x**2) / (1 + x + x**3))


def exact(x):
    """
    точное решение задачи Коши
    """
    # return np.exp(np.sin(x))
    return np.exp(-x**2 / 2) / (1 + x + x**3) + x**2


def sigmoid(x):
    return 1 / (1 + np.exp(-x))


def d_sigmoid(x):
    return sigmoid(x) * (1 - sigmoid(x))


def nn_output(w, x):
    """
    выход сетки
    """
    w0 = w[0]
    w1 = w[1]
    a1 = sigmoid(np.dot(x, w0))
    return np.dot(a1, w1)


def d_nn_output(w, x):
    """
    частные производные выхода сетки по параметрам x
    """
    w0 = w[0]
    w1 = w[1]
    return np.dot(np.dot(w1.T, w0.T), d_sigmoid(x))


def error(w, x):
    """
    считаем, насколько наше приближение отклонилось от уравнения y' = rhs(x, y)
    приближение: y ~ y(0) + x * nn(w, x)
    """
    err = 0
    for xi in x:
        nn_out = nn_output(w, xi)[0][0]
        y = y0 + xi * nn_out
        d_nn_out = d_nn_output(w, xi)[0][0]
        d_y = nn_out + xi * d_nn_out
        func = rhs(xi, y)
        err += (d_y - func)**2
    return err


def main():
    x = np.linspace(x0, x1, nx)
    dx = x[1] - x[0]

    y = exact(x)

    # приближённый расчёт через производную
    y_f = np.zeros_like(y)
    y_f[0] = y0
    for i in range(len(x)-1):
        y_f[i+1] = y_f[i] + dx * rhs(x[i], y_f[i])

    # начальные веса
    w = [2*random.rand(1, nx)-1, 2*random.rand(nx, 1)-1]

    lmb = 1e-3
    for i in range(1000):
        gradient = grad(error)(w, x)
        w[0] -= lmb * gradient[0]
        w[1] -= lmb * gradient[1]
    #   print(i)

    # смотрим, что получилось
    y_n = [1 + xi * nn_output(w, xi)[0][0] for xi in x]

    plt.figure()
    plt.plot(x, y, 'r.-')
    plt.plot(x, y_f, 'b.-')
    plt.plot(x, y_n, 'g.-')
    plt.show()


main()
