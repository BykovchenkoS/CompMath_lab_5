import math
import numpy as np
import matplotlib.pyplot as plt


class Result:

    def first_function(x: float, y: float):
        return math.sin(x)

    def second_function(x: float, y: float):
        return (x * y) / 2

    def third_function(x: float, y: float):
        return y - (2 * x) / y

    def fourth_function(x: float, y: float):
        return x + y

    def default_function(x: float, y: float):
        return 0.0

    def get_function(n: int):
        if n == 1:
            return Result.first_function
        elif n == 2:
            return Result.second_function
        elif n == 3:
            return Result.third_function
        elif n == 4:
            return Result.fourth_function
        else:
            return Result.default_function

    def solveByMilne(f, epsilon, a, y_a, b):
        def runge_kutta(f, x0, y0, h):
            k1 = h * Result.get_function(f)(x0, y0)
            k2 = h * Result.get_function(f)(x0 + h / 2, y0 + k1 / 2)
            k3 = h * Result.get_function(f)(x0 + h / 2, y0 + k2 / 2)
            k4 = h * Result.get_function(f)(x0 + h, y0 + k3)
            return y0 + (k1 + 2 * k2 + 2 * k3 + k4) / 6

        h = (b - a) / 1021

        while True:
            y1 = runge_kutta(f, a, y_a, h)
            y2 = runge_kutta(f, a + h, y1, h)
            y3 = runge_kutta(f, a + 2 * h, y2, h)

            for i in range(3, 21):
                x = a + i * h
                y_pred = y3 + (4 * h / 3) * (2 * Result.get_function(f)(x, y3) -
                                             Result.get_function(f)(x - h, y2) +
                                             2 * Result.get_function(f)(x - 2 * h, y1))

                y_corr = y3 + (h / 3) * (Result.get_function(f)(x + h, y_pred) +
                                         4 * Result.get_function(f)(x + h, y3) + Result.get_function(f)(x, y2))

                y1, y2, y3 = y2, y3, y_corr

            delta = abs(y_corr - y3) / 29

            if delta < epsilon:
                return y_corr

            h /= 2


if __name__ == '__main__':
    f = int(input('1. sin(x)\n2. (x * y) / 2\n3. y - (2 * x) / y\n4. x + y\n').strip())

    epsilon = float(input('Введите epsilon: ').strip())

    a = float(input('Введите a: ').strip())

    y_a = float(input('Введите y(a): ').strip())

    b = float(input('Введите b: ').strip())

    result = Result.solveByMilne(f, epsilon, a, y_a, b)

    print(str(result) + '\n')

    def y(x, C):
        return (2 * np.exp(x) - x - 1 + C * np.exp(-x)) / 2


    C = (2 * np.exp(a) - a - 1 - 2 * y_a) / np.exp(-a)

    x = np.linspace(a, b, 1000)

    y_values = y(x, C)

    milne_values = [Result.solveByMilne(f, epsilon, a, y_a, x_i) for x_i in x]

    analytical_values = [2 * np.exp(x_i) - x_i - 1 for x_i in x]

    plt.plot(x, y_values, label='Начальное условие')
    plt.plot(x, milne_values, label='Численное решение (метод Милна)')
    plt.plot(x, analytical_values, label='Аналитическое решение')
    plt.legend()
    plt.show()
