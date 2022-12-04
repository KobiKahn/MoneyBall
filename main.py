import word2number
from word2number import w2n
import numpy as np
from cmath import sqrt
import math
import pandas as pd
from matplotlib import pyplot as plt
from importlib import reload
import seaborn as sb
import statistics as stat


def cal_corr(list1, list2, option=0):
    if len(list1) == len(list2):
        i = -1
        prod_list = []
        min_sqr1 = []
        min_sqr2 = []
        for val in list1:
            i += 1
            prod_list.append(list1[i] * list2[i])
            min_sqr1.append(list1[i] ** 2)
            min_sqr2.append(list2[i] ** 2)

        numerator = ((len(list1) * sum(prod_list)) - (sum(list1) * sum(list2)))
        denominator = (sqrt(len(list1) * sum(min_sqr1) - sum(list1) ** 2) * sqrt(len(list2) * sum(min_sqr2) - sum(list2) ** 2))
        denominator = denominator.real
        correlation = numerator / denominator
        if option == 0:
            return (correlation)
        elif option == 1:
            return (sum(list1), sum(list2), sum(prod_list), sum(min_sqr1), len(list1))
    else:
        print('ERROR LISTS ARE NOT THE SAME LENGTH CANT COMPUTE')
        return False


def LSC(list1, list2):
    x_sum, y_sum, prod_sum, xsqr_sum, list_len = cal_corr(list1, list2, 1)
    matrix1 = [[xsqr_sum, x_sum], [x_sum, list_len]]
    matrix2 = [prod_sum, y_sum]
    matrix1 = np.array(matrix1)
    matrix2 = np.array(matrix2)
    inv_mat1 = np.linalg.inv(matrix1)
    solution = np.dot(inv_mat1, matrix2)
    # a = slope b = y int
    a = solution[0]
    b = solution[1]
    return(a, b)
    # scatter_plot(list1, list2, name, a, b)


def scatter_plot(data1, data2, name, slope=None, y_int=None):
    y_vals = []
    x_data = [min(data1), max(data1)]
    if slope and y_int != None:
        for val in range(2):
            ans = (slope * x_data[val]) + y_int
            y_vals.append(ans)
        plt.text(x_data[-1], y_vals[-1] + .2, f'Y={round(slope)}*X+{round(y_int)}', color='g')
        plt.plot(x_data, y_vals, '-r')
    plt.scatter(data1, data2)
    plt.title(f'{name}')
    plt.xlabel('X')
    plt.ylabel('Y')
    plt.show()


def comp_residual(x, y):
    a, b = LSC(x, y)
    val_list = []
    r_list = []
    for val in x:
        val_list.append(val*a + b)
    for i in range(len(y)):
        r_list.append(y[i] - val_list[i])
    r_mean = stat.mean(r_list)
    r_std = stat.stdev(r_list)
    return(r_list, r_mean, r_std, a, b)


def scatter_plot_residual(data1, data2, name, slope, y_int, n_dev, r_mean, r_std):
    y_vals = []
    l_bound = []
    u_bound = []
    x_data = [min(data1), max(data1)]
    for val in range(2):
        ans = (slope * x_data[val]) + y_int
        y_vals.append(ans)
    for val in range(2):
        l_bound.append((slope * x_data[val]) + y_int - (n_dev*r_std))
    for val in range(2):
        u_bound.append((slope * x_data[val]) + y_int + (n_dev*r_std))
    plt.text(x_data[-1], y_vals[-1] + .2, f'Y={round(slope)}*X+{round(y_int)}', color='g')
    plt.plot(x_data, y_vals, '-r')
    plt.plot(x_data, l_bound, '--r')
    plt.plot(x_data, u_bound, '--r')
    plt.scatter(data1, data2)
    plt.title(f'{name}')
    plt.xlabel('X')
    plt.ylabel('Y')
    plt.show()

x_list = [1.0,2.0,4.0,6.0,5.0,6.0,9.0,8.0,11.0,12.0]
y_list = [14.0,10.0,12.0,9.0,8.0,6.0,4.0,3.0,3.0,10.0]


res_list, res_mean, res_std, slope, y_intercept = comp_residual(x_list, y_list)

scatter_plot_residual(x_list, y_list, 'RESIDUAL_PLOT', slope, y_intercept, 2, res_mean, res_std)








