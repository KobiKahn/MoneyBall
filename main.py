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


# MAKE DATAFRAMES
hitting_df = pd.read_csv('Hitting_stats', delim_whitespace=True)
pitching_df = pd.read_csv('pitching_stats', delim_whitespace=True)
# TURN HITTING DATAFRAMES INTO INDIVIDUAL LISTS
OBP = list(hitting_df['OBP'])
SLG = list(hitting_df['SLG'])
AVG = list(hitting_df['AVG'])
RBI = list(hitting_df['RBI'])
H_WPCT = list(hitting_df['WPCT'])

# CALCULATE HITTING CORRELATION TO WINNING PERCENTAGE
OBP_to_W = cal_corr(OBP, H_WPCT)
SLG_to_W = cal_corr(SLG, H_WPCT)
AVG_to_W = cal_corr(AVG, H_WPCT)
RBI_to_W = cal_corr(RBI, H_WPCT)

# TURN PITCHING DATAFRAMES INTO INDIVIDUAL LISTS
ERA = list(pitching_df['ERA'])
SO = list(pitching_df['SO'])
R = list(pitching_df['R'])
HR = list(pitching_df['HR'])
P_WPCT = list(pitching_df['WPCT'])

# CALCULATE PITCHING CORRELATION TO WINNING PERCENTAGE
ERA_to_W = cal_corr(ERA, P_WPCT)
SO_to_W = cal_corr(SO, P_WPCT)
R_to_W = cal_corr(R, P_WPCT)
HR_to_W = cal_corr(HR, P_WPCT)

# MAKE CORRELATION TABLES
Hitting_corr_dict = {'Index':['H_WPCT'], 'OBP':[OBP_to_W], 'SLG':[SLG_to_W], 'AVG': [AVG_to_W], 'RBI': [RBI_to_W]}
Hitting_corr_df = pd.DataFrame.from_dict(Hitting_corr_dict)
print(Hitting_corr_df)
print()
Pitching_corr_dict = {'Index':['P_WPCT'], 'ERA':[ERA_to_W], 'SO':[SO_to_W], 'R': [R_to_W], 'HR': [HR_to_W]}
Pitching_corr_df = pd.DataFrame.from_dict(Pitching_corr_dict)
print(Pitching_corr_df)


H_res_list, H_res_mean, H_res_std, H_slope, H_y_intercept = comp_residual(RBI, H_WPCT)
scatter_plot_residual(RBI, H_WPCT, 'RUNS BATTED IN to WIN PERCENTAGE', H_slope, H_y_intercept, 2, H_res_mean, H_res_std)

P_res_list, P_res_mean, P_res_std, P_slope, P_y_intercept = comp_residual(SO, P_WPCT)
scatter_plot_residual(SO, P_WPCT, 'NUMBER OF STRIKEOUTS to WIN PERCENTAGE', P_slope, P_y_intercept, 2, P_res_mean, P_res_std)

H_RSME = sqrt(sum(H_res_list))
P_RSME = sqrt(sum(P_res_list))

print(f'HITING RSME: {H_RSME}\nPITCHING RSME: {P_RSME}')



