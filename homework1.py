import numpy as np
import pandas as pd
import math
import matplotlib.pyplot as plt

CAR_MAX = 100  # 车的容量

data_origin = pd.read_excel("./data.xlsx")
dt = data_origin.copy()  # 复制数据
dt.iloc[0, 3] = 0  # 处理仓库
data_e = np.array(dt.iloc[:, 1:3])  # 单独取出坐标
data_con = np.array(dt.iloc[:, 3])  # 客户点的量

C = np.zeros([data_e.shape[0], data_e.shape[0]])  # 距离矩阵，创建零矩阵
for i in range(C.shape[0]):
    for j in range(C.shape[1]):
        C[i][j] = math.floor(math.sqrt(
            (data_e[i][0] - data_e[j][0]) ** 2 + (
                    data_e[i][1] - data_e[j][1]) ** 2) * 10 ** 4) / 10 ** 4  # 计算并取四位小数

S = np.zeros([data_e.shape[0], data_e.shape[0]])  # 节约矩阵，创建零矩阵
for i in range(C.shape[0]):
    for j in range(C.shape[1]):
        S[i][j] = C[i][0] + C[0][j] - C[i][j]  # 节约矩阵

S_new = np.array([range(S.size), S.flatten()]).T  # [[编号（解码回矩阵），节约矩阵的saving]]
index = np.argsort(S.flatten())[::-1]  # 排序编码下标


def saving(S_new=S_new, index=index, data_con=data_con):
    """
    节约算法
    :param S_new: 节约矩阵
    :param index: 排序下标
    :param data_con: 客户需求
    :return: 解
    """
    solve = []  # 解的初始化
    solve_k = [[0], 0]  # 解的某一条回路,solve_k[0]是路线，solve[1]是当前装载量
    no_usd = list(range(1, 71))

    while S_new[index[0]][1] > 0:
        k = 0
        while S_new[index[k]][1] > 0:
            x, y = int(S_new[index[k]][0] // S.shape[0]), int(S_new[index[k]][0] % S.shape[0] - 1)
            if (x in no_usd) and (y in no_usd):
                solve_k = [[0, x, y, 0], data_con[x] + data_con[y]]
                no_usd.remove(x)
                no_usd.remove(y)
                index = np.delete(index, k)
                break
            else:
                k += 1

        if solve_k[1] == 0:  # 判断是否成功初始化
            break  # 意味着剩下的客户点不足以形成三个点及以上的回路
        else:
            k = 0  # 初始化成功，重新试着拓展回路
            while S_new[index[k]][1] > 0:
                x, y = int(S_new[index[k]][0] // S.shape[0]), int(S_new[index[k]][0] % S.shape[0] - 1)

                if (x in solve_k[0]) and (y not in solve_k[0]):
                    if (y in no_usd) and data_con[y] + solve_k[1] <= CAR_MAX:
                        no_usd.remove(y)
                        solve_k[1] = data_con[y] + solve_k[1]
                        c = solve_k[0].index(x)
                        if solve_k[0][c - 1] == 0:
                            solve_k[0].insert(1, y)
                        else:
                            solve_k[0].insert(-2, y)
                        index = np.delete(index, k)
                elif (y in solve_k[0]) and (x not in solve_k[0]):
                    if (x in no_usd) and data_con[x] + solve_k[1] <= CAR_MAX:
                        no_usd.remove(x)
                        solve_k[1] = data_con[x] + solve_k[1]
                        c = solve_k[0].index(y)
                        if solve_k[0][c - 1] == 0:
                            solve_k[0].insert(1, x)
                        else:
                            solve_k[0].insert(-2, x)
                        index = np.delete(index, k)

                k += 1

            solve.append(solve_k)
            solve_k = [[0, 0], 0]

    if len(no_usd) != 0:
        for k in no_usd:
            solve.append([[0, k, 0], data_con[i]])
    return solve


def desolve(solve, C=C):
    """
    求解解的总路程
    :param solve: 解
    :param C: 距离矩阵
    :return: 总距离total_distance
    """
    total_distance = 0
    for i in range(len(solve)):
        row_distance = 0
        for j in range(len(solve[i][0]) - 1):
            row_distance = row_distance + C[solve[i][0][j]][solve[i][0][j + 1]]
        solve[i].append(row_distance)
        total_distance += row_distance

    return total_distance, solve


def deal_solve(solve_k, data_e=data_e):
    """
    处理每一个子解的序列，使之方便绘图
    :param solve_k: 每一个子解
    :param data_e: 坐标
    :return: 绘图用x,y
    """
    decode_x, decode_y = data_e[solve_k[0], 0], data_e[solve_k[0], 1]
    x, y = [], []
    for ix in range(len(decode_x) - 1):
        x.append([decode_x[ix], decode_x[ix + 1]])
        y.append([decode_y[ix], decode_y[ix + 1]])
    return x, y


def solve_plot(solve, data_e=data_e):
    """
    对输入的解作图
    :param solve:解
    :param data_e:坐标
    :return:图
    """
    fig = plt.figure()
    ax1 = fig.add_subplot(1, 1, 1)
    ax1.set_title("solve")
    ax1.set_xlabel("x")
    ax1.set_ylabel("y")
    ax1.scatter(data_e[:, 0], data_e[:, 1], c='r', marker='.')  # 绘制初步的散点

    for i, txt in enumerate(range(data_e.shape[0])):  # 给点写上编号
        ax1.annotate(txt, (data_e[i, 0], data_e[i, 1]))

    for i in range(len(solve)):
        x, y = deal_solve(solve[i])
        for j in range(len(x)):
            plt.plot(x[j], y[j], color='r')

    print(data_e[solve[0][0], 0], data_e[solve[0][0], 1])
    plt.show()


def main():
    solve_greedy = saving()
    total_distance, initial = desolve(solve_greedy)
    solve_plot(initial)
    return initial


main()
