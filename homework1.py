import numpy as np
import pandas as pd
import math
import matplotlib.pyplot as plt

CAR_MAX = 100  # 车的容量

dt = pd.read_excel("./data.xlsx")  # 复制数据
dt.iloc[0, 3] = 0  # 处理仓库
data_e = np.array(dt.iloc[:, 1:3])  # 单独取出坐标
data_con = np.array(dt.iloc[:, 3])  # 客户点的量  0 - 70  data_con[k]即为第k个点的装载量

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
        if i == j:
            S[i][j] = 0

S = np.triu(S)
S_new = np.array([range(S.size), S.flatten()]).T  # [[编号（解码回矩阵），节约矩阵的saving]]
index = np.argsort(S.flatten())[::-1]  # 排序编码下标


def saving(S_new=S_new, index=index, data_con=data_con):
    solve = []
    for i in range(len(data_con) - 1):  # 初始化
        solve.append([[0, i + 1, 0], data_con[i + 1]])  # solve[k][0]第k+1个客户点,solve[k][1]现在的装载值   1 - 70

    can_use_left = [i + 1 for i in range(len(data_con) - 1)]  # 左边为0的点  1 - 70
    can_use_right = [i + 1 for i in range(len(data_con) - 1)]  # 右边为0的点  1 - 70

    k = 0
    while S_new[index[k]][1] > 0:  # 第k+1大的节约值是否大于0
        x, y = int(S_new[index[k]][0] // S.shape[0]), int(S_new[index[k]][0] % S.shape[0])  # x,y为第x个客户和第y个客户
        if (x in can_use_right) and (y in can_use_left):  # x-0-0-y
            index_x, index_y = 0, 0
            for i in range(len(solve)):
                if x in solve[i][0]:
                    index_x = i
                    break
            for i in range(len(solve)):
                if y in solve[i][0]:
                    index_y = i
                    break

            if (index_x != index_y) and (solve[index_x][1] + solve[index_y][1] < CAR_MAX):  # 确定可以合并
                can_use_left.remove(y)  # 删去x,y的各自的左、右0
                can_use_right.remove(x)
                solve[index_x][0].pop()  # 删去解上的0
                solve[index_y][0].pop(0)
                solve[index_x][0].extend(solve[index_y][0])  # 两列表合并
                solve[index_x][1] += solve[index_y][1]  # 解合并
                del solve[index_y]  # 删元素

        elif (y in can_use_right) and (x in can_use_left):  # y-0-0-x
            index_x, index_y = 0, 0
            for i in range(len(solve)):
                if x in solve[i][0]:
                    index_x = i
                    break
            for i in range(len(solve)):
                if y in solve[i][0]:
                    index_y = i
                    break

            if (index_x != index_y) and (solve[index_x][1] + solve[index_y][1] < CAR_MAX):  # 确定可以合并
                can_use_left.remove(x)  # 删去x,y的各自的左、右0
                can_use_right.remove(y)
                solve[index_y][0].pop()  # 删去解上的0
                solve[index_x][0].pop(0)
                solve[index_y][0].extend(solve[index_x][0])  # 两列表合并
                solve[index_y][1] += solve[index_x][1]  # 解合并
                del solve[index_x]  # 删元素

        elif (x in can_use_left) and (y in can_use_left):
            index_x, index_y = 0, 0
            for i in range(len(solve)):
                if x in solve[i][0]:
                    index_x = i
                    break
            for i in range(len(solve)):
                if y in solve[i][0]:
                    index_y = i
                    break

            if (index_x != index_y) and (solve[index_x][1] + solve[index_y][1] < CAR_MAX):
                can_use_left.remove(x)
                can_use_left.remove(y)
                solve[index_x][0].pop(0)
                solve[index_y][0].pop(0)
                c = solve[index_y][0][-2]
                can_use_right.remove(c)
                can_use_left.append(c)
                solve[index_y][0] = solve[index_y][0][::-1]
                solve[index_y][0].extend(solve[index_x][0])
                solve[index_y][1] += solve[index_x][1]
                del solve[index_x]

        elif (x in can_use_right) and (y in can_use_right):
            index_x, index_y = 0, 0
            for i in range(len(solve)):
                if x in solve[i][0]:
                    index_x = i
                    break
            for i in range(len(solve)):
                if y in solve[i][0]:
                    index_y = i
                    break

            if (index_x != index_y) and (solve[index_x][1] + solve[index_y][1] < CAR_MAX):
                can_use_right.remove(x)
                can_use_right.remove(y)
                solve[index_x][0].pop()
                solve[index_y][0].pop()
                c = solve[index_y][0][1]
                can_use_left.remove(c)
                can_use_right.append(c)
                solve[index_y][0] = solve[index_y][0][::-1]
                solve[index_x][0].extend(solve[index_y][0])
                solve[index_x][1] += solve[index_y][1]
                del solve[index_y]

        k += 1
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

    plt.show()


def main():
    solve_greedy = saving()
    total_distance, initial = desolve(solve_greedy)
    solve_plot(initial)
    return total_distance, initial


main()
