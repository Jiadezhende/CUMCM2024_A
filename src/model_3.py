from math import cos, pi, sin, sqrt
import time as tm
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import os

# 设置matplotlib支持中文字体
plt.rcParams['font.sans-serif'] = ['SimHei', 'Microsoft YaHei', 'DejaVu Sans']  # 指定默认字体
plt.rcParams['axes.unicode_minus'] = False  # 解决保存图像是负号'-'显示为方块的问题

def transform_matrix(pA_1, pA_2, pB_1, pB_2):
    """
    根据两个二维坐标系的 y 轴方向向量（未单位化）和原点坐标，
    构造从 A 到 B 的二维齐次变换矩阵。
    y 轴顺时针 90° 为 x 轴方向。
    @param pA_1: 第一个坐标系的连接点1的直角坐标
    @param pA_2: 第一个坐标系的连接点2的直角坐标
    @param pB_1: 第二个坐标系的连接点1的直角坐标
    @param pB_2: 第二个坐标系的连接点2的直角坐标
    """

    def normalize(v):
        if np.linalg.norm(v) == 0:
            raise ValueError("Zero vector cannot be normalized.")
        return v / np.linalg.norm(v)

    def rot90_cw(v):
        """将二维向量顺时针旋转 90°"""
        if len(v) != 2:
            raise ValueError("Input vector must be 2-dimensional.")
        return np.array([v[1], -v[0]])

    # 验证输入是否为 np.array 且维度正确
    for point in [pA_1, pA_2, pB_1, pB_2]:
        if not isinstance(point, np.ndarray):
            raise TypeError("All input points must be of type np.array.")
        if point.shape != (2,):
            raise ValueError("All input points must be 2-dimensional.")

    tA = (pA_1 + pA_2) / 2
    tB = (pB_1 + pB_2) / 2
    # print("tA-tB:")
    # print(tA - tB)

    # 单位化 y 轴向量
    yA_u = normalize(pA_1 - pA_2)
    yB_u = normalize(pB_1 - pB_2)

    # 得到 x 轴向量（y 轴顺时针 90°）
    xA_u = rot90_cw(yA_u)
    xB_u = rot90_cw(yB_u)

    # 构造 R_WA, R_WB (列向量分别是 x, y)
    R_WA = np.column_stack((xA_u, yA_u))
    R_WB = np.column_stack((xB_u, yB_u))

    # print("R_WA:")
    # print(R_WA)
    # print("R_WB:")
    # print(R_WB)

    # 检查 R_WB 是否可逆
    if np.linalg.det(R_WB) == 0:
        raise ValueError("Matrix R_WB is not invertible.")

    # 从 A 到 B 的旋转和平移
    R_BA = np.linalg.inv(R_WB) @ R_WA
    t_BA = np.linalg.inv(R_WB) @ (tA - tB)  # 一维数组被@解释为列向量，结果仍为一维数组
    # print("R_BA:")
    # print(R_BA)
    # print("t_BA:")
    # print(t_BA)

    # 齐次变换矩阵
    T_BA = np.eye(3)
    T_BA[0:2, 0:2] = R_BA
    T_BA[0:2, 2] = t_BA

    return T_BA

# Question1，分析板凳龙的盘入过程
class Model_3:
    def __init__(self, num, head_length, body_length, distance,extension,width):
        """
        @param num: 板凳龙节数
        @param head_length: 板凳龙头部的长度
        @param body_length: 板凳龙每段身体的长度
        @param distance: 螺距
        """
        self.distance = distance
        self.num = num
        self.head_length = head_length
        self.body_length = body_length
        self.extension = extension
        self.width = width

    def min_distance(self,r,lower_bound,upper_bound,iter,step):
        """
        二分法计算最小螺距
        @param r: 掉头空间半径
        @param lower_bound: 螺距下限
        @param upper_bound: 螺距上限
        @param iter: 迭代次数
        @return: 最小螺距
        """
        
        for i in range(iter):
            self.distance = (lower_bound + upper_bound) / 2
            print(f"Iteration {i+1}/{iter}: Current distance = {self.distance}")
            alpha=2 * pi*r / self.distance  # 螺距与极角的关系
            if self.__private_accessible(alpha, step):
                upper_bound = self.distance
            else:
                lower_bound = self.distance
        return upper_bound
    
    def __private_accessible(self,alpha,step):
        limit=alpha + 2.5*np.pi  # 计算极角的上限
        while alpha<limit:
            if  self.trace_collision(self.pos_trace_moment(alpha, 0.01)):
                return False
            alpha += step
        return True

    def pos_trace_moment(self, theta,step):
        """
        追踪板凳龙的当前时刻各点位置
        @param theta: 龙头极角
        @param step: 步长
        @return: 包含时刻和位置信息的字典
        """
        # 计算头部位置
        head_position = theta
        res=[]
        res.append(head_position)
        for i in range(self.num):
            if i == 0:
                # 第一节：头部连接到第一节身体
                next_pos = self.__private_next(res[i], self.head_length, step)
                res.append(next_pos)
            else:
                # 其他节：身体之间的连接
                next_pos = self.__private_next(res[i], self.body_length, step)
                res.append(next_pos)
        # print(f'位置生成完毕: {theta}')
        return {
            'head': theta,
            'positions': res
        }

    # 计算首个满足条件的点(固定步长定位+二分)
    def __private_next(self,theta,length,step,upper_bound=0, max_iter=10000):
        if step<=0: raise ValueError("__private_next: 步长必须为正数")
        start_time = tm.time()
        ans = theta
        tar = 4 * (length ** 2) * (np.pi ** 2) / (self.distance ** 2) - theta ** 2
        iter_count = 0
        timeout = 3  # 超时时间（秒），可根据需要调整
        if upper_bound:
            while ans < upper_bound:
                ans += step
                iter_count += 1
                if tm.time() - start_time > timeout or iter_count > max_iter:
                    raise TimeoutError("__private_next 超时或迭代次数过多")
                if tar < ans ** 2 - 2 * theta * ans * cos(theta - ans):
                    break
        else:
            while True:
                ans += step
                iter_count += 1
                if tm.time() - start_time > timeout or iter_count > max_iter:
                    raise TimeoutError("__private_next 超时或迭代次数过多")
                if tar < ans ** 2 - 2 * theta * ans * cos(theta - ans):
                    break
        # 至此，确定有一个数值解在(ans-step,ans)之间
        l = ans - step
        r = ans
        for i in range(100):
            ans = (l + r) / 2
            if tar < ans ** 2 - 2 * theta * ans * cos(theta - ans):
                l = ans
            else:
                r = ans
        return ans

    # 可视化函数：绘制等距螺线和连接各点
    def visualize(self, result_data, title=None):
        """
        绘制等距螺线和给定极角序列对应的点
        @param result_data: trace_moment方法返回的结果字典，包含time和positions
        @param title: 图表标题，如果为None则自动生成
        """
        head_position = result_data['head']
        theta_sequence = result_data['positions']
        
        if title is None:
            title = f"板凳龙轨迹可视化 (head={head_position}rad)"
        
        fig, ax = plt.subplots(1, 1, figsize=(10, 10), subplot_kw=dict(projection='polar'))
        
        # 绘制等距螺线（逆时针向外）
        theta_spiral = np.linspace(0, max(theta_sequence) * 1.2, 1000)
        rho_spiral = self.distance / (2 * pi) * theta_spiral
        ax.plot(theta_spiral, rho_spiral, 'grey', alpha=0.6, label='等距螺线')
        
        # 计算给定极角序列对应的极径
        rho_points = [self.distance / (2 * pi) * theta for theta in theta_sequence]
        
        # 在螺线上标出各点
        ax.scatter(theta_sequence, rho_points, c='red', s=15, zorder=5, label='连接点')
        
        # 用直线连接各点
        ax.plot(theta_sequence, rho_points, 'blue', linewidth=2, label='板凳龙', zorder=4)
        
        # 标注各点
        for i, (theta, rho) in enumerate(zip(theta_sequence, rho_points)):
            ax.annotate(f'P{i}', (theta, rho), xytext=(5, 5), 
                       textcoords='offset points', fontsize=8)
        
        ax.set_title(title, pad=20)
        ax.legend(loc='upper right', bbox_to_anchor=(1.2, 1.0))
        ax.grid(True, alpha=0.3)
        plt.tight_layout()
        plt.show()
        
        return fig, ax
    
    def trace_collision(self, pos_result):
        # 追踪位置并检测碰撞
        """
        该方法用于检测位置结果是否发生碰撞。
        @param pos_result: 位置结果
        @return: 是否发生碰撞
        """
        theta_list=pos_result['positions']
        for i in range(len(theta_list)):
            if i==0:
                for j in range(i+2,len(theta_list)-1):
                    if theta_list[j]-theta_list[i]>4*np.pi:
                        break
                    if self.__private_isCollided(theta_list[j],theta_list[j+1],theta_list[i],theta_list[i+1],True):
                        print(f"Collision detected between bench {i} and bench {j}. Head at {pos_result['head']} rad")
                        return True
            else:
                for j in range(i+2,len(theta_list)-1):
                    if theta_list[j]-theta_list[i]>4*np.pi:
                        break
                    if self.__private_isCollided(theta_list[j],theta_list[j+1],theta_list[i],theta_list[i+1]):
                        print(f"Collision detected between bench {i} and bench {j}. Head at {pos_result['head']} rad")
                        return True
        return False

    def __private_isCollided(self, pA_1, pA_2, pB_1, pB_2, isHead=False):
        """
        私有方法：判断AB两个板凳是否发生碰撞。
        @param pA_1: A板凳的连接点1的极角
        @param pA_2: A板凳的连接点2的极角
        @param pB_1: B板凳的连接点1的极角
        @param pB_2: B板凳的连接点2的极角
        @param isHead: B板凳是否为头部
        @return: 是否发生碰撞
        """
        if self.__private_isOverlapped(pA_1, pA_2, pB_1, pB_2, isHead) and self.__private_isOverlapped(pB_1, pB_2, pA_1, pA_2):
            self.__private_isOverlapped(pA_1, pA_2, pB_1, pB_2, isHead,True)
            return True
        return False

    def __private_isOverlapped(self, pA_1,pA_2,pB_1,pB_2,isHead=False,show=False):
        """
        私有方法：判断A在B坐标系上的投影是否有重合。
        @param pA_1: A板凳的连接点1的极角
        @param pA_2: A板凳的连接点2的极角
        @param pB_1: B板凳的连接点1的极角
        @param pB_2: B板凳的连接点2的极角
        @param isHead: B板凳是否为头部
        @return: 是否发生重合
        """

        # 极坐标转换为直角坐标
        pA_1_xy = np.array([self.distance / (2 * np.pi) * pA_1 * np.cos(pA_1), self.distance / (2 * np.pi) * pA_1 * np.sin(pA_1)])
        pA_2_xy = np.array([self.distance / (2 * np.pi) * pA_2 * np.cos(pA_2), self.distance / (2 * np.pi) * pA_2 * np.sin(pA_2)])
        pB_1_xy = np.array([self.distance / (2 * np.pi) * pB_1 * np.cos(pB_1), self.distance / (2 * np.pi) * pB_1 * np.sin(pB_1)])
        pB_2_xy = np.array([self.distance / (2 * np.pi) * pB_2 * np.cos(pB_2), self.distance / (2 * np.pi) * pB_2 * np.sin(pB_2)])

        # 计算变换矩阵
        T_BA = transform_matrix(pA_1_xy, pA_2_xy, pB_1_xy, pB_2_xy)

        A_length=self.body_length
        # A坐标下，四角坐标
        rectA = np.array([
            [-self.width/2,A_length/2 + self.extension],  #左上
            [self.width/2,A_length/2 + self.extension],  #右上
            [self.width/2,-A_length/2 - self.extension],  #右下
            [-self.width/2,-A_length/2 - self.extension]  #左下
        ])

        # print("RectA:")
        # print(rectA)

        if isHead:
            B_length=self.head_length
        else:
            B_length=self.body_length

        # 计算变换后的坐标
        rectA_transformed = np.array([T_BA @ np.array([*corner, 1]) for corner in rectA])

        # 检测碰撞
        if (np.min(rectA_transformed[:,0])>=self.width/2) or (np.max(rectA_transformed[:,0])<=-self.width/2):
            return False
        if (np.min(rectA_transformed[:,1])>=B_length/2 + self.extension) or (np.max(rectA_transformed[:,1])<=-B_length/2 - self.extension):
            return False
        
        if show:
            # 绘制碰撞示意图
            plt.figure(figsize=(6, 6))

            # 绘制 rectB，此处利用了A的数据
            rectA_closed = np.vstack([rectA, rectA[0]])  # 闭合矩形
            plt.plot(rectA_closed[:, 0], rectA_closed[:, 1], label="rectB", color="blue")

            # 绘制 rectA_transformed
            rectA_transformed_closed = np.vstack([rectA_transformed[:, :2], rectA_transformed[0, :2]])  # 闭合矩形
            plt.plot(rectA_transformed_closed[:, 0], rectA_transformed_closed[:, 1], label="rectA", color="red")

            # 设置图形属性
            plt.axhline(0, color='black', linewidth=0.5, linestyle='--')
            plt.axvline(0, color='black', linewidth=0.5, linestyle='--')
            plt.gca().set_aspect('equal', adjustable='box')
            plt.legend()
            plt.title("Collision Visualization")
            plt.xlabel("X")
            plt.ylabel("Y")
            plt.grid(True)
            plt.show()

        return True