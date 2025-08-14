from model_1 import Model_1
import numpy as np
import matplotlib.pyplot as plt

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

class Model_2(Model_1):
    def __init__(self, num, head_length, body_length, distance, extension, width):
        """
        初始化碰撞模型参数
        @param num: 板凳龙数量
        @param head_length: 头部长度
        @param body_length: 身体长度
        @param distance: 初始距离
        @param extension: 连接点到窄边的距离
        @param width: 宽度
        """
        super().__init__(num, head_length, body_length, distance)
        self.extension = extension
        self.width = width

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
                        print(f"Collision detected between bench {i} and bench {j} at time {pos_result['time']}")
                        return True
            else:
                for j in range(i+2,len(theta_list)-1):
                    if theta_list[j]-theta_list[i]>4*np.pi:
                        break
                    if self.__private_isCollided(theta_list[j],theta_list[j+1],theta_list[i],theta_list[i+1]):
                        print(f"Collision detected between bench {i} and bench {j} at time {pos_result['time']}")
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