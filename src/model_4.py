import numpy as np


class Model_4:
    """
    对于每一个position，均用一个标识位说明其所在曲线段
    盘入螺线，圆弧A，圆弧B，盘出螺线
    用极角恰可以表示连接点在取线段上的坐标
    """
    def __init__(self,num, head_length, body_length, distance, extension, width,r,k):
        """
        初始化完整运动模型参数
        @param num: 板凳龙数量
        @param head_length: 头部长度
        @param body_length: 身体长度
        @param distance: 螺距
        @param extension: 连接点到窄边的距离
        @param width: 宽度
        @param r: 掉头区域半径
        @param k: r_A/r_B
        """
        self.num = num
        self.head_length = head_length
        self.body_length = body_length
        self.distance = distance
        self.extension = extension
        self.width = width
        self.r = r
        self.k = k

        # 推理得出的量
        self.__private_alpha()
        self.__private_r_AB()
        self.__private_core_AB()

    def __private_alpha(self):
        """
        更新切点的极角
        @return: 切点的极角
        """
        self.alpha = 2 * np.pi * self.r / self.distance

    def __private_r_AB(self):
        """
        更新圆弧A，B的半径
        """
        self.__private_alpha()
        alpha=self.alpha
        self.r_B = self.r * np.sqrt(1 + alpha**2) / (np.abs(np.sin(alpha) + alpha * np.cos(alpha)) * (self.k+1))
        self.r_A =  self.k * self.r_B

    def __private_core_AB(self):
        """
        计算圆弧A，B的圆心，同时设定圆弧边界条件
        @return: 圆弧A，B的圆心坐标
        """
        self.__private_alpha()
        self.__private_r_AB()
        alpha=self.alpha
        # 首先计算法向量
        norm=np.sqrt(1 + alpha**2)
        n=np.array([-np.sin(alpha)-alpha*np.cos(alpha),np.cos(alpha)-alpha*np.sin(alpha)])
        # 切点坐标
        x_A = self.r * np.cos(alpha)
        y_A = self.r * np.sin(alpha)
        x_B = -x_A
        y_B = -y_A
        # 圆心坐标
        self.core_A = (x_A + n[0] * self.r_A/norm, y_A + n[1] * self.r_A/norm)
        self.core_B = (x_B - n[0] * self.r_B/norm, y_B - n[1] * self.r_B/norm)
        
        # 计算边界角度
        # 对于圆弧A：从-n向量角度到core_B-core_A向量角度
        n_angle = np.arctan2(n[1], n[0])  # n向量的角度
        neg_n_angle = np.arctan2(-n[1], -n[0])  # -n向量的角度
        
        # core_B - core_A 向量的角度
        diff_BA = np.array([self.core_B[0] - self.core_A[0], self.core_B[1] - self.core_A[1]])
        angle_BA = np.arctan2(diff_BA[1], diff_BA[0])
        
        # core_A - core_B 向量的角度
        diff_AB = np.array([self.core_A[0] - self.core_B[0], self.core_A[1] - self.core_B[1]])
        angle_AB = np.arctan2(diff_AB[1], diff_AB[0])
        
        # 将角度规范化到[0, 2π)
        def normalize_angle(angle):
            return angle % (2 * np.pi)
        
        neg_n_angle = normalize_angle(neg_n_angle)
        n_angle = normalize_angle(n_angle)
        angle_BA = normalize_angle(angle_BA)
        angle_AB = normalize_angle(angle_AB)
        
        # 圆弧A的边界：从-n向量角度到core_B-core_A向量角度
        self.boundary_A = (angle_BA,neg_n_angle)
        
        # 圆弧B的边界：从n向量角度到core_A-core_B向量角度
        self.boundary_B = (angle_AB,n_angle)

    def get_core_A(self):
        """
        获取圆弧A的圆心
        @return: 圆弧A的圆心坐标
        """
        return self.core_A

    def get_core_B(self):
        """
        获取圆弧B的圆心
        @return: 圆弧B的圆心坐标
        """
        return self.core_B
    
    def get_r_A(self):
        """
        获取圆弧A的半径
        @return: 圆弧A的半径
        """
        return self.r_A

    def get_r_B(self):
        """
        获取圆弧B的半径
        @return: 圆弧B的半径
        """
        return self.r_B
    
    def get_alpha(self):
        """
        获取切点的极角
        @return: 切点的极角
        """
        return self.alpha
    
    def get_boundary_A(self):
        """
        获取圆弧A的边界角度范围
        @return: (起始角度, 终止角度) - 角度范围为[0, 2π)
        """
        return self.boundary_A
    
    def get_boundary_B(self):
        """
        获取圆弧B的边界角度范围
        @return: (起始角度, 终止角度) - 角度范围为[0, 2π)
        """
        return self.boundary_B

    def is_in_A(self, angle):
        """
        判断给定角度是否在圆弧A的范围内
        @param angle: 待判断的角度
        @return: True - 在范围内，False - 不在范围内
        """
        # 将传入的角度规范化到[0, 2π)
        normalized_angle = angle % (2 * np.pi)
        
        start, end = self.boundary_A
        if start < end:
            return start <= normalized_angle <= end
        else:
            return start <= normalized_angle or normalized_angle <= end
        
    def is_in_B(self, angle):
        """
        判断给定角度是否在圆弧B的范围内
        @param angle: 待判断的角度
        @return: True - 在范围内，False - 不在范围内
        """
        # 将传入的角度规范化到[0, 2π)
        normalized_angle = angle % (2 * np.pi)
        
        start, end = self.boundary_B
        if start < end:
            return start <= normalized_angle <= end
        else:
            return start <= normalized_angle or normalized_angle <= end
        
    def head_trace(self,time,v):
        """
        获取龙头位置
        @return: 龙头位置
        """
        # 此处以掉头位置为0时刻
        if time>-1E-10:
            pass
        pass