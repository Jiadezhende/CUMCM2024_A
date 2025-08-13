from math import cos, pi, sin, sqrt
import time as tm
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import os

# 设置matplotlib支持中文字体
plt.rcParams['font.sans-serif'] = ['SimHei', 'Microsoft YaHei', 'DejaVu Sans']  # 指定默认字体
plt.rcParams['axes.unicode_minus'] = False  # 解决保存图像是负号'-'显示为方块的问题

# Question1，分析板凳龙的盘入过程
class Model_1:
    def __init__(self, num, head_length, body_length, distance):
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
        return

    # 忽略碰撞情况，且终点处极径不过小，不出现连接点回退
    
    def trace_period(self, theta_start,time,v):
        """
        追踪板凳龙的盘入轨迹
        @param time: 最大追踪时间
        @param v: 速度
        return
        """
        return

    def pos_trace_moment(self, theta_start,time, v,step):
        """
        追踪板凳龙的当前时刻各点位置
        @param theta_start: 起始极角
        @param time: 当前时间
        @param v: 速度
        @param step: 步长
        @return: 包含时刻和位置信息的字典
        """
        # 计算头部位置
        head_position = self.head_trace(theta_start,time,v)
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
        
        return {
            'time': time,
            'positions': res
        }

    # 计算首个满足条件的点(固定步长定位+二分)
    def __private_next(self,theta,length,step,upper_bound=0, max_iter=10000):
        if step<=0: raise ValueError("__private_next: 步长必须为正数")
        start_time = tm.time()
        ans = theta
        tar = 4 * (length ** 2) * (pi ** 2) / (self.distance ** 2) - theta ** 2
        iter_count = 0
        timeout = 5  # 超时时间（秒），可根据需要调整
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
    
    def __private_function1(self, theta1, theta2):
        """
        用于计算相邻连接点速度之间的比例系数
        @param theta1: 连接点的极角
        @param theta2: 后一个连接点的极角
        """
        p1=cos(theta2-theta1)
        p2=theta1*theta2*sin(theta2-theta1)
        p3=theta1**2
        p4=theta2**2
        return sqrt((p4+1)/(p3+1))*(theta2*p1+p2-theta1)/(-theta1*p1+p2+theta2)

    # 计算头部在某一时刻的极角
    def head_trace(self,theta_start,time,v):
        """
        计算头部在某一时刻的极角
        @param theta_start: 起始极角
        @param time: 当前时间
        @param v: 速度
        """
        return sqrt(theta_start**2 - 4*pi*v*time/self.distance)
    
    def v_trace_moment(self, pos_result, v):
        """
        计算各连接点在某一时刻的速度
        @param pos_result: trace_moment方法返回的结果字典，包含time和positions
        @param v: 龙头速度
        @return: 包含时刻和速度信息的字典
        """
        time = pos_result['time']
        theta_list = pos_result['positions']
        
        res=[v]
        for i in range(len(theta_list)-1):
            res.append(self.__private_function1(theta_list[i], theta_list[i+1]) * res[i])
        
        return {
            'time': time,
            'velocities': res
        }

    # 可视化函数：绘制等距螺线和连接各点
    def visualize(self, result_data, title=None):
        """
        绘制等距螺线和给定极角序列对应的点
        @param result_data: trace_moment方法返回的结果字典，包含time和positions
        @param title: 图表标题，如果为None则自动生成
        """
        time = result_data['time']
        theta_sequence = result_data['positions']
        
        if title is None:
            title = f"板凳龙轨迹可视化 (t={time}s)"
        
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

    def save_position(self, result_data, output_file):
        """
        保存某一时刻连接点位置
        @param result_data: trace_moment方法返回的结果字典，包含time和positions
        @param output_file: 输出文件路径
        """
        time = result_data['time']
        theta_list = result_data['positions']
        
        # 确保输出目录存在
        os.makedirs(os.path.dirname(output_file), exist_ok=True)
        
        # 将极角转换为笛卡尔坐标
        positions = []
        for theta in theta_list:
            rho = self.distance / (2 * pi) * theta
            x = rho * cos(theta)
            y = rho * sin(theta)
            positions.append((x, y))
        
        # 创建列名和数据
        column_name = f"{time} s"
        data = []
        
        # 添加龙头位置
        data.append(positions[0][0])  # 龙头x
        data.append(positions[0][1])  # 龙头y
        
        # 添加龙身位置
        for i in range(1, len(positions)):
            data.append(positions[i][0])  # 第i节龙身x
            data.append(positions[i][1])  # 第i节龙身y
        
        # 创建行索引
        row_labels = ['龙头x (m)', '龙头y (m)']
        for i in range(1, len(positions)):
            row_labels.append(f'第{i}节龙身x (m)')
            row_labels.append(f'第{i}节龙身y (m)')
        
        # 尝试读取现有文件
        if os.path.exists(output_file):
            try:
                df = pd.read_excel(output_file, index_col=0)
            except:
                # 如果文件损坏或格式不对，创建新的DataFrame
                df = pd.DataFrame(index=row_labels)
        else:
            # 创建新的DataFrame
            df = pd.DataFrame(index=row_labels)
        
        # 添加新列
        df[column_name] = data
        
        # 保存到Excel文件
        df.to_excel(output_file)
        print(f"位置数据已保存到: {output_file}")
        print(f"添加了时刻: {column_name}")

    def save_velocity(self, velocity_result_data, output_file):
        """
        保存某一时刻连接点速度
        @param velocity_result_data: v_trace方法返回的结果字典，包含time和velocities
        @param output_file: 输出文件路径
        """
        time = velocity_result_data['time']
        velocity_list = velocity_result_data['velocities']
        
        # 确保输出目录存在
        os.makedirs(os.path.dirname(output_file), exist_ok=True)
        
        # 创建列名和数据
        column_name = f"{time} s"
        data = []
        
        # 添加龙头速度
        data.append(velocity_list[0])  # 龙头速度
        
        # 添加龙身速度
        for i in range(1, len(velocity_list)):
            data.append(velocity_list[i])  # 第i节龙身速度
        
        # 创建行索引
        row_labels = ['龙头 (m/s)']
        for i in range(1, len(velocity_list)):
            row_labels.append(f'第{i}节龙身 (m/s)')
        
        # 尝试读取现有文件
        if os.path.exists(output_file):
            try:
                df = pd.read_excel(output_file, index_col=0)
            except:
                # 如果文件损坏或格式不对，创建新的DataFrame
                df = pd.DataFrame(index=row_labels)
        else:
            # 创建新的DataFrame
            df = pd.DataFrame(index=row_labels)
        
        # 添加新列
        df[column_name] = data
        
        # 保存到Excel文件
        df.to_excel(output_file)
        print(f"速度数据已保存到: {output_file}")
        print(f"添加了时刻: {column_name}")
