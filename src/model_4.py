import os
import numpy as np
import time as tm
import matplotlib.pyplot as plt
import pandas as pd

# 设置中文字体支持
plt.rcParams['font.sans-serif'] = ['SimHei', 'Microsoft YaHei', 'DejaVu Sans']  # 中文字体
plt.rcParams['axes.unicode_minus'] = False  # 正确显示负号


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
        self.__private_AB()
        print(f"Model_4 initialized with alpha={self.alpha:.4f}, r_A={self.r_A:.4f}, r_B={self.r_B:.4f}")

    def __private_alpha(self):
        """
        更新切点的极角
        @return: 切点的极角
        """
        self.alpha = 2 * np.pi * self.r / self.distance

    def __private_AB(self):
        """
        更新圆弧A，B的半径，AB坐标
        """
        self.__private_alpha()
        alpha=self.alpha
        # 首先计算法向量
        norm=np.sqrt(1 + alpha**2)
        n=((-np.sin(alpha)-alpha*np.cos(alpha))/norm,(np.cos(alpha)-alpha*np.sin(alpha))/norm)
        print(f"法向量: {n}")
        a=(-np.cos(alpha), -np.sin(alpha))
        # 计算a与n的夹角cos值
        cos_angle = np.abs(a[0] * n[0] + a[1] * n[1])

        # 计算半径
        self.r_B = self.r / cos_angle/(self.k+1)
        self.r_A =  self.k * self.r_B
        # 切点坐标
        x_A = self.r * np.cos(alpha)
        y_A = self.r * np.sin(alpha)
        x_B = -x_A
        y_B = -y_A
        # 圆心坐标
        self.core_A = (x_A + n[0] * self.r_A, y_A + n[1] * self.r_A)
        self.core_B = (x_B - n[0] * self.r_B, y_B - n[1] * self.r_B)
        
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

    def trace_period(self, min_time, max_time, v, step=0.01, output_dir="../data/output"):
        """
        追踪板凳龙的完整运动轨迹，记录每秒的位置和速度
        @param min_time: 最小追踪时间（秒）
        @param max_time: 最大追踪时间（秒）
        @param v: 龙头速度
        @param step: 计算步长
        @param output_dir: 输出目录
        @return: 包含所有时刻数据的字典
        """
        import datetime
        
        # 创建输出目录
        os.makedirs(output_dir, exist_ok=True)
        
        # 生成时间戳用于文件命名
        timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
        pos_file = os.path.join(output_dir, f"q4_positions_{timestamp}.xlsx")
        vel_file = os.path.join(output_dir, f"q4_velocities_{timestamp}.xlsx")
        
        print(f"开始追踪板凳龙完整轨迹，时间范围: {min_time}-{max_time}秒")
        print(f"龙头速度: {v} m/s")
        print(f"计算步长: {step}")
        
        all_positions = []
        all_velocities = []
        
        # 逐秒计算位置和速度
        for t in range(int(min_time), int(max_time) + 1):
            try:
                # 检查头部是否还在有效范围内
                head_position = self.head_trace(t, v)
                if head_position is None:
                    print(f"警告: 在t={t}s时头部位置无效，停止计算")
                    break
                
                # 计算当前时刻的位置
                pos_result = self.pos_trace_moment(t, v, step)
                if pos_result is None or any(pos is None for pos in pos_result['positions']):
                    print(f"警告: 在t={t}s时部分位置计算失败，跳过此时刻")
                    continue
                
                all_positions.append(pos_result)
                
                # 计算当前时刻的速度
                vel_result = self.v_trace_moment(pos_result, v)
                all_velocities.append(vel_result)
                
                # 保存到Excel文件（批量处理）
                self.save_position(pos_result, pos_file)
                self.save_velocity(vel_result, vel_file)
                
                if t % 10 == 0:  # 每10秒输出一次进度
                    print(f"已完成 t={t}s 的计算")
                    
            except Exception as e:
                print(f"在t={t}s时发生错误: {e}")
                continue
        
        print(f"轨迹追踪完成，共计算了 {len(all_positions)} 个时刻")
        print(f"位置数据保存至: {pos_file}")
        print(f"速度数据保存至: {vel_file}")
        
        return {
            'positions_timeline': all_positions,
            'velocities_timeline': all_velocities,
            'position_file': pos_file,
            'velocity_file': vel_file,
            'total_time_points': len(all_positions)
        }

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
        
    def pos_trace_moment(self, time, v,step):
        """
        追踪板凳龙的当前时刻各点位置
        @param theta_start: 起始极角
        @param time: 当前时间
        @param v: 速度
        @param step: 步长
        @return: 包含时刻和位置信息的字典
        """
        # 计算头部位置
        head_position = self.head_trace(time,v)
        print(f"头部位置: {head_position}")

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
    
    def v_trace_moment(self, pos_result, v):
        """
        计算各连接点在某一时刻的速度
        @param pos_result: trace_moment方法返回的结果字典，包含time和positions
        @param v: 龙头速度
        @return: 包含时刻和速度信息的字典
        """
        time = pos_result['time']
        pos_list = pos_result['positions']
        
        res=[v]
        for i in range(len(pos_list)-1):
            res.append(self.__private_frac_v(pos_list[i], pos_list[i+1]) * res[i])
        
        return {
            'time': time,
            'velocities': res
        }
    
    def __private_frac_v(self, pos1, pos2):
        """
        计算两个位置之间的速度比
        @param pos1: 第一个位置元组 (极角, 所在区域)
        @param pos2: 第二个位置元组 (极角, 所在区域)
        @return: 速度比v2/v1
        """
        # 首先获取速度的单位向量
        v1 = self.__private_e_v(pos1)
        v2 = self.__private_e_v(pos2)
        xy1=self.pos_t_xy(pos1)
        xy2=self.pos_t_xy(pos2)
        xy_diff = (xy2[0] - xy1[0], xy2[1] - xy1[1])
        return (v1[0] * xy_diff[0] + v1[1] * xy_diff[1]) / (v2[0] * xy_diff[0] + v2[1] * xy_diff[1])

    def __private_e_v(self, pos):
        """
        计算当前位置的速度单位向量
        @param pos: 当前位置 (极角，所在区域)
        @return: 速度单位向量 (vx, vy)
        """
        if pos[1] == 0:
            norm=np.sqrt(1 + pos[0]**2)
            return ((-np.cos(pos[0])+pos[0]*np.sin(pos[0]))/norm,(-np.sin(pos[0])-pos[0]*np.cos(pos[0]))/norm)
        elif pos[1] == 1 or pos[1] == 3:
            # case 1: 圆弧A上
            return (np.sin(pos[0]), -np.cos(pos[0]))
        elif pos[1] == 2:
            # case 2: 圆弧B上
            return (-np.sin(pos[0]), np.cos(pos[0]))
        else:
            return None

    def head_trace(self,time,v):
        """
        获取龙头位置
        @param time: 当前时刻
        @param v: 龙头速度
        @return: 龙头位置
        """
        # 此处以掉头位置为0时刻
        if time<1E-10:
            # case 0: 盘入螺线上
            return (np.sqrt(self.alpha**2-4*np.pi*v*time/self.distance), 0)
        else:
            s=v*time
            a=self.boundary_A[1]-self.boundary_A[0]
            if a<0: a+=2*np.pi
            s1=self.r_A*a
            s2=self.r_B*a
            if(s<s1):
                # case 1: 圆弧A上
                theta=self.boundary_A[1]-s/self.r_A
                return (theta% (2 * np.pi), 1)
            elif(s<s1+s2):
                # case 2: 圆弧B上
                theta=self.boundary_B[0]+s/self.r_B
                return (theta% (2 * np.pi), 2)
            else:
                # case 3: 盘出螺线上
                theta=np.sqrt(self.alpha**2+4*np.pi*(v*time-s1-s2)/self.distance)
                return (theta, 3)
    
    def pos_t_xy(self,position):
        """
        获取给定位置的笛卡尔坐标
        @param position: 位置元组 (极角, 所在区域)
        @return: 笛卡尔坐标 (x, y)
        """
        if position[1] == 0:
            # case 0: 盘入螺线上
            x=self.distance*position[0]*np.cos(position[0])/(2*np.pi)
            y=self.distance*position[0]*np.sin(position[0])/(2*np.pi)
            return (x, y)
        elif position[1] == 1:
            # case 1: 圆弧A上
            x=self.r_A * np.cos(position[0])+self.core_A[0]
            y=self.r_A * np.sin(position[0])+self.core_A[1]
            return (x, y)
        elif position[1] == 2:
            # case 2: 圆弧B上
            x=self.r_B * np.cos(position[0])+self.core_B[0]
            y=self.r_B * np.sin(position[0])+self.core_B[1]
            return (x, y)
        elif position[1] == 3:
            # case 3: 盘出螺线上
            x=-self.distance*position[0]*np.cos(position[0])/(2*np.pi)
            y=-self.distance*position[0]*np.sin(position[0])/(2*np.pi)
            return (x, y)
        else:
            return None

    def __private_next(self,position,length,step):
        """
        计算下一个位置，依次尝试各个区域直到找到有效位置。
        策略：依次尝试 private_next_i，其中 i <= position[1] 且 i 递减
        @param position: 当前的位置
        @param length: 龙头/龙身的长度
        @param step: 步长
        @return: 下一个位置
        """
        current_region = position[1]
        
        # 从当前区域开始，递减尝试各个区域
        for i in range(current_region, -1, -1):
            try:
                if i == 0:
                    result = self.__private_next_0(position, length, step)
                elif i == 1:
                    result = self.__private_next_1(position, length, step)
                elif i == 2:
                    result = self.__private_next_2(position, length, step)
                elif i == 3:
                    result = self.__private_next_3(position, length, step)
                else:
                    continue
                
                if result is not None:
                    return result
                    
            except (ValueError, TimeoutError):
                continue
        
        # 如果所有区域都没有找到有效位置，返回None
        return None
        
    def __private_next_0(self,position,length,step,upper_bound=0.0):
        """
        下一个点在盘入螺线上
        @param position: 当前的位置
        @param length: 龙头/龙身的长度
        @param step: 步长
        @param upper_bound: 上界
        @param max_iter: 最大迭代次数
        """
        if step<=0: raise ValueError("__private_next: 步长必须为正数")
        timeout = 3  # 超时时间（秒），可根据需要调整
        start_time = tm.time()
        if position[1] == 0:
            ans = position[0] + step    # 当前点在盘入螺线上，从当前位置开始尝试
        else:
            ans = self.alpha    # 当前点不在盘入螺线，从切点开始尝试
        current_pos = self.pos_t_xy(position)

        def func(cur,tar):
            return length**2-(cur[0]-tar[0])**2-(cur[1]-tar[1])**2

        if upper_bound>1e-6:
            while ans < upper_bound:
                if tm.time() - start_time > timeout:
                    raise TimeoutError("__private_next 超时")
                if func(current_pos, self.pos_t_xy((ans,0))) < 0:
                    break
                ans += step
        else:
            while True:
                if tm.time() - start_time > timeout:
                    raise TimeoutError("__private_next 超时")
                if func(current_pos, self.pos_t_xy((ans,0))) < 0:
                    break
                ans += step
        # 至此，确定有一个数值解在(ans-step,ans)之间
        l = ans-step
        r = ans
        for i in range(100):
            ans = (l + r) / 2
            if func(current_pos, self.pos_t_xy((ans,0))) < 0:
                r = ans
            else:
                l = ans
        return (ans, 0)

    def __private_next_1(self,position,length,step):
        """
        后续点在圆弧A
        @param position: 当前的位置
        @param length: 龙头/龙身的长度
        @param step: 步长
        @param upper_bound: 上界
        """
        if step<=0: raise ValueError("__private_next: 步长必须为正数")
        if position[1]<1: raise ValueError("__private_next: 当前点尚未未经过圆弧A")
        start_time = tm.time()
        timeout = 3  # 超时时间（秒），可根据需要调整

        if position[1] == 1:
            ans= position[0]+length/self.r_A
        else:
            ans=self.boundary_A[0]
            current_pos = self.pos_t_xy(position)
            def func(cur,tar):
                return length**2-(cur[0]-tar[0])**2-(cur[1]-tar[1])**2
            while self.is_in_A(ans):
                if tm.time() - start_time > timeout:
                    raise TimeoutError("__private_next 超时")
                if func(current_pos, self.pos_t_xy((ans,1))) < 0:
                    break
                ans += step
            # 至此，确定有一个数值解在(ans-step,ans)之间
            l = ans-step
            r = ans
            for i in range(100):
                ans = (l + r) / 2
                if func(current_pos, self.pos_t_xy((ans,1))) < 0:
                    r = ans
                else:
                    l = ans
        if self.is_in_A(ans):
            return (ans%(2*np.pi), 1)
        else:
            return None

    def __private_next_2(self,position,length,step):
        """
        后续点在圆弧B
        @param position: 当前的位置
        @param length: 龙头/龙身的长度
        @param step: 步长
        """
        if step<=0: raise ValueError("__private_next: 步长必须为正数")
        if position[1]<2: raise ValueError("__private_next: 当前点尚未未经过圆弧B")
        start_time = tm.time()
        timeout = 3  # 超时时间（秒），可根据需要调整

        if position[1] == 2:
            ans=position[0]-length/self.r_B
        else:
            ans=self.boundary_B[1]
            current_pos = self.pos_t_xy(position)
            def func(cur,tar):
                return length**2-(cur[0]-tar[0])**2-(cur[1]-tar[1])**2
            while self.is_in_B(ans):
                if tm.time() - start_time > timeout:
                    raise TimeoutError("__private_next 超时")
                if func(current_pos, self.pos_t_xy((ans,2))) < 0:
                    break
                ans -= step
            # 至此，确定有一个数值解在(ans,ans+step)之间
            l = ans
            r = ans + step
            for i in range(100):
                ans = (l + r) / 2
                if func(current_pos, self.pos_t_xy((ans,2))) < 0:
                    l = ans
                else:
                    r = ans
        if self.is_in_B(ans):
            return (ans%(2*np.pi), 2)
        else:
            return None

    def __private_next_3(self,position,length,step):
        """
        后续点在盘出螺线上
        @param position: 当前的位置
        @param length: 龙头/龙身的长度
        @param step: 步长
        """
        if step<=0: raise ValueError("__private_next: 步长必须为正数")
        if position[1]<3: raise ValueError("__private_next: 当前点尚未未经过盘出螺线")
        start_time = tm.time()
        timeout = 3  # 超时时间（秒），可根据需要调整

        ans = position[0]    # 从当前点开始尝试
        current_pos = self.pos_t_xy(position)

        def func(cur,tar):
            return length**2-(cur[0]-tar[0])**2-(cur[1]-tar[1])**2

        while ans>self.alpha:
            if tm.time() - start_time > timeout:
                raise TimeoutError("__private_next 超时")
            if func(current_pos, self.pos_t_xy((ans,3))) < 0:
                break
            ans -= step
        if ans<self.alpha:
            return None
        # 至此，确定有一个数值解在(ans,ans+step)之间
        l = ans
        r = ans+step
        for i in range(100):
            ans = (l + r) / 2
            if func(current_pos, self.pos_t_xy((ans,3))) < 0:
                l = ans
            else:
                r = ans

        return (ans, 3)
    
    def visualize(self, res, title="龙的轨迹可视化", figsize=(12, 8)):
        """
        可视化位置数组，将位置转换为笛卡尔坐标并渲染出连接的轨迹
        @param positions: 位置数组，每个元素为 (极角, 所在区域) 的元组
        @param title: 图表标题
        @param figsize: 图表大小
        """
        positions=res['positions']
        if not positions:
            print("位置数组为空，无法可视化")
            return
        
        # 将位置转换为笛卡尔坐标
        x_coords = []
        y_coords = []
        colors = []
        
        # 定义不同区域的颜色
        region_colors = {
            0: 'blue',    # 盘入螺线 - 蓝色
            1: 'green',   # 圆弧A - 绿色
            2: 'red',     # 圆弧B - 红色
            3: 'orange'   # 盘出螺线 - 橙色
        }
        
        region_names = {
            0: '盘入螺线',
            1: '圆弧A',
            2: '圆弧B', 
            3: '盘出螺线'
        }
        
        for position in positions:
            xy = self.pos_t_xy(position)
            if xy is not None:
                x_coords.append(xy[0])
                y_coords.append(xy[1])
                colors.append(region_colors.get(position[1], 'black'))
        
        if not x_coords:
            print("没有有效的坐标点，无法可视化")
            return
        
        # 创建图表
        plt.figure(figsize=figsize)
        
        # 绘制连接线
        plt.plot(x_coords, y_coords, 'k-', linewidth=1, alpha=0.7, label='轨迹连线')
        
        # 绘制点，按区域着色
        for i, (x, y, color) in enumerate(zip(x_coords, y_coords, colors)):
            region = positions[i][1]
            if i == 0 or positions[i-1][1] != region:  # 只在区域变化时添加标签
                plt.scatter(x, y, c=color, s=20, label=region_names.get(region, f'区域{region}'))
            else:
                plt.scatter(x, y, c=color, s=20)
        
        # 标记起点和终点
        if len(x_coords) > 1:
            plt.scatter(x_coords[0], y_coords[0], c='black', s=100, marker='o', 
                       label='起点', edgecolors='white', linewidth=2)
            plt.scatter(x_coords[-1], y_coords[-1], c='purple', s=100, marker='s', 
                       label='终点', edgecolors='white', linewidth=2)
        
        # 添加圆弧A和圆弧B的圆心标记
        plt.scatter(self.core_A[0], self.core_A[1], c='green', s=100, marker='+', 
                   linewidth=3, label='圆弧A圆心')
        plt.scatter(self.core_B[0], self.core_B[1], c='red', s=100, marker='+', 
                   linewidth=3, label='圆弧B圆心')
        
        # 绘制圆弧A和圆弧B的边界圆
        circle_A = plt.Circle(self.core_A, self.r_A, fill=False, color='green', 
                             linestyle='--', alpha=0.5, label='圆弧A边界')
        circle_B = plt.Circle(self.core_B, self.r_B, fill=False, color='red', 
                             linestyle='--', alpha=0.5, label='圆弧B边界')
        plt.gca().add_patch(circle_A)
        plt.gca().add_patch(circle_B)
        
        # 绘制等距螺线
        self._draw_spiral_lines(plt)
        
        # 设置图表属性
        plt.xlabel('X 坐标 (米)', fontsize=12)
        plt.ylabel('Y 坐标 (米)', fontsize=12)
        plt.title(f'{title} - 时刻: {res["time"]:.2f}s', fontsize=14, fontweight='bold')
        plt.grid(True, alpha=0.3)
        plt.axis('equal')
        
        # 调整图例位置和字体
        legend = plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left', fontsize=10)
        legend.set_title('图例说明', prop={'size': 11, 'weight': 'bold'})
        plt.tight_layout()
        
        # 显示统计信息
        print(f"=== 板凳龙轨迹统计信息 ===")
        print(f"时刻: {res['time']:.2f} 秒")
        print(f"总节数: {len(positions)} 节")
        
        region_count = {}
        for pos in positions:
            region = pos[1]
            region_count[region] = region_count.get(region, 0) + 1
        
        print("各区域分布:")
        for region, count in sorted(region_count.items()):
            percentage = (count / len(positions)) * 100
            print(f"  {region_names.get(region, f'区域{region}')}: {count} 节 ({percentage:.1f}%)")
        print("=" * 25)
        
        plt.show()
    
    def _draw_spiral_lines(self, plt):
        """
        绘制等距螺线（盘入和盘出螺线）
        @param plt: matplotlib.pyplot 对象
        """
        # 螺线参数
        spiral_step = 0.01  # 角度步长
        
        # 计算螺线范围
        theta_max = max(10 * np.pi, self.alpha + 10 * np.pi)  # 确保螺线足够长
        theta_range = np.arange(self.alpha, theta_max, spiral_step)
        
        # 盘入螺线 (顺时针方向)
        x_spiral_in = []
        y_spiral_in = []
        for theta in theta_range:
            x = self.distance * theta * np.cos(theta) / (2 * np.pi)
            y = self.distance * theta * np.sin(theta) / (2 * np.pi)
            x_spiral_in.append(x)
            y_spiral_in.append(y)
        
        # 盘出螺线 (逆时针方向)  
        x_spiral_out = []
        y_spiral_out = []
        for theta in theta_range:
            x = -self.distance * theta * np.cos(theta) / (2 * np.pi)
            y = -self.distance * theta * np.sin(theta) / (2 * np.pi)
            x_spiral_out.append(x)
            y_spiral_out.append(y)
        
        # 绘制螺线
        plt.plot(x_spiral_in, y_spiral_in, 'b:', linewidth=1.5, alpha=0.6, label='盘入螺线轨迹')
        plt.plot(x_spiral_out, y_spiral_out, 'orange', linestyle=':', linewidth=1.5, alpha=0.6, label='盘出螺线轨迹')
        
        # 标记切点
        cut_x_A = self.r * np.cos(self.alpha)
        cut_y_A = self.r * np.sin(self.alpha)
        cut_x_B = -cut_x_A
        cut_y_B = -cut_y_A
        
        plt.scatter(cut_x_A, cut_y_A, c='blue', s=80, marker='*', 
                   edgecolors='white', linewidth=1, label='切点A')
        plt.scatter(cut_x_B, cut_y_B, c='orange', s=80, marker='*', 
                   edgecolors='white', linewidth=1, label='切点B')
        
        # 绘制掉头区域的边界圆
        turn_circle = plt.Circle((0, 0), self.r, fill=False, color='black', 
                                linestyle='-', alpha=0.8, linewidth=2, label='掉头区域边界')
        plt.gca().add_patch(turn_circle)

    def save_position(self,result_data, output_file):
        """
        保存某一时刻连接点位置
        @param result_data: trace_moment方法返回的结果字典，包含time和positions
        @param output_file: 输出文件路径
        """
        time = result_data['time']
        pos_list = result_data['positions']
        
        # 确保输出目录存在
        os.makedirs(os.path.dirname(output_file), exist_ok=True)
        
        # 将极角转换为笛卡尔坐标
        positions = []
        for pos in pos_list:
            positions.append(self.pos_t_xy(pos))
        
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
        for i in range(1, len(positions)-2):
            row_labels.append(f'第{i}节龙身x (m)')
            row_labels.append(f'第{i}节龙身y (m)')
        row_labels.append('龙尾x (m)')
        row_labels.append('龙尾y (m)')
        row_labels.append('龙尾（后）x (m)')
        row_labels.append('龙尾（后）y (m)')

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
        @param velocity_result_data: v_trace_moment方法返回的结果字典，包含time和velocities
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
        for i in range(1, len(velocity_list)-2):
            row_labels.append(f'第{i}节龙身 (m/s)')
        row_labels.append('龙尾 (m/s)')
        row_labels.append('龙尾（后）(m/s)')

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
