import math

def solve_asinbcos(a, b, c, eps=1e-12):
    """
    解 a*sin(x) + b*cos(x) = c
    返回 [0, 2π) 范围内的所有解
    """
    R = math.hypot(a, b)  # sqrt(a^2 + b^2)
    if R < eps:
        raise ValueError("a 和 b 不能同时为 0")

    # 检查解存在性
    if abs(c) > R + eps:
        return []  # 无解

    # 计算相位 phi
    phi = math.atan2(b, a)  # 带象限的反正切

    # 特殊情况：abs(c/R) == 1 只会有一个角度 + 对应的补角
    ratio = max(-1.0, min(1.0, c / R))  # 防止数值误差超出 [-1,1]
    base_angle = math.asin(ratio)  # 主值 [-pi/2, pi/2]

    solutions = []

    # 两组解：
    # x + phi = base_angle + 2kπ
    # x + phi = π - base_angle + 2kπ
    for theta in [base_angle, math.pi - base_angle]:
        x = theta - phi
        # 将结果映射到 [0, 2π)
        x_mod = x % (2 * math.pi)
        solutions.append(x_mod)

    # 去重（可能两个解相同）
    solutions = sorted(set([round(sol, 12) for sol in solutions]))
    return solutions
