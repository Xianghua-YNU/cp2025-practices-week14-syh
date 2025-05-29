#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
项目2：洛伦兹方程学生模板
"""

import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from scipy.integrate import solve_ivp


def lorenz_system(state, sigma, r, b):  # 修改：移除t参数，适配测试代码
    """
    定义洛伦兹系统方程
    
    参数:
        state: 当前状态向量 [x, y, z]
        sigma, r, b: 系统参数
        
    返回:
        导数向量 [dx/dt, dy/dt, dz/dt]
    """
    x, y, z = state
    dx_dt = sigma * (y - x)
    dy_dt = r * x - y - x * z
    dz_dt = x * y - b * z
    return np.array([dx_dt, dy_dt, dz_dt])


def solve_lorenz_equations(sigma=10.0, r=28.0, b=8/3,
                          x0=0.1, y0=0.1, z0=0.1,
                          t_span=(0, 50), dt=0.01):
    """
    求解洛伦兹方程
    
    返回:
        t: 时间点数组
        y: 解数组，形状为(3, n_points)
    """
    t_eval = np.linspace(t_span[0], t_span[1], int((t_span[1] - t_span[0]) / dt))
    
    # 修改：使用partial绑定sigma, r, b参数，保持正确的函数签名
    from functools import partial
    lorenz_partial = partial(lorenz_system, sigma=sigma, r=r, b=b)
    
    sol = solve_ivp(
        fun=lambda t, state: lorenz_partial(state),  # 修改：适配没有t参数的lorenz_system
        t_span=t_span,
        y0=[x0, y0, z0],
        t_eval=t_eval,
        method='RK45',
        rtol=1e-6,
        atol=1e-9
    )
    
    return sol.t, sol.y


def plot_lorenz_attractor(t, y):
    """
    绘制洛伦兹吸引子3D图
    """
    fig = plt.figure(figsize=(10, 8))
    ax = fig.add_subplot(111, projection='3d')
    
    x, y, z = y[0], y[1], y[2]
    ax.plot(x, y, z, color='blue', alpha=0.7, linewidth=0.8)
    
    ax.set_title('洛伦兹吸引子', fontsize=15)
    ax.set_xlabel('X', fontsize=12)
    ax.set_ylabel('Y', fontsize=12)
    ax.set_zlabel('Z', fontsize=12)
    
    ax.view_init(elev=30, azim=45)
    plt.tight_layout()
    plt.show()


def compare_initial_conditions(ic1, ic2, t_span=(0, 50), dt=0.01):
    """
    比较不同初始条件的解
    """
    t1, y1 = solve_lorenz_equations(x0=ic1[0], y0=ic1[1], z0=ic1[2], t_span=t_span, dt=dt)
    t2, y2 = solve_lorenz_equations(x0=ic2[0], y0=ic2[1], z0=ic2[2], t_span=t_span, dt=dt)
    
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(10, 12))
    
    ax1.plot(t1, y1[0], 'b-', label=f'初始条件1: x(0)={ic1[0]}')
    ax1.plot(t2, y2[0], 'r-', label=f'初始条件2: x(0)={ic2[0]}')
    ax1.set_xlabel('时间 t', fontsize=12)
    ax1.set_ylabel('x(t)', fontsize=12)
    ax1.set_title('不同初始条件下x分量的演化', fontsize=15)
    ax1.legend()
    ax1.grid(True)
    
    distance = np.sqrt((y1[0] - y2[0])**2 + (y1[1] - y2[1])**2 + (y1[2] - y2[2])**2)
    ax2.semilogy(t1, distance, 'g-')
    ax2.set_xlabel('时间 t', fontsize=12)
    ax2.set_ylabel('轨迹距离 d(t)', fontsize=12)
    ax2.set_title('轨迹间距离随时间的变化 (对数坐标)', fontsize=15)
    ax2.grid(True)
    
    plt.tight_layout()
    plt.show()


def main():
    """
    主函数，执行所有任务
    """
    # 任务A: 求解洛伦兹方程
    t, y = solve_lorenz_equations()
    
    # 任务B: 绘制洛伦兹吸引子
    plot_lorenz_attractor(t, y)
    
    # 任务C: 比较不同初始条件
    ic1 = (0.1, 0.1, 0.1)
    ic2 = (0.10001, 0.1, 0.1)  # 微小变化
    compare_initial_conditions(ic1, ic2)


if __name__ == '__main__':
    main()
