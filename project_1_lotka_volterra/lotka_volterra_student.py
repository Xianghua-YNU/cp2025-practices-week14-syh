#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
项目1：Lotka-Volterra捕食者-猎物模型 - 学生代码模板

学生姓名：[请填写您的姓名]
学号：[请填写您的学号]
完成日期：[请填写完成日期]
"""

import numpy as np
import matplotlib.pyplot as plt
from typing import Tuple, List
from scipy.signal import find_peaks  # 添加缺失的导入

# 设置中文字体（可选）
plt.rcParams['font.sans-serif'] = ['SimHei']
plt.rcParams['axes.unicode_minus'] = False


def lotka_volterra_system(state: np.ndarray, t: float, alpha: float, beta: float, 
                          gamma: float, delta: float) -> np.ndarray:
    """
    Lotka-Volterra方程组的右端函数
    
    方程组：
    dx/dt = α*x - β*x*y  (猎物增长率 - 被捕食率)
    dy/dt = γ*x*y - δ*y  (捕食者增长率 - 死亡率)
    
    参数:
        state: np.ndarray, 形状为(2,), 当前状态向量 [x, y]
        t: float, 时间（本系统中未显式使用，但保持接口一致性）
        alpha: float, 猎物自然增长率
        beta: float, 捕食效率
        gamma: float, 捕食者从猎物获得的增长效率
        delta: float, 捕食者自然死亡率
    
    返回:
        np.ndarray, 形状为(2,), 导数向量 [dx/dt, dy/dt]
    """
    x, y = state
    
    dxdt = alpha * x - beta * x * y
    dydt = gamma * x * y - delta * y
    
    return np.array([dxdt, dydt])


def euler_method(f, y0: np.ndarray, t_span: Tuple[float, float], 
                 dt: float, *args) -> Tuple[np.ndarray, np.ndarray]:
    """
    欧拉法求解常微分方程组
    
    参数:
        f: 微分方程组的右端函数，签名为 f(y, t, *args)
        y0: np.ndarray, 初始条件向量
        t_span: Tuple[float, float], 时间范围 (t_start, t_end)
        dt: float, 时间步长
        *args: 传递给f的额外参数
    
    返回:
        t: np.ndarray, 时间数组
        y: np.ndarray, 解数组，形状为 (len(t), len(y0))
    """
    t_start, t_end = t_span
    t = np.arange(t_start, t_end + dt, dt)
    n_steps = len(t)
    n_vars = len(y0)
    
    y = np.zeros((n_steps, n_vars))
    y[0] = y0
    
    for i in range(n_steps - 1):
        y[i+1] = y[i] + dt * f(y[i], t[i], *args)
    
    return t, y


def improved_euler_method(f, y0: np.ndarray, t_span: Tuple[float, float], 
                         dt: float, *args) -> Tuple[np.ndarray, np.ndarray]:
    """
    改进欧拉法（2阶Runge-Kutta法）求解常微分方程组
    
    参数:
        f: 微分方程组的右端函数
        y0: np.ndarray, 初始条件向量
        t_span: Tuple[float, float], 时间范围
        dt: float, 时间步长
        *args: 传递给f的额外参数
    
    返回:
        t: np.ndarray, 时间数组
        y: np.ndarray, 解数组
    """
    t_start, t_end = t_span
    t = np.arange(t_start, t_end + dt, dt)
    n_steps = len(t)
    n_vars = len(y0)
    
    y = np.zeros((n_steps, n_vars))
    y[0] = y0
    
    for i in range(n_steps - 1):
        k1 = dt * f(y[i], t[i], *args)
        k2 = dt * f(y[i] + k1, t[i] + dt, *args)
        y[i+1] = y[i] + (k1 + k2) / 2
    
    return t, y


def runge_kutta_4(f, y0: np.ndarray, t_span: Tuple[float, float], 
                  dt: float, *args) -> Tuple[np.ndarray, np.ndarray]:
    """
    4阶龙格-库塔法求解常微分方程组
    
    参数:
        f: 微分方程组的右端函数
        y0: np.ndarray, 初始条件向量
        t_span: Tuple[float, float], 时间范围 (t_start, t_end)
        dt: float, 时间步长
        *args: 传递给f的额外参数
    
    返回:
        t: np.ndarray, 时间数组
        y: np.ndarray, 解数组，形状为 (len(t), len(y0))
    """
    t_start, t_end = t_span
    t = np.arange(t_start, t_end + dt, dt)
    n_steps = len(t)
    n_vars = len(y0)
    
    y = np.zeros((n_steps, n_vars))
    y[0] = y0
    
    for i in range(n_steps - 1):
        k1 = dt * f(y[i], t[i], *args)
        k2 = dt * f(y[i] + k1/2, t[i] + dt/2, *args)
        k3 = dt * f(y[i] + k2/2, t[i] + dt/2, *args)
        k4 = dt * f(y[i] + k3, t[i] + dt, *args)
        y[i+1] = y[i] + (k1 + 2*k2 + 2*k3 + k4) / 6
    
    return t, y


def solve_lotka_volterra(alpha: float, beta: float, gamma: float, delta: float,
                        x0: float, y0: float, t_span: Tuple[float, float], 
                        dt: float) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    使用4阶龙格-库塔法求解Lotka-Volterra方程组
    
    参数:
        alpha: float, 猎物自然增长率
        beta: float, 捕食效率
        gamma: float, 捕食者从猎物获得的增长效率
        delta: float, 捕食者自然死亡率
        x0: float, 初始猎物数量
        y0: float, 初始捕食者数量
        t_span: Tuple[float, float], 时间范围
        dt: float, 时间步长
    
    返回:
        t: np.ndarray, 时间数组
        x: np.ndarray, 猎物种群数量数组
        y: np.ndarray, 捕食者种群数量数组
    """
    y0_vec = np.array([x0, y0])
    t, y = runge_kutta_4(lotka_volterra_system, y0_vec, t_span, dt, alpha, beta, gamma, delta)
    x = y[:, 0]
    y = y[:, 1]
    
    return t, x, y


def compare_methods(alpha: float, beta: float, gamma: float, delta: float,
                   x0: float, y0: float, t_span: Tuple[float, float], 
                   dt: float) -> dict:
    """
    比较三种数值方法求解Lotka-Volterra方程组
    
    参数:
        alpha, beta, gamma, delta: 模型参数
        x0, y0: 初始条件
        t_span: 时间范围
        dt: 时间步长
    
    返回:
        dict: 包含三种方法结果的字典，格式为：
        {
            'euler': {'t': t_array, 'x': x_array, 'y': y_array},
            'improved_euler': {'t': t_array, 'x': x_array, 'y': y_array},
            'rk4': {'t': t_array, 'x': x_array, 'y': y_array}
        }
    """
    y0_vec = np.array([x0, y0])
    
    # 使用欧拉法求解
    t_euler, y_euler = euler_method(lotka_volterra_system, y0_vec, t_span, dt, alpha, beta, gamma, delta)
    x_euler = y_euler[:, 0]
    y_pred_euler = y_euler[:, 1]
    
    # 使用改进欧拉法求解
    t_imp_euler, y_imp_euler = improved_euler_method(lotka_volterra_system, y0_vec, t_span, dt, alpha, beta, gamma, delta)
    x_imp_euler = y_imp_euler[:, 0]
    y_pred_imp_euler = y_imp_euler[:, 1]
    
    # 使用4阶龙格-库塔法求解
    t_rk4, y_rk4 = runge_kutta_4(lotka_volterra_system, y0_vec, t_span, dt, alpha, beta, gamma, delta)
    x_rk4 = y_rk4[:, 0]
    y_pred_rk4 = y_rk4[:, 1]
    
    results = {
        'euler': {'t': t_euler, 'x': x_euler, 'y': y_pred_euler},
        'improved_euler': {'t': t_imp_euler, 'x': x_imp_euler, 'y': y_pred_imp_euler},
        'rk4': {'t': t_rk4, 'x': x_rk4, 'y': y_pred_rk4}
    }
    
    return results


def plot_population_dynamics(t: np.ndarray, x: np.ndarray, y: np.ndarray, 
                           title: str = "Lotka-Volterra种群动力学") -> None:
    """
    绘制种群动力学图
    
    参数:
        t: np.ndarray, 时间数组
        x: np.ndarray, 猎物种群数量
        y: np.ndarray, 捕食者种群数量
        title: str, 图标题
    """
    # 创建一个包含两个子图的图形
    fig = plt.figure(figsize=(14, 6))
    fig.suptitle(title, fontsize=16)
    
    # 第一个子图：时间序列图
    ax1 = fig.add_subplot(121)
    ax1.plot(t, x, 'b-', label='猎物 (兔子)')
    ax1.plot(t, y, 'r-', label='捕食者 (狐狸)')
    ax1.set_xlabel('时间')
    ax1.set_ylabel('种群数量')
    ax1.set_title('种群数量随时间的变化')
    ax1.legend()
    ax1.grid(True)
    
    # 第二个子图：相空间轨迹图
    ax2 = fig.add_subplot(122)
    ax2.plot(x, y, 'g-')
    ax2.set_xlabel('猎物数量')
    ax2.set_ylabel('捕食者数量')
    ax2.set_title('相空间轨迹图')
    ax2.grid(True)
    
    # 添加初始点标记
    ax2.plot(x[0], y[0], 'ko', markersize=8)
    ax2.text(x[0], y[0], '  初始点', fontsize=10)
    
    # 确保布局合理
    plt.tight_layout(rect=[0, 0, 1, 0.96])  # 为suptitle留出空间
    plt.show()


def plot_method_comparison(results: dict) -> None:
    """
    绘制不同数值方法的比较图
    
    参数:
        results: dict, compare_methods函数的返回结果
    """
    # 创建一个2x3的子图布局
    fig, axes = plt.subplots(2, 3, figsize=(18, 10))
    fig.suptitle('不同数值方法求解Lotka-Volterra方程组的比较', fontsize=16)
    
    methods = ['euler', 'improved_euler', 'rk4']
    method_names = ['欧拉法', '改进欧拉法', '4阶龙格-库塔法']
    colors = ['blue', 'green', 'red']
    
    # 上排：三种方法的时间序列图
    for i, method in enumerate(methods):
        t = results[method]['t']
        x = results[method]['x']
        y = results[method]['y']
        
        ax = axes[0, i]
        ax.plot(t, x, color=colors[0], label='猎物')
        ax.plot(t, y, color=colors[1], label='捕食者')
        ax.set_title(f'{method_names[i]} - 时间序列图')
        ax.set_xlabel('时间')
        ax.set_ylabel('种群数量')
        ax.legend()
        ax.grid(True)
    
    # 下排：三种方法的相空间图
    for i, method in enumerate(methods):
        x = results[method]['x']
        y = results[method]['y']
        
        ax = axes[1, i]
        ax.plot(x, y, color=colors[2])
        ax.set_title(f'{method_names[i]} - 相空间轨迹图')
        ax.set_xlabel('猎物数量')
        ax.set_ylabel('捕食者数量')
        ax.grid(True)
    
    plt.tight_layout(rect=[0, 0, 1, 0.96])  # 为suptitle留出空间
    plt.show()


def analyze_parameters() -> None:
    """
    分析不同参数对系统行为的影响
    
    分析内容：
    1. 不同初始条件的影响
    2. 守恒量验证
    """
    # 设置基本参数
    alpha, beta, gamma, delta = 1.0, 0.5, 0.5, 2.0
    t_span = (0, 30)
    dt = 0.01
    
    # 测试不同初始条件
    initial_conditions = [
        (2.0, 2.0),    # 基准条件
        (3.0, 2.0),    # 更多猎物
        (2.0, 3.0),    # 更多捕食者
        (1.0, 1.0),    # 更少的两者
        (5.0, 1.0)     # 很多猎物，很少捕食者
    ]
    
    # 创建图形
    plt.figure(figsize=(15, 12))
    
    # 绘制不同初始条件的相空间轨迹
    plt.subplot(2, 2, 1)
    for x0, y0 in initial_conditions:
        t, x, y = solve_lotka_volterra(alpha, beta, gamma, delta, x0, y0, t_span, dt)
        plt.plot(x, y, label=f'初始条件: x0={x0}, y0={y0}')
    
    plt.xlabel('猎物数量')
    plt.ylabel('捕食者数量')
    plt.title('不同初始条件下的相空间轨迹')
    plt.legend()
    plt.grid(True)
    
    # 计算并验证守恒量
    plt.subplot(2, 2, 2)
    x0, y0 = 2.0, 2.0
    t, x, y = solve_lotka_volterra(alpha, beta, gamma, delta, x0, y0, t_span, dt)
    
    # 计算守恒量 C = δ*ln(x) - γ*x + β*y - α*ln(y)
    C = delta * np.log(x) - gamma * x + beta * y - alpha * np.log(y)
    
    plt.plot(t, C)
    plt.xlabel('时间')
    plt.ylabel('守恒量 C')
    plt.title('系统守恒量随时间的变化')
    plt.grid(True)
    
    # 分析不同参数对系统的影响 - 改变alpha
    plt.subplot(2, 2, 3)
    alphas = [0.8, 1.0, 1.2]
    x0, y0 = 2.0, 2.0
    
    for a in alphas:
        t, x, y = solve_lotka_volterra(a, beta, gamma, delta, x0, y0, t_span, dt)
        plt.plot(t, x, label=f'α={a} (猎物)')
        plt.plot(t, y, '--', label=f'α={a} (捕食者)')
    
    plt.xlabel('时间')
    plt.ylabel('种群数量')
    plt.title('不同α值(猎物增长率)对种群动态的影响')
    plt.legend()
    plt.grid(True)
    
    # 分析不同参数对系统的影响 - 改变delta
    plt.subplot(2, 2, 4)
    deltas = [1.5, 2.0, 2.5]
    
    for d in deltas:
        t, x, y = solve_lotka_volterra(alpha, beta, gamma, d, x0, y0, t_span, dt)
        plt.plot(t, x, label=f'δ={d} (猎物)')
        plt.plot(t, y, '--', label=f'δ={d} (捕食者)')
    
    plt.xlabel('时间')
    plt.ylabel('种群数量')
    plt.title('不同δ值(捕食者死亡率)对种群动态的影响')
    plt.legend()
    plt.grid(True)
    
    plt.tight_layout()
    plt.show()


def main():
    """
    主函数：演示Lotka-Volterra模型的完整分析
    
    执行步骤：
    1. 设置参数并求解基本问题
    2. 比较不同数值方法
    3. 分析参数影响
    4. 输出数值统计结果
    """
    # 参数设置（根据题目要求）
    alpha, beta, gamma, delta = 1.0, 0.5, 0.5, 2.0
    x0, y0 = 2.0, 2.0
    t_span = (0, 30)
    dt = 0.01
    
    print("=== Lotka-Volterra捕食者-猎物模型分析 ===")
    print(f"参数: α={alpha}, β={beta}, γ={gamma}, δ={delta}")
    print(f"初始条件: x0={x0}, y0={y0}")
    print(f"时间范围: {t_span}, 步长: {dt}")
    
    try:
        # 1. 基本求解
        print("\n1. 使用4阶龙格-库塔法求解...")
        t, x, y = solve_lotka_volterra(alpha, beta, gamma, delta, x0, y0, t_span, dt)
        plot_population_dynamics(t, x, y, "Lotka-Volterra模型的种群动力学")
        
        # 2. 方法比较
        print("\n2. 比较不同数值方法...")
        results = compare_methods(alpha, beta, gamma, delta, x0, y0, t_span, dt)
        plot_method_comparison(results)
        
        # 3. 参数分析
        print("\n3. 分析参数影响...")
        analyze_parameters()
        
        # 4. 数值结果统计
        print("\n4. 数值结果统计:")
        print(f"模拟时间范围: {t[0]:.2f} 到 {t[-1]:.2f}")
        print(f"时间步数: {len(t)}")
        print(f"最大猎物数量: {np.max(x):.4f} (时间点: {t[np.argmax(x)]:.2f})")
        print(f"最小猎物数量: {np.min(x):.4f} (时间点: {t[np.argmin(x)]:.2f})")
        print(f"最大捕食者数量: {np.max(y):.4f} (时间点: {t[np.argmax(y)]:.2f})")
        print(f"最小捕食者数量: {np.min(y):.4f} (时间点: {t[np.argmin(y)]:.2f})")
        
        # 计算平均种群数量
        print(f"平均猎物数量: {np.mean(x):.4f}")
        print(f"平均捕食者数量: {np.mean(y):.4f}")
        
        # 计算振荡周期（粗略估计）
        # 寻找猎物数量的峰值
        peaks, _ = find_peaks(x)
        if len(peaks) >= 2:
            periods = np.diff(t[peaks])
            avg_period = np.mean(periods)
            print(f"猎物数量振荡的平均周期: {avg_period:.4f}")
        else:
            print("无法准确计算振荡周期（未找到足够的峰值）")
        
    except Exception as e:
        print(f"\n错误: {e}")
        print("请检查代码实现并确保所有函数正确完成。")


if __name__ == "__main__":
    main()
