import numpy as np
import matplotlib.pyplot as plt
from typing import Tuple, Callable, List

def harmonic_oscillator_ode(state: np.ndarray, t: float, omega: float = 1.0) -> np.ndarray:
    """
    简谐振子的一阶微分方程组。
    
    参数:
        state: np.ndarray, 形状为(2,)的数组，包含位置x和速度v
        t: float, 当前时间（在这个系统中实际上没有使用）
        omega: float, 角频率
    
    返回:
        np.ndarray: 形状为(2,)的数组，包含dx/dt和dv/dt
    """
    x, v = state
    dx_dt = v
    dv_dt = -omega**2 * x
    return np.array([dx_dt, dv_dt])

def anharmonic_oscillator_ode(state: np.ndarray, t: float, omega: float = 1.0) -> np.ndarray:
    """
    非谐振子的一阶微分方程组。
    
    参数:
        state: np.ndarray, 形状为(2,)的数组，包含位置x和速度v
        t: float, 当前时间（在这个系统中实际上没有使用）
        omega: float, 角频率
    
    返回:
        np.ndarray: 形状为(2,)的数组，包含dx/dt和dv/dt
    """
    x, v = state
    dx_dt = v
    dv_dt = -omega**2 * x**3
    return np.array([dx_dt, dv_dt])

def rk4_step(ode_func: Callable, state: np.ndarray, t: float, dt: float, **kwargs) -> np.ndarray:
    """
    使用四阶龙格-库塔方法进行一步数值积分。
    
    参数:
        ode_func: Callable, 微分方程函数
        state: np.ndarray, 当前状态
        t: float, 当前时间
        dt: float, 时间步长
        **kwargs: 传递给ode_func的额外参数
    
    返回:
        np.ndarray: 下一步的状态
    """
    k1 = ode_func(state, t, **kwargs)
    k2 = ode_func(state + 0.5 * dt * k1, t + 0.5 * dt, **kwargs)
    k3 = ode_func(state + 0.5 * dt * k2, t + 0.5 * dt, **kwargs)
    k4 = ode_func(state + dt * k3, t + dt, **kwargs)
    
    next_state = state + (dt / 6.0) * (k1 + 2 * k2 + 2 * k3 + k4)
    return next_state

def solve_ode(ode_func: Callable, initial_state: np.ndarray, t_span: Tuple[float, float], 
              dt: float, **kwargs) -> Tuple[np.ndarray, np.ndarray]:
    """
    求解常微分方程组。
    
    参数:
        ode_func: Callable, 微分方程函数
        initial_state: np.ndarray, 初始状态
        t_span: Tuple[float, float], 时间范围 (t_start, t_end)
        dt: float, 时间步长
        **kwargs: 传递给ode_func的额外参数
    
    返回:
        Tuple[np.ndarray, np.ndarray]: (时间点数组, 状态数组)
    """
    t_start, t_end = t_span
    num_steps = int((t_end - t_start) / dt) + 1
    t_points = np.linspace(t_start, t_end, num_steps)
    state_shape = initial_state.shape
    
    # 初始化状态数组
    states = np.zeros((num_steps, *state_shape))
    states[0] = initial_state
    
    # 使用RK4方法求解ODE
    for i in range(num_steps - 1):
        states[i+1] = rk4_step(ode_func, states[i], t_points[i], dt, **kwargs)
    
    return t_points, states

def plot_time_evolution(t: np.ndarray, states: np.ndarray, title: str) -> None:
    """
    绘制状态随时间的演化。
    
    参数:
        t: np.ndarray, 时间点数组
        states: np.ndarray, 状态数组
        title: str, 图标题
    """
    plt.figure(figsize=(10, 6))
    plt.plot(t, states[:, 0], 'b-', label='位移 x')
    plt.plot(t, states[:, 1], 'r-', label='速度 v')
    plt.title(title, fontsize=14)
    plt.xlabel('时间 t', fontsize=12)
    plt.ylabel('状态变量', fontsize=12)
    plt.grid(True)
    plt.legend()
    plt.tight_layout()
    plt.show()

def plot_phase_space(states: np.ndarray, title: str) -> None:
    """
    绘制相空间轨迹。
    
    参数:
        states: np.ndarray, 状态数组
        title: str, 图标题
    """
    plt.figure(figsize=(10, 6))
    plt.plot(states[:, 0], states[:, 1], 'g-')
    plt.title(title, fontsize=14)
    plt.xlabel('位移 x', fontsize=12)
    plt.ylabel('速度 v', fontsize=12)
    plt.grid(True)
    plt.axis('equal')  # 使坐标轴比例相等，更好地显示椭圆
    plt.tight_layout()
    plt.show()

def analyze_period(t: np.ndarray, states: np.ndarray) -> float:
    """
    分析振动周期。
    
    参数:
        t: np.ndarray, 时间点数组
        states: np.ndarray, 状态数组
    
    返回:
        float: 估计的振动周期
    """
    x = states[:, 0]
    peaks = []
    
    # 查找局部最大值（波峰） - 使用更严格的条件提高精度
    for i in range(1, len(x) - 1):
        if x[i] > x[i-1] and x[i] > x[i+1] and x[i] > 0.9 * np.max(x):
            peaks.append(t[i])
    
    # 计算平均周期
    if len(peaks) < 2:
        print("警告：未能找到足够的峰值来计算周期")
        return np.nan
    
    periods = np.diff(peaks)
    avg_period = np.mean(periods)
    
    return avg_period

def main():
    # 设置参数
    omega = 1.0
    t_span = (0, 50)
    dt = 0.01  # 减小时间步长可以提高精度，但会增加计算时间
    
    # 任务1 - 简谐振子的数值求解
    initial_state_harmonic = np.array([1.0, 0.0])  # x(0)=1, v(0)=0
    t_harmonic, states_harmonic = solve_ode(
        harmonic_oscillator_ode, 
        initial_state_harmonic, 
        t_span, 
        dt, 
        omega=omega
    )
    plot_time_evolution(t_harmonic, states_harmonic, "简谐振子的时间演化 (x₀=1)")
    period_harmonic_1 = analyze_period(t_harmonic, states_harmonic)
    print(f"简谐振子 (x₀=1) 的周期: {period_harmonic_1:.4f}")
    
    # 任务2 - 振幅对周期的影响分析
    initial_state_harmonic_2 = np.array([2.0, 0.0])  # x(0)=2, v(0)=0
    _, states_harmonic_2 = solve_ode(
        harmonic_oscillator_ode, 
        initial_state_harmonic_2, 
        t_span, 
        dt, 
        omega=omega
    )
    plot_time_evolution(t_harmonic, states_harmonic_2, "简谐振子的时间演化 (x₀=2)")
    period_harmonic_2 = analyze_period(t_harmonic, states_harmonic_2)
    print(f"简谐振子 (x₀=2) 的周期: {period_harmonic_2:.4f}")
    
    print("\n简谐振子等时性验证:")
    print(f"振幅变化时周期变化率: {(period_harmonic_2/period_harmonic_1 - 1)*100:.2f}%")
    
    # 任务3 - 非谐振子的数值分析
    initial_state_anharmonic_1 = np.array([1.0, 0.0])  # x(0)=1, v(0)=0
    t_anharmonic, states_anharmonic_1 = solve_ode(
        anharmonic_oscillator_ode, 
        initial_state_anharmonic_1, 
        t_span, 
        dt, 
        omega=omega
    )
    plot_time_evolution(t_anharmonic, states_anharmonic_1, "非谐振子的时间演化 (x₀=1)")
    period_anharmonic_1 = analyze_period(t_anharmonic, states_anharmonic_1)
    print(f"\n非谐振子 (x₀=1) 的周期: {period_anharmonic_1:.4f}")
    
    initial_state_anharmonic_2 = np.array([2.0, 0.0])  # x(0)=2, v(0)=0
    _, states_anharmonic_2 = solve_ode(
        anharmonic_oscillator_ode, 
        initial_state_anharmonic_2, 
        t_span, 
        dt, 
        omega=omega
    )
    plot_time_evolution(t_anharmonic, states_anharmonic_2, "非谐振子的时间演化 (x₀=2)")
    period_anharmonic_2 = analyze_period(t_anharmonic, states_anharmonic_2)
    print(f"非谐振子 (x₀=2) 的周期: {period_anharmonic_2:.4f}")
    
    print("\n非谐振子振幅依赖性:")
    print(f"振幅增大时周期变化率: {(period_anharmonic_2/period_anharmonic_1 - 1)*100:.2f}%")
    
    # 任务4 - 相空间分析
    plot_phase_space(states_harmonic, "简谐振子的相空间轨迹 (x₀=1)")
    plot_phase_space(states_anharmonic_1, "非谐振子的相空间轨迹 (x₀=1)")
    plot_phase_space(states_anharmonic_2, "非谐振子的相空间轨迹 (x₀=2)")

if __name__ == "__main__":
    main()
