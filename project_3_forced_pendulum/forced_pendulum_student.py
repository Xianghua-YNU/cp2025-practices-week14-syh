import numpy as np
import matplotlib.pyplot as plt
from scipy.integrate import solve_ivp

def forced_pendulum_ode(t, state, l, g, C, Omega):
    """
    受驱单摆的常微分方程
    state: [theta, omega]
    返回: [dtheta/dt, domega/dt]
    """
    theta, omega = state
    dtheta_dt = omega
    domega_dt = -(g/l) * np.sin(theta) + C * np.cos(theta) * np.sin(Omega * t)
    return [dtheta_dt, domega_dt]

def solve_pendulum(l=0.1, g=9.81, C=2, Omega=5, t_span=(0,100), y0=[0,0]):
    """
    求解受迫单摆运动方程
    返回: t, theta
    """
    t_eval = np.linspace(t_span[0], t_span[1], 2000)
    sol = solve_ivp(
        forced_pendulum_ode,
        t_span,
        y0,
        args=(l, g, C, Omega),
        t_eval=t_eval,
        method='RK45'
    )
    return sol.t, sol.y[0]

def find_resonance(l=0.1, g=9.81, C=2, Omega_range=None, t_span=(0,200), y0=[0,0]):
    """
    寻找共振频率
    返回: Omega_range, amplitudes
    """
    if Omega_range is None:
        omega_natural = np.sqrt(g/l)
        Omega_range = np.linspace(omega_natural/2, 2*omega_natural, 50)
    
    amplitudes = []
    for Omega in Omega_range:
        t, theta = solve_pendulum(l, g, C, Omega, t_span, y0)
        # 动态计算稳态时间点，确保有足够的数据点
        min_steady_points = 100  # 至少需要100个点来计算振幅
        min_steady_time = min(50, (t_span[1] - t_span[0]) * 0.25)  # 取50秒或总时长的25%中的较小值
        
        # 确保有足够的稳态数据点
        steady_start_idx = np.searchsorted(t, min_steady_time)
        if len(t) - steady_start_idx < min_steady_points:
            steady_start_idx = max(0, len(t) - min_steady_points)
        
        steady_state_theta = theta[steady_start_idx:]
        amplitude = np.max(np.abs(steady_state_theta))
        amplitudes.append(amplitude)
    
    return Omega_range, amplitudes

def plot_results(t, theta, title):
    """绘制结果"""
    plt.figure(figsize=(10, 5))
    plt.plot(t, theta)
    plt.title(title)
    plt.xlabel('Time (s)')
    plt.ylabel('Angle (rad)')
    plt.grid(True)
    plt.show()

def plot_resonance_curve(Omega_range, amplitudes):
    """绘制共振曲线"""
    plt.figure(figsize=(10, 5))
    plt.plot(Omega_range, amplitudes)
    plt.title('Resonance Curve: Amplitude vs Driving Frequency')
    plt.xlabel('Driving Frequency $\Omega$ (rad/s)')
    plt.ylabel('Steady-State Amplitude (rad)')
    plt.grid(True)
    plt.show()

def main():
    """主函数"""
    # 任务1: 特定参数下的数值解与可视化
    t, theta = solve_pendulum(Omega=5)
    plot_results(t, theta, 'Pendulum Angle vs Time ($\Omega$=5 rad/s)')
    
    # 任务2: 探究共振现象
    omega_natural = np.sqrt(9.81/0.1)
    Omega_range = np.linspace(omega_natural/2, 2*omega_natural, 50)
    Omega_range, amplitudes = find_resonance(Omega_range=Omega_range)
    plot_resonance_curve(Omega_range, amplitudes)
    
    # 找到共振频率并绘制共振情况
    resonance_index = np.argmax(amplitudes)
    omega_res = Omega_range[resonance_index]
    print(f"Resonant frequency: {omega_res:.3f} rad/s")
    
    t_res, theta_res = solve_pendulum(Omega=omega_res, t_span=(0,200))
    plot_results(t_res, theta_res, f'Pendulum Angle vs Time at Resonance ($\Omega$={omega_res:.3f} rad/s)')

if __name__ == '__main__':
    main()
