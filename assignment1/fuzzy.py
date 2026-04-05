import numpy as np
import matplotlib.pyplot as plt

# System constants
M_C = 1.0  # kg (Trolley mass)
M_P = 0.1  # kg (Pole mass)
L = 0.5    # m (Half-length of pole)
G = 9.81   # m/s^2 (Gravity)

# Membership functions
# x1: [-0.2, -0.1, 0, 0.1, 0.2]
# x2: [-1.0, -0.5, 0, 0.5, 1.0]
# F: [-10, -5, 0, 5, 10]

def trimf(x, abc):
    a, b, c = abc
    if x <= a or x >= c:
        return 0
    elif a < x <= b:
        return (x - a) / (b - a)
    elif b < x < c:
        return (c - x) / (c - b)
    return 0

def trapmf(x, abcd):
    a, b, c, d = abcd
    if x <= a: return 1.0 if a == -np.inf else 0.0
    if x >= d: return 1.0 if d == np.inf else 0.0
    if a < x < b: return (x - a) / (b - a)
    if b <= x <= c: return 1.0
    if c < x < d: return (d - x) / (d - c)
    return 0.0

# Membership functions for x1 (angular displacement)
x1_mfs = {
    'NB': lambda x: trapmf(x, [-np.inf, -0.2, -0.2, -0.1]),
    'NS': lambda x: trimf(x, [-0.2, -0.1, 0]),
    'ZE': lambda x: trimf(x, [-0.1, 0, 0.1]),
    'PS': lambda x: trimf(x, [0, 0.1, 0.2]),
    'PB': lambda x: trapmf(x, [0.1, 0.2, 0.2, np.inf])
}

# Membership functions for x2 (angular velocity)
x2_mfs = {
    'NB': lambda x: trapmf(x, [-np.inf, -1.0, -1.0, -0.5]),
    'NS': lambda x: trimf(x, [-1.0, -0.5, 0]),
    'ZE': lambda x: trimf(x, [-0.5, 0, 0.5]),
    'PS': lambda x: trimf(x, [0, 0.5, 1.0]),
    'PB': lambda x: trapmf(x, [0.5, 1.0, 1.0, np.inf])
}

# Force centers for defuzzification
f_centers = {
    'NB': -10,
    'NS': -5,
    'ZE': 0,
    'PS': 5,
    'PB': 10
}

# Rule Base
rule_base = {
    ('NB', 'NB'): 'NB', ('NB', 'NS'): 'NB', ('NB', 'ZE'): 'NS', ('NB', 'PS'): 'NS', ('NB', 'PB'): 'ZE',
    ('NS', 'NB'): 'NB', ('NS', 'NS'): 'NS', ('NS', 'ZE'): 'NS', ('NS', 'PS'): 'ZE', ('NS', 'PB'): 'PS',
    ('ZE', 'NB'): 'NS', ('ZE', 'NS'): 'NS', ('ZE', 'ZE'): 'ZE', ('ZE', 'PS'): 'PS', ('ZE', 'PB'): 'PS',
    ('PS', 'NB'): 'NS', ('PS', 'NS'): 'ZE', ('PS', 'ZE'): 'PS', ('PS', 'PS'): 'PS', ('PS', 'PB'): 'PB',
    ('PB', 'NB'): 'ZE', ('PB', 'NS'): 'PS', ('PB', 'ZE'): 'PS', ('PB', 'PS'): 'PB', ('PB', 'PB'): 'PB'
}

def get_force(x1, x2):
    # Fuzzification
    mu_x1 = {name: mf(x1) for name, mf in x1_mfs.items()}
    mu_x2 = {name: mf(x2) for name, mf in x2_mfs.items()}
    
    # Inference and Aggregation
    output_activations = {label: 0.0 for label in f_centers}
    
    for (x1_label, x2_label), f_label in rule_base.items():
        activation = min(mu_x1[x1_label], mu_x2[x2_label])
        output_activations[f_label] = max(output_activations[f_label], activation)
    
    # Defuzzification (Centroid method)
    num = sum(activation * f_centers[label] for label, activation in output_activations.items())
    den = sum(activation for label, activation in output_activations.items())
    
    if den == 0:
        return 0.0
    return num / den

def dynamics(state, f):
    theta, theta_dot = state
    sin_t = np.sin(theta)
    cos_t = np.cos(theta)
    
    # Nonlinear dynamics of inverted pendulum
    num = G * sin_t + cos_t * ((-f - M_P * L * (theta_dot ** 2) * sin_t) / (M_C + M_P))
    den = L * (4 / 3 - (M_P * (cos_t ** 2)) / (M_C + M_P))
    
    theta_ddot = num / den
    return np.array([theta_dot, theta_ddot])

def simulate():
    dt = 0.01 # Increased slightly for simulation speed, though plan said 0.001
    t_end = 5.0
    t_steps = int(t_end / dt)
    
    # Initial states x1(1) = 0.05 rad, x2(1) = -0.4 rad/s
    state = np.array([0.05, -0.4])
    
    history = {
        't': [],
        'theta': [],
        'theta_dot': [],
        'f': []
    }
    
    for i in range(t_steps):
        t = i * dt
        theta, theta_dot = state
        
        # Get control force
        f = get_force(theta, theta_dot)
        
        # Clamp force F=[−10, 10]
        f = np.clip(f, -10, 10)
        
        history['t'].append(t)
        history['theta'].append(theta)
        history['theta_dot'].append(theta_dot)
        history['f'].append(f)
        
        # RK4 Numerical Integration
        k1 = dynamics(state, f)
        k2 = dynamics(state + 0.5 * dt * k1, f)
        k3 = dynamics(state + 0.5 * dt * k2, f)
        k4 = dynamics(state + dt * k3, f)
        
        state = state + (dt / 6.0) * (k1 + 2 * k2 + 2 * k3 + k4)
        
    return history

def plot_results(history):
    fig, axes = plt.subplots(3, 1, figsize=(10, 12), sharex=True)
    
    # Modern Styling
    plt.style.use('seaborn-v0_8-muted')
    
    axes[0].plot(history['t'], history['theta'], label='Angular Displacement (rad)', color='#2563eb', linewidth=2)
    axes[0].axhline(0, color='black', linestyle='--', alpha=0.3)
    axes[0].set_ylabel('Angle (rad)')
    axes[0].set_title('Inverted Pendulum Fuzzy Logic Control Performance', fontsize=14, pad=20)
    axes[0].grid(True, alpha=0.3)
    axes[0].legend()
    
    axes[1].plot(history['t'], history['theta_dot'], label='Angular Velocity (rad/s)', color='#dc2626', linewidth=2)
    axes[1].axhline(0, color='black', linestyle='--', alpha=0.3)
    axes[1].set_ylabel('Velocity (rad/s)')
    axes[1].grid(True, alpha=0.3)
    axes[1].legend()
    
    axes[2].plot(history['t'], history['f'], label='Control Force (N)', color='#16a34a', linewidth=2)
    axes[2].axhline(0, color='black', linestyle='--', alpha=0.3)
    axes[2].set_ylabel('Force (N)')
    axes[2].set_xlabel('Time (s)')
    axes[2].grid(True, alpha=0.3)
    axes[2].legend()
    
    plt.tight_layout()
    plt.savefig('fuzzy_results.png')
    print("Results plot saved to 'fuzzy_results.png'")

if __name__ == "__main__":
    history = simulate()
    # Print final state
    print(f"Initial State: theta=0.05 rad, theta_dot=-0.4 rad/s")
    print(f"Final State (t=5s): theta={history['theta'][-1]:.6f} rad, theta_dot={history['theta_dot'][-1]:.6f} rad/s")
    plot_results(history)
