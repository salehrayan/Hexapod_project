import math
import numpy as np
from scipy.integrate import odeint
import matplotlib.pyplot as plt

# Define parameters
a = 150  # Convergence factor
num_oscillators = 1  # Number of oscillators

# Intrinsic amplitudes and frequencies
mu = np.array([1.])
omega = np.array([1.]) * 2* math.pi



# Differential equations for the CPG
def cpg_dynamics(y, t, a, mu, omega):
    num_osc = len(mu)
    r = y[:num_osc]
    r_dot = y[num_osc:2 * num_osc]
    theta = y[2 * num_osc:]
    theta_dot = omega

    r_ddot = a * (a * (mu - r) / 4 - r_dot)

    return np.concatenate([r_dot, r_ddot, theta_dot])


# Initial conditions
r0 = np.random.rand(num_oscillators)
r_dot0 = np.zeros(num_oscillators)  # Initial velocities set to zero
theta0 = np.random.rand(num_oscillators) * 2 * np.pi
y0 = np.concatenate([r0, r_dot0, theta0])

# Time vector
t = np.linspace(0, 20, 2000)

# Integrate the differential equations
solution = odeint(cpg_dynamics, y0, t, args=(a, mu, omega))

# Extract results
r = solution[:, :num_oscillators]
r_dot = solution[:, num_oscillators:2 * num_oscillators]
theta = solution[:, 2 * num_oscillators:]

# Calculate output signals
output_signals = r * np.cos(theta)

# Plot the results
plt.figure(figsize=(12, 6))
for i in range(num_oscillators):
    plt.plot(t, r[:, i], label=f'Oscillator {i + 1} Amplitude')
plt.xlabel('Time')
plt.ylabel('Amplitude')
plt.legend()
plt.title('Oscillator Amplitudes Over Time')
plt.show()

plt.figure(figsize=(12, 6))
for i in range(num_oscillators):
    plt.plot(t, theta[:, i], label=f'Oscillator {i + 1} Phase')
plt.xlabel('Time')
plt.ylabel('Phase')
plt.legend()
plt.title('Oscillator Phases Over Time')
plt.show()

plt.figure(figsize=(12, 6))
for i in range(num_oscillators):
    plt.plot(t, output_signals[:, i], label=f'Oscillator {i + 1} Output')
plt.xlabel('Time')
plt.ylabel('Output')
plt.legend()
plt.title('Oscillator Outputs Over Time')
plt.show()
