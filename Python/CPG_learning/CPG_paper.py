import numpy as np
from scipy.integrate import odeint
import matplotlib.pyplot as plt

# Define parameters
a = 1.0  # Convergence factor
num_oscillators = 3  # Number of oscillators

# Intrinsic amplitudes and frequencies
mu = np.array([1.0, 1.0, 1.0])
omega = np.array([1.0, 0.8, 1.2])

# Coupling weights and phase biases
w = np.array([[0, 0.5, 0.5],
              [0.5, 0, 0.5],
              [0.5, 0.5, 0]])
# w = np.zeros_like(w)
phi = np.array([[0, 0, 0],
                [0, 0, 0],
                [0, 0, 0]])


# Differential equations for the CPG
def cpg_dynamics(y, t, a, mu, omega, w, phi):
    num_osc = len(mu)
    r = y[:num_osc]
    r_dot = y[num_osc:2 * num_osc]
    theta = y[2 * num_osc:]

    r_ddot = a * (a * (mu - r) / 4 - r_dot)
    dthetadt = np.zeros(num_osc)

    for i in range(num_osc):
        dthetadt[i] = omega[i] + np.sum(
            [r[j] * w[i][j] * np.sin(theta[j] - theta[i] - phi[i][j]) for j in range(num_osc)])

    return np.concatenate([r_dot, r_ddot, dthetadt])


# Initial conditions
r0 = np.random.rand(num_oscillators)
r_dot0 = np.zeros(num_oscillators)  # Initial velocities set to zero
theta0 = np.random.rand(num_oscillators) * 2 * np.pi
y0 = np.concatenate([r0, r_dot0, theta0])

# Time vector
t = np.linspace(0, 20, 2000)

# Integrate the differential equations
solution = odeint(cpg_dynamics, y0, t, args=(a, mu, omega, w, phi))

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
