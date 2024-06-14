import numpy as np
import matplotlib.pyplot as plt

# Parameters
tau = 0.5
w_f = 0  # Mutual inhibitory coupling weight
w_e = 0  # Self-excitatory weight
I = 0  # External input
b = 0.0  # Bias term

# Initial state with more pronounced values
v1 = 0.1
v2 = -0.1

# Time parameters
dt = 0.01
time = np.arange(0, 50, dt)  # Longer time to ensure observation of oscillations

# Storage for results
v1_vals = []
v2_vals = []
y1_vals = []
y2_vals = []

for t in time:
    u1 = max(0, v1)
    u2 = max(0, v2)
    y1 = u1 + b
    y2 = u2 + b

    dv1 = (-v1 - w_f * y2 - w_e * u1 + I) / tau
    dv2 = (-v2 - w_f * y1 - w_e * u2 + I) / tau

    v1 += dv1 * dt
    v2 += dv2 * dt

    v1_vals.append(v1)
    v2_vals.append(v2)
    y1_vals.append(y1)
    y2_vals.append(y2)

# Plotting the Matsuoka Oscillator Outputs
plt.figure(figsize=(12, 6))
plt.plot(time, y1_vals, label='y1')
plt.plot(time, y2_vals, label='y2')
plt.xlabel('Time')
plt.ylabel('Output')
plt.title('Matsuoka Oscillator Outputs')
plt.legend()
plt.grid(True)
plt.show()
