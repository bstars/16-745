"""
This file implements different integrators for the pendulum system (no input)

	[\theta . ] = [     \theta .   ]
	[\theta ..] = [-g/l sin(\theta)]

including
	1. Forward Euler
	2.
"""

import numpy as np
import matplotlib.pyplot as plt


def pendulum_dynamics(x):
	l = 1.0
	g = 9.81
	theta = x[0]
	thetadot = x[1]

	return np.array([
		thetadot,
		-g / l * np.sin(theta)
	])

def euler_forward(x0, f, T, dt):
	T = int(T / dt)
	x_hist = [x0]

	for _ in range(T):
		x_hist.append(
			x_hist[-1] + dt * f(x_hist[-1])
		)
	return np.stack(x_hist)

def rk4_forward(x0, f, T, dt):
	T = int(T / dt)
	x_hist = [x0]

	for _ in range(T):
		x = x_hist[-1]
		f1 = f(x)
		f2 = f(x + 0.5 * dt * f1)
		f3 = f(x + 0.5 * dt * f2)
		f4 = f(x + dt * f3)
		x_hist.append(
			x + dt / 6. * (f1 + 2 * f2 + 2 * f3 + f4)
		)
	return np.stack(x_hist)




if __name__ == '__main__':

	x_init = np.array([0.1, 0.])

	x_hist_euler = euler_forward(x_init, pendulum_dynamics, 50, 0.01)
	x_hist_rk4 = rk4_forward(x_init, pendulum_dynamics, 50, 0.01)
	plt.plot(x_hist_euler[:,0], label='Euler')
	plt.plot(x_hist_rk4[:, 0], label='RK-4')
	plt.legend()
	plt.show()