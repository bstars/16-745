"""
This file implements DIRCOL for
cartpole-swing-up problem,

https://deepnote.com/workspace/Underactuated-2ed1518a-973b-4145-bd62-1768b49956a8/project/096cffe7-416e-4d51-a471-5fd526ec8fab/notebook/cartpole_balancing-378a5603bb1d465182289bc04b4dc77b


where the goal state is [0, pi/2, 0, 0], with stage cost

"""

from gekko import GEKKO
import numpy as np
from torch import nn
import matplotlib.pyplot as plt

g = 9.81
l = 1.
mc = 1.
mp = 1.

h = 0.1
nx = 4
nu = 1
Tfinal = 5.0 # Final time
Nt = int(Tfinal / h) + 1 # number of discrete time steps
Q = np.eye(nx)
QN = 10 * np.eye(nx)
R = 0.05 * np.eye(nu)
x_goal = np.array([0., np.pi, 0, 0])

def cartpole_dynamics(x_, u, lib):
	"""

	:param x: [x, theta, x_dot, theta_dot]
	:param u: [u, ]
	:return:
	"""
	x, theta, xd, thetad = x_
	coef = 1 / (mc + mp * lib.sin(theta) ** 2)
	xdd = coef * (
			u[0] + mp * lib.sin(theta) *
			(
				l * thetad**2 + g * lib.cos(theta)
			)
	)
	thetadd = coef / l * (
		-u[0] * lib.cos(theta)
		-mp * l * thetad**2 * lib.cos(theta) * lib.sin(theta)
		-(mc + mp) * g * lib.sin(theta)
	)

	return xd, thetad, xdd, thetadd

def cartpole_dynamics_rk4(x, u):
	""" RK-4 integrator """
	f1 = np.stack(cartpole_dynamics(x, u, np))
	f2 = np.stack(cartpole_dynamics(x + 0.5 * h * f1, u, np))
	f3 = np.stack(cartpole_dynamics(x + 0.5 * h * f2, u, np))
	f4 = np.stack(cartpole_dynamics(x + h * f3, u, np))
	return x + (h / 6.0) * (f1 + 2 * f2 + 2 * f3 + f4)

def solve_dir_col():
	m = GEKKO()
	xhist = m.Array(m.Var, dim=(Nt, nx))
	uhist = m.Array(m.Var, dim=(Nt, nu))


	# construct equality constraints for DIRCOL
	# f(x_{k+1/2}, u_{k+1/2}) = \dot x_{k+1/2}
	fxu = m.Array(m.Var, dim=(Nt, nx))

	for t in range(Nt):
		temp = cartpole_dynamics(xhist[t], uhist[t], lib=m)
		m.Equations([
			temp[j] == fxu[t,j] for j in range(nx)
		])

	for t in range(Nt-1):
		xmid = 0.5 * (xhist[t] + xhist[t+1]) + h/8 * (fxu[t] - fxu[t+1])
		umid = 0.5 * (uhist[t] + uhist[t+1])
		xdmid = -3/(2*h) * (xhist[t] - xhist[t+1]) - 0.25 * (fxu[t] + fxu[t+1])
		temp = cartpole_dynamics(xmid, umid, lib=m)
		m.Equations([
			temp[i] == xdmid[i] for i in range(nx)
		])

	m.Minimize(
		m.sum([0.5 * (xhist[t] - x_goal) @ Q @ (xhist[t] - x_goal) for t in range(Nt-1)])
		+ 0.5 * (xhist[-1] - x_goal) @ QN @ (xhist[-1] - x_goal)
		+ m.sum([0.5 * uhist[t] @ R @ uhist[t] for t in range(Nt)])
	)

	# m.Minimize(
	# 	m.sum( [m.qobj(-Q @ x_goal, Q, x=xhist[t]) for t in range(Nt)] ) +
	# 	m.sum( [m.qobj(np.zeros(nu), R, x=uhist[t]) for t in range(Nt)] )
	# )

	m.solve()
	return xhist, uhist

if __name__ == '__main__':
	xhist, uhist = solve_dir_col()
	uhist = np.array([u.value[0] for u in uhist[:,0]])[:,None]

	traj = [np.zeros(nx)]
	for u in uhist:
		traj.append(cartpole_dynamics_rk4(traj[-1], u))
	traj = np.stack(traj, axis=0)
	fig, (ax1, ax2) = plt.subplots(1, 2)
	ax1.plot([x.value[0] for x in xhist[:,0]], label='x')
	ax1.plot([x.value[0] for x in xhist[:, 1]], label='theta')
	ax2.plot(traj[:,0], label='x')
	ax2.plot(traj[:,1], label='theta')
	plt.legend()
	plt.show()






