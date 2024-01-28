"""
This file implements Differential Dynamic Programming for
cartpole-swing-up problem,

https://deepnote.com/workspace/Underactuated-2ed1518a-973b-4145-bd62-1768b49956a8/project/096cffe7-416e-4d51-a471-5fd526ec8fab/notebook/cartpole_balancing-378a5603bb1d465182289bc04b4dc77b


where the goal state is [0, pi/2, 0, 0], with stage cost

"""

import numpy as np
import cvxpy
import torch
from torch import nn
import matplotlib.pyplot as plt

g = 9.81
l = 1.
mc = 1.
mp = 1.


h = 0.05 # timestep, 20hz
nx = 4
nu = 1
Tfinal = 5.0 # final time
Nt = int(Tfinal / h) + 1 # number of discrete time steps
Q = np.eye(nx)
QN = 100 * np.eye(nx)
R = 0.05 * np.eye(nu)
x_goal = np.array([0., np.pi, 0, 0])

def cartpole_dynamics(x_, u, grad=True):
	"""

	:param x: [x, theta, x_dot, theta_dot]
	:param u: [u, ]
	:return:
	"""
	lib = torch if grad else np
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

	return lib.stack([xd, thetad, xdd, thetadd])

def cartpole_dynamics_rk4(x, u, grad=False):
	""" RK-4 integrator """
	f1 = cartpole_dynamics(x, u, grad)
	f2 = cartpole_dynamics(x + 0.5 * h * f1, u, grad)
	f3 = cartpole_dynamics(x + 0.5 * h * f2, u, grad)
	f4 = cartpole_dynamics(x + h * f3, u, grad)
	return x + (h / 6.0) * (f1 + 2 * f2 + 2 * f3 + f4)

def rollout(x_init, u_hist):
	x_hist = [x_init]
	for u in u_hist:
		x_hist.append(
			cartpole_dynamics_rk4(x_hist[-1], u, grad=False)
		)
	return np.stack(x_hist)

def dynamics_jacobian_aux(xu):
	jacobian = torch.autograd.functional.jacobian(
		lambda xu: cartpole_dynamics_rk4(xu[:nx], xu[nx:], grad=True),
		(xu,),
	)[0]
	return jacobian

def dynamics_obj_jacobian_hessian(x, u):
	"""
	Compute the function value, jacobian, and hessian of
		x = quadrotor_rk4(x,u)
	:param x: np.array, [nx, ]
	:param u: np.array, [nu, ]
	:return:
		val: [nx, ]
		jacobian: [nx, nx+nu]
		hessian: [nx, nx+nu, nx+nu]
			hessian[i, j, k] = d^2 f(x,u)[i] / dxu[j] dxu[k]
	"""
	val = cartpole_dynamics_rk4(x, u, grad=False)

	xu = torch.from_numpy(np.concatenate([x, u]))
	jacobian = dynamics_jacobian_aux(xu)
	# hessian = torch.autograd.functional.jacobian(
	# 	dynamics_jacobian_aux,
	# 	(xu, ),
	# )[0]
	return val, jacobian.detach().numpy() # , hessian.detach().numpy()

def compute_cost(x_hist, u_hist):
	dx = x_hist - x_goal
	J = 0
	J += 0.5 * np.sum( (dx[:-1] @ Q) * dx[:-1] )
	J += 0.5 * np.sum( dx[-1] @ QN @ dx[-1] )
	J += 0.5 * np.sum( (u_hist @ R) * u_hist )
	return J

def differential_dynamic_programming(x_init):
	np.set_printoptions(1)
	u_hist = np.random.randn(Nt - 1, nu) * 0.1
	x_hist = rollout(x_init, u_hist)
	J = compute_cost(x_hist, u_hist)
	num_iter = 0
	while True:
		num_iter += 1

		print(num_iter, J)

		p_next = QN @ (x_hist[-1] - x_goal)
		P_next = QN

		Ks = []
		ds = []
		dJ = 0

		# backward pass
		for t in reversed(range(Nt-1)):
			# Compute the gradient and hessian of cost-to-go
			f, Df = dynamics_obj_jacobian_hessian(x_hist[t], u_hist[t])

			g = Df.T @ p_next
			gx = g[:nx] + Q @ (x_hist[t] - x_goal)
			gu = g[nx:] + R @ u_hist[t]

			G = Df.T @ P_next @ Df
			Gxx = G[:nx, :nx] + Q
			Guu = G[nx:, nx:] + R
			Gxu = G[:nx, nx:]
			Gux = G[nx:, :nx]

			# Compute the feedback policy
			Guuinv = np.linalg.inv(Guu)
			K = Guuinv @ Gux
			d = Guuinv @ gu

			dJ += gu @ d
			Ks.append(K.copy())
			ds.append(d.copy())

			# Update the gradient and hessian of cost-to-go
			P_next = Gxx + K.T @ Guu @ K - Gxu @ K - K.T @ Gux
			p_next = gx - K.T @ gu + K.T @ Guu @ d - Gxu @ d

		Ks = np.stack(Ks[::-1])
		ds = np.stack(ds[::-1])

		if np.max(np.abs(ds)) < 5e-2:
			return x_hist, u_hist

		# forward pass with line search
		x_new = [x_init]
		u_new = []
		step = 1.

		for t in range(len(Ks)):
			u = u_hist[t] - step * ds[t] - Ks[t] @ (x_new[t] - x_hist[t])
			u_new.append(u.copy())
			x_new.append(
				cartpole_dynamics_rk4(x_new[t], u, grad=False)
			)
		x_new = np.array(x_new)
		u_new = np.array(u_new)
		Jnew = compute_cost(x_new, u_new)

		while Jnew > J - 0.01 * step * dJ:
			step *= 0.5

			x_new = [x_init]
			u_new = []
			for t in range(len(Ks)):
				u = u_hist[t] - step * ds[t] - Ks[t] @ (x_new[t] - x_hist[t])
				u_new.append(u.copy())
				x_new.append(
					cartpole_dynamics_rk4(x_new[t], u, grad=False)
				)
			x_new = np.array(x_new)
			u_new = np.array(u_new)
			Jnew = compute_cost(x_new, u_new)

		x_hist = x_new.copy()
		u_hist = u_new.copy()
		J = Jnew

if __name__ == '__main__':
	x_init = np.array([0, 0, 0, 0.])
	x_hist, u_hist = differential_dynamic_programming(x_init)
	plt.plot(x_hist[:, 0], label='x')
	plt.plot(x_hist[:, 1], label='theta')
	plt.legend()
	plt.show()


