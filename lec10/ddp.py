"""
This file implements Differential Dynamic Programming for the quadrotor
	m * x'' = - (u1 + u2) * sin(\theta)
	m * y'' = (u1 + u2) * cos(\theta) - mg
	J * \theta'' = 0.5 * l * (u2 - u1)

with stage cost
	l_n(x, u) = 0.5 * (x - x_hover)' Q (x - x_hover) + 0.5 * (u - u_hover)' R (u - u_hover)

"""

import numpy as np
import cvxpy
import torch
from torch import nn
import matplotlib.pyplot as plt


g = 9.81 # m/s^2
m = 1. # kg
l = 0.3 # meter
J = 0.2 * m * l * l
h = 0.05 # timestep, 20hz
nx = 6
nu = 2
Tfinal = 7.0 # final time
Nt = int(Tfinal / h) # number of discrete time steps
Q = np.eye(nx)
R = 0.1 * np.eye(nu)

# steady state for hovering
x_hover = np.array([0., 1., 0., 0., 0., 0.]) # 1 meter above ground
u_hover = np.ones(2) * 0.5 * m * g


def quadrotor_dynamics(x, u, grad):
	"""
	m * x'' = - (u1 + u2) * sin(\theta)
	m * y'' = (u1 + u2) * cos(\theta) - mg
	J * \theta'' = 0.5 * l * (u2 - u1)

	:param x: [x, y, theta, x', y', \theta']
	:param u: [u1, u2]
	:return:
	"""

	lib = torch if grad else np

	x, y, theta, xd, yd, thetad = x
	u1, u2 = u

	xdd = 1 / m * (u1 + u2) * lib.sin(theta)
	ydd = 1 / m * (u1 + u2) * lib.cos(theta) - g
	thetadd = (1 / J) * (l / 2) * (u2 - u1)
	return lib.stack([xd, yd, thetad, xdd, ydd, thetadd])

def quadrotor_dynamics_rk4(x, u, grad=True):
	""" RK-4 integrator """
	f1 = quadrotor_dynamics(x, u, grad)
	f2 = quadrotor_dynamics(x + 0.5 * h * f1, u, grad)
	f3 = quadrotor_dynamics(x + 0.5 * h * f2, u, grad)
	f4 = quadrotor_dynamics(x + h * f3, u, grad)
	return x + (h / 6.0) * (f1 + 2 * f2 + 2 * f3 + f4)

def dynamics_jacobian_aux(xu):
	"""
	:param: torch.tensor, [nx+nu]
	:return:
		jacobian: [nx, nx + nu]
	"""
	jacobian = torch.autograd.functional.jacobian(
		lambda xu: quadrotor_dynamics_rk4(xu[:nx], xu[:nu], grad=True),
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
	val = quadrotor_dynamics_rk4(x, u, grad=False)

	xu = torch.from_numpy(np.concatenate([x, u]))
	jacobian = dynamics_jacobian_aux(xu)
	hessian = torch.autograd.functional.jacobian(
		dynamics_jacobian_aux,
		(xu, ),
	)[0]
	return val, jacobian.detach().numpy(), hessian.detach().numpy()

def compute_cost(x_hist, u_hist):
	"""
	:param x_hist: [N, nx]
	:param u_hist: [N-1, nu]
	:return:
	"""
	J = 0
	J += 0.5 * np.sum( ((x_hist - x_hover) @ Q) * (x_hist - x_hover) )
	J += 0.5 * np.sum( ((u_hist - u_hover) @ R) * (u_hist - u_hover))

	return J

def rollout(x_init, u_hist):
	"""
	:param x_init: [nx, ]
	:param u_hist: [N-1, nu]
	:return:
		x_hist: [N, nx]
	"""
	xs = [x_init]
	for t in range(len(u_hist)):
		x_new = quadrotor_dynamics_rk4(xs[-1], u_hist[t], grad=False)
		xs.append(x_new.copy())
	return np.stack(xs)

def differential_dynamic_programming(x_init):
	np.set_printoptions(1)
	u_hist = np.random.randn(Nt-1, nu) * 0.01
	x_hist = rollout(x_init, u_hist)
	J = compute_cost(x_hist, u_hist)

	while True:
		print(J)

		# backward pass
		p_next = Q @ (x_hist[-1] - x_hover)
		P_next = Q

		Ks = []
		ds = []
		dJ = 0

		for t in reversed(range(Nt-1)):

			# Compute the gradient and hessian of the cost-to-go
			f, Df, D2f = dynamics_obj_jacobian_hessian(x_hist[t], u_hist[t])
			g = Df.T @ p_next # (P_next @ f + p_next)
			gx = g[:nx] + Q @ (x_hist[t] - x_hover)
			gu = g[nx:] + R @ (u_hist[t] - u_hover)

			# 1st-order approximation for the dynamics
			G = Df.T @ P_next @ Df # + np.einsum('i,ijk -> jk', p_next, D2f)
			Gxx = G[:nx, :nx] + Q
			Guu = G[nx:, nx:] + R
			Gxu = G[:nx, nx:]
			Gux = G[nx:, :nx]

			# Compute the feedback policy
			Guu_inv = np.linalg.inv(Guu)
			K = Guu_inv @ Gux
			d = Guu_inv @ gu


			dJ += gu @ d
			Ks.append(K.copy())
			ds.append(d.copy())

			# Update the gradient and hessian of cost-to-go
			P_next = Gxx + K.T @ Guu @ K - Gxu @ K - K.T @ Gux
			p_next = gx - K.T @ gu + K.T @ Guu @ d - Gxu @ d
		Ks = np.stack(Ks[::-1])
		ds = np.stack(ds[::-1])

		# forward pass with line search
		x_new = [x_init]
		u_new = []
		step = 0.5

		for t in range(len(Ks)):
			u = u_hist[t] - step * ds[t] - Ks[t] @ (x_new[t] - x_hist[t])
			u_new.append(u.copy())
			x_new.append(
				quadrotor_dynamics_rk4(x_new[t], u, grad=False)
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
					quadrotor_dynamics_rk4(x_new[-1], u, grad=False)
				)
			x_new = np.array(x_new)
			u_new = np.array(u_new)
			Jnew = compute_cost(x_new, u_new)


		if np.max(np.abs(u_hist - u_new)) < 1e-5:
			return x_new, u_new

		x_hist = x_new.copy()
		u_hist = u_new.copy()
		J = Jnew



if __name__ == '__main__':
	x_init = np.array([2., 2., 0., 0., 0., 0.])
	x_hist, u_hist = differential_dynamic_programming(x_init)
	# plt.scatter(x_hist[0,0], x_hist[0, 1], 'ro')
	plt.plot(x_hist[:, 0], x_hist[:, 1], 'o--', markersize=4)
	plt.show()