import numpy as np
import cvxpy
import torch
import matplotlib.pyplot as plt
import control

g = 9.81 # m/s^2
m = 1. # kg
l = 0.3 # meter
J = 0.2 * m * l * l
h = 0.05 # timestep, 20hz


# input constraints
umin = np.array([0.2 * m * g, 0.2 * m * g])
umax = np.array([0.6 * m * g, 0.6 * m * g])


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

# steady state for hovering
x_hover = np.array([0., 1., 0., 0., 0., 0.]) # 1 meter above ground
u_hover = np.ones(2) * 0.5 * m * g

# Linearize about the steady state x_hover = f(x_hover, u_hover)
# x_{t+1} = f(x_t, u_t)
#         ~ f(x_hover, u_hover) + Dxf(x_hover, u_hover) (x_t - x_hover) + Duf(x_hover, u_hover) (u_t - u_hover)
#         = x_hover + A (x_t - x_hover) + B(u_t - u_hover)
# dx_{t+1} = A dx_t + B du_t
A, B = torch.autograd.functional.jacobian(
	lambda x, u : quadrotor_dynamics_rk4(x, u, grad=True),
	inputs=(
		torch.from_numpy(x_hover),
		torch.from_numpy(u_hover)
	)
)

A = A.detach().numpy()
B = B.detach().numpy()


nx = 6 # number of states
nu = 2 # number of inputs
Tfinal = 5.0 # final time
Nt = int(Tfinal / h) # number of discrete time steps
Q = np.eye(nx)
R = 0.01 * np.eye(nu)


# LQR Controller (with no input constraint)
P, _, _ = control.dare(A, B, Q, R)
K, _, _ = control.dlqr(A, B, Q, R)
def lqr_controller(x):
	return u_hover - K @ (x - x_hover)

def mpc_controller(x, H=10):
	xvar = cvxpy.Variable(H * nx)
	uvar = cvxpy.Variable((H - 1) * nu)
	obj = cvxpy.Constant(0)
	constraints = [
		xvar[0: nx] == x - x_hover
	]

	for i in range(H - 1):
		obj += cvxpy.quad_form(xvar[i * nx: (i + 1) * nx], 0.5 * Q)
		obj += cvxpy.quad_form(uvar[i * nu: (i + 1) * nu], 0.5 * R)
		constraints.append(
			A @ xvar[i * nx: (i + 1) * nx] + B @ uvar[i * nu: (i + 1) * nu] == xvar[(i + 1) * nx: (i + 2) * nx]
		)

		constraints.append(umin <= uvar[i * nu: (i + 1) * nu] + u_hover)
		constraints.append(uvar[i * nu: (i + 1) * nu] + u_hover <= umax)

	obj += cvxpy.quad_form(xvar[-nx:], 0.5 * P)
	problem = cvxpy.Problem(
		cvxpy.Minimize(obj),
		constraints=constraints
	)
	problem.solve()
	return u_hover + uvar.value[:nu]


def close_loop(x0, controller, N):
	x_hist = [x0.copy()]
	for _ in range(N):
		u = controller(x_hist[-1])
		u = np.maximum(umin, u)
		u = np.minimum(umax, u)
		x = quadrotor_dynamics_rk4(x_hist[-1], u, grad=False)
		x_hist.append(x.copy())
	return np.stack(x_hist)


x0 = np.array([3.5, 3., 0., 0., 0., 0.])
trajectory_mpc = close_loop(x0, mpc_controller, N=Nt)
trajectory_lqr = close_loop(x0, lqr_controller, N=Nt)

plt.plot(trajectory_mpc[:,0], trajectory_mpc[:,1], 'o--', label='MPC', markersize=4)
plt.plot(trajectory_lqr[:,0], trajectory_lqr[:,1], 'o--', label='LQR', markersize=4)
plt.legend()
plt.show()
