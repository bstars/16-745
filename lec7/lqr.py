"""
This file solve LQR problem

	\min_{u[1:N], x[1:N+1]}.    \sum_{n=1}^{N} [0.5 x_n Q_n x_n + 0.5 u_n R_n u_n] + 0.5 x_N Q_N x_N
	s.t.                        x_{n+1} = A x_n + B u_n

for the (discretized) double integrator

	x_{n+1} = [1 h] x_{n} + [1/2 h^2] u_n   (h is the time interval for each step)
			  [0 1]         [   h   ]


with
	1. Implicit shooting
	2. Riccati
"""


import numpy as np
import matplotlib.pyplot as plt
import cvxpy



# Define hyperparameters
h = 0.1
T = 5.
N = int(T / h)
A = np.array([
	[1., h],
	[0., 1.]
])


B = np.array([0.5 * h**2, h])

Q = np.eye(2)
R = 0.1
x_init = np.array([1., 0.])
# x_init = np.array([-50, -50.])


def compute_cost(x_hist, u_hist):
	"""
	:param x_hist: [N+1, 2]
	:param u_hist: [N,]
	:return:
	"""
	J = 0
	J += 0.5 * np.sum( (x_hist @ Q) * x_hist )
	J += 0.5 * R * np.sum(u_hist**2)
	return J

def rollout(u_hist):
	"""
	:param u_hist: [N,]
	:return:
		x_hist: [N+1, 2]
	"""
	xs = [x_init]
	for t in range(N):
		x_new = A @ xs[-1] + B * u_hist[t]
		xs.append(x_new.copy())
	return np.stack(xs)



def implicit_shooting():
	u_hist = np.zeros([N,])
	RinvBT = 1/ R * B.T
	alpha = 0.01 # line search parameter
	beta = 0.5 # line search parameter
	while True:

		# forward pass
		x_hist = rollout(u_hist)

		# backward pass
		lambs = [Q @ x_hist[-1]]
		for t in reversed(range(1, N)):
			lamb_new = Q @ x_hist[t] + A.T @ lambs[-1]
			lambs.append(lamb_new.copy())
		lambs = lambs[::-1]
		lambs = np.stack(lambs)

		u_hist_new = - (RinvBT @ lambs.T).T
		du = u_hist_new - u_hist
		du2 = np.sum(np.square(du))
		if du2 < 1e-4:
			return u_hist

		s = 1
		J = compute_cost(rollout(u_hist), u_hist)

		while compute_cost(rollout(u_hist + s * du), u_hist + s * du) > J - alpha * s * du2:

			s *= beta
		# print(J, s, x_hist.shape, u_hist.shape, lambs.shape)
		u_hist += s * du

def riccati():
	"""
	This is actually an MPC policy with LQR
	"""
	P = Q.copy()

	for _ in range(N):
		mid = 1 / (R + B.T @ P @ B)
		BtPA = B.T @ P @ A
		P = Q + A.T @ P  @ A - BtPA.T @  BtPA * mid

	K = (B.T @ P @ A) / (R + B.T @ P @ B)

	x_hist = [x_init]
	for _ in range(N):
		u = - K @ x_hist[-1]
		x = A @ x_hist[-1] + B * u
		x_hist.append(x.copy())

	return np.stack(x_hist)


if __name__ == '__main__':
	# u_hist = implicit_shooting()
	# x_hist = rollout(u_hist)

	x_hist = riccati()
	plt.plot(x_hist[:,0], label='position')
	plt.plot(x_hist[:,1], label='velocity')
	plt.legend()
	plt.show()








