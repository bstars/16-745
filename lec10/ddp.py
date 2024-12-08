import numpy as np
import torch
import time
import matplotlib.pyplot as plt



class RK4_Dynamics():
    def __init__(self, f, n, m, dt=0.01):
        """
        """
        self.f = f
        self.dt = dt
        self.n = n
        self.m = m

        


    def step(self, x, u, lib=np):
        f1 = self.f(x, u, lib)
        f2 = self.f(x + 0.5 * self.dt * f1, u, lib)
        f3 = self.f(x + 0.5 * self.dt * f2, u, lib)
        f4 = self.f(x + self.dt * f3, u, lib)
        return x + (self.dt / 6.0) * (f1 + 2 * f2 + 2 * f3 + f4)
    
    def step_aux(self, xu):
        return self.step(xu[:self.n], xu[self.n:], lib=torch)
    
    def step_with_derivative(self, x, u, compute_hessian=False):
        xnext = self.step(x, u)
        xu = torch.from_numpy(np.concatenate([x, u]))
        jacobian = torch.autograd.functional.jacobian(self.step_aux, xu)

        if not compute_hessian:
            return xnext, jacobian.detach().numpy()
        
        hessian = torch.autograd.functional.jacobian(
            lambda xu : torch.autograd.functional.jacobian(self.step_aux, xu), xu
        )
        return xnext, jacobian.detach().numpy(), hessian.detach().numpy()
    
    
    
def ddp(dynamics:RK4_Dynamics, Q, Qf, qs, qf, R, rs, x_init, u_hist=None):


    """
    This function approximately solves the LQR problem
        min{x,u}.   \sum_{t=0}^{N-1}    0.5 * x[t]*Q*x[t] + q[t]*x[t] 
                                        + 0.5 * u[t]*R*u[t] + r[t]*u[t]
                    + 0.5 * x[N]*Qf*x[N] + qf*x[N] 

        s.t.        x[t+1] = dynamics.step(x[t], u[t]) for t=0,...N-1
                    x[0] = x_init

        ( Since x[0] is give, sometime we ignore the objective related to x[0] in computation )
    """


    def rollout(x_init, u_hist):
        x_hist = [x_init]
        for u in u_hist:
            x_hist.append(
                dynamics.step(x_hist[-1], u)
            )
        return np.stack(x_hist)
    
    
    def compute_cost(x_hist, u_hist):
        # dx = x_hist - x_goal
        # J = 0
        # J += 0.5 * np.sum( (dx[:-1] @ Q) * dx[:-1] )
        # J += 0.5 * np.sum( dx[-1] @ QN @ dx[-1] )
        # J += 0.5 * np.sum( (u_hist @ R) * u_hist )

        J = 0
        J += 0.5 * np.sum( (x_hist[:-1] @ Q) * x_hist[:-1] ) + np.sum(x_hist[:-1] * qs)
        J += 0.5 * np.sum( x_hist[-1] @ Qf @ x_hist[-1] ) + np.sum(x_hist[-1] * qf)
        J += 0.5 * np.sum( (u_hist @ R) * u_hist ) + np.sum(u_hist * rs)
        return J
    
    nx = Q.shape[0]
    nu = R.shape[0]
    Nt = len(rs)
    if u_hist is None:
        u_hist = np.random.randn(Nt, nu)
    
    x_hist = rollout(x_init, u_hist)
    J = compute_cost(x_hist, u_hist)
    num_iter = 0
    
    while True:
        num_iter += 1

        # p = QN @ (x_hist[-1] - x_goal)
        # P = QN

        p = Qf @ x_hist[-1] + qf
        P = Qf

        Ks = []
        ds = []
        dJ = 0

        for t in reversed(range(Nt)):

            f, df, ddf = dynamics.step_with_derivative(x_hist[t], u_hist[t], compute_hessian=True)
            dfdx = df[:,:nx]
            dfdu = df[:,nx:]
            temp = np.einsum('i, ijk -> jk', p, ddf)

            gx = Q @ x_hist[t] + qs[t] + dfdx.T @ p
            gu = R @ u_hist[t] + rs[t] + dfdu.T @ p

            Gxx = Q + dfdx.T @ P @ dfdx + temp[:nx, :nx]
            Guu = R + dfdu.T @ P @ dfdu + temp[nx:, nx:]
            Gxu = dfdx.T @ P @ dfdu + temp[:nx, nx:]
            Gux = dfdu.T @ P @ dfdx + temp[nx:, :nx]


            # f, df = dynamics.step_with_derivative(x_hist[t], u_hist[t], compute_hessian=False)
            # dfdx = df[:,:nx]
            # dfdu = df[:,nx:]

            # gx = Q @ x_hist[t] + qs[t] + dfdx.T @ p
            # gu = R @ u_hist[t] + rs[t] + dfdu.T @ p

            # Gxx = Q + dfdx.T @ P @ dfdx 
            # Guu = R + dfdu.T @ P @ dfdu
            # Gxu = dfdx.T @ P @ dfdu
            # Gux = dfdu.T @ P @ dfdx


            Guuinv = np.linalg.inv(Guu)
            K = Guuinv @ Gux
            d = Guuinv @ gu

            dJ += gu @ d
            Ks.append(K.copy())
            ds.append(d.copy())

            P = Gxx + K.T @ Guu @ K - Gxu @ K - K.T @ Gux
            p = gx - K.T @ gu + K.T @ Guu @ d - Gxu @ d

        Ks = np.stack(Ks[::-1])
        ds = np.stack(ds[::-1])

        if np.max(np.abs(ds)) <= 1e-3:
            return x_hist, u_hist
        
        x_new = [x_init]
        u_new = []
        step = 1.

        for t in range(len(Ks)):
            u = u_hist[t] - step * ds[t] - Ks[t] @ (x_new[t] - x_hist[t])
            u_new.append(u.copy())
            x_new.append(
                dynamics.step(x_new[t], u)
            )
            
        x_new = np.stack(x_new)
        u_new = np.stack(u_new)
        Jnew = compute_cost(x_new, u_new)

        while Jnew > J - 0.9 * step * dJ:
            step *= 0.5
            x_new = [x_init]
            u_new = []

            for t in range(len(Ks)):
                u = u_hist[t] - step * ds[t] - Ks[t] @ (x_new[t] - x_hist[t])
                u_new.append(u.copy())
                x_new.append(
                    dynamics.step(x_new[t], u)
                )
                
            x_new = np.stack(x_new)
            u_new = np.stack(u_new)
            Jnew = compute_cost(x_new, u_new)

        x_hist = x_new.copy()
        u_hist = u_new.copy()
        J = Jnew

        print(num_iter, J)


if __name__ == '__main__':

    """
    This class solves the LQR problem
        min{x,u}.   \sum_{t=0}^{N-1}    0.5 * (x[t] - x_goal) * Q * (x[t] - x_goal) + 0.5 * u[t]*R*u[t]
                    + 0.5 * x[N]*Qf*x[N] + qf*x[N] 

        s.t.        x[t+1] = dynamics.step(x[t], u[t]) for t=0,...N-1
                    x[0] = x_init

        ( Since x[0] is give, sometime we ignore the objective related to x[0] in computation )

    The main function solves the cartpole swing-up problem
        https://deepnote.com/workspace/Underactuated-2ed1518a-973b-4145-bd62-1768b49956a8/project/096cffe7-416e-4d51-a471-5fd526ec8fab/notebook/cartpole_balancing-378a5603bb1d465182289bc04b4dc77b
    where the target state is straight up [0., np.pi, 0, 0]
    and the starting state is straight down [0., 0., 0, 0]
    """

    dt = 0.05 # timestep, 20hz
    nx = 4
    nu = 1
    Tfinal = 2 # final time
    Nt = int(Tfinal / dt) # number of discrete time steps
    Q = 1 * np.eye(nx)
    Qf = 100 * np.eye(nx)
    R = 0.05 * np.eye(nu)
    x_goal = np.array([0., np.pi, 0, 0])
    q = -Q @ x_goal
    qs = np.array([q for _ in range(Nt)])
    qf = -Qf @ x_goal
    rs = np.zeros([Nt, nu])


    def cartpole_dynamics(x_, u, lib=np):
        g = 9.81
        l = 1.
        mc = 1.
        mp = 1.

        x, theta, xd, thetad = x_
        coef = 1 / (mc + mp * lib.sin(theta) ** 2)
        xdd = coef * (
            u[0] + mp * lib.sin(theta) * (
                l * thetad**2 + g * lib.cos(theta)
            )
        )

        thetadd = coef / l * (
            -u[0] * lib.cos(theta)
            -mp * l * thetad**2 * lib.cos(theta) * lib.sin(theta)
            -(mc + mp) * g * lib.sin(theta)
        )

        return lib.stack([xd, thetad, xdd, thetadd])
    
    dynamics = RK4_Dynamics(cartpole_dynamics, m=nu, n=nx, dt=dt)
    x_hist, u_hist = ddp(dynamics, Q, Qf, qs, qf, R, rs, x_init=np.array([0., 0., 0., 0.]))
    plt.plot(x_hist[:,0])
    plt.plot(x_hist[:,1])
    plt.show()

   
