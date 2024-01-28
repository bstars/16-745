from gekko import GEKKO
import numpy as np
m = GEKKO()
x = m.Array(m.Var,4,value=1,lb=1,ub=5)
x1,x2,x3,x4 = x
# change initial values
x2.value = 5; x3.value = 5
m.Equation(x1*x2*x3*x4>=25)
m.Equation(x1**2+x2**2+x3**2+x4**2==40)

m.Minimize(x1*x4*(x1+x2+x3)+x3)
m.solve()
print('x: ', x)
print('Objective: ',m.options.OBJFCNVAL)