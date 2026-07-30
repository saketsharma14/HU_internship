import time
import matplotlib.pyplot as plt
import math

import numpy as np

from coppeliasim_zmqremoteapi_client import RemoteAPIClient

client = RemoteAPIClient()
sim = client.require('sim')

OBJECT_PATH="/PioneerP3DX"

robotHandle=sim.getObject(OBJECT_PATH)

X,Y,Z=[],[],[]

sim.startSimulation()

Kp=0.01
Ki=0
Kd=0

prev_error=0,0

Exp_X=[]
Exp_Y=[]

SD=0

sim.setStepping(True)

while (t := sim.getSimulationTime()) < 20:
    # s = f'Simulation time: {t:.2f} [s] (simulation running asynchronously '\
    #     'to client, i.e. non-stepping)'
    # print(s)
    x,y,z=sim.getObjectPosition(robotHandle)
    exp_x=math.sin(t)
    exp_y=math.cos(t)
    Exp_X.append(exp_x)
    Exp_Y.append(exp_y)

    D=math.sqrt((x-exp_x)**2+(y-exp_y)**2)


    P=Kp * D
    I=Ki * SD
    SD+=D
    DE=Kd * (D-prev_error)
    prev_error=D

    myu= P + I + DE


    # phi=sim.getObjectOrientation(robotHandle)
    # print(phi)
    X.append(x)
    Y.append(y)
    Z.append(z)

    sim.step()


sim.stopSimulation()

print("stopped....")

k=np.linspace(0,15,100)

# 2. Create figure
fig = plt.figure()
ax = plt.axes()

# 3. Plot the 3D line
ax.plot(X, Y, 'red')

# 4. Set labels
ax.set_xlabel('X Label')
ax.set_ylabel('Y Label')

plt.show()

