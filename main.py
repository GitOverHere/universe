import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

# Set up initial conditions
N = 100  # number of particles
L = 10  # length of box
dt = 0.01  # time step
T = 10  # total simulation time
pos = L * np.random.rand(N, 3)
vel = np.random.randn(N, 3)

# Define the force function
# Define the force function
def force(pos, L):
    # Calculates the pairwise Lennard-Jones force
    dr = pos[:, np.newaxis, :] - pos[np.newaxis, :, :]
    dr = dr - np.round(dr / L) * L  # Apply periodic boundary conditions
    r2 = np.sum(dr**2, axis=2)
    mask = r2 < 2.5**2
    r2[mask] = 2.5**2
    r6 = r2**3
    dr_masked = dr[:,:,np.newaxis] * mask[:,:,np.newaxis]
    f = 48 * (1 / r6**2 - 0.5 / r6)[:,:,np.newaxis] * dr_masked
    return np.sum(f, axis=1)


# Initialize arrays to store positions and velocities
pos_history = np.zeros((int(T/dt)+1, N, 3))
vel_history = np.zeros_like(pos_history)

# Record initial positions and velocities
pos_history[0] = pos
vel_history[0] = vel

# Perform the simulation
for i in range(int(T/dt)):
    # Update positions
    pos_new = pos + vel*dt + 0.5*force(pos, L)*dt**2
    pos_new = pos_new % L  # Apply periodic boundary conditions
    # Update velocities
    vel_new = vel + 0.5*(force(pos, L) + force(pos_new, L))*dt
    # Record new positions and velocities
    pos_history[i+1] = pos_new
    vel_history[i+1] = vel_new
    # Update positions and velocities for the next time step
    pos = pos_new
    vel = vel_new

# Plot the final positions of the particles
fig = plt.figure()
ax = fig.add_subplot(111, projection='3d')
ax.scatter(pos[:, 0], pos[:, 1], pos[:, 2])
ax.set_xlim([0, L])
ax.set_ylim([0, L])
ax.set_zlim([0, L])
plt.show()
