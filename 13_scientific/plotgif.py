import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation as animation

nx = 41
ny = 41

def read_csv(filename):
    return np.loadtxt(filename, delimiter=',')

fig, ax = plt.subplots(figsize=(11, 7), dpi=100)

def update_plot(n):
    ax.clear()
    u = read_csv(f'u_{n}.csv')
    v = read_csv(f'v_{n}.csv')
    p = read_csv(f'p_{n}.csv')

    x = np.linspace(0, 2, nx)
    y = np.linspace(0, 2, ny)
    X, Y = np.meshgrid(x, y)

    contour = ax.contourf(X, Y, p, alpha=0.5, cmap=plt.cm.coolwarm)
    # if n == 0:
    #     fig.colorbar(contour)
    ax.quiver(X, Y, u, v)
    ax.set_xlabel('X')
    ax.set_ylabel('Y')
    ax.set_title(f'Time step: {n * 10}')

ani = animation.FuncAnimation(fig, update_plot, frames=range(0, 500, 10), repeat=False)

ani.save('navier_stokes_simulation.gif', writer='imagemagick')

plt.show()