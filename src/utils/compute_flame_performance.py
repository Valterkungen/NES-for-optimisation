# compute_flame_performance.py
import math
import numpy as np
import matplotlib.pyplot as plt
from flameLib import getPsi, getUV, getF, getPerf

def compute_flame_performance(nozc, nozw, refine=1, plot=False):
    # Set up grid parameters
    imax = 40*refine*2
    nmax = 80*refine**2

    x1 = -2.5
    x2 = 2.5
    y1 = 0.0
    y2 = 10.0
    dx = (x2 - x1) / float(imax - 1)
    dy = (y2 - y1) / float(nmax - 1)
    x = np.zeros((imax, nmax))
    y = np.zeros((imax, nmax))

    for n in range(0, nmax):
        x[:, n] = np.linspace(x1, x2, imax)

    for i in range(0, imax):
        y[i, :] = np.linspace(y1, y2, nmax)

    # Load stream function and velocities
    psi = np.zeros((imax, nmax))
    u = np.zeros((imax, nmax))
    v = np.zeros((imax, nmax))
    crn = np.zeros((imax, nmax))

    for n in range(0, nmax):
        for i in range(0, imax):
            psi[i, n] = getPsi(x[i, n], y[i, n])
            u[i, n], v[i, n] = getUV(x[i, n], y[i, n])
            crn[i, n] = u[i, n] * dy / (v[i, n] * dx)

    # Initialize mass fraction and set inflow conditions
    Z = np.zeros((imax, nmax))
    F = np.zeros((imax, nmax))

    xnozLeft = nozc - 0.5 * nozw
    xnozRight = nozc + 0.5 * nozw
    mfar = 0.0075 / nozw

    for i in range(0, imax):
        if x[i, 0] > xnozLeft and x[i, 0] < xnozRight:
            dist = x[i, 0] - nozc
            Z[i, 0] = mfar * 0.5 * (1.0 + math.cos(6.2832 * (dist / nozw - 1.0)))

    # March explicitly in y to solve the mass fraction
    for n in range(0, nmax - 1):
        for i in range(0, imax):
            F[i, n] = getF(u[i, n], v[i, n], Z[i, n])

        # Update left boundary
        Z[0, n + 1] = Z[0, n] - u[0, n] * dy * (Z[1, n] - Z[0, n]) / (v[0, n] * dx) + F[0, n] * dy / v[0, n]

        # Update interior nodes
        for i in range(1, imax - 1):
            if u[i, n] < 0:
                Z[i, n + 1] = Z[i, n] - (Z[i + 1, n] - Z[i, n]) * dy * u[i, n] / (v[i, n] * dx) + F[i, n] * dy / v[i, n]
            else:
                Z[i, n + 1] = Z[i, n] - (Z[i, n] - Z[i - 1, n]) * dy * u[i, n] / (v[i, n] * dx) + F[i, n] * dy / v[i, n]

        # Update right boundary
        Z[imax - 1, n + 1] = Z[imax - 1, n] - u[imax - 1, n] * dy * (Z[imax - 1, n] - Z[imax - 2, n]) / (v[imax - 1, n] * dx) + F[imax - 1, n] * dy / v[imax - 1, n]

    # Evaluate performance
    perf = getPerf(imax, nmax, x, y, dx, dy, F)

    # Print Parameters and Flame Performance
    #print('Parameters', [nozc, nozw])
    #print('Flame performance:', perf)

    # Optionally plot results
    if plot:
        fig = plt.figure(figsize=(18, 8))
        ax4 = plt.subplot2grid((2, 5), (0, 4), rowspan=1)
        ax4.set_title(r"$Z$ - Mass fraction")
        a = ax4.contourf(x, y, Z, cmap=plt.cm.gnuplot)
        plt.colorbar(a, ax=ax4)
        plt.show()

    return perf

# Test case for the function
if __name__ == '__main__':
    # Example input values for testing
    nozc = -0.6568471175908313
    nozw = 1.7327949784402639
    performance = compute_flame_performance(nozc, nozw, refine=3, plot=True)
    print('Test flame performance:', performance)
