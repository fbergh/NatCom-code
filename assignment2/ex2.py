import numpy as np
import numpy.random as rand
import matplotlib.pyplot as plt
import sys

### Seed & Parameter ###
rand.seed(1)
GLOBAL_MINIMUM = 0

### Functions ###
def init_swarm_pos(n_particles, lower_bound, upper_bound):
    return lower_bound + rand.rand(n_particles) * (upper_bound-lower_bound)


# Define fitness function
f = lambda x: x**2


def plot_trajectory(n_particles, trajectory, margin=10, title="", is_save=True, filename="img/trajectory.png"):
    max_val = trajectory.max()
    min_val = trajectory.min()
    x_bound = -min_val if abs(min_val) > max_val else max_val
    x_vals = np.linspace(-x_bound-margin, x_bound+margin, num=1000)
    plt.plot(x_vals, f(x_vals), label="f(x)")
    for i in range(n_particles):
        plt.plot(trajectory[:, i], f(trajectory[:, i]), 'o')
        plt.plot(trajectory[:, i], f(trajectory[:, i]))
        plt.plot(trajectory[0, i], f(trajectory[0, i]), 'o', label="Start", c="red")
        plt.plot(trajectory[-1, i], f(trajectory[-1, i]), 'o', label="End", c="blue")
    plt.ylabel("Y")
    plt.xlabel("X")
    plt.title(title)
    plt.legend()
    if is_save:
        plt.savefig(filename)
    plt.show()


def pso(n_particles, omega, iterations=10000, epsilon=5, a1=1.5, a2=1.5, lower_bound=-100, upper_bound=100):
    # Initialise
    swarm_pos = init_swarm_pos(n_particles, lower_bound, upper_bound)
    trajectory = [swarm_pos.copy()] 
    velocities = np.ones((n_particles))
    local_bests = upper_bound * np.ones((n_particles))
    global_best = upper_bound
    min_found = False

    for _ in range(iterations):
        for i in range(n_particles):
            # Update rule
            r1, r2 = rand.uniform(0.001, 1, size=2)
            velocities[i] = omega*velocities[i] + a1*r1*(local_bests[i]-swarm_pos[i]) + a2*r2*(global_best-swarm_pos[i])
            swarm_pos[i] += velocities[i]

            # Update local and global minimum
            if f(swarm_pos[i]) < f(local_bests[i]):
                local_bests[i] = swarm_pos[i]
            if f(swarm_pos[i]) < f(global_best):
                global_best = swarm_pos[i]

            # Convergence check
            min_found = abs(swarm_pos[i]) < GLOBAL_MINIMUM+epsilon

        trajectory.append(swarm_pos.copy())
        # If converged quit
        if min_found:
            break

    return trajectory


def main():
    n_particles = 1
    omegas = [0, 0.25, 0.5, 0.75]
    for omega in omegas:
        trajectory = pso(n_particles, omega)
        plot_trajectory(n_particles, np.array(trajectory), title="Trajectory for omega="+str(omega)+" ("+str(len(trajectory)-1)+" epochs)", 
                        filename="img/omega"+str(omega).replace('.','_')+".png")


if __name__ == "__main__":
    main()
