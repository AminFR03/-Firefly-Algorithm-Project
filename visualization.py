import matplotlib.pyplot as plt
import numpy as np

def plot_convergence(history, title="Firefly Algorithm Convergence"):
    history = np.array(history)

    plt.figure()
    plt.plot(history[:, 0], history[:, 1], marker="o")
    plt.scatter(0, 0)
    plt.xlabel("x")
    plt.ylabel("y")
    plt.title(title)
    plt.grid(True)
    plt.show()
