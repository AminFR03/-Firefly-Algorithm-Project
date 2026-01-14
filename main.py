from firefly_algorithm import FireflyAlgorithm
from benchmark_functions import sphere
from visualization import plot_convergence

def main():
    bounds = [(-5, 5), (-5, 5)]

    fa = FireflyAlgorithm(
        objective_function=sphere,
        bounds=bounds,
        n_fireflies=30,
        alpha=0.3,
        beta0=1.0,
        gamma=1.0,
        max_iter=40
    )

    best_position, best_value, history = fa.optimize()

    print("Best position found:", best_position)
    print("Best objective value:", best_value)

    plot_convergence(history)

if __name__ == "__main__":
    main()
