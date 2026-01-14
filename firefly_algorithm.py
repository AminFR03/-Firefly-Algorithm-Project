import numpy as np

class FireflyAlgorithm:
    def __init__(
        self,
        objective_function,
        bounds,
        n_fireflies=25,
        alpha=0.3,
        beta0=1.0,
        gamma=1.0,
        max_iter=50
    ):
        self.f = objective_function
        self.bounds = np.array(bounds)
        self.n = n_fireflies
        self.alpha = alpha
        self.beta0 = beta0
        self.gamma = gamma
        self.max_iter = max_iter

        self.dim = len(bounds)
        self.population = self._initialize_fireflies()
        self.fitness = np.array([self.f(x) for x in self.population])

    def _initialize_fireflies(self):
        lower = self.bounds[:, 0]
        upper = self.bounds[:, 1]
        return np.random.uniform(lower, upper, (self.n, self.dim))

    def _distance(self, x, y):
        return np.linalg.norm(x - y)

    def _attractiveness(self, r):
        return self.beta0 * np.exp(-self.gamma * r**2)

    def optimize(self):
        best_history = []

        for _ in range(self.max_iter):
            for i in range(self.n):
                for j in range(self.n):
                    if self.fitness[j] < self.fitness[i]:
                        r = self._distance(self.population[i], self.population[j])
                        beta = self._attractiveness(r)
                        rand = self.alpha * (np.random.rand(self.dim) - 0.5)

                        self.population[i] += (
                            beta * (self.population[j] - self.population[i]) + rand
                        )

                        # Apply bounds
                        self.population[i] = np.clip(
                            self.population[i],
                            self.bounds[:, 0],
                            self.bounds[:, 1]
                        )

                        self.fitness[i] = self.f(self.population[i])

            best_idx = np.argmin(self.fitness)
            best_history.append(self.population[best_idx].copy())

        return (
            self.population[best_idx],
            self.fitness[best_idx],
            best_history
        )
