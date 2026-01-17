import random


class GRASPSolver:
    """
    A generic GRASP (Greedy Randomized Adaptive Search Procedure) solver.
    This class is a template and the following methods should be implemented
    in a subclass for a specific problem:
    - construct_greedy_randomized_solution
    - local_search
    - evaluate
    """

    def __init__(self, max_iterations=100, seed=None):
        self.max_iterations = max_iterations
        self.seed = seed
        self.best_solution = None
        self.best_solution_value = float("inf")

    def solve(self, problem):
        """The main GRASP loop."""
        if self.seed is not None:
            random.seed(self.seed)

        self.best_solution = None
        self.best_solution_value = float("inf")

        for i in range(self.max_iterations):
            # Construction phase
            solution = self.construct_greedy_randomized_solution(problem)

            # Local search phase
            solution = self.local_search(solution, problem)

            # Evaluate and update best solution found
            value = self.evaluate(solution, problem)

            if value < self.best_solution_value:
                self.best_solution = solution
                self.best_solution_value = value
                print(f"Iteration {i}: New best solution found with value {value}")

        return self.best_solution, self.best_solution_value

    def construct_greedy_randomized_solution(self, problem):
        """
        Builds a solution using a greedy randomized approach.
        This method must be implemented by a subclass.
        """
        raise NotImplementedError

    def local_search(self, solution, problem):
        """
        Improves a given solution by exploring its neighborhood.
        This method must be implemented by a subclass.
        """
        raise NotImplementedError

    def evaluate(self, solution, problem):
        """
        Evaluates the quality of a solution.
        This method must be implemented by a subclass.
        """
        raise NotImplementedError