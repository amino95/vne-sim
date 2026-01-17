import random
import unittest

from grasp_solver import GRASPSolver


class NumberSelectionProblem:
    """
    Dummy problem for testing GRASP.
    Select a subset of numbers whose sum is greater than a constraint,
    while minimizing the sum.
    """

    def __init__(self, numbers, constraint_min_sum):
        self.numbers = numbers
        self.constraint_min_sum = constraint_min_sum


class NumberSelectionGRASP(GRASPSolver):
    """A GRASP implementation for the NumberSelectionProblem."""

    def __init__(self, max_iterations=100, seed=None, alpha=0.3):
        super().__init__(max_iterations, seed)
        self.alpha = alpha

    def construct_greedy_randomized_solution(self, problem):
        """Construct a solution by randomly selecting from the best candidates."""
        solution = []
        current_sum = 0
        candidates = list(problem.numbers)

        while current_sum < problem.constraint_min_sum and candidates:
            # Build a Restricted Candidate List (RCL)
            candidates.sort()  # smaller numbers are better

            rcl_size = int(len(candidates) * self.alpha)
            if rcl_size == 0 and candidates:
                rcl_size = 1

            rcl = candidates[:rcl_size]
            if not rcl:
                break

            selected = random.choice(rcl)
            solution.append(selected)
            current_sum += selected
            candidates.remove(selected)

        return solution

    def local_search(self, solution, problem):
        """
        A simple local search: try to swap a number in the solution
        with a number not in the solution to see if it improves the solution.
        """
        is_improved = True
        while is_improved:
            is_improved = False
            best_neighbor = list(solution)
            best_neighbor_value = self.evaluate(best_neighbor, problem)

            for i in range(len(solution)):
                # Try removing a number
                if len(solution) > 1:
                    neighbor = solution[:i] + solution[i + 1 :]
                    neighbor_value = self.evaluate(neighbor, problem)
                    if neighbor_value < best_neighbor_value:
                        best_neighbor = neighbor
                        best_neighbor_value = neighbor_value
                        is_improved = True

                # Try swapping a number
                numbers_not_in_solution = [n for n in problem.numbers if n not in solution]
                for num_to_add in numbers_not_in_solution:
                    neighbor = solution[:i] + solution[i + 1 :] + [num_to_add]
                    neighbor_value = self.evaluate(neighbor, problem)
                    if neighbor_value < best_neighbor_value:
                        best_neighbor = neighbor
                        best_neighbor_value = neighbor_value
                        is_improved = True
                        solution = best_neighbor
        return solution

    def evaluate(self, solution, problem):
        """
        Minimize the sum of selected numbers, with a penalty for infeasible solutions.
        """
        solution_sum = sum(solution)
        if solution_sum < problem.constraint_min_sum:
            return float("inf")
        return solution_sum


class TestGraspSolver(unittest.TestCase):
    def test_number_selection_problem(self):
        numbers = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 20, 30, 40]
        min_sum = 50

        problem = NumberSelectionProblem(numbers, min_sum)
        grasp = NumberSelectionGRASP(max_iterations=50, seed=42, alpha=0.4)

        print("\nSolving NumberSelectionProblem with GRASP...")
        best_solution, best_value = grasp.solve(problem)

        print(f"\nBest solution found: {best_solution}")
        print(f"Best solution value: {best_value}")
        # Check that the solution is feasible
        self.assertIsNotNone(best_solution)
        self.assertGreaterEqual(sum(best_solution), min_sum)

        # In this specific case, we can manually find a good solution
        # to see if GRASP is reasonable.
        # e.g., [10, 40] = 50, [20, 30] = 50
        # The optimal is 50.
        self.assertEqual(best_value, 50)


if __name__ == "__main__":
    unittest.main()
