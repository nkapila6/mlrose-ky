"""Unit tests for algorithms/gd.py"""

# Authors: Kyle Nakamura
# License: BSD 3-clause

import numpy as np
import pytest

from mlrose_ky.algorithms import gradient_descent
from tests.globals import SEED


# noinspection PyMissingOrEmptyDocstring
class MockProblem:
    """Mock problem class for testing gradient_descent."""

    def __init__(self):
        self.state = np.array([0.0])
        self.maximize = 1
        self.fitness = self.evaluate_fitness(self.state)
        self.fitness_evaluations = 0
        self.can_stop_flag = False

    def reset(self):
        self.state = np.array([0.0])
        self.fitness = self.evaluate_fitness(self.state)

    def set_state(self, state):
        self.state = state
        self.fitness = self.evaluate_fitness(state)

    def get_state(self):
        return self.state

    def get_fitness(self):
        return self.fitness

    def get_maximize(self):
        return self.maximize

    def calculate_updates(self):
        # Return gradient towards the optimal state at 5.0
        return np.array([5.0 - self.state[0]])

    def update_state(self, updates):
        return self.state + updates

    def eval_fitness(self, state):
        self.fitness_evaluations += 1
        return self.evaluate_fitness(state)

    @staticmethod
    def evaluate_fitness(state):
        # Quadratic function with maximum at state = 5.0, fitness = 25.0
        return -((state[0] - 5.0) ** 2) + 25.0

    def can_stop(self):
        return self.can_stop_flag

    def get_adjusted_fitness(self):
        return self.fitness


# noinspection PyMissingOrEmptyDocstring
@pytest.fixture
def iteration_counter():
    class IterationCounter:
        def __init__(self):
            self.iterations = 0

    return IterationCounter()


# noinspection PyMissingOrEmptyDocstring
@pytest.fixture
def state_fitness_callback():
    # noinspection PyMissingOrEmptyDocstring
    def callback(iteration, state, fitness, user_data, attempt=None, max_attempts_reached=None, curve=None, **kwargs):
        if iteration > 0:
            user_data.iterations += 1
        if iteration >= 5:
            return False  # Stop iterating
        return True  # Continue iterating

    return callback


class TestGradientDescent:
    """Unit tests for the gradient_descent function."""

    def test_gradient_descent_basic(self):
        """Test basic functionality with default parameters."""
        problem = MockProblem()
        best_state, best_fitness, _ = gradient_descent(problem)
        assert np.allclose(best_state, np.array([5.0]), atol=1e-2)
        assert np.isclose(best_fitness, 25.0, atol=1e-2)

    def test_gradient_descent_random_state(self):
        """Test setting a random seed."""
        problem = MockProblem()
        gradient_descent(problem, random_state=SEED)

    def test_gradient_descent_with_init_state(self):
        """Test initializing with a custom state."""
        problem = MockProblem()
        init_state = np.array([10.0])
        best_state, best_fitness, _ = gradient_descent(problem, init_state=init_state)
        assert np.allclose(best_state, np.array([5.0]), atol=1e-2)
        assert np.isclose(best_fitness, 25.0, atol=1e-2)

    def test_gradient_descent_with_curve(self):
        """Test collecting the fitness curve."""
        problem = MockProblem()
        best_state, best_fitness, fitness_curve = gradient_descent(problem, curve=True)
        assert fitness_curve is not None
        assert len(fitness_curve) > 0

    def test_gradient_descent_with_callback(self, iteration_counter, state_fitness_callback):
        """Test using a state fitness callback."""
        problem = MockProblem()
        _user_data = iteration_counter
        gradient_descent(problem, state_fitness_callback=state_fitness_callback, callback_user_info=_user_data)
        assert _user_data.iterations == 5

    def test_gradient_descent_when_max_attempts_set(self):
        """Test terminating when max_attempts is reached."""
        problem = MockProblem()
        problem.evaluate_fitness = lambda state: 0.0
        gradient_descent(problem, max_attempts=1)

    def test_gradient_descent_when_max_iters_set(self):
        """Test terminating when max_iters is reached."""
        problem = MockProblem()
        gradient_descent(problem, max_iters=1)

    def test_gradient_descent_minimization(self):
        """Test the algorithm on a minimization problem."""
        problem = MockProblem()
        problem.maximize = -1
        problem.evaluate_fitness = lambda state: (state[0] - 5.0) ** 2
        problem.fitness = problem.evaluate_fitness(problem.state)
        best_state, best_fitness, _ = gradient_descent(problem)
        assert np.allclose(best_state, problem.get_state(), atol=1e-2)
