## Algorithms
Functions to implement the randomized optimization and search algorithms.

> [!NOTE] Recommendation
> The below functions are implemented within mlrose-ky. However, it is highly recommended to use the [Runners](/runners/) for assignment.
### Hill Climbing 
Use standard hill climbing to find the optimum for a given optimization problem.

```python
hill_climb(
    problem,
    max_iters=float('inf'),
    restarts=0,
    init_state=None,
    curve=False,
    random_state=None
) 
```

**Parameters** [[source]](https://github.com/knakamura13/mlrose-ky/blob/123e66187276cdd7a188c48aff97360a6803d5f2/mlrose_ky/algorithms/hc.py#L12)

*   **problem** (_optimization object_) – Object containing fitness function optimization problem to be solved. For example, `DiscreteOpt()`, `ContinuousOpt()` or `TSPOpt()`.
*   **max\_iters** (_int, default: np.inf_) – Maximum number of iterations of the algorithm for each restart.
*   **restarts** (_int, default: 0_) – Number of random restarts.
*   **init\_state** (_array, default: None_) – 1-D Numpy array containing starting state for algorithm. If `None`, then a random state is used.
*   **curve** (_bool, default: False_) – Boolean to keep fitness values for a curve. If `False`, then no curve is stored. If `True`, then a history of fitness values is provided as a third return value.
*   **random\_state** (_int, default: None_) – If random\_state is a positive integer, random\_state is the seed used by np.random.seed(); otherwise, the random seed is not set.

**Returns:**

*   **best\_state** (_array_) – Numpy array containing state that optimizes the fitness function.
*   **best\_fitness** (_float_) – Value of fitness function at best state.
*   **fitness\_curve** (_array_) – Numpy array containing the fitness at every iteration. Only returned if input argument `curve` is `True`.

#### References

Russell, S. and P. Norvig (2010). _Artificial Intelligence: A Modern Approach_, 3rd edition. Prentice Hall, New Jersey, USA.

### Random Hill Climbing
Use randomized hill climbing to find the optimum for a given optimization problem.

```python
random_hill_climb(
    problem,
    max_attempts=10,
    max_iters=float('inf'),
    restarts=0,
    init_state=None,
    curve=False,
    random_state=None
)
```

**Parameters** [[source]](https://github.com/knakamura13/mlrose-ky/blob/123e66187276cdd7a188c48aff97360a6803d5f2/mlrose_ky/algorithms/rhc.py#L12)

*   **problem** (_optimization object_) – Object containing fitness function optimization problem to be solved. For example, `DiscreteOpt()`, `ContinuousOpt()` or `TSPOpt()`.
*   **max\_attempts** (_int, default: 10_) – Maximum number of attempts to find a better neighbor at each step.
*   **max\_iters** (_int, default: np.inf_) – Maximum number of iterations of the algorithm.
*   **restarts** (_int, default: 0_) – Number of random restarts.
*   **init\_state** (_array, default: None_) – 1-D Numpy array containing starting state for algorithm. If `None`, then a random state is used.
*   **curve** (_bool, default: False_) – Boolean to keep fitness values for a curve. If `False`, then no curve is stored. If `True`, then a history of fitness values is provided as a third return value.
*   **random\_state** (_int, default: None_) – If random\_state is a positive integer, random\_state is the seed used by np.random.seed(); otherwise, the random seed is not set.

**Returns**:

*   **best\_state** (_array_) – Numpy array containing state that optimizes the fitness function.
*   **best\_fitness** (_float_) – Value of fitness function at best state.
*   **fitness\_curve** (_array_) – Numpy array containing the fitness at every iteration. Only returned if input argument `curve` is `True`.

#### References

Brownlee, J (2011). _Clever Algorithms: Nature-Inspired Programming Recipes_. [http://www.cleveralgorithms.com](http://www.cleveralgorithms.com/).

### Simulated Annealing
Use simulated annealing to find the optimum for a given optimization problem.

```python
simulated_annealing(
    problem,
    schedule=<mlrose_ky.decay.GeomDecay object>,
    max_attempts=10,
    max_iters=float('inf'),
    init_state=None,
    curve=False,
    random_state=None
)
```

**Parameters** [[source]](https://github.com/knakamura13/mlrose-ky/blob/123e66187276cdd7a188c48aff97360a6803d5f2/mlrose_ky/algorithms/sa.py#L13)

*   **problem** (_optimization object_) – Object containing fitness function optimization problem to be solved. For example, `DiscreteOpt()`, `ContinuousOpt()` or `TSPOpt()`.
*   **schedule** (schedule object, default: `mlrose_ky.GeomDecay()`) – Schedule used to determine the value of the temperature parameter.
*   **max\_attempts** (_int, default: 10_) – Maximum number of attempts to find a better neighbor at each step.
*   **max\_iters** (_int, default: np.inf_) – Maximum number of iterations of the algorithm.
*   **init\_state** (_array, default: None_) – 1-D Numpy array containing starting state for algorithm. If `None`, then a random state is used.
*   **curve** (_bool, default: False_) – Boolean to keep fitness values for a curve. If `False`, then no curve is stored. If `True`, then a history of fitness values is provided as a third return value.
*   **random\_state** (_int, default: None_) – If random\_state is a positive integer, random\_state is the seed used by np.random.seed(); otherwise, the random seed is not set.

**Returns**:

*   **best\_state** (_array_) – Numpy array containing state that optimizes the fitness function.
*   **best\_fitness** (_float_) – Value of fitness function at best state.
*   **fitness\_curve** (_array_) – Numpy array containing the fitness at every iteration. Only returned if input argument `curve` is `True`.

#### References

Russell, S. and P. Norvig (2010). _Artificial Intelligence: A Modern Approach_, 3rd edition. Prentice Hall, New Jersey, USA.

### Genetic Algorithms
Use a standard genetic algorithm to find the optimum for a given optimization problem.

```python
genetic_alg(
    problem,
    pop_size=200,
    mutation_prob=0.1,
    max_attempts=10,
    max_iters=float('inf'),
    curve=False,
    random_state=None
)
```

**Parameters** [[source]](https://github.com/knakamura13/mlrose-ky/blob/123e66187276cdd7a188c48aff97360a6803d5f2/mlrose_ky/algorithms/ga.py#L101)

*   **problem** (_optimization object_) – Object containing fitness function optimization problem to be solved. For example, `DiscreteOpt()`, `ContinuousOpt()` or `TSPOpt()`.
*   **pop\_size** (_int, default: 200_) – Size of population to be used in genetic algorithm.
*   **mutation\_prob** (_float, default: 0.1_) – Probability of a mutation at each element of the state vector during reproduction, expressed as a value between 0 and 1.
*   **max\_attempts** (_int, default: 10_) – Maximum number of attempts to find a better state at each step.
*   **max\_iters** (_int, default: np.inf_) – Maximum number of iterations of the algorithm.
*   **curve** (_bool, default: False_) – Boolean to keep fitness values for a curve. If `False`, then no curve is stored. If `True`, then a history of fitness values is provided as a third return value.
*   **random\_state** (_int, default: None_) – If random\_state is a positive integer, random\_state is the seed used by np.random.seed(); otherwise, the random seed is not set.

**Returns**:

*   **best\_state** (_array_) – Numpy array containing state that optimizes the fitness function.
*   **best\_fitness** (_float_) – Value of fitness function at best state.
*   **fitness\_curve** (_array_) – Numpy array of arrays containing the fitness of the entire population at every iteration. Only returned if input argument `curve` is `True`.

#### References

Russell, S. and P. Norvig (2010). _Artificial Intelligence: A Modern Approach_, 3rd edition. Prentice Hall, New Jersey, USA.

### MIMIC
Use MIMIC to find the optimum for a given optimization problem.

```python
mimic(
    problem,
    pop_size=200,
    keep_pct=0.2,
    max_attempts=10,
    max_iters=float('inf'),
    curve=False,
    random_state=None,
    fast_mimic=False
)
```
> [!DANGER] Warning
> MIMIC cannot be used for solving continuous-state optimization problems.

**Parameters** [[source]](https://github.com/knakamura13/mlrose-ky/blob/123e66187276cdd7a188c48aff97360a6803d5f2/mlrose_ky/algorithms/mimic.py#L12)

*   **problem** (_optimization object_) – Object containing fitness function optimization problem to be solved. For example, `DiscreteOpt()` or `TSPOpt()`.
*   **pop\_size** (_int, default: 200_) – Size of population to be used in algorithm.
*   **keep\_pct** (_float, default: 0.2_) – Proportion of samples to keep at each iteration of the algorithm, expressed as a value between 0 and 1.
*   **max\_attempts** (_int, default: 10_) – Maximum number of attempts to find a better neighbor at each step.
*   **max\_iters** (_int, default: np.inf_) – Maximum number of iterations of the algorithm.
*   **curve** (_bool, default: False_) – Boolean to keep fitness values for a curve. If `False`, then no curve is stored. If `True`, then a history of fitness values is provided as a third return value.
*   **random\_state** (_int, default: None_) – If random\_state is a positive integer, random\_state is the seed used by np.random.seed(); otherwise, the random seed is not set.
*   **fast\_mimic** (_bool, default: False_) – Activate fast mimic mode to compute the mutual information in vectorized form. Faster speed but requires more memory.

**Returns**:

*   **best\_state** (_array_) – Numpy array containing state that optimizes the fitness function.
*   **best\_fitness** (_float_) – Value of fitness function at best state.
*   **fitness\_curve** (_array_) – Numpy array containing the fitness at every iteration. Only returned if input argument `curve` is `True`.

#### References
De Bonet, J., C. Isbell, and P. Viola (1997). MIMIC: Finding Optima by Estimating Probability Densities. In _Advances in Neural Information Processing Systems_ (NIPS) 9, pp. 424–430.

