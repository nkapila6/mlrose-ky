## Optimization Problem Types
Classes for defining optimization problem objects.

#### Discrete Optimization Problem
Class for defining discrete-state optimization problems.

>[!INFO] class
>_class_ `DiscreteOpt`(_length_, _fitness\_fn_, _maximize=True_, _max\_val=2_)

**Parameters**:

*   **length** (_int_) – Number of elements in state vector.
*   **fitness\_fn** (_fitness function object_) – Object to implement fitness function for optimization.
*   **maximize** (_bool, default: True_) – Whether to maximize the fitness function. Set `False` for minimization problem.
*   **max\_val** (_int, default: 2_) – Number of unique values that each element in the state vector can take. Assumes values are integers in the range 0 to (max\_val - 1), inclusive.

#### Continuous Optimization Problem
Class for defining continuous-state optimization problems.
>[!INFO] class
>_class_ `ContinuousOpt`(_length_, _fitness\_fn_, _maximize=True_, _min\_val=0_, _max\_val=1_, _step=0.1_)

**Parameters**:

*   **length** (_int_) – Number of elements in state vector.
*   **fitness\_fn** (_fitness function object_) – Object to implement fitness function for optimization.
*   **maximize** (_bool, default: True_) – Whether to maximize the fitness function. Set `False` for minimization problem.
*   **min\_val** (_float, default: 0_) – Minimum value that each element of the state vector can take.
*   **max\_val** (_float, default: 1_) – Maximum value that each element of the state vector can take.
*   **step** (_float, default: 0.1_) – Step size used in determining neighbors of current state.

#### Travelling Salesperson Optimization Problem
Class for defining travelling salesperson optimization problems.
>[!INFO] class
>_class_ `TSPOpt`(_length_, _fitness\_fn=None_, _maximize=False_, _coords=None_, _distances=None_)

**Parameters**:

*   **length** (_int_) – Number of elements in state vector. Must equal number of nodes in the tour.
*   **fitness\_fn** (_fitness function object, default: None_) – Object to implement fitness function for optimization. If `None`, then `TravellingSales(coords=coords, distances=distances)` is used by default.
*   **maximize** (_bool, default: False_) – Whether to maximize the fitness function. Set `False` for minimization problem.
*   **coords** (_list of pairs, default: None_) – Ordered list of the (x, y) co-ordinates of all nodes. This assumes that travel between all pairs of nodes is possible. If this is not the case, then use distances instead. This argument is ignored if fitness\_fn is not `None`.
*   **distances** (_list of triples, default: None_) – List giving the distances, d, between all pairs of nodes, u and v, for which travel is possible, with each list item in the form (u, v, d). Order of the nodes does not matter, so (u, v, d) and (v, u, d) are considered to be the same. If a pair is missing from the list, it is assumed that travel between the two nodes is not possible. This argument is ignored if fitness\_fn or coords is not `None`.