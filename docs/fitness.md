## Fitness Functions
Classes for defining fitness functions.

### One Max

#### Formula

Fitness function for One Max optimization problem. Evaluates the fitness of a state vector \( x = [x_0, x_1, \dots, x_{n-1}] \) as:

\[ \text{Fitness}(x) = \sum_{i=0}^{n-1} x_i \]

#### Class declaration
```python
class OneMax
```
>[!WARNING]- Note
> The One Max fitness function is suitable for use in either discrete or continuous-state optimization problems.

#### Class method
Evaluate the fitness of a state vector.
```python
evaluate(state)
```
**Parameters**: `state` (_array_) – State array for evaluation.

**Returns**: `fitness` (_float_) – Value of fitness function.

#### Example
```
>>> import mlrose_ky
>>> import numpy as np
>>> fitness = mlrose_ky.OneMax()
>>> state = np.array(\[0, 1, 0, 1, 1, 1, 1\])
>>> fitness.evaluate(state)
5
```
### Flip Flops

#### Formula
Fitness function for Flip Flop optimization problem. Evaluates the fitness of a state vector \( x \) as the total number of pairs of consecutive elements of \( x \), \((x_i\) and \(x_{i+1})\) where \( x_i \neq x_{i+1} \).

#### Class declaration
```python
class FlipFlop
```
> [!WARNING]- Note
> The Flip Flop fitness function is suitable for use in discrete-state optimization problems _only_.
#### Class method
Evaluate the fitness of a state vector.
```python
evaluate()
```

**Parameters**: `state` (_array_) – State array for evaluation.

**Returns**: `fitness` (_float_) – Value of fitness function.
#### Example
```
>>> import mlrose_ky
>>> import numpy as np
>>> fitness = mlrose_ky.FlipFlop()
>>> state = np.array([0, 1, 0, 1, 1, 1, 1])
>>> fitness.evaluate(state)
3
```

### Four Peaks
Fitness function for Four Peaks optimization problem. 
#### Formula
Evaluates the fitness of an n-dimensional state vector \( x \), given parameter \( T \), as:

\[\text{Fitness}(x, T) = \max(\text{tail}(0, x), \text{head}(1, x)) + R(x, T)\]

where:

- \( \text{tail}(b, x) \) is the number of trailing \( b \)'s in \( x \);
- \( \text{head}(b, x) \) is the number of leading \( b \)'s in \( x \);
- \( R(x, T) = n \), if \( \text{tail}(0, x) > T \) and \( \text{head}(1, x) > T \); and
- \( R(x, T) = 0 \), otherwise.
#### Class declaration
```python
class FourPeaks(t_pct=0.1)
```
> [!WARNING]- Note
> The Four Peaks fitness function is suitable for use in bit-string (discrete-state with `max_val = 2`) optimization problems _only_.

**Parameters:**  

- `t_pct` (*float*, *default: 0.1*) – Threshold parameter (\( T \)) for Four Peaks fitness function, expressed as a percentage of the state space dimension, \( n \) (i.e. \( T = t_{pct} \times n \)).
#### Class method
Evaluate the fitness of a state vector.
```python
evaluate(state)
```

**Parameters**: `state` (_array_) – State array for evaluation.

**Returns**: `fitness` (_float_) – Value of fitness function.
#### Example
```python
>>> import mlrose_ky
>>> import numpy as np
>>> fitness = mlrose_ky.FourPeaks(t_pct=0.15)
>>> state = np.array([1, 1, 1, 0, 1, 0, 0, 1, 0, 0, 0, 0])
>>> fitness.evaluate(state)
16
```

#### References

De Bonet, J., C. Isbell, and P. Viola (1997). MIMIC: Finding Optima by Estimating Probability Densities. In _Advances in Neural Information Processing Systems_ (NIPS) 9, pp. 424–430.

### Six Peaks
Fitness function for Six Peaks optimization problem.
#### Formula
Evaluates the fitness of an n-dimensional state vector \( x \), given parameter \( T \), as:

\[\text{Fitness}(x, T) = \max(\text{tail}(0, x), \text{head}(1, x)) + R(x, T)\]

where:

- \( \text{tail}(b, x) \) is the number of trailing \( b \)'s in \( x \);
- \( \text{head}(b, x) \) is the number of leading \( b \)'s in \( x \);
- \( R(x, T) = n \), if \( \text{tail}(0, x) > T \) and \( \text{head}(1, x) > T \) or \( \text{tail}(1, x) > T \) and \( \text{head}(0, x) > T \); and
- \( R(x, T) = 0 \), otherwise.

#### Class declaration
```python
class SixPeaks(t_pct=0.1)
```
> [!WARNING]- Note
> The Six Peaks fitness function is suitable for use in bit-string (discrete-state with `max_val = 2`) optimization problems only.

**Parameters:**  
`t_pct` (*float*, *default: 0.1*) – Threshold parameter (\( T \)) for Six Peaks fitness function, expressed as a percentage of the state space dimension, \( n \) (i.e. \( T = t_{pct} \times n \)).
#### Class method
```python
evaluate(state)
```

**Parameters**:
`state` (_array_) – State array for evaluation.

**Returns**:
`fitness` (_float_) – Value of fitness function.
#### Example
```python
>>> import mlrose_ky
>>> import numpy as np
>>> fitness = mlrose_ky.SixPeaks(t_pct=0.15)
>>> state = np.array([0, 0, 0, 1, 0, 1, 1, 0, 1, 1, 1, 1])
>>> fitness.evaluate(state)
12
```
#### References
De Bonet, J., C. Isbell, and P. Viola (1997). MIMIC: Finding Optima by Estimating Probability Densities. In _Advances in Neural Information Processing Systems_ (NIPS) 9, pp. 424–430.

### Continuous Peaks
Fitness function for Continuous Peaks optimization problem. 
#### Formula
Evaluates the fitness of an n-dimensional state vector \( x \), given parameter \( T \), as:

\[
\text{Fitness}(x, T) = \max(\text{max\_run}(0, x), \text{max\_run}(1, x)) + R(x, T)
\]

where:

- \( \text{max\_run}(b, x) \) is the length of the maximum run of \( b \)'s in \( x \);
- \( R(x, T) = n \), if \( \text{max\_run}(0, x) > T \) and \( \text{max\_run}(1, x) > T \); and
- \( R(x, T) = 0 \), otherwise.
#### Class declaration
```python
class ContinuousPeaks(t_pct=0.1)
```
> [!WARNING]- Note
> The Continuous Peaks fitness function is suitable for use in bit-string (discrete-state with `max_val = 2`) optimization problems _only_.

**Parameters:**  
`t_pct` (*float*, *default: 0.1*) – Threshold parameter (\( T \)) for Continuous Peaks fitness function, expressed as a percentage of the state space dimension, \( n \) (i.e. \( T = t_{pct} \times n \)).
#### Class method
```python
evaluate(state)
```

**Parameters**:
`state` (_array_) – State array for evaluation.

Returns:
`fitness` (_float_) – Value of fitness function.

#### Example
```python
>>> import mlrose_ky
>>> import numpy as np
>>> fitness = mlrose_ky.ContinuousPeaks(t_pct=0.15)
>>> state = np.array([0, 0, 0, 1, 0, 0, 0, 0, 0, 1, 1, 1])
>>> fitness.evaluate(state)
17
```

### Knapsack
Fitness function for Knapsack optimization problem.
#### Formula
Given a set of \( n \) items, where item \( i \) has known weight \( w_i \) and known value \( v_i \), and maximum knapsack capacity \( W \), the Knapsack fitness function evaluates the fitness of a state vector \( x = [x_0, x_1, \dots, x_{n-1}] \) as:

\[
\text{Fitness}(x) = \sum_{i=0}^{n-1} v_i x_i, \quad \text{if} \quad \sum_{i=0}^{n-1} w_i x_i \leq W, \quad \text{and } 0, \text{ otherwise},
\]

where \( x_i \) denotes the number of copies of item \( i \) included in the knapsack.
#### Class declaration
```python
class Knapsack(weights, values, max_weight_pct=0.35)
```
> [!WARNING]- Info
> The Knapsack fitness function is suitable for use in discrete-state optimization problems _only_.

**Parameters:**
- **`weights`** (*list*) – List of weights for each of the \( n \) items.
- **`values`** (*list*) – List of values for each of the \( n \) items.
- **`max_weight_pct`** (*float*, *default: 0.35*) – Parameter used to set maximum capacity of knapsack (\( W \)) as a percentage of the total of the weights list (\( W = \text{max_weight_pct} \times \text{total_weight} \)).
#### Class method
```python
evaluate(state)
```

**Parameters**:
`state` (_array_) – State array for evaluation.

Returns:
`fitness` (_float_) – Value of fitness function.
#### Example
```python
>>> import mlrose_ky
>>> import numpy as np
>>> weights = [10, 5, 2, 8, 15]
>>> values = [1, 2, 3, 4, 5]
>>> max_weight_pct = 0.6
>>> fitness = mlrose_ky.Knapsack(weights, values, max_weight_pct)
>>> state = np.array([1, 0, 2, 1, 0])
>>> fitness.evaluate(state)
11
```
 
### Travelling Salesman (TSP)
Fitness function for Travelling Salesman optimization problem. 
#### Formula
Evaluates the fitness of a tour of n nodes, represented by state vector \[x\], giving the order in which the nodes are visited, as the total distance travelled on the tour (including the distance travelled between the final node in the state vector and the first node in the state vector during the return leg of the tour). Each node must be visited exactly once for a tour to be considered valid.
#### Class declaration
```python
class TravellingSales(coords=None, distances=None)
```
> [!WARNING]- Note
> 1.  The TravellingSales fitness function is suitable for use in travelling salesperson (tsp) optimization problems _only_.
> 2.  It is necessary to specify at least one of `coords` and `distances` in initializing a TravellingSales fitness function object.

**Parameters**:
*   `coords` (_list of pairs, default: None_) – Ordered list of the (x, y) coordinates of all nodes (where element i gives the coordinates of node i). This assumes that travel between all pairs of nodes is possible. If this is not the case, then use `distances` instead.
*   `distances` (_list of triples, default: None_) – List giving the distances, d, between all pairs of nodes, u and v, for which travel is possible, with each list item in the form (u, v, d). Order of the nodes does not matter, so (u, v, d) and (v, u, d) are considered to be the same. If a pair is missing from the list, it is assumed that travel between the two nodes is not possible. This argument is ignored if coords is not `None`.
#### Class method
```python
evaluate(state)
```

**Parameters**:
`state` (_array_) – State array for evaluation.

Returns:
`fitness` (_float_) – Value of fitness function.
#### Example
```python
>>> import mlrose_ky
>>> import numpy as np
>>> coords = [(0, 0), (3, 0), (3, 2), (2, 4), (1, 3)]
>>> dists = [(0, 1, 3), (0, 2, 5), (0, 3, 1), (0, 4, 7), (1, 3, 6),
             (4, 1, 9), (2, 3, 8), (2, 4, 2), (3, 2, 8), (3, 4, 4)]
>>> fitness_coords = mlrose_ky.TravellingSales(coords=coords)
>>> state = np.array([0, 1, 4, 3, 2])
>>> fitness_coords.evaluate(state)
13.86138...
>>> fitness_dists = mlrose_ky.TravellingSales(distances=dists)
>>> fitness_dists.evaluate(state)
29
```
 
### N-Queens
Fitness function for N-Queens optimization problem.
#### Formula
Evaluates the fitness of an n-dimensional state vector \( x = [x_0, x_1, \dots, x_{n-1}] \), where \( x_i \) represents the row position (between 0 and \( n-1 \), inclusive) of the ‘queen’ in column \( i \), as the number of pairs of attacking queens.
#### Class declaration
```python
class Queens
```
> [!WARNING]- Note
> The Queens fitness function is suitable for use in discrete-state optimization problem only.
#### Class method
```python
evaluate(state)
```

**Parameters**:
`state` (_array_) – State array for evaluation.

Returns:
`fitness` (_float_) – Value of fitness function.
#### Example
```python
>>> import mlrose_ky
>>> import numpy as np
>>> fitness = mlrose_ky.Queens()
>>> state = np.array([1, 4, 1, 3, 5, 5, 2, 7])
>>> fitness.evaluate(state)
6
```
#### References
Russell, S. and P. Norvig (2010). _Artificial Intelligence: A Modern Approach_, 3rd edition. Prentice Hall, New Jersey, USA.

### Max K Color
Fitness function for Max-k color optimization problem. 
#### Formula
Fitness function for Max-k color optimization problem. Evaluates the fitness of an n-dimensional state vector \( x = [x_0, x_1, \dots, x_{n-1}] \), where \( x_i \) represents the color of node \( i \), as the number of pairs of adjacent nodes of the same color.
#### Class declaration
```python
class MaxKColor(edges)
```
> [!WARNING]- Note
> The MaxKColor fitness function is suitable for use in discrete-state optimization problems _only_.

**Parameters**:
`edges` (_list of pairs_) – List of all pairs of connected nodes. Order does not matter, so (a, b) and (b, a) are considered to be the same.
#### Class method
Evaluate the fitness of a state vector.
```python
evaluate(state)
```

**Parameters**:
`state` (_array_) – State array for evaluation.

Returns:
`fitness` (_float_) – Value of fitness function.
#### Example
```python
>>> import mlrose_ky
>>> import numpy as np
>>> edges = [(0, 1), (0, 2), (0, 4), (1, 3), (2, 0), (2, 3), (3, 4)]
>>> fitness = mlrose_ky.MaxKColor(edges)
>>> state = np.array([0, 1, 0, 1, 1])
>>> fitness.evaluate(state)
3
```

### Write your own fitness function
Class for generating your own fitness function.
#### Class declaration
```python
class CustomFitness(fitness_fn, problem_type='either', **kwargs)
```

Parameters:

* `fitness_fn` (_callable_) – Function for calculating fitness of a state with the signature `fitness_fn(state, **kwargs)`
*   `problem_type` (_string, default: ‘either’_) – Specifies problem type as ‘discrete’, ‘continuous’, ‘tsp’ or ‘either’ (denoting either discrete or continuous).
*   `kwargs` (_additional arguments_) – Additional parameters to be passed to the fitness function.
#### Class method
Evaluate the fitness of a state vector.
```python
evaluate(state)
```

**Parameters**:
`state` (_array_) – State array for evaluation.

**Returns**:
`fitness` (_float_) – Value of fitness function.
#### Example
```python
>>> import mlrose_ky
>>> import numpy as np
>>> def cust_fn(state, c): return c*np.sum(state)
>>> kwargs = {'c': 10}
>>> fitness = mlrose_ky.CustomFitness(cust_fn, **kwargs)
>>> state = np.array([1, 2, 3, 4, 5])
>>> fitness.evaluate(state)
150
```