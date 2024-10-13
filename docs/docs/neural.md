## Machine Learning Weight Optimization 
Classes for defining neural network weight optimization problems.

### Neural Network
Class for defining neural network classifier weights optimization problem.

```python
class NeuralNetwork(
    hidden_nodes=None,
    activation='relu',
    algorithm='random_hill_climb',
    max_iters=100,
    bias=True,
    is_classifier=True,
    learning_rate=0.1,
    early_stopping=False,
    clip_max=1e10,
    restarts=0,
    schedule=<mlrose_ky.decay.GeomDecay object>,
    pop_size=200,
    mutation_prob=0.1,
    max_attempts=10,
    random_state=None,
    curve=False
)
```

**Parameters**:

*   `hidden_nodes` (_list of ints_) – List giving the number of nodes in each hidden layer.
*   `activation` (_string, default: ‘relu’_) – Activation function for each of the hidden layers. Must be one of: ‘identity’, ‘relu’, ‘sigmoid’ or ‘tanh’.
*   `algorithm` (_string, default: ‘random\_hill\_climb’_) – Algorithm used to find optimal network weights. Must be one of:’random\_hill\_climb’, ‘simulated\_annealing’, ‘genetic\_alg’ or ‘gradient\_descent’.
*   `max_iters` (_int, default: 100_) – Maximum number of iterations used to fit the weights.
*   `bias` (_bool, default: True_) – Whether to include a bias term.
*   **is\_classifer** (_bool, default: True_) – Whether the network is for classification or regression. Set `True` for classification and `False` for regression.
*   `learning_rate` (_float, default: 0.1_) – Learning rate for gradient descent or step size for randomized optimization algorithms.
*   `early_stopping` (_bool, default: False_) – Whether to terminate algorithm early if the loss is not improving. If `True`, then stop after max\_attempts iters with no improvement.
*   `clip_max` (_float, default: 1e+10_) – Used to limit weights to the range \[-1\*clip\_max, clip\_max\].
*   `restarts` (_int, default: 0_) – Number of random restarts. Only required if `algorithm = 'random_hill_climb'`.
*   `schedule` (_schedule object, default = mlrose_ky.GeomDecay()_) – Schedule used to determine the value of the temperature parameter. Only required if `algorithm = 'simulated_annealing'`.
*   `pop_size` (_int, default: 200_) – Size of population. Only required if `algorithm = 'genetic_alg'`.
*   `mutation_prob` (_float, default: 0.1_) – Probability of a mutation at each element of the state vector during reproduction, expressed as a value between 0 and 1. Only required if `algorithm = 'genetic_alg'`.
*   `max_attempts` (_int, default: 10_) – Maximum number of attempts to find a better state. Only required if `early_stopping = True`.
*   `random_state` (_int, default: None_) – If random\_state is a positive integer, random\_state is the seed used by np.random.seed(); otherwise, the random seed is not set.
*   `curve` (_bool, default: False_) – If bool is True, fitness\_curve containing the fitness at each training iteration is returned.

Variables:

*   `fitted_weights` (_array_) – Numpy array giving the fitted weights when `fit` is performed.
*   `loss` (_float_) – Value of loss function for fitted weights when `fit` is performed.
*   `predicted_probs` (_array_) – Numpy array giving the predicted probabilities for each class when `predict` is performed for multi-class classification data; or the predicted probability for class 1 when `predict` is performed for binary classification data.
*   `fitness_curve` (_array_) – Numpy array giving the fitness at each training iteration.

### Linear Regression
Class for defining linear regression weights optimization problem. Inherits `fit` and `predict` methods from `NeuralNetwork()` class.

```python
class LinearRegression(
    algorithm='random_hill_climb',
    max_iters=100,
    bias=True,
    learning_rate=0.1,
    early_stopping=False,
    clip_max=1e10,
    restarts=0,
    schedule=<mlrose_ky.decay.GeomDecay object>,
    pop_size=200,
    mutation_prob=0.1,
    max_attempts=10,
    random_state=None,
    curve=False
)
```
 
**Parameters**:

*   `algorithm` (_string, default: ‘random\_hill\_climb’_) – Algorithm used to find optimal network weights. Must be one of:’random\_hill\_climb’, ‘simulated\_annealing’, ‘genetic\_alg’ or ‘gradient\_descent’.
*   `max_iters` (_int, default: 100_) – Maximum number of iterations used to fit the weights.
*   `bias` (_bool, default: True_) – Whether to include a bias term.
*   `learning_rate` (_float, default: 0.1_) – Learning rate for gradient descent or step size for randomized optimization algorithms.
*   `early_stopping` (_bool, default: False_) – Whether to terminate algorithm early if the loss is not improving. If `True`, then stop after max\_attempts iters with no improvement.
*   `clip_max` (_float, default: 1e+10_) – Used to limit weights to the range \[-1\*clip\_max, clip\_max\].
*   `restarts` (_int, default: 0_) – Number of random restarts. Only required if `algorithm = 'random_hill_climb'`.
*   `schedule` (_schedule object, default = mlrose_ky.GeomDecay()_) – Schedule used to determine the value of the temperature parameter. Only required if `algorithm = 'simulated_annealing'`.
*   `pop_size` (_int, default: 200_) – Size of population. Only required if `algorithm = 'genetic_alg'`.
*   `mutation_prob` (_float, default: 0.1_) – Probability of a mutation at each element of the state vector during reproduction, expressed as a value between 0 and 1. Only required if `algorithm = 'genetic_alg'`.
*   `max_attempts` (_int, default: 10_) – Maximum number of attempts to find a better state. Only required if `early_stopping = True`.
*   `random_state` (_int, default: None_) – If random\_state is a positive integer, random\_state is the seed used by np.random.seed(); otherwise, the random seed is not set.
*   `curve` (_bool, default: False_) – If bool is true, curve containing the fitness at each training iteration is returned.

**Variables**:

*   `fitted_weights` (_array_) – Numpy array giving the fitted weights when `fit` is performed.
*   `loss` (_float_) – Value of loss function for fitted weights when `fit` is performed.
*   `fitness_curve` (_array_) – Numpy array giving the fitness at each training iteration.

### Logistic Regression
Class for defining logistic regression weights optimization problem. Inherits `fit` and `predict` methods from `NeuralNetwork()` class.

```python
class LogisticRegression(
    algorithm='random_hill_climb',
    max_iters=100,
    bias=True,
    learning_rate=0.1,
    early_stopping=False,
    clip_max=1e10,
    restarts=0,
    schedule=<mlrose_ky.decay.GeomDecay object>,
    pop_size=200,
    mutation_prob=0.1,
    max_attempts=10,
    random_state=None,
    curve=False
)
```

**Parameters**:

*  `algorithm` (_string, default: ‘random\_hill\_climb’_) – Algorithm used to find optimal network weights. Must be one of:’random\_hill\_climb’, ‘simulated\_annealing’, ‘genetic\_alg’ or ‘gradient\_descent’.
*  `max_iters` (_int, default: 100_) – Maximum number of iterations used to fit the weights.
*  `bias` (_bool, default: True_) – Whether to include a bias term.
*  `learning_rate` (_float, default: 0.1_) – Learning rate for gradient descent or step size for randomized optimization algorithms.
*  `early_stopping` (_bool, default: False_) – Whether to terminate algorithm early if the loss is not improving. If `True`, then stop after max\_attempts iters with no improvement.
* `clip_max` (_float, default: 1e+10_) – Used to limit weights to the range \[-1\*clip\_max, clip\_max\].
* `restarts` (_int, default: 0_) – Number of random restarts. Only required if `algorithm = 'random_hill_climb'`.
* `schedule` (_schedule object, default = mlrose_ky.GeomDecay()_) – Schedule used to determine the value of the temperature parameter. Only required if `algorithm = 'simulated_annealing'`.
*  `pop_size` (_int, default: 200_) – Size of population. Only required if `algorithm = 'genetic_alg'`.
* `mutation_prob` (_float, default: 0.1_) – Probability of a mutation at each element of the state vector during reproduction, expressed as a value between 0 and 1. Only required if `algorithm = 'genetic_alg'`.
* `max_attempts` (_int, default: 10_) – Maximum number of attempts to find a better state. Only required if `early_stopping = True`.
* `random_state` (_int, default: None_) – If random\_state is a positive integer, random\_state is the seed used by np.random.seed(); otherwise, the random seed is not set.
* `curve` (_bool, default: False_) – If bool is true, curve containing the fitness at each training iteration is returned.

**Variables**:

* `  fitted_weights` (_array_) – Numpy array giving the fitted weights when `fit` is performed.
*   `loss` (_float_) – Value of loss function for fitted weights when `fit` is performed.
*   `fitness_curve` (_array_) – Numpy array giving the fitness at each training iteration.