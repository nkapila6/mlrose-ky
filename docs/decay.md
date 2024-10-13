## Decay Schedules 

Classes for defining decay schedules for simulated annealing.

### Geometric Decay
Schedule for geometrically decaying the simulated annealing temperature parameter T according to the formula:

#### Formula
\[ T(t) = \max(T_0 \times r^t, T_{min}) \]

where
- \( T_0 \) is the initial temperature (at time \( t = 0 \));
- \( r \) is the rate of geometric decay; and
- \( T_{min} \) is the minimum temperature value.

#### Class declaration
```python
class GeomDecay(init_temp=1.0, decay=0.99, min_temp=0.001)
```

**Parameters** [[source]](https://github.com/knakamura13/mlrose-ky/blob/123e66187276cdd7a188c48aff97360a6803d5f2/mlrose_ky/algorithms/decay/geometric_decay.py#L7)

- **init\_temp** (_float, default: 1.0_) – Initial value of temperature parameter T. Must be greater than 0.
- **decay** (_float, default: 0.99_) – Temperature decay parameter, r. Must be between 0 and 1.
- **min\_temp** (_float, default: 0.001_) – Minimum value of temperature parameter. Must be greater than 0.

#### Class method
Evaluate the temperature parameter at time t.
```python
evaluate(t)
```

**Parameters**: **t** (_int_) – Time at which the temperature parameter T is evaluated.

**Returns**: **temp** (_float_) – Temperature parameter at time t.

#### Example
```python
>>> import mlrose_ky
>>> schedule = mlrose_ky.GeomDecay(init_temp=10, decay=0.95, min_temp=1)
>>> schedule.evaluate(5)
7.73780...
```
### Arithmetic Decay
Schedule for arithmetically decaying the simulated annealing temperature parameter T according to the formula:
#### Formula
\[ T(t) = \max(T_{0} - rt, T_{min}) \]

where
*   \( T_{0} \) is the initial temperature (at time t = 0);
*   \( r \) is the rate of arithmetic decay; and
*   \( T_{min} \) is the minimum temperature value.

#### Class declaration
```python
class ArithDecay(init_temp=1.0, decay= 0.0001, min_temp=0.001)
```

**Parameters** [[source]](https://github.com/knakamura13/mlrose-ky/blob/123e66187276cdd7a188c48aff97360a6803d5f2/mlrose_ky/algorithms/decay/arithmetic_decay.py#L7)

*   **init\_temp** (_float, default: 1.0_) – Initial value of temperature parameter T. Must be greater than 0.
*   **decay** (_float, default: 0.0001_) – Temperature decay parameter, r. Must be greater than 0.
*   **min\_temp** (_float, default: 0.001_) – Minimum value of temperature parameter. Must be greater than 0.
#### Class method
Evaluate the temperature parameter at time t.
```python
evaluate(t)
```

**Parameters**: **t** (_int_) – Time at which the temperature paramter T is evaluated.

**Returns**: **temp** (_float_) – Temperature parameter at time t.

#### Example
```python
>>> import mlrose_ky
>>> schedule = mlrose_ky.ArithDecay(init_temp\=10, decay=0.95, min_temp\=1)
>>> schedule.evaluate(5)
5.25
```

### Exponential Decay
Schedule for exponentially decaying the simulated annealing temperature parameter T according to the formula.
#### Formula
\[ T(t) = \max(T_{0} e^{-rt}, T_{min}) \]

where:
*   \( T_{0} \) is the initial temperature (at time t = 0);
*   \( r \) is the rate of arithmetic decay; and
*   \( T_{min} \) is the minimum temperature value.

#### Class declaration
```python
class ExpDecay(init_temp=1.0, exp_const=0.005, min_temp=0.001)
```

**Parameters** [[source]](https://github.com/knakamura13/mlrose-ky/blob/123e66187276cdd7a188c48aff97360a6803d5f2/mlrose_ky/algorithms/decay/exponential_decay.py#L9)
*   **init\_temp** (_float, default: 1.0_) – Initial value of temperature parameter T. Must be greater than 0.
*   **exp\_const** (_float, default: 0.005_) – Exponential constant parameter, r. Must be greater than 0.
*   **min\_temp** (_float, default: 0.001_) – Minimum value of temperature parameter. Must be greater than 0.

#### Class method
Evaluate the temperature parameter at time t.
```python
evaluate(t)
```

**Parameters**: **t** (_int_) – Time at which the temperature paramter T is evaluated.

**Returns**: **temp** (_float_) – Temperature parameter at time t.

#### Example
```python
>>> import mlrose_ky
>>> schedule = mlrose_ky.ExpDecay(init_temp=10, exp_const=0.05, min_temp=1)
>>> schedule.evaluate(5)
7.78800...
```
### Write your own custom schedule
Class for generating your own temperature schedule.
```python
class CustomSchedule(schedule, **kwargs)
```

**Parameters** [[source]](https://github.com/knakamura13/mlrose-ky/blob/123e66187276cdd7a188c48aff97360a6803d5f2/mlrose_ky/algorithms/decay/custom_decay.py#L9)
*   **schedule** (_callable_) – Function for calculating the temperature at time t with the signature `schedule(t, **kwargs)`.
*   **kwargs** (_additional arguments_) – Additional parameters to be passed to schedule.

#### Example
```python
>>> import mlrose_ky
>>> def custom(t, c): return t + c
>>> kwargs \= {'c': 10}
>>> schedule \= mlrose_ky.CustomSchedule(custom, \*\*kwargs)
>>> schedule.evaluate(5)
15
```
#### Class method
Evaluate the temperature parameter at time t.
```python
evaluate(t)
```

**Parameters**: **t** (_int_) – Time at which the temperature paramter T is evaluated.

**Returns**: **temp** (_float_) – Temperature parameter at time t.