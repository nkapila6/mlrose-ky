## Parallelize mlrose randomized optimizations
Faster experimentation using the new parallel script.
*Tutorial by Nikhil Kapila - 21.02.2025*

!!! warning 
    21.02.2025: I do not have access to the PyPI project so the code can be copied from GitHub and copied into your local environment, i.e. this will not work if you install it through `pip`.

    The source can be viewed [here](https://github.com/knakamura13/mlrose-ky/blob/main/src/mlrose_ky/utils/parallel.py).

```python
import mlrose_ky as mlrose
from mlrose_ky.utils import parallel
import numpy as np
import time
from pprint import pprint
```

## Creating the parameter grid
In this section, we initialize all the different parameters required by our algorithms. 
We use an example of Four Peaks for this notebook to explain the parallelized *FASTER* runs.


```python
# We use 3 seeds
seeds = [1, 2, 3]

# for Four Peaks problem
t_pct = 0.05

# for all algorithms
max_attempt = [10, 25, 50]
max_iter = [5000]

# RHC 
restart = [1, 2, 5, 10, 15, 25, 50, 100]

# pop_size for GA and MIMIC
# mutation_prob for GA
# keep_pct for MIMIC
pop_size = [50, 100, 200, 400, 800]
mutation_prob = [0.05, 0.1, 0.2, 0.3, 0.4]
keep_pct = [0.1, 0.2, 0.3, 0.4]

# decays for SA, you can pick the other types too --> refer docs
decays = []
for t in [1, 2, 5]:
    for decay in [0.0001, 0.00025, 0.0005]:
        decays.append(mlrose.ArithDecay(init_temp=t, decay=decay, min_temp=0.001))
```

## RHC
We use a single sized RHC, pass in all our previous parameters, and generate the output results.
`view_params` displays how your input parameters are expanded into all possible combinations of their values, generating a full set of parameter permutations.

### With `view_params` as `False`


```python
for size in [20]: #, 60, 100]:
    print("fourpeaks of size:", size)
    problem = mlrose.DiscreteOpt(
        length=size, fitness_fn=mlrose.FourPeaks(t_pct=t_pct), maximize=True
    )
    problem.set_mimic_fast_mode(fast_mode=True)

    # Hyperparameter tunning section
    print("rhc")
    rhc_grid = {
        "problem": [problem],
        "max_attempt": max_attempt,
        "max_iter": max_iter,
        "restart": restart,
        "seeds": [seeds],
    }

    rhc_results = parallel.get_results(rhc_grid, parallel.rhc_run, verbose=True, view_params=False)
```

    fourpeaks of size: 20
    rhc
    Number of params: 24


    [Parallel(n_jobs=-1)]: Using backend LokyBackend with 8 concurrent workers.
    [Parallel(n_jobs=-1)]: Done  24 out of  24 | elapsed:    3.8s finished


### With `view_params` as `True`
You can see how the different combinations come out.


```python
for size in [20]: #, 60, 100]:
    print("fourpeaks of size:", size)
    problem = mlrose.DiscreteOpt(
        length=size, fitness_fn=mlrose.FourPeaks(t_pct=t_pct), maximize=True
    )
    problem.set_mimic_fast_mode(fast_mode=True)

    # Hyperparameter tunning section
    print("rhc")
    rhc_grid = {
        "problem": [problem],
        "max_attempt": max_attempt,
        "max_iter": max_iter,
        "restart": restart,
        "seeds": [seeds],
    }

    rhc_results = parallel.get_results(rhc_grid, parallel.rhc_run, verbose=True, view_params=True)
```

    [Parallel(n_jobs=-1)]: Using backend LokyBackend with 8 concurrent workers.


    fourpeaks of size: 20
    rhc
    [{'max_attempt': 10,
      'max_iter': 5000,
      'problem': <mlrose_ky.opt_probs.discrete_opt.DiscreteOpt object at 0x11cf53a70>,
      'restart': 1,
      'seeds': [1, 2, 3]},
     {'max_attempt': 10,
      'max_iter': 5000,
      'problem': <mlrose_ky.opt_probs.discrete_opt.DiscreteOpt object at 0x11cf53a70>,
      'restart': 2,
      'seeds': [1, 2, 3]},
     {'max_attempt': 10,
      'max_iter': 5000,
      'problem': <mlrose_ky.opt_probs.discrete_opt.DiscreteOpt object at 0x11cf53a70>,
      'restart': 5,
      'seeds': [1, 2, 3]},
     {'max_attempt': 10,
      'max_iter': 5000,
      'problem': <mlrose_ky.opt_probs.discrete_opt.DiscreteOpt object at 0x11cf53a70>,
      'restart': 10,
      'seeds': [1, 2, 3]},
     {'max_attempt': 10,
      'max_iter': 5000,
      'problem': <mlrose_ky.opt_probs.discrete_opt.DiscreteOpt object at 0x11cf53a70>,
      'restart': 15,
      'seeds': [1, 2, 3]},
     {'max_attempt': 10,
      'max_iter': 5000,
      'problem': <mlrose_ky.opt_probs.discrete_opt.DiscreteOpt object at 0x11cf53a70>,
      'restart': 25,
      'seeds': [1, 2, 3]},
     {'max_attempt': 10,
      'max_iter': 5000,
      'problem': <mlrose_ky.opt_probs.discrete_opt.DiscreteOpt object at 0x11cf53a70>,
      'restart': 50,
      'seeds': [1, 2, 3]},
     {'max_attempt': 10,
      'max_iter': 5000,
      'problem': <mlrose_ky.opt_probs.discrete_opt.DiscreteOpt object at 0x11cf53a70>,
      'restart': 100,
      'seeds': [1, 2, 3]},
     {'max_attempt': 25,
      'max_iter': 5000,
      'problem': <mlrose_ky.opt_probs.discrete_opt.DiscreteOpt object at 0x11cf53a70>,
      'restart': 1,
      'seeds': [1, 2, 3]},
     {'max_attempt': 25,
      'max_iter': 5000,
      'problem': <mlrose_ky.opt_probs.discrete_opt.DiscreteOpt object at 0x11cf53a70>,
      'restart': 2,
      'seeds': [1, 2, 3]},
     {'max_attempt': 25,
      'max_iter': 5000,
      'problem': <mlrose_ky.opt_probs.discrete_opt.DiscreteOpt object at 0x11cf53a70>,
      'restart': 5,
      'seeds': [1, 2, 3]},
     {'max_attempt': 25,
      'max_iter': 5000,
      'problem': <mlrose_ky.opt_probs.discrete_opt.DiscreteOpt object at 0x11cf53a70>,
      'restart': 10,
      'seeds': [1, 2, 3]},
     {'max_attempt': 25,
      'max_iter': 5000,
      'problem': <mlrose_ky.opt_probs.discrete_opt.DiscreteOpt object at 0x11cf53a70>,
      'restart': 15,
      'seeds': [1, 2, 3]},
     {'max_attempt': 25,
      'max_iter': 5000,
      'problem': <mlrose_ky.opt_probs.discrete_opt.DiscreteOpt object at 0x11cf53a70>,
      'restart': 25,
      'seeds': [1, 2, 3]},
     {'max_attempt': 25,
      'max_iter': 5000,
      'problem': <mlrose_ky.opt_probs.discrete_opt.DiscreteOpt object at 0x11cf53a70>,
      'restart': 50,
      'seeds': [1, 2, 3]},
     {'max_attempt': 25,
      'max_iter': 5000,
      'problem': <mlrose_ky.opt_probs.discrete_opt.DiscreteOpt object at 0x11cf53a70>,
      'restart': 100,
      'seeds': [1, 2, 3]},
     {'max_attempt': 50,
      'max_iter': 5000,
      'problem': <mlrose_ky.opt_probs.discrete_opt.DiscreteOpt object at 0x11cf53a70>,
      'restart': 1,
      'seeds': [1, 2, 3]},
     {'max_attempt': 50,
      'max_iter': 5000,
      'problem': <mlrose_ky.opt_probs.discrete_opt.DiscreteOpt object at 0x11cf53a70>,
      'restart': 2,
      'seeds': [1, 2, 3]},
     {'max_attempt': 50,
      'max_iter': 5000,
      'problem': <mlrose_ky.opt_probs.discrete_opt.DiscreteOpt object at 0x11cf53a70>,
      'restart': 5,
      'seeds': [1, 2, 3]},
     {'max_attempt': 50,
      'max_iter': 5000,
      'problem': <mlrose_ky.opt_probs.discrete_opt.DiscreteOpt object at 0x11cf53a70>,
      'restart': 10,
      'seeds': [1, 2, 3]},
     {'max_attempt': 50,
      'max_iter': 5000,
      'problem': <mlrose_ky.opt_probs.discrete_opt.DiscreteOpt object at 0x11cf53a70>,
      'restart': 15,
      'seeds': [1, 2, 3]},
     {'max_attempt': 50,
      'max_iter': 5000,
      'problem': <mlrose_ky.opt_probs.discrete_opt.DiscreteOpt object at 0x11cf53a70>,
      'restart': 25,
      'seeds': [1, 2, 3]},
     {'max_attempt': 50,
      'max_iter': 5000,
      'problem': <mlrose_ky.opt_probs.discrete_opt.DiscreteOpt object at 0x11cf53a70>,
      'restart': 50,
      'seeds': [1, 2, 3]},
     {'max_attempt': 50,
      'max_iter': 5000,
      'problem': <mlrose_ky.opt_probs.discrete_opt.DiscreteOpt object at 0x11cf53a70>,
      'restart': 100,
      'seeds': [1, 2, 3]}]
    Number of params: 24


    [Parallel(n_jobs=-1)]: Done  24 out of  24 | elapsed:    0.9s finished


## Understanding the output
The output is a list of DataFrames, we had 24 different combinations and hence, there are 24 different output dataframes wherein each row in each output refers to a specific seed.


```python
len(rhc_results)
```




    24



Looking at the first dataframe, we can see our hyperparams of `Restart`=1 and the time taken per seed.


```python
rhc_results[0]
```




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>Seed</th>
      <th>Best State</th>
      <th>Best Fitness</th>
      <th>Fitness Value</th>
      <th>Fevals</th>
      <th>Max Attempt</th>
      <th>Max Iters</th>
      <th>Restart</th>
      <th>Problem</th>
      <th>Time</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>1</td>
      <td>[1, 1, 1, 1, 1, 1, 1, 1, 1, 0, 0, 0, 1, 1, 1, ...</td>
      <td>29.0</td>
      <td>[4.0, 4.0, 4.0, 4.0, 4.0, 4.0, 4.0, 5.0, 8.0, ...</td>
      <td>[18.0, 19.0, 20.0, 21.0, 22.0, 23.0, 24.0, 26....</td>
      <td>10</td>
      <td>5000</td>
      <td>1</td>
      <td>&lt;mlrose_ky.opt_probs.discrete_opt.DiscreteOpt ...</td>
      <td>0.002735</td>
    </tr>
    <tr>
      <th>1</th>
      <td>2</td>
      <td>[0, 0, 0, 1, 1, 1, 1, 0, 0, 0, 0, 1, 0, 1, 0, ...</td>
      <td>6.0</td>
      <td>[6.0, 6.0, 6.0, 6.0, 6.0, 6.0, 6.0, 6.0, 6.0, ...</td>
      <td>[25.0, 26.0, 27.0, 28.0, 29.0, 30.0, 31.0, 32....</td>
      <td>10</td>
      <td>5000</td>
      <td>1</td>
      <td>&lt;mlrose_ky.opt_probs.discrete_opt.DiscreteOpt ...</td>
      <td>0.002735</td>
    </tr>
    <tr>
      <th>2</th>
      <td>3</td>
      <td>[1, 1, 1, 0, 0, 0, 1, 0, 0, 0, 0, 1, 0, 1, 1, ...</td>
      <td>3.0</td>
      <td>[0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, ...</td>
      <td>[1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0, ...</td>
      <td>10</td>
      <td>5000</td>
      <td>1</td>
      <td>&lt;mlrose_ky.opt_probs.discrete_opt.DiscreteOpt ...</td>
      <td>0.002735</td>
    </tr>
  </tbody>
</table>
</div>



To access fitness value of `Restart`=1 for `Seed`=1, we can do the following.


```python
rhc_results[1].loc[0, 'Fitness Value']
# or rhc_results[1]['Fitness Value'][0]
# view more ways here: https://pandas.pydata.org/docs/reference/api/pandas.Series.html
```




    [4.0,
     4.0,
     4.0,
     4.0,
     4.0,
     4.0,
     4.0,
     5.0,
     8.0,
     28.0,
     28.0,
     29.0,
     29.0,
     29.0,
     29.0,
     29.0,
     29.0,
     29.0,
     29.0,
     29.0,
     29.0,
     29.0]



## Similarly, for SA


```python
sa_grid = {
    "problem": [problem],
    "max_attempt": max_attempt,
    "max_iter": max_iter,
    "decay": decays,
    "seeds": [seeds],
}
sa_results = parallel.get_results(sa_grid, parallel.sa_run, verbose=True, view_params=False)
```

    [Parallel(n_jobs=-1)]: Using backend LokyBackend with 8 concurrent workers.


    Number of params: 27


    [Parallel(n_jobs=-1)]: Done  27 out of  27 | elapsed:    1.2s finished



```python
sa_results[0]
```




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>Seed</th>
      <th>Best State</th>
      <th>Best Fitness</th>
      <th>Fitness Value</th>
      <th>Fevals</th>
      <th>Max Attempt</th>
      <th>Max Iters</th>
      <th>Decay</th>
      <th>Problem</th>
      <th>Time</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>1</td>
      <td>[1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 0, 1, 0, 0, ...</td>
      <td>31.0</td>
      <td>[22.0, 22.0, 22.0, 22.0, 22.0, 22.0, 22.0, 26....</td>
      <td>[2.0, 4.0, 6.0, 8.0, 9.0, 11.0, 12.0, 14.0, 16...</td>
      <td>10</td>
      <td>5000</td>
      <td>ArithDecay(init_temp=1, decay=0.0001, min_temp...</td>
      <td>&lt;mlrose_ky.opt_probs.discrete_opt.DiscreteOpt ...</td>
      <td>0.132708</td>
    </tr>
    <tr>
      <th>1</th>
      <td>2</td>
      <td>[1, 1, 0, 1, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, ...</td>
      <td>34.0</td>
      <td>[2.0, 2.0, 2.0, 2.0, 2.0, 2.0, 0.0, 0.0, 0.0, ...</td>
      <td>[2.0, 4.0, 6.0, 8.0, 10.0, 12.0, 14.0, 16.0, 1...</td>
      <td>10</td>
      <td>5000</td>
      <td>ArithDecay(init_temp=1, decay=0.0001, min_temp...</td>
      <td>&lt;mlrose_ky.opt_probs.discrete_opt.DiscreteOpt ...</td>
      <td>0.132708</td>
    </tr>
    <tr>
      <th>2</th>
      <td>3</td>
      <td>[1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, ...</td>
      <td>36.0</td>
      <td>[3.0, 3.0, 2.0, 2.0, 2.0, 2.0, 2.0, 2.0, 2.0, ...</td>
      <td>[2.0, 4.0, 6.0, 8.0, 10.0, 12.0, 14.0, 16.0, 1...</td>
      <td>10</td>
      <td>5000</td>
      <td>ArithDecay(init_temp=1, decay=0.0001, min_temp...</td>
      <td>&lt;mlrose_ky.opt_probs.discrete_opt.DiscreteOpt ...</td>
      <td>0.132708</td>
    </tr>
  </tbody>
</table>
</div>



## GA


```python
ga_grid = {
    "problem": [problem],
    "pop_size": pop_size,
    "mutation_prob": mutation_prob,
    "max_attempt": max_attempt,
    "max_iter": max_iter,
    "seeds": [seeds],
}
ga_results = parallel.get_results(ga_grid, parallel.ga_run, verbose=True)
```

    [Parallel(n_jobs=-1)]: Using backend LokyBackend with 8 concurrent workers.


    Number of params: 75


    [Parallel(n_jobs=-1)]: Done  52 tasks      | elapsed:    9.1s
    [Parallel(n_jobs=-1)]: Done  60 out of  75 | elapsed:   11.0s remaining:    2.8s
    [Parallel(n_jobs=-1)]: Done  75 out of  75 | elapsed:   20.6s finished



```python
ga_results[0]
```




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>Seed</th>
      <th>Best State</th>
      <th>Best Fitness</th>
      <th>Fitness Value</th>
      <th>Fevals</th>
      <th>Max Attempt</th>
      <th>Max Iters</th>
      <th>Pop Size</th>
      <th>Mutation Prob</th>
      <th>Problem</th>
      <th>Time</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>1</td>
      <td>[1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, ...</td>
      <td>38.0</td>
      <td>[27.0, 27.0, 27.0, 27.0, 28.0, 29.0, 29.0, 29....</td>
      <td>[102.0, 153.0, 204.0, 255.0, 307.0, 359.0, 410...</td>
      <td>10</td>
      <td>5000</td>
      <td>50</td>
      <td>0.05</td>
      <td>&lt;mlrose_ky.opt_probs.discrete_opt.DiscreteOpt ...</td>
      <td>0.162858</td>
    </tr>
    <tr>
      <th>1</th>
      <td>2</td>
      <td>[1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, ...</td>
      <td>37.0</td>
      <td>[26.0, 27.0, 27.0, 28.0, 28.0, 30.0, 30.0, 33....</td>
      <td>[102.0, 154.0, 205.0, 257.0, 308.0, 360.0, 411...</td>
      <td>10</td>
      <td>5000</td>
      <td>50</td>
      <td>0.05</td>
      <td>&lt;mlrose_ky.opt_probs.discrete_opt.DiscreteOpt ...</td>
      <td>0.162858</td>
    </tr>
    <tr>
      <th>2</th>
      <td>3</td>
      <td>[1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, ...</td>
      <td>38.0</td>
      <td>[28.0, 29.0, 31.0, 31.0, 33.0, 34.0, 34.0, 34....</td>
      <td>[102.0, 154.0, 206.0, 257.0, 309.0, 361.0, 412...</td>
      <td>10</td>
      <td>5000</td>
      <td>50</td>
      <td>0.05</td>
      <td>&lt;mlrose_ky.opt_probs.discrete_opt.DiscreteOpt ...</td>
      <td>0.162858</td>
    </tr>
  </tbody>
</table>
</div>



## MIMIC


```python
mimic_grid = {
    "problem": [problem],
    "pop_size": pop_size,
    "keep_pct": keep_pct,
    "max_attempt": max_attempt,
    "max_iter": max_iter,
    "seeds": [seeds],
}

mimic_results = parallel.get_results(mimic_grid, parallel.mimic_run, verbose=True)
```

    [Parallel(n_jobs=-1)]: Using backend LokyBackend with 8 concurrent workers.


    Number of params: 60


    [Parallel(n_jobs=-1)]: Done  34 tasks      | elapsed:    2.6s
    [Parallel(n_jobs=-1)]: Done  60 out of  60 | elapsed:    9.4s finished



```python
mimic_results[0]
```




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>Seed</th>
      <th>Best State</th>
      <th>Best Fitness</th>
      <th>Fitness Value</th>
      <th>Fevals</th>
      <th>Max Attempt</th>
      <th>Max Iters</th>
      <th>Pop Size</th>
      <th>Keep Pct</th>
      <th>Problem</th>
      <th>Time</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>1</td>
      <td>[1, 1, 1, 0, 0, 0, 0, 0, 0, 1, 1, 0, 1, 0, 1, ...</td>
      <td>25.0</td>
      <td>[25.0, 25.0, 25.0, 25.0, 25.0, 25.0, 25.0, 25....</td>
      <td>[102.0, 153.0, 204.0, 255.0, 306.0, 357.0, 408...</td>
      <td>10</td>
      <td>5000</td>
      <td>50</td>
      <td>0.1</td>
      <td>&lt;mlrose_ky.opt_probs.discrete_opt.DiscreteOpt ...</td>
      <td>0.295165</td>
    </tr>
    <tr>
      <th>1</th>
      <td>2</td>
      <td>[1, 1, 1, 1, 1, 0, 0, 1, 0, 1, 1, 0, 0, 0, 0, ...</td>
      <td>29.0</td>
      <td>[27.0, 29.0, 29.0, 29.0, 29.0, 29.0, 29.0, 29....</td>
      <td>[102.0, 154.0, 205.0, 256.0, 307.0, 358.0, 409...</td>
      <td>10</td>
      <td>5000</td>
      <td>50</td>
      <td>0.1</td>
      <td>&lt;mlrose_ky.opt_probs.discrete_opt.DiscreteOpt ...</td>
      <td>0.295165</td>
    </tr>
    <tr>
      <th>2</th>
      <td>3</td>
      <td>[1, 1, 1, 0, 1, 0, 0, 1, 0, 0, 0, 0, 0, 1, 0, ...</td>
      <td>26.0</td>
      <td>[26.0, 26.0, 26.0, 26.0, 26.0, 26.0, 26.0, 26....</td>
      <td>[102.0, 153.0, 204.0, 255.0, 306.0, 357.0, 408...</td>
      <td>10</td>
      <td>5000</td>
      <td>50</td>
      <td>0.1</td>
      <td>&lt;mlrose_ky.opt_probs.discrete_opt.DiscreteOpt ...</td>
      <td>0.295165</td>
    </tr>
  </tbody>
</table>
</div>


