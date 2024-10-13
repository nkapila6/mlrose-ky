# mlrose_ky Generator and Runner Usage Examples - Andrew Rollings
*Modified by Kyle Nakamura*

## Overview

These examples will not solve assignment 2 for you, but they will give you some idea on how to use the problem generator and runner classes.

Hopefully this will result in slightly fewer "How do I \<insert basic usage here\>" questions every semester...

Also, and in case it hasn't been made clear enough by the TAs, using any of the visualizations from this tutorial for your report is a bad idea for two reasons: 
1. It provides nothing useful as far as the assignment goes, and 
2. The TAs will undoubtedly frown upon it.

Visualization is part of the analysis and, for the most part, you're supposed to do that by yourself. Just including
images of the before/after state of a problem really isn't useful in terms of what you're supposed to be analyzing.

## Import Libraries


```python
%pip install chess IPython
```

    Requirement already satisfied: chess in /Users/kylenakamura/anaconda3/envs/machine-learning/lib/python3.11/site-packages (1.10.0)
    Requirement already satisfied: IPython in /Users/kylenakamura/anaconda3/envs/machine-learning/lib/python3.11/site-packages (8.25.0)
    Requirement already satisfied: decorator in /Users/kylenakamura/anaconda3/envs/machine-learning/lib/python3.11/site-packages (from IPython) (5.1.1)
    Requirement already satisfied: jedi>=0.16 in /Users/kylenakamura/anaconda3/envs/machine-learning/lib/python3.11/site-packages (from IPython) (0.18.1)
    Requirement already satisfied: matplotlib-inline in /Users/kylenakamura/anaconda3/envs/machine-learning/lib/python3.11/site-packages (from IPython) (0.1.6)
    Requirement already satisfied: prompt-toolkit<3.1.0,>=3.0.41 in /Users/kylenakamura/anaconda3/envs/machine-learning/lib/python3.11/site-packages (from IPython) (3.0.47)
    Requirement already satisfied: pygments>=2.4.0 in /Users/kylenakamura/anaconda3/envs/machine-learning/lib/python3.11/site-packages (from IPython) (2.18.0)
    Requirement already satisfied: stack-data in /Users/kylenakamura/anaconda3/envs/machine-learning/lib/python3.11/site-packages (from IPython) (0.2.0)
    Requirement already satisfied: traitlets>=5.13.0 in /Users/kylenakamura/anaconda3/envs/machine-learning/lib/python3.11/site-packages (from IPython) (5.14.3)
    Requirement already satisfied: typing-extensions>=4.6 in /Users/kylenakamura/anaconda3/envs/machine-learning/lib/python3.11/site-packages (from IPython) (4.12.2)
    Requirement already satisfied: pexpect>4.3 in /Users/kylenakamura/anaconda3/envs/machine-learning/lib/python3.11/site-packages (from IPython) (4.9.0)
    Requirement already satisfied: parso<0.9.0,>=0.8.0 in /Users/kylenakamura/anaconda3/envs/machine-learning/lib/python3.11/site-packages (from jedi>=0.16->IPython) (0.8.3)
    Requirement already satisfied: ptyprocess>=0.5 in /Users/kylenakamura/anaconda3/envs/machine-learning/lib/python3.11/site-packages (from pexpect>4.3->IPython) (0.7.0)
    Requirement already satisfied: wcwidth in /Users/kylenakamura/anaconda3/envs/machine-learning/lib/python3.11/site-packages (from prompt-toolkit<3.1.0,>=3.0.41->IPython) (0.2.13)
    Requirement already satisfied: executing in /Users/kylenakamura/anaconda3/envs/machine-learning/lib/python3.11/site-packages (from stack-data->IPython) (0.8.3)
    Requirement already satisfied: asttokens in /Users/kylenakamura/anaconda3/envs/machine-learning/lib/python3.11/site-packages (from stack-data->IPython) (2.0.5)
    Requirement already satisfied: pure-eval in /Users/kylenakamura/anaconda3/envs/machine-learning/lib/python3.11/site-packages (from stack-data->IPython) (0.2.2)
    Requirement already satisfied: six in /Users/kylenakamura/anaconda3/envs/machine-learning/lib/python3.11/site-packages (from asttokens->stack-data->IPython) (1.16.0)
    Note: you may need to restart the kernel to use updated packages.



```python
from IPython.display import HTML

import numpy as np
import logging
import networkx as nx
import matplotlib.pyplot as plt
import string

from ast import literal_eval
import chess

from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler, OneHotEncoder
from sklearn.metrics import accuracy_score

import mlrose_ky as mlrose
from mlrose_ky.generators import QueensGenerator, MaxKColorGenerator, TSPGenerator
from mlrose_ky.runners import SARunner, GARunner, NNGSRunner

# Hide warnings from libraries
logging.basicConfig(level=logging.WARNING)
```

## Example 1: Solving the 8-Queens problem using the SA algorithm

### Initializing and viewing the problem

First, we'll use the `QueensGenerator` to create an instance of the 8-Queens problem.


```python
# Generate a new 8-Queens optimization problem using a fixed seed
problem = QueensGenerator.generate(seed=123456, size=8)
```

The initial, un-optimized state can be seen below, both as a list and as a chess board.


```python
# View the initial state as a list
state = problem.get_state()
print('Initial state:', state)

# View the initial state as a chess board
board_layout = "/".join(["".join(([str(s)] if s > 0 else []) + ["Q"] + ([str((7 - s))] if s < 7 else [])) for s in state])
chess.Board(board_layout)  # You may need to "trust" this notebook for the board visualization to work
```

    Initial state: [2 3 3 2 7 0 1 1]





    
![svg](problem_examples_files/problem_examples_8_1.svg)
    



### Solving 8-Queens using a Runner (i.e., grid search)

Runners are used to execute "grid search" experiments on optimization problems.

We'll use the Simulated Annealing Runner (SARunner) to perform a grid search on the 8-Queens problem and then extract the optimal SA hyperparameters.

Here is a brief explanation of the SARunner parameters used in the example below:
- `max_attempts`: A list of maximum attempts to try improving the fitness score before terminating a run
- `temperature_list`: A list of temperatures to try when initializing the SA algorithm's decay function (e.g., `GeomDecay(init_temp=1.0)`)
- `decay_list`: A list of decay schedules to try as the SA algorithm's decay function (e.g., `GeomDecay`, `ExpDecay`, etc.)
- `iteration_list`: A list of iterations to snapshot the state of the algorithm at (only determines the rows that the Runner will output)

*Disclaimer: the values used here are just toy values picked specifically for this example. 
You will have to choose your own range of values for your experiments. 
I strongly recommend you don't just copy these, or you will find that the grading is unlikely to go the way you would like.* 


```python
# Create an SA Runner instance to solve the problem
sa = SARunner(
    problem=problem,
    experiment_name="queens_8_sa",
    seed=123456,
    output_directory=None,  # Note: specify an output directory (str) to have these results saved to disk
    max_attempts=100,
    temperature_list=[0.1, 0.5, 0.75, 1.0, 2.0, 5.0],
    decay_list=[mlrose.GeomDecay],
    iteration_list=2 ** np.arange(11),  # list of 11 integers from 2^0 to 2^11
)

# Run the SA Runner and retrieve its results
df_run_stats, df_run_curves = sa.run()
```


```python
# Calculate some simple stats about the experiment
temperatures_per_run = max(1, len(sa.temperature_list))
decays_per_run = max(1, len(sa.decay_list))
iters_per_run = len(sa.iteration_list) + 1
total_runs = temperatures_per_run * decays_per_run

print(f"The experiment executed {total_runs} runs, each with {iters_per_run} snapshots at different iterations.")
print(f"In total, the output dataframe should contain {total_runs * iters_per_run} rows.")
```

    The experiment executed 6 runs, each with 12 snapshots at different iterations.
    In total, the output dataframe should contain 72 rows.


The `df_run_stats` dataframe contains snapshots of the state of the algorithm at the iterations specified in the `iteration_list`.

Since `iterations_list` contains 11 numbers, and iteration 0 is always included in the results, each run will take up 12 rows of the dataframe.

The first 12 rows (i.e., results from the first run) are shown below:


```python
HTML(df_run_stats[["Iteration", "Fitness", "FEvals", "Time", "State"]][:iters_per_run].to_html(index=False))
```




<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th>Iteration</th>
      <th>Fitness</th>
      <th>FEvals</th>
      <th>Time</th>
      <th>State</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <td>0</td>
      <td>11.0</td>
      <td>0</td>
      <td>0.000856</td>
      <td>[1, 2, 2, 1, 0, 3, 7, 3]</td>
    </tr>
    <tr>
      <td>1</td>
      <td>9.0</td>
      <td>2</td>
      <td>0.002331</td>
      <td>[1, 2, 2, 0, 0, 3, 7, 3]</td>
    </tr>
    <tr>
      <td>2</td>
      <td>8.0</td>
      <td>4</td>
      <td>0.002864</td>
      <td>[1, 2, 2, 0, 0, 3, 7, 5]</td>
    </tr>
    <tr>
      <td>4</td>
      <td>8.0</td>
      <td>7</td>
      <td>0.003399</td>
      <td>[1, 2, 2, 5, 0, 3, 7, 5]</td>
    </tr>
    <tr>
      <td>8</td>
      <td>5.0</td>
      <td>13</td>
      <td>0.004127</td>
      <td>[1, 2, 7, 5, 0, 3, 5, 5]</td>
    </tr>
    <tr>
      <td>16</td>
      <td>4.0</td>
      <td>24</td>
      <td>0.005042</td>
      <td>[1, 2, 7, 5, 3, 0, 5, 5]</td>
    </tr>
    <tr>
      <td>32</td>
      <td>4.0</td>
      <td>47</td>
      <td>0.006737</td>
      <td>[1, 5, 7, 5, 0, 0, 3, 4]</td>
    </tr>
    <tr>
      <td>64</td>
      <td>1.0</td>
      <td>86</td>
      <td>0.009472</td>
      <td>[1, 5, 2, 6, 3, 0, 7, 4]</td>
    </tr>
    <tr>
      <td>128</td>
      <td>1.0</td>
      <td>155</td>
      <td>0.014421</td>
      <td>[1, 5, 2, 6, 3, 0, 4, 7]</td>
    </tr>
    <tr>
      <td>256</td>
      <td>1.0</td>
      <td>295</td>
      <td>0.025960</td>
      <td>[1, 7, 2, 6, 3, 5, 0, 4]</td>
    </tr>
    <tr>
      <td>512</td>
      <td>0.0</td>
      <td>461</td>
      <td>0.044061</td>
      <td>[1, 5, 0, 6, 3, 7, 2, 4]</td>
    </tr>
    <tr>
      <td>1024</td>
      <td>0.0</td>
      <td>461</td>
      <td>0.044061</td>
      <td>[1, 5, 0, 6, 3, 7, 2, 4]</td>
    </tr>
  </tbody>
</table>



Some information was intentionally excluded from the previous output. Let's now preview the entirety of the first run:


```python
HTML(df_run_stats[:iters_per_run].to_html(index=False))
```




<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th>Iteration</th>
      <th>Fitness</th>
      <th>FEvals</th>
      <th>Time</th>
      <th>State</th>
      <th>schedule_type</th>
      <th>schedule_init_temp</th>
      <th>schedule_decay</th>
      <th>schedule_min_temp</th>
      <th>schedule_current_value</th>
      <th>Temperature</th>
      <th>max_iters</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <td>0</td>
      <td>11.0</td>
      <td>0</td>
      <td>0.000856</td>
      <td>[1, 2, 2, 1, 0, 3, 7, 3]</td>
      <td>geometric</td>
      <td>0.1</td>
      <td>0.99</td>
      <td>0.001</td>
      <td>0.099999</td>
      <td>0.1</td>
      <td>1024</td>
    </tr>
    <tr>
      <td>1</td>
      <td>9.0</td>
      <td>2</td>
      <td>0.002331</td>
      <td>[1, 2, 2, 0, 0, 3, 7, 3]</td>
      <td>geometric</td>
      <td>0.1</td>
      <td>0.99</td>
      <td>0.001</td>
      <td>0.099998</td>
      <td>0.1</td>
      <td>1024</td>
    </tr>
    <tr>
      <td>2</td>
      <td>8.0</td>
      <td>4</td>
      <td>0.002864</td>
      <td>[1, 2, 2, 0, 0, 3, 7, 5]</td>
      <td>geometric</td>
      <td>0.1</td>
      <td>0.99</td>
      <td>0.001</td>
      <td>0.099997</td>
      <td>0.1</td>
      <td>1024</td>
    </tr>
    <tr>
      <td>4</td>
      <td>8.0</td>
      <td>7</td>
      <td>0.003399</td>
      <td>[1, 2, 2, 5, 0, 3, 7, 5]</td>
      <td>geometric</td>
      <td>0.1</td>
      <td>0.99</td>
      <td>0.001</td>
      <td>0.099997</td>
      <td>0.1</td>
      <td>1024</td>
    </tr>
    <tr>
      <td>8</td>
      <td>5.0</td>
      <td>13</td>
      <td>0.004127</td>
      <td>[1, 2, 7, 5, 0, 3, 5, 5]</td>
      <td>geometric</td>
      <td>0.1</td>
      <td>0.99</td>
      <td>0.001</td>
      <td>0.099996</td>
      <td>0.1</td>
      <td>1024</td>
    </tr>
    <tr>
      <td>16</td>
      <td>4.0</td>
      <td>24</td>
      <td>0.005042</td>
      <td>[1, 2, 7, 5, 3, 0, 5, 5]</td>
      <td>geometric</td>
      <td>0.1</td>
      <td>0.99</td>
      <td>0.001</td>
      <td>0.099995</td>
      <td>0.1</td>
      <td>1024</td>
    </tr>
    <tr>
      <td>32</td>
      <td>4.0</td>
      <td>47</td>
      <td>0.006737</td>
      <td>[1, 5, 7, 5, 0, 0, 3, 4]</td>
      <td>geometric</td>
      <td>0.1</td>
      <td>0.99</td>
      <td>0.001</td>
      <td>0.099993</td>
      <td>0.1</td>
      <td>1024</td>
    </tr>
    <tr>
      <td>64</td>
      <td>1.0</td>
      <td>86</td>
      <td>0.009472</td>
      <td>[1, 5, 2, 6, 3, 0, 7, 4]</td>
      <td>geometric</td>
      <td>0.1</td>
      <td>0.99</td>
      <td>0.001</td>
      <td>0.099990</td>
      <td>0.1</td>
      <td>1024</td>
    </tr>
    <tr>
      <td>128</td>
      <td>1.0</td>
      <td>155</td>
      <td>0.014421</td>
      <td>[1, 5, 2, 6, 3, 0, 4, 7]</td>
      <td>geometric</td>
      <td>0.1</td>
      <td>0.99</td>
      <td>0.001</td>
      <td>0.099986</td>
      <td>0.1</td>
      <td>1024</td>
    </tr>
    <tr>
      <td>256</td>
      <td>1.0</td>
      <td>295</td>
      <td>0.025960</td>
      <td>[1, 7, 2, 6, 3, 5, 0, 4]</td>
      <td>geometric</td>
      <td>0.1</td>
      <td>0.99</td>
      <td>0.001</td>
      <td>0.099974</td>
      <td>0.1</td>
      <td>1024</td>
    </tr>
    <tr>
      <td>512</td>
      <td>0.0</td>
      <td>461</td>
      <td>0.044061</td>
      <td>[1, 5, 0, 6, 3, 7, 2, 4]</td>
      <td>geometric</td>
      <td>0.1</td>
      <td>0.99</td>
      <td>0.001</td>
      <td>0.099956</td>
      <td>0.1</td>
      <td>1024</td>
    </tr>
    <tr>
      <td>1024</td>
      <td>0.0</td>
      <td>461</td>
      <td>0.044061</td>
      <td>[1, 5, 0, 6, 3, 7, 2, 4]</td>
      <td>geometric</td>
      <td>0.1</td>
      <td>0.99</td>
      <td>0.001</td>
      <td>0.099956</td>
      <td>0.1</td>
      <td>1024</td>
    </tr>
  </tbody>
</table>



**What does all this information tell us about the first run of our experiment?**

1. `Iteration` shows the index of the snapshot, with 12 snapshots per run in this example.
2. `Fitness` shows the fitness score of the state at the corresponding iteration, where 0.0 is optimal for minimization problems.
3. `FEvals` shows the number of fitness function evaluations performed by the algorithm at the corresponding iteration.
4. `Time` shows the time elapsed (calculated using `time.perf_counter()`) up to the corresponding iteration.
5. `State` shows the state of the algorithm at the corresponding iteration (see `mlrose_ky.fitness.queens` for more details).
6. `Temperature` shows the decay function (and its parameters) that was used for this run. We will have 6 unique values in this column, one for each run.
7. `max_iters` shows the maximum number of iterations allowed for the algorithm to run. It defaults to `max(iteration_list)` for all Runners, which is 1024 in this case.

To pick out the most performant run from the dataframe, we need to find the row with the best fitness.
Since Queens is a minimization problem, we're looking for the row with minimal fitness (i.e., zero).

It's likely that multiple runs will achieve the same fitness, so we need to find the run that achieved the best `Fitness` in the fewest `FEvals` (*Note: we could make this selection using `Iterations` or `Time` if we so desired.*)


```python
best_fitness = df_run_stats["Fitness"].min()  # Should be 0.0 in this case

# Get all runs with the best fitness value
best_runs = df_run_stats[df_run_stats["Fitness"] == best_fitness]
best_runs
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
      <th>Iteration</th>
      <th>Fitness</th>
      <th>FEvals</th>
      <th>Time</th>
      <th>State</th>
      <th>schedule_type</th>
      <th>schedule_init_temp</th>
      <th>schedule_decay</th>
      <th>schedule_min_temp</th>
      <th>schedule_current_value</th>
      <th>Temperature</th>
      <th>max_iters</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>10</th>
      <td>512</td>
      <td>0.0</td>
      <td>461</td>
      <td>0.044061</td>
      <td>[1, 5, 0, 6, 3, 7, 2, 4]</td>
      <td>geometric</td>
      <td>0.1</td>
      <td>0.99</td>
      <td>0.001</td>
      <td>0.099956</td>
      <td>0.1</td>
      <td>1024</td>
    </tr>
    <tr>
      <th>11</th>
      <td>1024</td>
      <td>0.0</td>
      <td>461</td>
      <td>0.044061</td>
      <td>[1, 5, 0, 6, 3, 7, 2, 4]</td>
      <td>geometric</td>
      <td>0.1</td>
      <td>0.99</td>
      <td>0.001</td>
      <td>0.099956</td>
      <td>0.1</td>
      <td>1024</td>
    </tr>
    <tr>
      <th>22</th>
      <td>512</td>
      <td>0.0</td>
      <td>461</td>
      <td>0.040817</td>
      <td>[1, 5, 0, 6, 3, 7, 2, 4]</td>
      <td>geometric</td>
      <td>0.5</td>
      <td>0.99</td>
      <td>0.001</td>
      <td>0.499795</td>
      <td>0.5</td>
      <td>1024</td>
    </tr>
    <tr>
      <th>23</th>
      <td>1024</td>
      <td>0.0</td>
      <td>461</td>
      <td>0.040817</td>
      <td>[1, 5, 0, 6, 3, 7, 2, 4]</td>
      <td>geometric</td>
      <td>0.5</td>
      <td>0.99</td>
      <td>0.001</td>
      <td>0.499795</td>
      <td>0.5</td>
      <td>1024</td>
    </tr>
    <tr>
      <th>58</th>
      <td>512</td>
      <td>0.0</td>
      <td>427</td>
      <td>0.044455</td>
      <td>[7, 1, 3, 0, 6, 4, 2, 5]</td>
      <td>geometric</td>
      <td>2.0</td>
      <td>0.99</td>
      <td>0.001</td>
      <td>1.999107</td>
      <td>2.0</td>
      <td>1024</td>
    </tr>
    <tr>
      <th>59</th>
      <td>1024</td>
      <td>0.0</td>
      <td>427</td>
      <td>0.044455</td>
      <td>[7, 1, 3, 0, 6, 4, 2, 5]</td>
      <td>geometric</td>
      <td>2.0</td>
      <td>0.99</td>
      <td>0.001</td>
      <td>1.999107</td>
      <td>2.0</td>
      <td>1024</td>
    </tr>
    <tr>
      <th>70</th>
      <td>512</td>
      <td>0.0</td>
      <td>583</td>
      <td>0.061001</td>
      <td>[6, 0, 2, 7, 5, 3, 1, 4]</td>
      <td>geometric</td>
      <td>5.0</td>
      <td>0.99</td>
      <td>0.001</td>
      <td>4.996936</td>
      <td>5.0</td>
      <td>1024</td>
    </tr>
    <tr>
      <th>71</th>
      <td>1024</td>
      <td>0.0</td>
      <td>583</td>
      <td>0.061001</td>
      <td>[6, 0, 2, 7, 5, 3, 1, 4]</td>
      <td>geometric</td>
      <td>5.0</td>
      <td>0.99</td>
      <td>0.001</td>
      <td>4.996936</td>
      <td>5.0</td>
      <td>1024</td>
    </tr>
  </tbody>
</table>
</div>



This gives us our candidates for the best run. 

The best run will be the one that achieved the best fitness in the fewest evaluations.


```python
minimum_evaluations = best_runs["FEvals"].min()  # Should be 461 in this case

# Extract the best run with the minimum number of evaluations
best_run = best_runs[best_runs["FEvals"] == minimum_evaluations]
```

The best run using these criteria is as follows:


```python
print(best_run.iloc[0])
```

    Iteration                                      512
    Fitness                                        0.0
    FEvals                                         427
    Time                                      0.044455
    State                     [7, 1, 3, 0, 6, 4, 2, 5]
    schedule_type                            geometric
    schedule_init_temp                             2.0
    schedule_decay                                0.99
    schedule_min_temp                            0.001
    schedule_current_value                    1.999107
    Temperature                                    2.0
    max_iters                                     1024
    Name: 58, dtype: object


Which has the following identifying state information:


```python
best_temperature_param = best_run["Temperature"].iloc[0].init_temp
best_temperature_param
```




    2.0



To map this result back to the original output of the Runner, we are looking for all rows in `df_run_stats` where the temperature is equal to 2.


```python
run_stats_best_run = df_run_stats[df_run_stats["schedule_init_temp"] == best_temperature_param]
run_stats_best_run[["Iteration", "Fitness", "FEvals", "Time", "State"]]
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
      <th>Iteration</th>
      <th>Fitness</th>
      <th>FEvals</th>
      <th>Time</th>
      <th>State</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>48</th>
      <td>0</td>
      <td>11.0</td>
      <td>0</td>
      <td>0.000136</td>
      <td>[1, 2, 2, 1, 0, 3, 7, 3]</td>
    </tr>
    <tr>
      <th>49</th>
      <td>1</td>
      <td>9.0</td>
      <td>2</td>
      <td>0.001460</td>
      <td>[1, 2, 2, 0, 0, 3, 7, 3]</td>
    </tr>
    <tr>
      <th>50</th>
      <td>2</td>
      <td>8.0</td>
      <td>4</td>
      <td>0.002754</td>
      <td>[1, 2, 2, 0, 0, 3, 7, 5]</td>
    </tr>
    <tr>
      <th>51</th>
      <td>4</td>
      <td>8.0</td>
      <td>7</td>
      <td>0.004162</td>
      <td>[1, 2, 2, 5, 0, 3, 7, 5]</td>
    </tr>
    <tr>
      <th>52</th>
      <td>8</td>
      <td>7.0</td>
      <td>14</td>
      <td>0.005876</td>
      <td>[1, 2, 2, 5, 0, 3, 5, 5]</td>
    </tr>
    <tr>
      <th>53</th>
      <td>16</td>
      <td>6.0</td>
      <td>27</td>
      <td>0.007869</td>
      <td>[3, 2, 3, 5, 0, 1, 5, 5]</td>
    </tr>
    <tr>
      <th>54</th>
      <td>32</td>
      <td>4.0</td>
      <td>57</td>
      <td>0.011020</td>
      <td>[3, 5, 6, 5, 5, 0, 4, 7]</td>
    </tr>
    <tr>
      <th>55</th>
      <td>64</td>
      <td>5.0</td>
      <td>114</td>
      <td>0.015731</td>
      <td>[2, 0, 3, 6, 1, 2, 1, 7]</td>
    </tr>
    <tr>
      <th>56</th>
      <td>128</td>
      <td>3.0</td>
      <td>205</td>
      <td>0.022950</td>
      <td>[2, 0, 6, 3, 5, 0, 4, 3]</td>
    </tr>
    <tr>
      <th>57</th>
      <td>256</td>
      <td>2.0</td>
      <td>358</td>
      <td>0.036341</td>
      <td>[7, 1, 3, 6, 6, 4, 0, 5]</td>
    </tr>
    <tr>
      <th>58</th>
      <td>512</td>
      <td>0.0</td>
      <td>427</td>
      <td>0.044455</td>
      <td>[7, 1, 3, 0, 6, 4, 2, 5]</td>
    </tr>
    <tr>
      <th>59</th>
      <td>1024</td>
      <td>0.0</td>
      <td>427</td>
      <td>0.044455</td>
      <td>[7, 1, 3, 0, 6, 4, 2, 5]</td>
    </tr>
  </tbody>
</table>
</div>



And the best state associated with this is:


```python
best_state = run_stats_best_run[["schedule_current_value", "schedule_init_temp", "schedule_min_temp"]].tail(1)
best_state
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
      <th>schedule_current_value</th>
      <th>schedule_init_temp</th>
      <th>schedule_min_temp</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>59</th>
      <td>1.999107</td>
      <td>2.0</td>
      <td>0.001</td>
    </tr>
  </tbody>
</table>
</div>



The final state is as follows:


```python
state = literal_eval(run_stats_best_run["State"].tail(1).values[0])
print(state)

board_layout = "/".join(["".join(([str(s)] if s > 0 else []) + ["Q"] + ([str((7 - s))] if s < 7 else [])) for s in state])
board = chess.Board(board_layout)
board
```

    [7, 1, 3, 0, 6, 4, 2, 5]





    
![svg](problem_examples_files/problem_examples_29_1.svg)
    



### Example 2: Generating and Running Max K Color using the GA algorithm


```python
# Generate a new Max K problem using a fixed seed.
problem = MaxKColorGenerator().generate(seed=123456, number_of_nodes=10, max_connections_per_node=3, max_colors=3)
```

The input graph generated for the problem looks like this:


```python
nx.draw(problem.source_graph, pos=nx.spring_layout(problem.source_graph, seed=3))
plt.show()
```


    
![png](problem_examples_files/problem_examples_33_0.png)
    



```python
# create a runner class and solve the problem
ga = GARunner(
    problem=problem,
    experiment_name="max_k_ga",
    output_directory=None,  # note: specify an output directory to have results saved to disk
    seed=123456,
    iteration_list=2 ** np.arange(11),
    population_sizes=[10, 20, 50],
    mutation_rates=[0.1, 0.2, 0.5],
)

# the two data frames will contain the results
df_run_stats, df_run_curves = ga.run()
```

The preceding code will run the `GA` algorithm nine times for at most 1024 iterations per run.
Each run is a permutation from the list of `population_sizes` and `mutation_rates`.

Note that the initial state parameters here are just toy values picked specifically
for this example. You will have to choose your own range of values for your
assignment. I strongly recommend you don't just copy these, or you will find
that the grading is unlikely to go the way you would like.

Really. I mean it... A mutation rate of 0.5 is little better than a pure random search.

The output in the `df_run_stats` dataframe contains snapshots of the state of the algorithm at the iterations
specified in the `iteration_list` passed into the runner class.

The first row (corresponding to the first run of this algorithm) are as follows:


```python
df_run_stats[["Iteration", "Fitness", "FEvals", "Time", "State"]][0:1]
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
      <th>Iteration</th>
      <th>Fitness</th>
      <th>FEvals</th>
      <th>Time</th>
      <th>State</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>0</td>
      <td>3.0</td>
      <td>10</td>
      <td>0.000702</td>
      <td>[1, 2, 2, 1, 0, 0, 0, 0, 2, 2]</td>
    </tr>
  </tbody>
</table>
</div>



The state information is excluded from the previous output.

A sample of this is below:


```python
state_sample = df_run_stats[["Population Size", "Mutation Rate"]][:1]
state_sample
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
      <th>Population Size</th>
      <th>Mutation Rate</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>10</td>
      <td>0.1</td>
    </tr>
  </tbody>
</table>
</div>



So, to pick out the most performant run from the dataframe, you need to find the row with the best fitness.
As Max-K-Color is a minimization problem, you'd pick the row with the minimum fitness.

However, I'm going to look in the `run_curves` (which stores minimal basic information every iteration) to
find out which input state achieved the best fitness in the fewest fitness evaluations.


```python
best_fitness = df_run_curves["Fitness"].min()
best_runs = df_run_curves[df_run_curves["Fitness"] == best_fitness]
best_runs
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
      <th>Iteration</th>
      <th>Time</th>
      <th>Fitness</th>
      <th>FEvals</th>
      <th>Population Size</th>
      <th>Mutation Rate</th>
      <th>max_iters</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>4</th>
      <td>4</td>
      <td>0.003967</td>
      <td>0.0</td>
      <td>57.0</td>
      <td>10</td>
      <td>0.1</td>
      <td>1024</td>
    </tr>
    <tr>
      <th>9</th>
      <td>4</td>
      <td>0.003967</td>
      <td>0.0</td>
      <td>57.0</td>
      <td>10</td>
      <td>0.2</td>
      <td>1024</td>
    </tr>
    <tr>
      <th>18</th>
      <td>8</td>
      <td>0.001445</td>
      <td>0.0</td>
      <td>101.0</td>
      <td>10</td>
      <td>0.5</td>
      <td>1024</td>
    </tr>
    <tr>
      <th>24</th>
      <td>5</td>
      <td>0.000106</td>
      <td>0.0</td>
      <td>127.0</td>
      <td>20</td>
      <td>0.1</td>
      <td>1024</td>
    </tr>
    <tr>
      <th>27</th>
      <td>2</td>
      <td>0.003304</td>
      <td>0.0</td>
      <td>64.0</td>
      <td>20</td>
      <td>0.2</td>
      <td>1024</td>
    </tr>
    <tr>
      <th>31</th>
      <td>3</td>
      <td>0.003797</td>
      <td>0.0</td>
      <td>85.0</td>
      <td>20</td>
      <td>0.5</td>
      <td>1024</td>
    </tr>
    <tr>
      <th>41</th>
      <td>9</td>
      <td>0.001615</td>
      <td>0.0</td>
      <td>511.0</td>
      <td>50</td>
      <td>0.1</td>
      <td>1024</td>
    </tr>
    <tr>
      <th>45</th>
      <td>3</td>
      <td>0.003797</td>
      <td>0.0</td>
      <td>205.0</td>
      <td>50</td>
      <td>0.2</td>
      <td>1024</td>
    </tr>
    <tr>
      <th>50</th>
      <td>4</td>
      <td>0.003967</td>
      <td>0.0</td>
      <td>256.0</td>
      <td>50</td>
      <td>0.5</td>
      <td>1024</td>
    </tr>
  </tbody>
</table>
</div>



This gives us nine candidates for the best run. We are going to pick the one with
that reached the best fitness value in the fewest number of evaluations.

(We could also have chosen to use `Iterations` as our criteria.)


```python
minimum_evaluations = best_runs["FEvals"].min()

best_run = best_runs[best_runs["FEvals"] == minimum_evaluations]
```

The best runs using these criteria is as follows:


```python
best_run
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
      <th>Iteration</th>
      <th>Time</th>
      <th>Fitness</th>
      <th>FEvals</th>
      <th>Population Size</th>
      <th>Mutation Rate</th>
      <th>max_iters</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>4</th>
      <td>4</td>
      <td>0.003967</td>
      <td>0.0</td>
      <td>57.0</td>
      <td>10</td>
      <td>0.1</td>
      <td>1024</td>
    </tr>
    <tr>
      <th>9</th>
      <td>4</td>
      <td>0.003967</td>
      <td>0.0</td>
      <td>57.0</td>
      <td>10</td>
      <td>0.2</td>
      <td>1024</td>
    </tr>
  </tbody>
</table>
</div>



We will arbitrarily pick the first row for this example,
which has the following identifying state information:


```python
best_mr = best_run["Mutation Rate"].iloc()[0]
best_pop_size = best_run["Population Size"].iloc()[0]
print(f"Best Mutation Rate: {best_mr}, best Population Size: {best_pop_size}")
```

    Best Mutation Rate: 0.1, best Population Size: 10


To map this back to the `run_stats` we look at the configuration data included in
the curve data. The curve data includes at least the minimum identifying information
to determine which run each row came from.

In this case, the values we are looking for are the `Mutation Rate` and `Population Size`.

So, we are looking for all rows in `df_run_stats` where the mutation rate and population size are equal to our best values.


```python
run_stats_best_run = df_run_stats[(df_run_stats["Mutation Rate"] == best_mr) & (df_run_stats["Population Size"] == best_pop_size)]
run_stats_best_run[["Iteration", "Fitness", "FEvals", "Time"]]
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
      <th>Iteration</th>
      <th>Fitness</th>
      <th>FEvals</th>
      <th>Time</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>0</td>
      <td>3.0</td>
      <td>10</td>
      <td>0.000702</td>
    </tr>
    <tr>
      <th>1</th>
      <td>1</td>
      <td>3.0</td>
      <td>21</td>
      <td>0.002124</td>
    </tr>
    <tr>
      <th>2</th>
      <td>2</td>
      <td>2.0</td>
      <td>33</td>
      <td>0.003304</td>
    </tr>
    <tr>
      <th>3</th>
      <td>4</td>
      <td>0.0</td>
      <td>57</td>
      <td>0.003967</td>
    </tr>
    <tr>
      <th>4</th>
      <td>8</td>
      <td>0.0</td>
      <td>57</td>
      <td>0.003967</td>
    </tr>
    <tr>
      <th>5</th>
      <td>16</td>
      <td>0.0</td>
      <td>57</td>
      <td>0.003967</td>
    </tr>
    <tr>
      <th>6</th>
      <td>32</td>
      <td>0.0</td>
      <td>57</td>
      <td>0.003967</td>
    </tr>
    <tr>
      <th>7</th>
      <td>64</td>
      <td>0.0</td>
      <td>57</td>
      <td>0.003967</td>
    </tr>
    <tr>
      <th>8</th>
      <td>128</td>
      <td>0.0</td>
      <td>57</td>
      <td>0.003967</td>
    </tr>
    <tr>
      <th>9</th>
      <td>256</td>
      <td>0.0</td>
      <td>57</td>
      <td>0.003967</td>
    </tr>
    <tr>
      <th>10</th>
      <td>512</td>
      <td>0.0</td>
      <td>57</td>
      <td>0.003967</td>
    </tr>
    <tr>
      <th>11</th>
      <td>1024</td>
      <td>0.0</td>
      <td>57</td>
      <td>0.003967</td>
    </tr>
  </tbody>
</table>
</div>



And the best state associated with this is:


```python
best_state = run_stats_best_run[["State"]].tail(1)
best_state
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
      <th>State</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>11</th>
      <td>[0, 1, 2, 0, 0, 2, 0, 1, 2, 1]</td>
    </tr>
  </tbody>
</table>
</div>



For the following node ordering:


```python
print([n for n in problem.source_graph.nodes])
```

    [0, 2, 8, 1, 3, 4, 6, 7, 9, 5]


Reordering the state by ascending node number gives the following:


```python
color_indexes = literal_eval(run_stats_best_run["State"].tail(1).values[0])
ordered_state = [color_indexes[n] for n in problem.source_graph.nodes]
print(ordered_state)
```

    [0, 2, 2, 1, 0, 0, 0, 1, 1, 2]


Which results in a graph looking like this:


```python
colors = ["lightcoral", "lightgreen", "yellow"]
node_color_map = [colors[s] for s in ordered_state]

nx.draw(problem.source_graph, pos=nx.spring_layout(problem.source_graph, seed=3), with_labels=True, node_color=node_color_map)
plt.show()
```


    
![png](problem_examples_files/problem_examples_56_0.png)
    


### Example 3: Generating and Running TSP using the GA algorithm


```python
# Generate a new TSP problem using a fixed seed.
problem = TSPGenerator().generate(seed=123456, number_of_cities=20)
```

The input graph generated for the problem looks like this:


```python
fig, ax = plt.subplots(1)  # Prepare 2 plots
ax.set_yticklabels([])
ax.set_xticklabels([])
for i, (x, y) in enumerate(problem.coords):
    ax.scatter(x, y, s=200, c="cornflowerblue")  # plot A
node_labels = {k: str(v) for k, v in enumerate(string.ascii_uppercase) if k < len(problem.source_graph.nodes)}
for i in node_labels.keys():
    x, y = problem.coords[i]
plt.text(
    x,
    y,
    node_labels[i],
    ha="center",
    va="center",
    c="white",
    fontweight="bold",
    bbox=dict(boxstyle=f"circle,pad=0.15", fc="cornflowerblue"),
)

plt.tight_layout()
plt.show()
```


    
![png](problem_examples_files/problem_examples_60_0.png)
    



```python
# create a runner class and solve the problem
ga = GARunner(
    problem=problem,
    experiment_name="tsp_ga",
    output_directory=None,  # note: specify an output directory to have results saved to disk
    seed=123456,
    iteration_list=2 ** np.arange(11),
    population_sizes=[10, 20],
    mutation_rates=[0.1, 0.25, 0.5],
)

# the two data frames will contain the results
df_run_stats, df_run_curves = ga.run()
```

The preceding code will run the `GA` algorithm nine times for at most 1024 iterations per run.
Each run is a permutation from the list of `population_sizes` and `mutation_rates`.

Note that the initial state parameters here are just toy values picked specifically
for this example. You will have to choose your own range of values for your
assignment. I strongly recommend you don't just copy these, or you will find
that the grading is unlikely to go the way you would like.

Really. I mean it... A mutation rate of 0.5 is little better than a pure random search.

The output in the `df_run_stats` dataframe contains snapshots of the state of the algorithm at the iterations
specified in the `iteration_list` passed into the runner class.

The first row (corresponding to the first run of this algorithm) are as follows:


```python
df_run_stats[["Iteration", "Fitness", "FEvals", "Time", "State"]][:1]
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
      <th>Iteration</th>
      <th>Fitness</th>
      <th>FEvals</th>
      <th>Time</th>
      <th>State</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>0</td>
      <td>2722.031402</td>
      <td>10</td>
      <td>0.00062</td>
      <td>[19, 13, 12, 9, 5, 6, 2, 3, 18, 8, 4, 7, 0, 14...</td>
    </tr>
  </tbody>
</table>
</div>



The state information is excluded from the previous output.

A sample of this is below:


```python
state_sample = df_run_stats[["Population Size", "Mutation Rate"]][:1]
state_sample
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
      <th>Population Size</th>
      <th>Mutation Rate</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>10</td>
      <td>0.1</td>
    </tr>
  </tbody>
</table>
</div>



So, to pick out the most performant run from the dataframe, you need to find the row with the best fitness.
As TSP is a minimization problem, you'd pick the row with the minimum fitness.

However, I'm going to look in the `run_curves` (which stores minimal basic information every iteration) to
find out which input state achieved the best fitness in the fewest fitness evaluations.


```python
best_fitness = df_run_curves["Fitness"].min()
best_runs = df_run_curves[df_run_curves["Fitness"] == best_fitness]
best_runs[:10]
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
      <th>Iteration</th>
      <th>Time</th>
      <th>Fitness</th>
      <th>FEvals</th>
      <th>Population Size</th>
      <th>Mutation Rate</th>
      <th>max_iters</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>3709</th>
      <td>707</td>
      <td>0.296142</td>
      <td>941.582778</td>
      <td>14903.0</td>
      <td>20</td>
      <td>0.1</td>
      <td>1024</td>
    </tr>
    <tr>
      <th>3710</th>
      <td>708</td>
      <td>0.296608</td>
      <td>941.582778</td>
      <td>14924.0</td>
      <td>20</td>
      <td>0.1</td>
      <td>1024</td>
    </tr>
    <tr>
      <th>3711</th>
      <td>709</td>
      <td>0.297102</td>
      <td>941.582778</td>
      <td>14945.0</td>
      <td>20</td>
      <td>0.1</td>
      <td>1024</td>
    </tr>
    <tr>
      <th>3712</th>
      <td>710</td>
      <td>0.297565</td>
      <td>941.582778</td>
      <td>14966.0</td>
      <td>20</td>
      <td>0.1</td>
      <td>1024</td>
    </tr>
    <tr>
      <th>3713</th>
      <td>711</td>
      <td>0.298021</td>
      <td>941.582778</td>
      <td>14987.0</td>
      <td>20</td>
      <td>0.1</td>
      <td>1024</td>
    </tr>
    <tr>
      <th>3714</th>
      <td>712</td>
      <td>0.298481</td>
      <td>941.582778</td>
      <td>15008.0</td>
      <td>20</td>
      <td>0.1</td>
      <td>1024</td>
    </tr>
    <tr>
      <th>3715</th>
      <td>713</td>
      <td>0.298940</td>
      <td>941.582778</td>
      <td>15029.0</td>
      <td>20</td>
      <td>0.1</td>
      <td>1024</td>
    </tr>
    <tr>
      <th>3716</th>
      <td>714</td>
      <td>0.299380</td>
      <td>941.582778</td>
      <td>15050.0</td>
      <td>20</td>
      <td>0.1</td>
      <td>1024</td>
    </tr>
    <tr>
      <th>3717</th>
      <td>715</td>
      <td>0.299820</td>
      <td>941.582778</td>
      <td>15071.0</td>
      <td>20</td>
      <td>0.1</td>
      <td>1024</td>
    </tr>
    <tr>
      <th>3718</th>
      <td>716</td>
      <td>0.300268</td>
      <td>941.582778</td>
      <td>15092.0</td>
      <td>20</td>
      <td>0.1</td>
      <td>1024</td>
    </tr>
  </tbody>
</table>
</div>



This gives us nine candidates for the best run. We are going to pick the one with
that reached the best fitness value in the fewest number of evaluations.

(We could also have chosen to use `Iterations` as our criteria.)


```python
minimum_evaluations = best_runs["FEvals"].min()
best_run = best_runs[best_runs["FEvals"] == minimum_evaluations]
```

The best runs using these criteria is as follows:


```python
best_run
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
      <th>Iteration</th>
      <th>Time</th>
      <th>Fitness</th>
      <th>FEvals</th>
      <th>Population Size</th>
      <th>Mutation Rate</th>
      <th>max_iters</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>3709</th>
      <td>707</td>
      <td>0.296142</td>
      <td>941.582778</td>
      <td>14903.0</td>
      <td>20</td>
      <td>0.1</td>
      <td>1024</td>
    </tr>
  </tbody>
</table>
</div>



This has the following identifying state information:


```python
best_mr = best_run["Mutation Rate"].iloc()[0]
best_pop_size = best_run["Population Size"].iloc()[0]
print(f"Best Mutation Rate: {best_mr}, best Population Size: {best_pop_size}")
```

    Best Mutation Rate: 0.1, best Population Size: 20


To map this back to the `run_stats` we look at the configuration data included in
the curve data. The curve data includes at least the minimum identifying information
to determine which run each row came from.

In this case, the values we are looking for are the `Mutation Rate` and `Population Size`.

So, we are looking for all rows in `df_run_stats` where the mutation rate and population size are equal to our best values.


```python
run_stats_best_run = df_run_stats[(df_run_stats["Mutation Rate"] == best_mr) & (df_run_stats["Population Size"] == best_pop_size)]
run_stats_best_run[["Iteration", "Fitness", "FEvals", "Time"]]
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
      <th>Iteration</th>
      <th>Fitness</th>
      <th>FEvals</th>
      <th>Time</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>36</th>
      <td>0</td>
      <td>2722.031402</td>
      <td>20</td>
      <td>0.000232</td>
    </tr>
    <tr>
      <th>37</th>
      <td>1</td>
      <td>2141.868537</td>
      <td>42</td>
      <td>0.002583</td>
    </tr>
    <tr>
      <th>38</th>
      <td>2</td>
      <td>2141.868537</td>
      <td>63</td>
      <td>0.004905</td>
    </tr>
    <tr>
      <th>39</th>
      <td>4</td>
      <td>2141.868537</td>
      <td>105</td>
      <td>0.007919</td>
    </tr>
    <tr>
      <th>40</th>
      <td>8</td>
      <td>2044.319536</td>
      <td>190</td>
      <td>0.012401</td>
    </tr>
    <tr>
      <th>41</th>
      <td>16</td>
      <td>1807.055513</td>
      <td>360</td>
      <td>0.019740</td>
    </tr>
    <tr>
      <th>42</th>
      <td>32</td>
      <td>1785.714484</td>
      <td>697</td>
      <td>0.032535</td>
    </tr>
    <tr>
      <th>43</th>
      <td>64</td>
      <td>1479.922718</td>
      <td>1376</td>
      <td>0.059770</td>
    </tr>
    <tr>
      <th>44</th>
      <td>128</td>
      <td>1151.614761</td>
      <td>2731</td>
      <td>0.114027</td>
    </tr>
    <tr>
      <th>45</th>
      <td>256</td>
      <td>1081.456468</td>
      <td>5421</td>
      <td>0.211686</td>
    </tr>
    <tr>
      <th>46</th>
      <td>512</td>
      <td>970.009048</td>
      <td>10803</td>
      <td>0.410439</td>
    </tr>
    <tr>
      <th>47</th>
      <td>1024</td>
      <td>941.582778</td>
      <td>21560</td>
      <td>0.831667</td>
    </tr>
  </tbody>
</table>
</div>



And the best state associated with this is:


```python
best_state = run_stats_best_run[["State"]].tail(1)
best_state
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
      <th>State</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>47</th>
      <td>[9, 0, 19, 17, 14, 15, 16, 8, 12, 5, 1, 6, 18,...</td>
    </tr>
  </tbody>
</table>
</div>



Which results in a graph looking like this:


```python
ordered_state = literal_eval(run_stats_best_run["State"].tail(1).values[0])
edge_labels = {(ordered_state[i], ordered_state[(i + 1) % len(ordered_state)]): f"{str(i + 1)}➜" for i in range(len(ordered_state))}
```


```python
fig, ax = plt.subplots(1)  # Prepare 2 plots
ax.set_yticklabels([])
ax.set_xticklabels([])
for i, (x, y) in enumerate(problem.coords):
    ax.scatter(x, y, s=1, c="green" if i == 5 else "cornflowerblue")  # plot A

for i in range(len(ordered_state)):
    start_node = ordered_state[i]
end_node = ordered_state[(i + 1) % len(ordered_state)]
start_pos = problem.coords[start_node]
end_pos = problem.coords[end_node]
ax.annotate(
    "",
    xy=start_pos,
    xycoords="data",
    xytext=end_pos,
    textcoords="data",
    c="red",
    arrowprops=dict(arrowstyle="->", ec="red", connectionstyle="arc3"),
)
node_labels = {k: str(v) for k, v in enumerate(string.ascii_uppercase) if k < len(problem.source_graph.nodes)}

for i in node_labels.keys():
    x, y = problem.coords[i]
plt.text(
    x,
    y,
    node_labels[i],
    ha="center",
    va="center",
    c="white",
    fontweight="bold",
    bbox=dict(boxstyle=f"circle,pad=0.15", fc="green" if i == ordered_state[0] else "cornflowerblue"),
)

plt.tight_layout()
plt.show()
```


    
![png](problem_examples_files/problem_examples_80_0.png)
    


And, to verify that the route is correct (or at least, the shortest one found):


```python
all_edge_lengths = {(x, y): d for x, y, d in problem.distances}
all_edge_lengths.update({(y, x): d for x, y, d in problem.distances})

route_length = sum([all_edge_lengths[k] for k in edge_labels.keys()])
print(f"route_length: ({round(route_length, 6)}) equal to best_fitness: ({round(best_fitness, 6)})")
```

    route_length: (941.582778) equal to best_fitness: (941.582778)


### Example 4: Using the NNGSRunner with the RHC algorithm


```python
# Load and Split data into training and test sets
data = load_iris()
X_train, X_test, y_train, y_test = train_test_split(
    data.data, data.target, test_size=0.3, random_state=123456
)

# Normalize feature data
scaler = MinMaxScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# One hot encode target values
one_hot = OneHotEncoder()
y_train_hot = np.asarray(one_hot.fit_transform(y_train.reshape(-1, 1)).todense())
y_test_hot = np.asarray(one_hot.transform(y_test.reshape(-1, 1)).todense())

grid_search_parameters = {
    'max_iters': [1000],  # nn params
    'learning_rate': [1e-2],  # nn params
    'activation': [mlrose.relu],  # nn params
    'restarts': [1],  # rhc params
}

nnr = NNGSRunner(x_train=X_train_scaled, y_train=y_train_hot, x_test=X_test_scaled, y_test=y_test_hot, experiment_name='nn_test_rhc',
                 algorithm=mlrose.algorithms.rhc.random_hill_climb, grid_search_parameters=grid_search_parameters,
                 iteration_list=[1, 10, 50, 100, 250, 500, 1000], hidden_layer_sizes=[[2]], clip_max=5, n_jobs=5, seed=123456)

run_stats_df, curves_df, cv_results_df, grid_search_cv = nnr.run()
```

    Fitting 5 folds for each of 1 candidates, totalling 5 fits


The runner returns the `run_stats` and `curves` corresponding to *best* hyperparameter combination,
as well as the cross validation results and the underlying `GridSearchCV` object used in the run.


```python
y_test_pred = grid_search_cv.predict(X_test_scaled)
y_test_accuracy = accuracy_score(np.asarray(y_test_hot), y_test_pred)
y_test_accuracy
```




    0.3333333333333333




```python
y_train_pred = grid_search_cv.predict(X_train_scaled)
y_train_accuracy = accuracy_score(y_train_hot, y_train_pred)
y_train_accuracy
```




    0.3333333333333333



Run stats dataframe


```python
run_stats_df[["current_restart", "Iteration", "Fitness", "FEvals", "Time", "learning_rate"]][:14]
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
      <th>current_restart</th>
      <th>Iteration</th>
      <th>Fitness</th>
      <th>FEvals</th>
      <th>Time</th>
      <th>learning_rate</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>0</td>
      <td>0</td>
      <td>1.306448</td>
      <td>1</td>
      <td>0.001905</td>
      <td>0.01</td>
    </tr>
    <tr>
      <th>1</th>
      <td>0</td>
      <td>1</td>
      <td>1.290536</td>
      <td>3</td>
      <td>0.004327</td>
      <td>0.01</td>
    </tr>
    <tr>
      <th>2</th>
      <td>0</td>
      <td>10</td>
      <td>1.246087</td>
      <td>14</td>
      <td>0.009143</td>
      <td>0.01</td>
    </tr>
    <tr>
      <th>3</th>
      <td>0</td>
      <td>50</td>
      <td>1.115004</td>
      <td>67</td>
      <td>0.027937</td>
      <td>0.01</td>
    </tr>
    <tr>
      <th>4</th>
      <td>0</td>
      <td>100</td>
      <td>1.044654</td>
      <td>127</td>
      <td>0.048129</td>
      <td>0.01</td>
    </tr>
    <tr>
      <th>5</th>
      <td>0</td>
      <td>250</td>
      <td>0.877647</td>
      <td>308</td>
      <td>0.107263</td>
      <td>0.01</td>
    </tr>
    <tr>
      <th>6</th>
      <td>0</td>
      <td>500</td>
      <td>0.739120</td>
      <td>614</td>
      <td>0.212110</td>
      <td>0.01</td>
    </tr>
    <tr>
      <th>7</th>
      <td>0</td>
      <td>1000</td>
      <td>0.732424</td>
      <td>1175</td>
      <td>0.433252</td>
      <td>0.01</td>
    </tr>
    <tr>
      <th>8</th>
      <td>1</td>
      <td>0</td>
      <td>1.306448</td>
      <td>1175</td>
      <td>0.436525</td>
      <td>0.01</td>
    </tr>
    <tr>
      <th>9</th>
      <td>1</td>
      <td>1</td>
      <td>1.306448</td>
      <td>1176</td>
      <td>0.438441</td>
      <td>0.01</td>
    </tr>
    <tr>
      <th>10</th>
      <td>1</td>
      <td>10</td>
      <td>1.247648</td>
      <td>1188</td>
      <td>0.444550</td>
      <td>0.01</td>
    </tr>
    <tr>
      <th>11</th>
      <td>1</td>
      <td>50</td>
      <td>1.148455</td>
      <td>1239</td>
      <td>0.465959</td>
      <td>0.01</td>
    </tr>
    <tr>
      <th>12</th>
      <td>1</td>
      <td>100</td>
      <td>1.105441</td>
      <td>1298</td>
      <td>0.491722</td>
      <td>0.01</td>
    </tr>
    <tr>
      <th>13</th>
      <td>1</td>
      <td>250</td>
      <td>0.931072</td>
      <td>1490</td>
      <td>0.570107</td>
      <td>0.01</td>
    </tr>
  </tbody>
</table>
</div>



curves dataframe


```python
curves_df[["current_restart", "Iteration", "Fitness", "FEvals", "Time", "learning_rate"]][:20]
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
      <th>current_restart</th>
      <th>Iteration</th>
      <th>Fitness</th>
      <th>FEvals</th>
      <th>Time</th>
      <th>learning_rate</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>0</td>
      <td>0</td>
      <td>1.290536</td>
      <td>1.0</td>
      <td>0.001905</td>
      <td>0.01</td>
    </tr>
    <tr>
      <th>1</th>
      <td>0</td>
      <td>1</td>
      <td>1.290536</td>
      <td>3.0</td>
      <td>0.004327</td>
      <td>0.01</td>
    </tr>
    <tr>
      <th>2</th>
      <td>0</td>
      <td>2</td>
      <td>1.290536</td>
      <td>4.0</td>
      <td>0.005450</td>
      <td>0.01</td>
    </tr>
    <tr>
      <th>3</th>
      <td>0</td>
      <td>3</td>
      <td>1.290536</td>
      <td>5.0</td>
      <td>0.005843</td>
      <td>0.01</td>
    </tr>
    <tr>
      <th>4</th>
      <td>0</td>
      <td>4</td>
      <td>1.263664</td>
      <td>7.0</td>
      <td>0.006530</td>
      <td>0.01</td>
    </tr>
    <tr>
      <th>5</th>
      <td>0</td>
      <td>5</td>
      <td>1.263664</td>
      <td>8.0</td>
      <td>0.006857</td>
      <td>0.01</td>
    </tr>
    <tr>
      <th>6</th>
      <td>0</td>
      <td>6</td>
      <td>1.246087</td>
      <td>10.0</td>
      <td>0.007742</td>
      <td>0.01</td>
    </tr>
    <tr>
      <th>7</th>
      <td>0</td>
      <td>7</td>
      <td>1.246087</td>
      <td>11.0</td>
      <td>0.008119</td>
      <td>0.01</td>
    </tr>
    <tr>
      <th>8</th>
      <td>0</td>
      <td>8</td>
      <td>1.246087</td>
      <td>12.0</td>
      <td>0.008459</td>
      <td>0.01</td>
    </tr>
    <tr>
      <th>9</th>
      <td>0</td>
      <td>9</td>
      <td>1.246087</td>
      <td>13.0</td>
      <td>0.008787</td>
      <td>0.01</td>
    </tr>
    <tr>
      <th>10</th>
      <td>0</td>
      <td>10</td>
      <td>1.246087</td>
      <td>14.0</td>
      <td>0.009143</td>
      <td>0.01</td>
    </tr>
    <tr>
      <th>11</th>
      <td>0</td>
      <td>11</td>
      <td>1.246087</td>
      <td>15.0</td>
      <td>0.010364</td>
      <td>0.01</td>
    </tr>
    <tr>
      <th>12</th>
      <td>0</td>
      <td>12</td>
      <td>1.246087</td>
      <td>16.0</td>
      <td>0.010792</td>
      <td>0.01</td>
    </tr>
    <tr>
      <th>13</th>
      <td>0</td>
      <td>13</td>
      <td>1.246087</td>
      <td>17.0</td>
      <td>0.011168</td>
      <td>0.01</td>
    </tr>
    <tr>
      <th>14</th>
      <td>0</td>
      <td>14</td>
      <td>1.246087</td>
      <td>18.0</td>
      <td>0.011561</td>
      <td>0.01</td>
    </tr>
    <tr>
      <th>15</th>
      <td>0</td>
      <td>15</td>
      <td>1.246087</td>
      <td>19.0</td>
      <td>0.011916</td>
      <td>0.01</td>
    </tr>
    <tr>
      <th>16</th>
      <td>0</td>
      <td>16</td>
      <td>1.246087</td>
      <td>20.0</td>
      <td>0.012291</td>
      <td>0.01</td>
    </tr>
    <tr>
      <th>17</th>
      <td>0</td>
      <td>17</td>
      <td>1.232142</td>
      <td>22.0</td>
      <td>0.013562</td>
      <td>0.01</td>
    </tr>
    <tr>
      <th>18</th>
      <td>0</td>
      <td>18</td>
      <td>1.216927</td>
      <td>24.0</td>
      <td>0.014337</td>
      <td>0.01</td>
    </tr>
    <tr>
      <th>19</th>
      <td>0</td>
      <td>19</td>
      <td>1.216927</td>
      <td>25.0</td>
      <td>0.014832</td>
      <td>0.01</td>
    </tr>
  </tbody>
</table>
</div>



cv results dataframe


```python
cv_results_df[
    [
        "mean_test_score",
        "rank_test_score",
        "mean_train_score",
        "param_activation",
        "param_hidden_layer_sizes",
        "param_learning_rate",
        "param_max_iters",
        "param_restarts",
    ]
]
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
      <th>mean_test_score</th>
      <th>rank_test_score</th>
      <th>mean_train_score</th>
      <th>param_activation</th>
      <th>param_hidden_layer_sizes</th>
      <th>param_learning_rate</th>
      <th>param_max_iters</th>
      <th>param_restarts</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>0.333333</td>
      <td>1</td>
      <td>0.333333</td>
      <td>relu</td>
      <td>[2]</td>
      <td>0.01</td>
      <td>1000</td>
      <td>1</td>
    </tr>
  </tbody>
</table>
</div>


