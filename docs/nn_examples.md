# mlrose_ky Generator and Neural Network Runner Usage Examples - Andrew Rollings

## Overview

These examples will not solve assignment 2 for you, but they will give you
some idea on how to use the problem generator and runner classes.

Hopefully this will result in slightly fewer
"How do I &lt;insert basic usage here&gt;?" questions every semester...

Also, and in case it hasn't been made clear enough by the TAs, using any of the visualization code in here for your
assignment is a bad idea, for two reasons... (1) It provides nothing useful as far as the assignment goes,
and (2) the TAs will undoubtedly frown upon it.

Visualization is part of the analysis and, for the most part, you're supposed to do that by yourself. Just including
images of the before/after state of a problem really isn't useful in terms of what you're supposed to be analyzing.

I also strongly recommend against using the synthetic data class used within for your assignment. It's a toy, with no real
value for the assignment. There's much better data out there.

### Import Libraries


```python
from IPython.core.display import display, HTML  # for some notebook formatting.

import mlrose_ky
import numpy as np
import pandas as pd
import logging
import networkx as nx
import matplotlib.pyplot as plt
import string


from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler, OneHotEncoder
from sklearn.metrics import accuracy_score
from mlrose_ky import SyntheticData, plot_synthetic_dataset
from mlrose_ky import SKMLPRunner, SARunner, GARunner, NNGSRunner

# switch off the chatter
logging.basicConfig(level=logging.WARNING)
```

### Generating sample data...

This sample data will be used in subsequent examples.


```python
sd = SyntheticData(seed=123456)

clean_data, clean_features, clean_classes, _ = sd.get_synthetic_data(x_dim=20, y_dim=20)

print(f"Features: {clean_features}")
print(f"Classes: {clean_classes}")

cx, cy, cx_tr, cx_ts, cy_tr, cy_ts = sd.setup_synthetic_data_test_train(clean_data)
plot_synthetic_dataset(x_train=cx_tr, y_train=cy_tr, x_test=cx_ts, y_test=cy_ts, transparent_bg=False, bg_color="black")
```

    Features: ['(1) A', '(2) B']
    Classes: ['RED', 'BLUE']



    
![png](nn_examples_files/nn_examples_6_1.png)
    


This is our standard 20x20 sample set with no noise or errors, with a 70/30 split between training and test data.

Each point represents a data sample.
Black-outlined points represent training data, and white-outlined points represent test data.

A sample of the clean data can be seen below, with the first two columns being the (x,y) position of the data point, and the third
column representing the color.


```python
HTML(pd.DataFrame(columns=["x", "y", "c"], data=clean_data)[150:160].to_html())
```




<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>x</th>
      <th>y</th>
      <th>c</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>150</th>
      <td>7</td>
      <td>10</td>
      <td>0</td>
    </tr>
    <tr>
      <th>151</th>
      <td>7</td>
      <td>11</td>
      <td>0</td>
    </tr>
    <tr>
      <th>152</th>
      <td>7</td>
      <td>12</td>
      <td>0</td>
    </tr>
    <tr>
      <th>153</th>
      <td>7</td>
      <td>13</td>
      <td>1</td>
    </tr>
    <tr>
      <th>154</th>
      <td>7</td>
      <td>14</td>
      <td>1</td>
    </tr>
    <tr>
      <th>155</th>
      <td>7</td>
      <td>15</td>
      <td>1</td>
    </tr>
    <tr>
      <th>156</th>
      <td>7</td>
      <td>16</td>
      <td>1</td>
    </tr>
    <tr>
      <th>157</th>
      <td>7</td>
      <td>17</td>
      <td>1</td>
    </tr>
    <tr>
      <th>158</th>
      <td>7</td>
      <td>18</td>
      <td>1</td>
    </tr>
    <tr>
      <th>159</th>
      <td>7</td>
      <td>19</td>
      <td>1</td>
    </tr>
  </tbody>
</table>



We can generate "dirty" data in two ways: by adding an column of random data,
or by introducing noisy duplicate data points that either reinforce or contradict
existing data points.


```python
noisy_data, noisy_features, noisy_classes, _ = sd.get_synthetic_data(x_dim=20, y_dim=20, add_noise=0.025)
print(f"Features: {noisy_features}")
print(f"Classes: {noisy_classes}")

nx, ny, nx_tr, nx_ts, ny_tr, ny_ts = sd.setup_synthetic_data_test_train(noisy_data)
plot_synthetic_dataset(x_train=nx_tr, y_train=ny_tr, x_test=nx_ts, y_test=ny_ts, transparent_bg=False, bg_color="black")
```

    Features: ['(1) A', '(2) B']
    Classes: ['RED', 'BLUE']



    
![png](nn_examples_files/nn_examples_10_1.png)
    


This is our data with 2.5% added noise.
A sample of the noisy data can be seen below, with the first two columns being the (x,y) position of the data point, and the third
column representing the color. Note that the sample shown represent some of the noisy data,
which is added to the end of the clean data.


```python
HTML(pd.DataFrame(columns=["x", "y", "c"], data=noisy_data)[-10:].to_html())
```




<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>x</th>
      <th>y</th>
      <th>c</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>410</th>
      <td>17</td>
      <td>17</td>
      <td>1</td>
    </tr>
    <tr>
      <th>411</th>
      <td>15</td>
      <td>18</td>
      <td>0</td>
    </tr>
    <tr>
      <th>412</th>
      <td>15</td>
      <td>19</td>
      <td>0</td>
    </tr>
    <tr>
      <th>413</th>
      <td>0</td>
      <td>6</td>
      <td>0</td>
    </tr>
    <tr>
      <th>414</th>
      <td>10</td>
      <td>14</td>
      <td>1</td>
    </tr>
    <tr>
      <th>415</th>
      <td>7</td>
      <td>8</td>
      <td>1</td>
    </tr>
    <tr>
      <th>416</th>
      <td>10</td>
      <td>19</td>
      <td>1</td>
    </tr>
    <tr>
      <th>417</th>
      <td>10</td>
      <td>9</td>
      <td>0</td>
    </tr>
    <tr>
      <th>418</th>
      <td>16</td>
      <td>1</td>
      <td>0</td>
    </tr>
    <tr>
      <th>419</th>
      <td>6</td>
      <td>0</td>
      <td>1</td>
    </tr>
  </tbody>
</table>



If we had chosen to add the random column of data, then the plots
would look that same as the above two, depending on whether we had
chosen to additionally add noise or not.

However, the features are different, as shown below, and we can see the extra
column in the data sample.


```python
extra_data, extra_features, extra_classes, _ = sd.get_synthetic_data(x_dim=20, y_dim=20, add_redundant_column=True)
print(f"Features: {extra_features}")
print(f"Classes: {extra_classes}")

ex, ey, ex_tr, ex_ts, ey_tr, ey_ts = sd.setup_synthetic_data_test_train(extra_data)
```

    Features: ['(1) A', '(2) B', '(3) R']
    Classes: ['RED', 'BLUE']



```python
HTML(pd.DataFrame(columns=["x", "y", "r", "c"], data=extra_data)[150:160].to_html())
```




<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>x</th>
      <th>y</th>
      <th>r</th>
      <th>c</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>150</th>
      <td>7.0</td>
      <td>10.0</td>
      <td>0.972641</td>
      <td>0.0</td>
    </tr>
    <tr>
      <th>151</th>
      <td>7.0</td>
      <td>11.0</td>
      <td>0.726259</td>
      <td>0.0</td>
    </tr>
    <tr>
      <th>152</th>
      <td>7.0</td>
      <td>12.0</td>
      <td>0.412651</td>
      <td>0.0</td>
    </tr>
    <tr>
      <th>153</th>
      <td>7.0</td>
      <td>13.0</td>
      <td>0.990003</td>
      <td>1.0</td>
    </tr>
    <tr>
      <th>154</th>
      <td>7.0</td>
      <td>14.0</td>
      <td>0.535660</td>
      <td>1.0</td>
    </tr>
    <tr>
      <th>155</th>
      <td>7.0</td>
      <td>15.0</td>
      <td>0.559253</td>
      <td>1.0</td>
    </tr>
    <tr>
      <th>156</th>
      <td>7.0</td>
      <td>16.0</td>
      <td>0.867020</td>
      <td>1.0</td>
    </tr>
    <tr>
      <th>157</th>
      <td>7.0</td>
      <td>17.0</td>
      <td>0.019276</td>
      <td>1.0</td>
    </tr>
    <tr>
      <th>158</th>
      <td>7.0</td>
      <td>18.0</td>
      <td>0.123097</td>
      <td>1.0</td>
    </tr>
    <tr>
      <th>159</th>
      <td>7.0</td>
      <td>19.0</td>
      <td>0.808300</td>
      <td>1.0</td>
    </tr>
  </tbody>
</table>



### Preparing the experiment parameters


```python
# ensure defaults are in grid search
default_grid_search_parameters = {
    "max_iters": [5000],
    "learning_rate_init": [0.1, 0.2, 0.4, 0.8],
    "hidden_layer_sizes": [[4, 4, 4]],
    "activation": [mlrose_ky.neural.activation.relu],
}

default_parameters = {
    "seed": 123456,
    "iteration_list": 2 ** np.arange(13),
    "max_attempts": 5000,
    "override_ctrl_c_handler": False,  # required for running in notebook
    "n_jobs": 5,
    "cv": 5,
}
```

### Example 1: Running the SKMLPRunner

#### (a) Clean Data


```python
skmlp_grid_search_parameters = {
    **default_grid_search_parameters,
    "max_iters": [5000],
    "learning_rate_init": [0.0001],
    "activation": [mlrose_ky.neural.activation.sigmoid],
}

skmlp_default_parameters = {**default_parameters, "early_stopping": True, "tol": 1e-05, "alpha": 0.001, "solver": "lbfgs"}
```


```python
cx_skr = SKMLPRunner(
    x_train=cx_tr,
    y_train=cy_tr,
    x_test=cx_ts,
    y_test=cx_ts,
    experiment_name="skmlp_clean",
    grid_search_parameters=skmlp_grid_search_parameters,
    **skmlp_default_parameters,
)

run_stats_df, curves_df, cv_results_df, cx_sr = cx_skr.run()
```

    Fitting 5 folds for each of 1 candidates, totalling 5 fits


    [Parallel(n_jobs=5)]: Using backend LokyBackend with 5 concurrent workers.
    [Parallel(n_jobs=5)]: Done   2 out of   5 | elapsed:    2.0s remaining:    3.0s
    [Parallel(n_jobs=5)]: Done   5 out of   5 | elapsed:  1.2min finished


The following plot shows the baseline predictions made for the clean dataset using the SKLearn runner.

Note that the background coloring shows the prediction made by the learner for that area.
The intensity of the color is indicative of the prediction confidence level.


```python
print(f'Hidden layer size: {cx_sr.best_params_["hidden_layer_sizes"]}')
plot_synthetic_dataset(x_train=cx_tr, y_train=cy_tr, x_test=cx_ts, y_test=cy_ts, classifier=cx_sr.best_estimator_.mlp)
```

    Hidden layer size: [4, 4, 4]



    
![png](nn_examples_files/nn_examples_22_1.png)
    


#### (b) Noisy Data


```python
nx1_skr = SKMLPRunner(
    x_train=nx_tr,
    y_train=ny_tr,
    x_test=nx_ts,
    y_test=nx_ts,
    experiment_name="skmlp_noisy_1",
    grid_search_parameters=skmlp_grid_search_parameters,
    **skmlp_default_parameters,
)

nx1_run_stats_df, nx1_curves_df, nx1_cv_results_df, nx1_sr = nx1_skr.run()
```

    Fitting 5 folds for each of 1 candidates, totalling 5 fits


    [Parallel(n_jobs=5)]: Using backend LokyBackend with 5 concurrent workers.
    [Parallel(n_jobs=5)]: Done   2 out of   5 | elapsed:    0.4s remaining:    0.6s
    [Parallel(n_jobs=5)]: Done   5 out of   5 | elapsed:  1.5min finished


The following plot shows the predictions made for the noisy dataset using the sklearn runner
with the same hidden layer size as the best clean set run.


```python
print(f'Hidden layer size: {nx1_sr.best_params_["hidden_layer_sizes"]}')
plot_synthetic_dataset(x_train=nx_tr, y_train=ny_tr, x_test=nx_ts, y_test=ny_ts, classifier=nx1_sr.best_estimator_.mlp)
```

    Hidden layer size: [4, 4, 4]



    
![png](nn_examples_files/nn_examples_26_1.png)
    



```python
noisy_data_grid_search_parameters = {**skmlp_grid_search_parameters, "hidden_layer_sizes": [[4, 4, 4, 4]]}
nx2_skr = SKMLPRunner(
    x_train=nx_tr,
    y_train=ny_tr,
    x_test=nx_ts,
    y_test=nx_ts,
    experiment_name="skmlp_noisy_2",
    grid_search_parameters=noisy_data_grid_search_parameters,
    **skmlp_default_parameters,
)

nx2_run_stats_df, nx2_curves_df, nx2_cv_results_df, nx2_sr = nx2_skr.run()
```

    Fitting 5 folds for each of 1 candidates, totalling 5 fits


    [Parallel(n_jobs=5)]: Using backend LokyBackend with 5 concurrent workers.
    [Parallel(n_jobs=5)]: Done   2 out of   5 | elapsed:  2.0min remaining:  3.0min
    [Parallel(n_jobs=5)]: Done   5 out of   5 | elapsed:  2.1min finished


The following plot shows the predictions made for the noisy dataset using the sklearn runner
with the optimal hidden layer size.


```python
print(f'Hidden layer size: {nx2_sr.best_params_["hidden_layer_sizes"]}')
plot_synthetic_dataset(x_train=nx_tr, y_train=ny_tr, x_test=nx_ts, y_test=ny_ts, classifier=nx2_sr.best_estimator_.mlp)
```

    Hidden layer size: [4, 4, 4, 4]



    
![png](nn_examples_files/nn_examples_29_1.png)
    


#### (c) Extra Data


```python
ex1_skr = SKMLPRunner(
    x_train=ex_tr,
    y_train=ey_tr,
    x_test=ex_ts,
    y_test=ex_ts,
    experiment_name="skmlp_extra_1",
    grid_search_parameters=skmlp_grid_search_parameters,
    **skmlp_default_parameters,
)

ex1_run_stats_df, ex1_curves_df, ex1_cv_results_df, ex1_sr = ex1_skr.run()
```

    Fitting 5 folds for each of 1 candidates, totalling 5 fits


    [Parallel(n_jobs=5)]: Using backend LokyBackend with 5 concurrent workers.
    [Parallel(n_jobs=5)]: Done   2 out of   5 | elapsed:  1.8min remaining:  2.6min
    [Parallel(n_jobs=5)]: Done   5 out of   5 | elapsed:  1.9min finished


The following plot shows the baseline predictions made for the extra-column dataset using the sklearn runner
with the same hidden layer size as the best clean set run:


```python
print(f'Hidden layer size: {ex1_sr.best_params_["hidden_layer_sizes"]}')
plot_synthetic_dataset(x_train=ex_tr, y_train=ey_tr, x_test=ex_ts, y_test=ey_ts, classifier=ex1_sr.best_estimator_.mlp)
```

    Hidden layer size: [4, 4, 4]



    
![png](nn_examples_files/nn_examples_33_1.png)
    



```python
# override the hidden_layer_sizes grid search.
extra_data_grid_search_parameters = {**skmlp_grid_search_parameters, "hidden_layer_sizes": [[6, 6, 6]]}

ex2_skr = SKMLPRunner(
    x_train=ex_tr,
    y_train=ey_tr,
    x_test=ex_ts,
    y_test=ex_ts,
    experiment_name="skmlp_extra_2",
    grid_search_parameters=extra_data_grid_search_parameters,
    **skmlp_default_parameters,
)

ex2_run_stats_df, ex2_curves_df, ex2_cv_results_df, ex2_sr = ex2_skr.run()
```

    Fitting 5 folds for each of 1 candidates, totalling 5 fits


    [Parallel(n_jobs=5)]: Using backend LokyBackend with 5 concurrent workers.
    [Parallel(n_jobs=5)]: Done   2 out of   5 | elapsed:  1.3min remaining:  1.9min
    [Parallel(n_jobs=5)]: Done   5 out of   5 | elapsed:  2.0min finished


The following plot shows the baseline predictions made for the extra-column dataset using the sklearn runner
with the optimal hidden layer size:


```python
print(f'Hidden layer size: {ex2_sr.best_params_["hidden_layer_sizes"]}')
plot_synthetic_dataset(x_train=ex_tr, y_train=ey_tr, x_test=ex_ts, y_test=ey_ts, classifier=ex2_sr.best_estimator_.mlp)
```

    Hidden layer size: [6, 6, 6]



    
![png](nn_examples_files/nn_examples_36_1.png)
    

