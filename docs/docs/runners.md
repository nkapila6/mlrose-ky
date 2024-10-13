## Tutorial - How to use Runners?

>[!INFO] Recommendation
>It is highly recommended that you use the [`Runners class`](https://github.com/knakamura13/mlrose-ky/tree/main/src/mlrose_ky/runners) for the second assignment on randomized optimizations.
### An example with RHC Runner
The below example illustrates how to initialize an RHCRunner object.

```python
import mlrose_ky

# initialize the following variables
problem = # An opt prob object from [[opt_probs]]
seed = # a seed so experiments are reproducible
iteration_list = 2**np.arange(10)
max_attempts = 5000
restart_list = [25,75,100]

rhc = mlrose_ky.RHCRunner(problem=problem,
						experiment_name="exp_name",
						seed=seed
						iteration_list = iteration_list
						max_attempts=max_attempts,
						restart_list=restart_list
						 )
```

The `.run()` method can be called on the `rhc` to return 2 dataframes that contain the results.

```python
df_run_stats, df_run_curves = rhc.run()
```

The `df_run_stats` dataframe records the information based on  the parameters in the restart list, i.e. 25 restarts, 75 restarts and 100 restarts. For the other runners, the `df_run_stats` follows the iteration list instead, i.e. `df_run_stats` has information based on list mentioned in `iteration_list`

However, we use `df_run_curves` for plotting that has all the information, i.e. all restarts from 0 to 100.

>[!INFO] Recommendation
>Since the experiments take a lot of time to run, it is highly recommended that you save the  `df_run_curves` output using the `pandas.to_csv()` method.>

This saved `df_run_curves` returns a nice dataframe that can be used with plotting functions (which is a WIP in `mlrose-ky`).

The output has the following items:
- `Fitness` score
- `FEvals` (Fitness Evaluations)
- `Time` (in seconds)
- Other algorithm variables depending on Runner used, eg: `Population Size` and `Keep Percent` for MIMIC.

>[!WARNING]- Recommendation
> Simulated Annealing does not track `decay` types so it is best to record different `decay` types into `df_run_curves` before saving the runner.

### Use runners to make your own custom wrapper

The best way to use runners would be to wrap and call the runners in your own function. One example workflow is below.

1. Pass the following input to the function.
	1. Generate a problem using one of the [problem generators](https://github.com/knakamura13/mlrose-ky/tree/main/src/mlrose_ky/generators).
	2. Pass in the `seed`, `iteration_list`, `max_attempts`, and `restart_list`
	3. Other parameters based on algorithm used, e.g. `temperature` and `decay` for Simulated Annealing, etc.
	4. Number of for loops to run the experiment on, i.e. to plot mean and average.
	
2. Run the runners and record the output, record any other necessary information you feel could be useful for plotting into `df_run_curves`.

3. Save `df_run_curves` as a `csv` file.

4. Use information from `df_run_curves` to make the necessary plots.

Feel free to innovate based on your requirements, the workflow above is just a guideline.