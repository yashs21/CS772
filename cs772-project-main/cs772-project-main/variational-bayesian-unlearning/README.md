
# Variational Bayesian Unlearning through Shards

Modified from https://github.com/qphong/variational-bayesian-unlearning

### Prerequisites

```
python = 3.7
tensorflow = 1.14.0
tensorflow-probability = 0.7.0
matplotlib
numpy
scipy
pickle
argparse
```

### Running the experiments

(Added functionality to divide data into shards, use `sharded_run.py`, `sharded_compute_kl_distance.py` and `sharded_plot_moon_gauss.py` instead)

1.
```
python sharded_run.py --exper moon --appr gauss_fullcov --nsample 100 --ntrain 3000 --shards 3 --folder result
```
2.
```
python3 sharded_compute_kl_distance.py --folder result --outfolder result/plot_data --exper moon --appr gauss_fullcov --shards 3
```
3.
```
python3 plot_kl_distance_mean_std.py --folder result/plot_data --exper moon --appr gauss_fullcov
```
4.
```
python3 sharded_plot_moon_gauss.py --exper moon --shards 3
```

## Synthetic Moon Classification Dataset

1. To run the training with VI on full data, retraining with VI on remaining data, and unlearning using EUBO, rKL
```
python run.py --exper moon --appr gauss_fullcov --nsample 1000 --ntrain 30000 --folder result
```
The result is written to folder `result`.

2. To compute the averaged KL distance between approximate predictive distributions
```
python compute_kl_distance.py --folder result --outfolder plot_data --exper moon --appr gauss_fullcov
```
The KL distances are written to folder `plot_data`.

3. To plot the averaged KL distance between approximate predictive distributions
```
python plot_kl_distance_mean_std.py --folder plot_data --exper moon --appr gauss_fullcov
```

4. To plot the posterior mean and variance of the latent function
```
python plot_moon_gauss.py --exper moon
```

5. To plot the predictive distributions
```
python plot_moon_marginal_prob.py --exper moon
```
