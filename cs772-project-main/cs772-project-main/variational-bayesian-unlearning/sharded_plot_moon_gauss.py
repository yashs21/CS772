import numpy as np 
import pickle


import matplotlib

from matplotlib import rc
rc('text', usetex=False)

matplotlib.rcParams['pdf.fonttype'] = 42
matplotlib.rcParams['ps.fonttype'] = 42


import matplotlib.pyplot as plt 
plt.style.use('ggplot')

plt.rcParams['lines.linewidth']=1.5
plt.rcParams['axes.facecolor']='w'

colors = plt.rcParams['axes.prop_cycle'].by_key()['color']

import model_config
from run import make_approximate


def num2str(num):
    return str(num).replace('.', '_')
    

import argparse


parser = argparse.ArgumentParser(description='Plot the data, posterior mean and variance of the synthetic moon classification dataset.')

parser.add_argument('--exper', help='in {moon, moon_random, moon_rm_30, moon_rm_40, moon_rm_50}',
                    required=False,
                    type=str,
                    default='moon')

parser.add_argument('--shards', help='number if shards data is divided into',
                    required=True,
                    type=str)

args = parser.parse_args()

num_shards = int(args.shards)
nbijector = 15
nhidden = 5

experiment = args.exper
approximate = "gauss_fullcov"


approximate_dist, approximate_config = make_approximate(approximate, nbijector, nhidden)
experiment_data = model_config.get_experiment(experiment)

dim = experiment_data['dim']
nparam = experiment_data['nparam']
model = experiment_data['model']
data = experiment_data['data']
remain_data = experiment_data['remain_data']
removed_data = experiment_data['removed_data']
ndata = data.shape[0]

prefix = "result/{}/{}".format(experiment, approximate)
selected_percentages = [1e-5, 1e-9, 0.0]


n = 50
plot_xmin = -1.5
plot_xmax = 2.5

x1d = np.linspace(plot_xmin, plot_xmax, n)
x1,x2 = np.meshgrid(x1d, x1d)
x = np.stack([x1.flatten(), x2.flatten()]).T

full_learned_params = [pickle.load(open("{}/full_data_post_shard_{}.p".format(prefix, i), "rb")) for i in range(num_shards)]
full_meanf, full_varf = [], []
for params in full_learned_params:
    meanf, varf = model.predict_f(x, 
                            params['loc'], 
                            params['sqrt_cov'].dot(params['sqrt_cov'].T))
    full_meanf.append(meanf)
    full_varf.append(varf)
full_meanf = np.mean(np.stack(full_meanf, axis=-1), axis=-1)
full_varf = np.mean(np.stack(full_varf, axis=-1), axis=-1)

remain_learned_params = [pickle.load(open("{}/remain_data_retrain_post_shard_{}.p".format(prefix, i), "rb")) for i in range(num_shards)]
remain_meanf, remain_varf = [], []
for params in remain_learned_params:
    meanf, varf = model.predict_f(x, 
                            params['loc'], 
                            params['sqrt_cov'].dot(params['sqrt_cov'].T))
    remain_meanf.append(meanf)
    remain_varf.append(varf)
remain_meanf = np.mean(np.stack(remain_meanf, axis=-1), axis=-1)
remain_varf = np.mean(np.stack(remain_varf, axis=-1), axis=-1)

elbo_meanf = {}
elbo_varf = {}

eubo_meanf = {}
eubo_varf = {}

for percentage in selected_percentages:
    elbo_learned_params = [pickle.load(open("{}/data_remain_data_by_unlearn_elbo_shard_{}_mode_{}.p".format(prefix, i, percentage), "rb")) for i in range(num_shards)]
    elbo_meanf_list, elbo_varf_list = [], []
    for params in elbo_learned_params:
        meanf, varf = model.predict_f(x, 
                                params['loc'], 
                                params['sqrt_cov'].dot(params['sqrt_cov'].T))
        elbo_meanf_list.append(meanf)
        elbo_varf_list.append(varf)
    elbo_meanf[percentage] = np.mean(np.stack(elbo_meanf_list, axis=-1), axis=-1)
    elbo_varf[percentage] = np.mean(np.stack(elbo_varf_list, axis=-1), axis=-1)

    eubo_learned_params = [pickle.load(open("{}/data_remain_data_by_unlearn_eubo_shard_{}_mode_{}.p".format(prefix, i, percentage), "rb")) for i in range(num_shards)]
    eubo_meanf_list, eubo_varf_list = [], []
    for params in eubo_learned_params:
        meanf, varf = model.predict_f(x, 
                                params['loc'], 
                                params['sqrt_cov'].dot(params['sqrt_cov'].T))
        eubo_meanf_list.append(meanf)
        eubo_varf_list.append(varf)
    eubo_meanf[percentage] = np.mean(np.stack(eubo_meanf_list, axis=-1), axis=-1)
    eubo_varf[percentage] = np.mean(np.stack(eubo_varf_list, axis=-1), axis=-1)

figsize = (2.2*6, 2.*3)

fig, axs = plt.subplots(3,6, figsize=figsize, tight_layout=True)

axs[0,0].scatter(remain_data[np.where(remain_data[:,-1] == 0),0], 
           remain_data[np.where(remain_data[:,-1] == 0),1], 
           marker='o', c=colors[1], s=20)
axs[0,0].scatter(remain_data[np.where(remain_data[:,-1] == 1),0], 
           remain_data[np.where(remain_data[:,-1] == 1),1], 
           marker='o', c=colors[4], s=20)


sc = axs[0,0].scatter(removed_data[np.where(removed_data[:,-1] == 0),0], 
           removed_data[np.where(removed_data[:,-1] == 0),1], 
           marker='X', c=colors[1], s=30)
sc.set_edgecolor('#C7006E')
sc = axs[0,0].scatter(removed_data[np.where(removed_data[:,-1] == 1),0], 
           removed_data[np.where(removed_data[:,-1] == 1),1], 
           marker='X', c=colors[4], s=30)
sc.set_edgecolor('#C7006E')

axs[0,0].set_xlim(plot_xmin, plot_xmax)
axs[0,0].set_ylim(plot_xmin, plot_xmax)

axs[0,0].set_xlabel(r'$x_0$')
axs[0,0].set_ylabel(r'$x_1$')
axs[0,0].set_title("Data")

contour = axs[0,1].contour(x1, x2, full_meanf.reshape(n,n), origin='lower', colors='black')
axs[0,1].clabel(contour, inline=True, fontsize=8)
axs[0,1].set_xlabel(r'$x_0$')
axs[0,1].set_ylabel(r'$x_1$')
axs[0,1].grid(False)
axs[0,1].set_title(r"full data: $\mu_x$")

contour = axs[0,2].contour(x1, x2, full_varf.reshape(n,n), origin='lower', colors='black')
axs[0,2].clabel(contour, inline=True, fontsize=8)
axs[0,2].set_xlabel(r'$x_0$')
axs[0,2].set_ylabel(r'$x_1$')
axs[0,2].grid(False)
axs[0,2].set_title(r"full data: $\sigma^2_x$")


contour = axs[0,3].contour(x1, x2, remain_meanf.reshape(n,n), origin='lower', colors='black')
axs[0,3].clabel(contour, inline=True, fontsize=8)
axs[0,3].set_xlabel(r'$x_0$')
axs[0,3].set_ylabel(r'$x_1$')
axs[0,3].grid(False)
axs[0,3].set_title(r"retrain: $\mu_x$")

contour = axs[0,4].contour(x1, x2, remain_varf.reshape(n,n), origin='lower', colors='black')
axs[0,4].clabel(contour, inline=True, fontsize=8)
axs[0,4].set_xlabel(r'$x_0$')
axs[0,4].set_ylabel(r'$x_1$')
axs[0,4].grid(False)
axs[0,4].set_title(r"retrain: $\sigma^2_x$")

plot_idx = 5
for percentage in selected_percentages:
    contour = axs[int(plot_idx/6),plot_idx%6].contour(x1, x2, elbo_meanf[percentage].reshape(n,n), origin='lower', colors='black')
    axs[int(plot_idx/6),plot_idx%6].clabel(contour, inline=True, fontsize=8)
    axs[int(plot_idx/6),plot_idx%6].set_xlabel(r'$x_0$')
    axs[int(plot_idx/6),plot_idx%6].set_ylabel(r'$x_1$')
    axs[int(plot_idx/6),plot_idx%6].grid(False)
    axs[int(plot_idx/6),plot_idx%6].set_title(r"rKL: $\mu_x$, $\lambda={}$".format(percentage))

    plot_idx += 1

    contour = axs[int(plot_idx/6),plot_idx%6].contour(x1, x2, elbo_varf[percentage].reshape(n,n), origin='lower', colors='black')
    axs[int(plot_idx/6),plot_idx%6].clabel(contour, inline=True, fontsize=8)
    axs[int(plot_idx/6),plot_idx%6].set_xlabel(r'$x_0$')
    axs[int(plot_idx/6),plot_idx%6].set_ylabel(r'$x_1$')
    axs[int(plot_idx/6),plot_idx%6].grid(False)
    axs[int(plot_idx/6),plot_idx%6].set_title(r"rKL: $\sigma^2_x$, $\lambda={}$".format(percentage))

    plot_idx += 1

    contour = axs[int(plot_idx/6),plot_idx%6].contour(x1, x2, eubo_meanf[percentage].reshape(n,n), origin='lower', colors='black')
    axs[int(plot_idx/6),plot_idx%6].clabel(contour, inline=True, fontsize=8)
    axs[int(plot_idx/6),plot_idx%6].set_xlabel(r'$x_0$')
    axs[int(plot_idx/6),plot_idx%6].set_ylabel(r'$x_1$')
    axs[int(plot_idx/6),plot_idx%6].grid(False)
    axs[int(plot_idx/6),plot_idx%6].set_title(r"EUBO: $\mu_x$, $\lambda={}$".format(percentage))

    plot_idx += 1

    contour = axs[int(plot_idx/6),plot_idx%6].contour(x1, x2, eubo_varf[percentage].reshape(n,n), origin='lower', colors='black')
    axs[int(plot_idx/6),plot_idx%6].clabel(contour, inline=True, fontsize=8)
    axs[int(plot_idx/6),plot_idx%6].set_xlabel(r'$x_0$')
    axs[int(plot_idx/6),plot_idx%6].set_ylabel(r'$x_1$')
    axs[int(plot_idx/6),plot_idx%6].grid(False)
    axs[int(plot_idx/6),plot_idx%6].set_title(r"EUBO: $\sigma^2_x$, $\lambda={}$".format(percentage))

    plot_idx += 1

plt.show()