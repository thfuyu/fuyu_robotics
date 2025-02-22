import os
import warnings
warnings.simplefilter(action='ignore', category=FutureWarning)

import sim_utils
import argparse
import pandas as pd
import config
import numpy as np
import config

# 创建目标文件夹（如果不存在）
outdir = r'D:\desktop1\2-12-surgecast\plume'
if not os.path.exists(outdir):
    os.makedirs(outdir)

parser = argparse.ArgumentParser(description='Generate plume simulations')
parser.add_argument('--duration',  metavar='d',  type=int, 
  help='simulation duration in seconds', default = config.duration)
parser.add_argument('--cores',  metavar='c',  type=int, 
  help='number of cores to use', default=24)
parser.add_argument('--dataset_name',  type=str, default='test')
parser.add_argument('--fname_suffix',  type=str, default='')
parser.add_argument('--dt',  type=float, 
	help='time per step (seconds)', default = config.dt)
parser.add_argument('--wind_magnitude',  type=float, 
	help='m/s', default = config.wind_magnitude)
parser.add_argument('--wind_y_varx',  type=float, default = config.wind_y_varx)
parser.add_argument('--birth_rate',  type=float, 
	help='poisson birth_rate parameter', default = config.birth_rate)
parser.add_argument('--outdir',  type=str, default=outdir)  # 修改为新的文件夹

args = parser.parse_args()
print(args)

wind_df = sim_utils.get_wind_xyt(
	args.duration+1, 
	dt=args.dt,
	wind_magnitude=args.wind_magnitude,
	regime=args.dataset_name
	)
wind_df['tidx'] = np.arange(len(wind_df), dtype=int) 
fname = os.path.join(args.outdir, f'wind_data_{args.dataset_name}{args.fname_suffix}.pickle')  # 使用os.path.join合成路径
wind_df.to_pickle(fname)
print(wind_df.head(n=5))
print(wind_df.tail(n=5))
print("Saved", fname)

# Older ODEINT version
# wind_y_var = args.wind_magnitude/np.sqrt(args.wind_y_varx)
# puff_df = sim_utils.get_puffs_df_oneshot(wind_df, wind_y_var,
# 	args.birth_rate, args.cores, verbose=True)

# Using faster vectorized version
wind_y_var = args.wind_magnitude/np.sqrt(args.wind_y_varx)
puff_df = sim_utils.get_puffs_df_vector(wind_df, wind_y_var, args.birth_rate, verbose=True)

fname = os.path.join(args.outdir, f'puff_data_{args.dataset_name}{args.fname_suffix}.pickle')  # 使用os.path.join合成路径
puff_df.to_pickle(fname)
print('puff_df.shape', puff_df.shape)
print(puff_df.tail())
print(puff_df.head())
print("Saved", fname)


## -- Extra Viz -- ##
# Plot puffs - also serves a good test
# Need to add concentration & radius data before plotting
import sim_analysis # load config later, eek!
data_puffs, data_wind = sim_analysis.load_plume(f'{args.dataset_name}{args.fname_suffix}')
t_val = data_puffs['time'].iloc[-1]
fig, ax = sim_analysis.plot_puffs_and_wind_vectors(
	data_puffs, 
	data_wind, 
	t_val, 
    fname='', 
    plotsize=(8,8))
fig.savefig(os.path.join(args.outdir, f'{args.dataset_name}{args.fname_suffix}_t{t_val:3.3f}.png'))  # 使用os.path.join合成路径
ax.set_xlim(-1, 12)
ax.set_ylim(-1.8, +1.8)
if 'switch' in args.dataset_name:
    ax.set_xlim(-1, +10) # if switching
    ax.set_ylim(-5, +5) # if switching
fig.savefig(os.path.join(args.outdir, f'{args.dataset_name}{args.fname_suffix}_t{t_val:3.3f}z.png'))  # 使用os.path.join合成路径
