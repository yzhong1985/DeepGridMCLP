import pandas as pd
import numpy as np
import torch
import os

num_sample = 128000  # n samples
n_demand_pts = 1000
n_facilities = 100  # number of facilities
r = 0.15  # coverage radius
p = 20  # select p facilities

uses = 'train'
# uses = 'valid'
# uses = 'test'

# demand pts [0, 1]
def gen_random_demand_points(num_points=500):
    demand_X = np.random.rand(num_points)
    demand_Y = np.random.rand(num_points)
    demand_vals = np.random.rand(num_points)  # [0,1]
    demand_points = np.vstack([demand_X, demand_Y]).T
    return demand_points, demand_vals

# facilites pts [0, 1]
def gen_fixed_facilities(num_facilities=1000):
    facilities_X = np.random.rand(num_facilities)
    facilities_Y = np.random.rand(num_facilities)
    facilities = np.vstack([facilities_X, facilities_Y]).T
    return facilities

# random select m from fixed_facilities
def select_random_facilities(fixed_facilities, m):
    selected_indices = np.random.choice(len(fixed_facilities), m, replace=False)
    selected_facilities = fixed_facilities[selected_indices]
    return selected_facilities

# random data
def gen_random_data(num_sample, n_facilities, p, r, demand_points, demand_vals, fixed_facilities):
    random_datasets = []
    for i in range(num_sample):
        random_facilities = select_random_facilities(fixed_facilities, n_facilities)
        random_data = {}
        random_data["users"] = torch.tensor(demand_points).to(torch.float32)  
        random_data["facilities"] = torch.tensor(random_facilities).to(torch.float32)  
        random_data['demand'] = torch.tensor(demand_vals).to(torch.float32)
        random_data["p"] = p
        random_data["r"] = r  
        random_datasets.append(random_data)
        print(f"generate sample {i+1}/{num_sample}")
    return random_datasets

def save_dataset_as_pkl(dataset, filename):
    pd.to_pickle(dataset, filename)
    print(f"data saved to {filename}")

def generate_random_data(num_sample, n_demand_pts, n_facilities, p, r):
    demand_points, demand_vals = gen_random_demand_points(n_demand_pts) 
    fixed_facilities = gen_fixed_facilities(n_facilities * 5)  # 5 times facilities cls
   # filename = os.path.join(r".\data\MCLP\MCLP_" + str(r) + "_" + str(p), f"MCLP_100_" + str(p) + "_" + uses + "_Random_Facilities.pkl")
    filename = os.path.join(r".\data", f"mclp_{n_demand_pts}_{n_facilities}_{p}_random_{uses}.pkl")
    dataset = gen_random_data(num_sample, n_facilities, p, r, demand_points, demand_vals, fixed_facilities)
    save_dataset_as_pkl(dataset, filename)


generate_random_data(num_sample, n_demand_pts, n_facilities, p, r)