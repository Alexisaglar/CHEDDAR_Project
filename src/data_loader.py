import numpy as np
import torch
from torch_geometric.data import Data
import h5py
import time
import matplotlib.pyplot as plt
import pandas as pd
import math

def load_network_data(file, network_key):
    with h5py.File(file, 'r') as f:
        net_group = f[network_key]
        static_data = {
            'line': net_group['network_config/line'][:],
            'bus': net_group['network_config/bus'][:],
        }
    return static_data

def create_dataset(file, seq_length=24):
    data_list = []

    with h5py.File(file, 'r') as f:
        network_keys = [key for key in f.keys() if key.startswith('network_')]
        
        for network_key in network_keys:
            static_data = load_network_data(file, network_key)
            net_group = f[network_key]
            season_keys = ['season_0']

            for season_key in season_keys:
                season = int(season_key.split('_')[-1])
                if season_key in net_group:
                    season_group = net_group[season_key]

                    for time_step_key in sorted(season_group.keys(), key=lambda x: int(x.split('_')[-1])):
                        time_step_group = season_group[time_step_key]
                        # Extract node features 
                        mode = np.tile(np.random.randint(0, 3), (33,1))
                        loads = time_step_group['res_bus'][:, 2:4]  
                        loads = np.where(loads > 0, loads, 0) 
                        random_values = np.random.uniform(0.5, 1.5, size=(33,1))
                        random_loads = loads * np.hstack((random_values, random_values))
                        bus_class = static_data['bus'][:,3]
                        node_values = (np.arange(33).reshape(-1,1))
                        node_features = np.hstack((random_loads, bus_class.reshape(-1,1), mode))
                        node_total_data = np.hstack((node_values, random_loads, bus_class.reshape(-1,1), mode))

                        # we sort the loads by higher demand per node
                        load_sorted = np.argsort(node_features[:, 0])[::-1] 
                        sorted_array = node_total_data[load_sorted]
                        # amount of classes
                        _, counts = np.unique(sorted_array[:, 3], return_counts=True)

                        priority_target = np.zeros(33)
                        # from which 40 % is industrial, 30% commercial and 30% residential
                        if mode[0] == 0:
                            com_total = 33
                            num_ind, num_com, num_res = math.floor(com_total * 0.40), math.floor(com_total * 0.30), math.floor(com_total * 0.30)
                            low_prio = sorted_array[sorted_array[:, 3] == 1]
                            med_prio = sorted_array[sorted_array[:, 3] == 2]
                            high_prio = sorted_array[sorted_array[:, 3] == 3]

                            priority_target[low_prio[:num_res, 0].astype(int)] = 1
                            priority_target[med_prio[:num_com, 0].astype(int)] = 2
                            priority_target[high_prio[:num_ind, 0].astype(int)] = 3


                        elif mode[0] == 1:
                            com_total = 20 
                            num_ind, num_com, num_res = math.floor(com_total * 0.40), math.floor(com_total * 0.30), math.floor(com_total * 0.30)
                            low_prio = sorted_array[sorted_array[:, 3] == 1]
                            med_prio = sorted_array[sorted_array[:, 3] == 2]
                            high_prio = sorted_array[sorted_array[:, 3] == 3]

                            priority_target[low_prio[:num_res, 0].astype(int)] = 1
                            priority_target[med_prio[:num_com, 0].astype(int)] = 2
                            priority_target[high_prio[:num_ind, 0].astype(int)] = 3

                        elif mode[0] == 2:
                            com_total = 10 
                            num_ind, num_com, num_res = math.floor(com_total * 0.40), math.floor(com_total * 0.30), math.floor(com_total * 0.30)
                            low_prio = sorted_array[sorted_array[:, 3] == 1]
                            med_prio = sorted_array[sorted_array[:, 3] == 2]
                            high_prio = sorted_array[sorted_array[:, 3] == 3]

                            priority_target[low_prio[:num_res, 0].astype(int)] = 1
                            priority_target[med_prio[:num_com, 0].astype(int)] = 2
                            priority_target[high_prio[:num_ind, 0].astype(int)] = 3


                        # Create edge index (remains constant for all time steps)
                        edge_index = np.vstack((static_data['line'][:, 0], static_data['line'][:, 1])).astype(int)
                        edge_features = static_data['line'][:, 2:5]  # Edge attributes (x, r, length)
                        
                        # Convert to torch tensors
                        edge_index = torch.tensor(edge_index, dtype=torch.long)
                        edge_features = torch.tensor(edge_features, dtype=torch.float)
                                           
                        #     # Convert to tensors
                        target = torch.tensor(priority_target, dtype=torch.long)  # Target as scalar float
                        node_features = torch.tensor(node_features, dtype=torch.float)  # Target as scalar float
                        
                        # Create the Data object
                        data = Data(
                            x=node_features,
                            edge_index=edge_index,
                            edge_attr=edge_features,
                            target = target,
                        )
                        
                        data_list.append(data)
                        
        # print(data_list)
        return data_list

# create_dataset("data/priority_dataset.h5")
