import numpy as np
import torch
from torch_geometric.data import Data
import h5py
import time
import matplotlib.pyplot as plt
import pandas as pd

def load_network_data(file, network_key):
    with h5py.File(file, 'r') as f:
        net_group = f[network_key]
        static_data = {
            'line': net_group['network_config/line'][:],
            'bus': net_group['network_config/bus'][:],
            'pv_potential': net_group['network_config/pv_potential'][:]
        }
    return static_data

def create_dataset(file, seq_length=24):
    data_list = []

    with h5py.File(file, 'r') as f:
        network_keys = [key for key in f.keys() if key.startswith('network_')]
        
        for network_key in network_keys:
            static_data = load_network_data(file, network_key)
            net_group = f[network_key]
            season_keys = ['season_0', 'season_1', 'season_2', 'season_3']
            features_season = []
            load_sums_season = []
            pv_potential_season = []
            bus_sensitivity_season = []

            for season_key in season_keys:
                season = int(season_key.split('_')[-1])
                # start_col = season * 4
                # end_col = (season + 1) * 4

                if season_key in net_group:
                    season_group = net_group[season_key]
                    features_time_step = []
                    load_sums_time_step = []
                    grid_req_time_step = []
                    sensitivity_time_step = []

                    for time_step_key in sorted(season_group.keys(), key=lambda x: int(x.split('_')[-1])):
                        time_step_group = season_group[time_step_key]
                        time_step = int(time_step_key.split('_')[-1])
                        
                        # target values
                        target_class = []
                        sensitivity_t = time_step_group['res_sensitivity'][:]

                        pv_potential_t = static_data['pv_potential'][time_step, season::4]

                        # Extract node features 
                        loads = time_step_group['res_bus'][:, 2:4]  
                        loads = np.where(loads > 0, loads, 0) 
                        pv_potential_buses = np.tile(pv_potential_t, (33,1))
                        # for LSF calculation
                        delta_p = np.tile(0.1, (33,1))
                        # for Size calculation
                        grid_factor = static_data['bus'][:, 4:5]  
                        # Stack node features
                        node_features = np.hstack((loads, grid_factor, delta_p, pv_potential_buses))
                        # print(node_features)


                        # Calculate load sum (p) scaled by grid factor
                        load_sum = np.sum(loads)  # Single scalar value
                        grid_required = np.sum(loads * grid_factor)  # Single scalar value
                        
                        features_time_step.append(node_features)
                        load_sums_time_step.append(load_sum)
                        grid_req_time_step.append(grid_required)
                        sensitivity_time_step.append(sensitivity_t)

                remain_load = np.array(load_sums_time_step) - np.array(grid_req_time_step)
                max_pv_potentials = np.max(static_data['pv_potential'][:, season::4].T, axis=1)
                indices_max_pv_potentials = np.argmax(static_data['pv_potential'][:, season::4].T, axis=1)

                for i in range(indices_max_pv_potentials.shape[0]):
                    load_pv_potential = remain_load[indices_max_pv_potentials[i]] / max_pv_potentials[i]
                    pv_potential_season.append(load_pv_potential)
                load_sums_season.append(load_sums_time_step)
                features_season.append(features_time_step) 
                bus_mean_sensitivity = np.mean(np.array(sensitivity_time_step), axis=0)
                bus_sensitivity_season.append(bus_mean_sensitivity)
                
            pv_target = np.divide(pv_potential_season , 100)
            bus_target = static_data['bus'][:,3]
            lsf_target = np.mean(np.array(bus_sensitivity_season), axis = 0)

            features_season = np.stack(features_season, axis=0)
            seasons = 4
            num_nodes= 33
            node_features = 8 
            features_season = features_season.reshape((seq_length*seasons), num_nodes, node_features)

            # Create edge index (remains constant for all time steps)
            edge_index = np.vstack((static_data['line'][:, 0], static_data['line'][:, 1])).astype(int)
            edge_features = static_data['line'][:, 2:5]  # Edge attributes (x, r, length)

            # Convert to torch tensors
            edge_index = torch.tensor(edge_index, dtype=torch.long)
            edge_features = torch.tensor(edge_features, dtype=torch.float)
                                
            #     # Convert to tensors
            node_feature_sequence = torch.tensor(features_season, dtype=torch.float)
            pv_target = torch.tensor(pv_target, dtype=torch.float)  # Target as scalar float
            bus_target = torch.tensor(bus_target, dtype=torch.long)  # Target as scalar float
            lsf_target = torch.tensor(lsf_target, dtype=torch.float)  # Target as scalar float

            # Create the Data object
            data = Data(
                x=node_feature_sequence,
                edge_index=edge_index,
                edge_attr=edge_features,
                target = bus_target,
            )
            
            data_list.append(data)

        return data_list

# create_dataset('data/dataset_MAMSTGAT.h5')
