import numpy as np
import time
import pandas as pd
import pandapower as pp
import pandapower.networks as nw
import networkx as nx
import matplotlib.pyplot as plt
from copy import deepcopy
import h5py
from torch.autograd import variable
from utils.constants import *
import random

# Constants
re_configuration = False 
network = nw.case33bw()
NUM_NETWORKS_TO_SIMULATE = 200

class PowerFlowSimulator:
    def __init__(self, net):
        self.net = net
        self.load_factors = pd.DataFrame(data=[COMMERCIAL_LOAD_FACTOR, INDUSTRIAL_LOAD_FACTOR, RESIDENTIAL_LOAD_FACTOR]).T
        self.season_factor = pd.DataFrame(data=[WINTER_LOAD_FACTOR]).T
        self.successful_nets = []
        self.all_results = {}
        self.original_loads = self.net.load[['p_mw', 'q_mvar']].copy()
        self.run_simulation()
        
    def run_simulation(self):
        while len(self.successful_nets) < NUM_NETWORKS_TO_SIMULATE:
            seasonal_results = self.simulate_loads()
            if seasonal_results:
                self.successful_nets.append(deepcopy(self.net))
                self.all_results[len(self.successful_nets)] = seasonal_results
                print(f"Successfully saved configuration {len(self.successful_nets)}.")
                self.plot_network(self.net, len(self.successful_nets))  # Plot each successful network
            else:
                print("Failed to converge for all seasons. Trying a new configuration...")

    def simulate_loads(self):
        # Drop not in service lines and not neccesary columns
        line_data = self.net.line[['from_bus', 'to_bus', 'length_km', 'r_ohm_per_km', 'x_ohm_per_km']]
        bus_data = self.net.bus[['vn_kv', 'max_vm_pu', 'min_vm_pu']]

        variable_NODE_TYPE = NODE_TYPE
        for i in range(len(NODE_TYPE)):
            if i > 0: 
                variable_NODE_TYPE[i] = random.choice(['industrial', 'residential', 'commercial'])

        vectorized_mapping = np.vectorize(CLASS_MAPPING.get)
        bus_data['node_type'] = vectorized_mapping(variable_NODE_TYPE)
        
        # Save network configuration data
        seasonal_results = {'network_config': {
            'line': deepcopy(line_data[self.net.line['in_service']].values),
            'bus': deepcopy(bus_data[self.net.bus['in_service']].values),
        }}
        print(seasonal_results)

        for season in self.season_factor.columns:
            time_step_results = {}
            for time_step in range(self.load_factors.shape[0]):
                self.reset_and_apply_loads(time_step, season, variable_NODE_TYPE)
                try:
                    pp.runpp(self.net, verbose=True, numba=False)
                    if np.any((self.net.res_bus.vm_pu < 0.8) | (self.net.res_bus.vm_pu > 1.1)):
                        print(f"Voltage out of bounds at time step {time_step}, season {season}. Simulation aborted for this step.")
                        return None  # Skip saving this time ste
                    lfa_results = {
                        'res_bus': deepcopy(self.net.res_bus.values),
                        'res_line': deepcopy(self.net.res_line[self.net.line['in_service']].values)
                    }
                    time_step_results[time_step] = lfa_results

                except pp.LoadflowNotConverged:
                    print(f'Load flow did not converge for time step {time_step}, season {season}.')
                    return None  # Terminate and return None to indicate failure

            seasonal_results[season] = time_step_results
        return seasonal_results

    def reset_and_apply_loads(self, time_step, season, variable_NODE_TYPE):
        # Reset loads to original before applying scaling factors
        self.net.load['p_mw'] = self.original_loads['p_mw']
        self.net.load['q_mvar'] = self.original_loads['q_mvar']

        ######### adding a "single season" based increase factor
        self.net.load.loc[variable_NODE_TYPE[1:] == 'residential', 'p_mw'] *= RESIDENTIAL_LOAD_FACTOR[time_step]
        self.net.load.loc[variable_NODE_TYPE[1:] == 'commerical', 'p_mw'] *= COMMERCIAL_LOAD_FACTOR[time_step]
        self.net.load.loc[variable_NODE_TYPE[1:] == 'industrial', 'p_mw'] *= INDUSTRIAL_LOAD_FACTOR[time_step]

        # scaling reactive power (starting from [1:] to avoid slack bus)
        self.net.load.loc[variable_NODE_TYPE[1:] == 'residential', 'q_mvar'] *= RESIDENTIAL_LOAD_FACTOR[time_step]
        self.net.load.loc[variable_NODE_TYPE[1:] == 'commerical', 'q_mvar'] *= COMMERCIAL_LOAD_FACTOR[time_step]
        self.net.load.loc[variable_NODE_TYPE[1:] == 'industrial', 'q_mvar'] *= INDUSTRIAL_LOAD_FACTOR[time_step]
        
        print('*' *  100)



    def plot_network(self, net, config_number):
        graph = pp.topology.create_nxgraph(net)
        pos = nx.spring_layout(graph, k=1, iterations=1001)
        plt.figure(figsize=(10, 6))
        nx.draw_networkx(graph, pos, with_labels=True, node_color='black', node_size=300, font_color='white')
        plt.title(f'Power Network Topology - Configuration {config_number}')
        plt.savefig(f'plots/Network_{config_number}', dpi=300)

    def save_results(self):
        with h5py.File('data/priority_dataset.h5', 'w') as f:
            for net_id, net_data in self.all_results.items():
                net_group = f.create_group(f'network_{net_id}')
                static_group = net_group.create_group('network_config')
                static_group.create_dataset('line', data=net_data['network_config']['line'])
                static_group.create_dataset('bus', data=net_data['network_config']['bus'])
                
                for season, time_step_data in net_data.items():
                    if season == 'network_config':
                        continue
                    season_group = net_group.create_group(f'season_{season}')
                    for time_step, results in time_step_data.items():
                        print(f'network id: {net_id}, time step: {time_step}, season: {season}')
                        time_step_group = season_group.create_group(f'time_step_{time_step}')
                        time_step_group.create_dataset('res_bus', data=results['res_bus'])
                        time_step_group.create_dataset('res_line', data=results['res_line'])

if __name__ == '__main__':
    simulator = PowerFlowSimulator(network)
    if simulator.successful_nets:
        simulator.save_results()

