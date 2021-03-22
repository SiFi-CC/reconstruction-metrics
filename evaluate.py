# Author: Awal Awal
# Date: Mar 2021
# Email: awal.nova@gmail.com

import numpy as np
import os
import uproot
from sificc_lib import Simulation, Event, utils
import argparse

parser = argparse.ArgumentParser(description='Evaluates the reconstructed events for the SiFi-CC')
parser.add_argument('-f', '--file', required=True, type=str, 
                    help='The root file containing the reconstructed events')
parser.add_argument('-d', '--source_dir', default='.', type=str, 
                    help='Directory of the source simulation root file. Default directory is ./')
args = parser.parse_args()

try:
    recon_file = uproot.open(args.file)
except Exception as e:
    print('Error:', str(e))

stat_tt = recon_file[b'TreeStat;1']
cone_tt = recon_file[b'ConeList;1']

np.warnings.filterwarnings('ignore', category=np.VisibleDeprecationWarning)
source_path = os.path.join(args.source_dir, str(stat_tt['InputFilename'].array()[0]))
start_entry = stat_tt['StartEvent'].array()[0]
stop_entry = stat_tt['StopEvent'].array()[0]+1
n_events = stat_tt['TotalSimNev'].array()[0]

assert stop_entry - start_entry == n_events , \
        'The field TreeStat/TotalSimNev does not match with difference' \
        ' between StartEvent & StopEvent'

l_entries = cone_tt.array('GlobalEventNumber')
assert l_entries.shape[0] == np.unique(l_entries).shape[0], \
        'There are duplicated entries in "{}"'.format(recon_path)

e_pos = np.concatenate([
    cone_tt.array('x_1').reshape((-1,1)),
    cone_tt.array('y_1').reshape((-1,1)),
    cone_tt.array('z_1').reshape((-1,1)),
], axis=1)

p_pos = np.concatenate([
    cone_tt.array('x_2').reshape((-1,1)),
    cone_tt.array('y_2').reshape((-1,1)),
    cone_tt.array('z_2').reshape((-1,1)),
], axis=1)

e_energy = cone_tt.array('E1').reshape((-1,1))

p_energy = cone_tt.array('E2').reshape((-1,1))

try:
    simulation = Simulation(source_path)
except Exception as e:
    print('Error:', str(e))
    

entry = start_entry
l_matches = []
l_euc_diff = []
valid_events = 0

for event in simulation.iterate_events(entry_start=start_entry, entry_stop=stop_entry, 
                                       bar_update_size=1, desc='evaluating recons.'):
    if event.is_distributed_clusters:
        valid_events += 1
        
        if event.is_ideal_compton:
            event_location = np.where(np.equal(l_entries, entry))[0]
            if event_location.shape[0] == 0:
                l_matches.append(0)
                
            else:
                recon_event_no = event_location[0]

                l_euc_diff.append(utils.euclidean_distance_np(utils.vec_as_np(event.real_e_position).reshape((1,-1)), 
                                                              e_pos[recon_event_no].reshape((1,-1)))[0])
                l_euc_diff.append(utils.euclidean_distance_np(utils.vec_as_np(event.real_p_position).reshape((1,-1)), 
                                                              p_pos[recon_event_no].reshape((1,-1)))[0])
                
                if np.abs(event.real_e_position.x - e_pos[recon_event_no][0]) <= 2.6 \
                        and np.abs(event.real_e_position.y - e_pos[recon_event_no][1]) <= 10 \
                        and np.abs(event.real_e_position.z - e_pos[recon_event_no][2]) <= 2.6 \
                        and np.abs(event.real_p_position.x - p_pos[recon_event_no][0]) <= 2.6 \
                        and np.abs(event.real_p_position.y - p_pos[recon_event_no][1]) <= 10 \
                        and np.abs(event.real_p_position.z - p_pos[recon_event_no][2]) <= 2.6 \
                        and np.abs(event.real_e_energy - e_energy) <= (event.real_e_energy * .12) \
                        and np.abs(event.real_p_energy - p_energy) <= (event.real_p_energy * .12):
                    l_matches.append(1)
                else:
                    l_matches.append(0)
        
    entry += 1
    
print()
print('Processed events: {}'.format(n_events))
print('Valid events:     {}'.format(valid_events))
print('Compton events:   {}'.format(len(l_matches)))
print('Matches:          {}'.format(sum(l_matches)))
print('Efficiency        {:.3f}'.format(sum(l_matches)/len(l_matches)))
print('Purity:           {:.3f}'.format(sum(l_matches)/len(l_entries)))
print('Euc mean:         {:.2f} mm'.format(np.mean(l_euc_diff)))
print('Euc std:          {:.2f} mm'.format(np.std(l_euc_diff)))
