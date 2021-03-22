# Author: Awal Awal
# Date: Mar 2021
# Email: awal.nova@gmail.com

import numpy as np
import os
import uproot
from sificc_lib import Simulation, Event, utils
import argparse

def main():
    # declare the script parameters and the default values
    parser = argparse.ArgumentParser(description='Evaluates the reconstructed events for the SiFi-CC')
    parser.add_argument('-f', '--file', required=True, type=str, 
                        help='The root file containing the reconstructed events')
    parser.add_argument('-s', '--source', default='.', type=str, metavar='PATH',
                        help='The path of the source simulation root file. Default path is ./')
    parser.add_argument('--e_pos_x', default=2.6, type=float, metavar='VALUE',
                        help='The distance limit for the x-axis of the electon. Default is 2.6 mm')
    parser.add_argument('--e_pos_y', default=10, type=float, metavar='VALUE',
                        help='The distance limit for the y-axis of the electon. Default is 10 mm')
    parser.add_argument('--e_pos_z', default=2.6, type=float, metavar='VALUE',
                        help='The distance limit for the z-axis of the electon. Default is 2.6 mm')
    parser.add_argument('--p_pos_x', default=2.6, type=float, metavar='VALUE',
                        help='The distance limit for the x-axis of the photon. Default is 2.6 mm')
    parser.add_argument('--p_pos_y', default=10, type=float, metavar='VALUE',
                        help='The distance limit for the y-axis of the photon. Default is 10 mm')
    parser.add_argument('--p_pos_z', default=2.6, type=float, metavar='VALUE',
                        help='The distance limit for the z-axis of the photon. Default is 2.6 mm')
    parser.add_argument('--e_energy', default=.12, type=float, metavar='VALUE',
                        help='The energy difference limit relative to the electron energy. Default is .12')
    parser.add_argument('--p_energy', default=.12, type=float, metavar='VALUE',
                        help='The energy difference limit relative to the photon energy. Default is .12')
    args = parser.parse_args()

    # open the reconstruction root file
    try:
        recon_file = uproot.open(args.file)
    except Exception as e:
        print('Error:', str(e))

    stat_tt = recon_file[b'TreeStat;1']
    cone_tt = recon_file[b'ConeList;1']

    # read the evaluation settings
    np.warnings.filterwarnings('ignore', category=np.VisibleDeprecationWarning)
    if 'InputFilename' in stat_tt:
        source_path = os.path.join(args.source, str(stat_tt['InputFilename'].array()[0]))
    else:
        source_path = args.source
    start_entry = stat_tt['StartEvent'].array()[0]
    stop_entry = stat_tt['StopEvent'].array()[0]+1
    n_events = stat_tt['TotalSimNev'].array()[0]

    # assert the difference between entries = number of events
    assert stop_entry - start_entry == n_events , \
            'The field TreeStat/TotalSimNev does not match with difference' \
            ' between StartEvent & StopEvent'

    # read the events sequence
    l_entries = cone_tt.array('GlobalEventNumber')

    # assert there is no duplicated reconstructed event
    assert l_entries.shape[0] == np.unique(l_entries).shape[0], \
            'There are duplicated entries in "{}"'.format(recon_path)

    # read the electron positions
    e_pos = np.concatenate([
        -cone_tt.array('z_1').reshape((-1,1)),
        cone_tt.array('x_1').reshape((-1,1)),
        -cone_tt.array('y_1').reshape((-1,1)),
    ], axis=1)

    # read the photon positions
    p_pos = np.concatenate([
        -cone_tt.array('z_2').reshape((-1,1)),
        cone_tt.array('x_2').reshape((-1,1)),
        -cone_tt.array('y_2').reshape((-1,1)),
    ], axis=1)

    # read the energies of the electron and photon
    e_energy = cone_tt.array('E1').reshape((-1,1))
    p_energy = cone_tt.array('E2').reshape((-1,1))

    # open the simulation root file
    try:
        simulation = Simulation(source_path)
    except Exception as e:
        print('Error:', str(e))


    entry = start_entry
    l_euc_diff = []
    valid_events = 0

    l_e_pos_x_matches = []
    l_e_pos_y_matches = []
    l_e_pos_z_matches = []
    l_p_pos_x_matches = []
    l_p_pos_y_matches = []
    l_p_pos_z_matches = []
    l_e_enrg_matches = []
    l_p_enrg_matches = []
    l_matches = []

    # iterate through the evaluated events from the simulation root file
    for event in simulation.iterate_events(entry_start=start_entry, entry_stop=stop_entry, 
                                           bar_update_size=1, desc='evaluating recons.'):
        # any event with at least one cluster in both modules is a valid event
        if event.is_distributed_clusters:
            valid_events += 1

            # check if the event is an ideal Compton event
            if event.is_ideal_compton:

                # find the Compton event exists the reconstructed events
                event_location = np.where(np.equal(l_entries, entry))[0]

                # if no matches within the reconstructed events, it's a mismatch
                if event_location.shape[0] == 0:
                    l_matches.append(0)
                # else it exists within the reconstructed events
                else:
                    # get the location of the reconstructed event
                    recon_event_no = event_location[0]

                    # add the euclidean distance between the real and reconstructed positions
                    l_euc_diff.append(utils.euclidean_distance_np(utils.vec_as_np(event.real_e_position).reshape((1,-1)), 
                                                                  e_pos[recon_event_no].reshape((1,-1)))[0])
                    l_euc_diff.append(utils.euclidean_distance_np(utils.vec_as_np(event.real_p_position).reshape((1,-1)), 
                                                                  p_pos[recon_event_no].reshape((1,-1)))[0])

                    # validate all the parameters of the event
                    e_pos_x_matche = np.abs(event.real_e_position.x - e_pos[recon_event_no][0]) <= args.e_pos_x
                    e_pos_y_matche = np.abs(event.real_e_position.y - e_pos[recon_event_no][1]) <= args.e_pos_y
                    e_pos_z_matche = np.abs(event.real_e_position.z - e_pos[recon_event_no][2]) <= args.e_pos_z
                    p_pos_x_matche = np.abs(event.real_p_position.x - p_pos[recon_event_no][0]) <= args.p_pos_x
                    p_pos_y_matche = np.abs(event.real_p_position.y - p_pos[recon_event_no][1]) <= args.p_pos_y
                    p_pos_z_matche = np.abs(event.real_p_position.z - p_pos[recon_event_no][2]) <= args.p_pos_z
                    e_enrg_matche = np.abs(event.real_e_energy - e_energy[recon_event_no]) <= \
                                    (event.real_e_energy * args.e_energy)
                    p_enrg_matche = np.abs(event.real_p_energy - p_energy[recon_event_no]) <= \
                                    (event.real_p_energy * args.p_energy)

                    # store the validation parameters
                    l_e_pos_x_matches.append(e_pos_x_matche)
                    l_e_pos_y_matches.append(e_pos_y_matche)
                    l_e_pos_z_matches.append(e_pos_z_matche)
                    l_p_pos_x_matches.append(p_pos_x_matche)
                    l_p_pos_y_matches.append(p_pos_y_matche)
                    l_p_pos_z_matches.append(p_pos_z_matche)
                    l_e_enrg_matches.append(e_enrg_matche)
                    l_p_enrg_matches.append(p_enrg_matche)

                    # check if all the validation parameters are met
                    if e_pos_x_matche and e_pos_y_matche and e_pos_z_matche \
                            and p_pos_x_matche and p_pos_y_matche and p_pos_z_matche \
                            and e_enrg_matche and p_enrg_matche:
                        l_matches.append(1)
                    else:
                        # if any parameter is not met, it's a mismatch
                        l_matches.append(0)

        entry += 1

    # print the evaluations
    print()
    print('Processed events: {:10,d}'.format(n_events))
    print('Valid events:     {:10,d}'.format(valid_events))
    print('Compton events:   {:10,d}'.format(len(l_matches)))
    print('Recon. Comptons:  {:10,d}\n'.format(len(l_entries)))
    print('Matches:  {:9,d}'.format(sum(l_matches)))
    print('  Efficiency: {:5.3f}'.format(sum(l_matches)/len(l_matches)))
    print('  Purity:     {:5.3f}'.format(sum(l_matches)/len(l_entries)))
    print('  Euc mean:  {:6.2f} mm'.format(np.mean(l_euc_diff)))
    print('  Euc std:   {:6.2f} mm\n'.format(np.std(l_euc_diff)))
    print('Mismatches breakdown')
    print('  e_pos_x :{:8,d}'.format(len(l_entries)-np.sum(l_e_pos_x_matches)))
    print('  e_pos_y :{:8,d}'.format(len(l_entries)-np.sum(l_e_pos_y_matches)))
    print('  e_pos_z :{:8,d}'.format(len(l_entries)-np.sum(l_e_pos_z_matches)))
    print('  p_pos_x :{:8,d}'.format(len(l_entries)-np.sum(l_p_pos_x_matches)))
    print('  p_pos_y :{:8,d}'.format(len(l_entries)-np.sum(l_p_pos_y_matches)))
    print('  p_pos_z :{:8,d}'.format(len(l_entries)-np.sum(l_p_pos_z_matches)))
    print('  e_energy:{:8,d}'.format(len(l_entries)-np.sum(l_e_enrg_matches)))
    print('  p_energy:{:8,d}'.format(len(l_entries)-np.sum(l_p_enrg_matches)))

if __name__ == '__main__':
    main()