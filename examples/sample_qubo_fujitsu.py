"""
This example show how to sample a QUBO using neal.

To run this example:

    - run the `build_qubo.py` script using NO placeholder (i.e. if you change the options in the script,
        you will have to update this scripts as well)

"""

import sys
import logging
import pickle
import dimod
import json
import os
import tempfile
from os.path import join as path_join

from hepqpr.qallse import *

# === Fujitsu sampler
class FujitsuSampler:
    """ Fujitsu annealing sampler """
    def __init__(self):
        pass

    def sample_qubo(self, Q):
        info = {
            "beta_range": 0,
            "beta_schedule_type": 'aa'
        }
        dlets    = {}
        dletsInv = {}
        nn = 0
        for k, v in Q.items():
            for ele in k:
                if ele not in dlets:
                    dlets[ele] = nn
                    dletsInv[nn] = ele
                    nn = nn + 1

        # Run DA
        qubo_request = {}
        binary_polynomial = {}
        terms = []

        for k, v in Q.items():
            term = {}
            term['coefficient'] = v
            term['polynomials'] = [dlets[k[0]], dlets[k[1]]] if k[0] != k[1] else [dlets[k[0]]]
            terms.append(term)
        binary_polynomial['terms'] = terms

        fujitsuDAPT = {'number_iterations'    : 1000000,
                       'number_replicas'      : 100,
                       'offset_increase_rate' : 1000,
                       'solution_mode'        : 'COMPLETE'}
        guidance_config = {}
        for v in dlets.values():
            guidance_config[v] = False
        fujitsuDAPT['guidance_config'] = guidance_config
        qubo_request['binary_polynomial']   = binary_polynomial
        qubo_request['fujitsuDAPT']        = fujitsuDAPT;

        tmpfile = tempfile.NamedTemporaryFile(mode='w+t')
        json.dump(qubo_request, tmpfile)
        tmpfile.flush()
        command = "curl -H 'X-DA-Access-Key:APIKEY' -H 'Accept:application/json' -H 'Content-type:application/json' -X POST -d @" + \
                  tmpfile.name + " URL/v1/qubo/solve"
        try:
            ret = os.popen(command).read()
        finally:
            tmpfile.close()
        json_data = json.loads(ret)

        # Print result
        qubo_solution = json_data['qubo_solution']
        result_status = qubo_solution['result_status']
        solutions     = qubo_solution['solutions']
        timing        = qubo_solution['timing']
        print('===  Fujitsu DA results ===')
        print('result_status: %r' % result_status)
        print('--- timing ---')
        print('cpu_time: %s '           % timing['cpu_time'])
        print('queue_time: %s '         % timing['queue_time'])
        print('solve_time: %s '         % timing['solve_time'])
        print('total_elapsed_time: %s ' % timing['total_elapsed_time'])
        print('aneeal_time: %s '        % timing['detailed']['anneal_time'])

        samples = []
        energies = []
        for solution in solutions:
            print('--- solution ---')
            print('energy : %s' % solution['energy'])
            print('frequency : %s' % solution['frequency'])
            energies.append(solution['energy'])
            sample = {}
            for k, v in solution['configuration'].items():
                sample[dletsInv[int(k)]] = 1 if v else 0
            samples.append(sample)

        # Get response
        response = dimod.SampleSet.from_samples(
            samples,
            energy=energies,
            info=info,
            vartype=dimod.SPIN
        )
        return response

# ==== RUN CONFIG

loglevel = logging.DEBUG

input_path = '../mini1/event000001000-hits.csv'  # TODO change it !
qubo_path = '../mini1'  # TODO change it

sampler = FujitsuSampler()

# ==== configure logging

logging.basicConfig(
    stream=sys.stderr,
    format="%(asctime)s.%(msecs)03d [%(name)-15s %(levelname)-5s] %(message)s",
    datefmt='%Y-%m-%dT%H:%M:%S')

logging.getLogger('hepqpr').setLevel(loglevel)

# ==== build model

# load data
dw = DataWrapper.from_path(input_path)
with open(path_join(qubo_path, 'qubo.pickle'), 'rb') as f:
    Q = pickle.load(f)

# sample qubo
response = sampler.sample_qubo(Q)

# get the results
all_doublets = Qallse.process_sample(next(response.samples()))
final_tracks, final_doublets = TrackRecreaterD().process_results(all_doublets)

# compute stats
en0 = dw.compute_energy(Q)
en = response.record.energy[0]
occs = response.record.num_occurrences

p, r, ms = dw.compute_score(final_doublets)
trackml_score = dw.compute_trackml_score(final_tracks)

# print stats
print(f'SAMPLE -- energy: {en:.4f}, ideal: {en0:.4f} (diff: {en-en0:.6f})')
print(f'          best sample occurrence: {occs[0]}/{occs.sum()}')

print(f'SCORE  -- precision (%): {p * 100}, recall (%): {r * 100}, missing: {len(ms)}')
print(f'          tracks found: {len(final_tracks)}, trackml score (%): {trackml_score * 100}')
