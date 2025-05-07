"""
Source code for the paper: 
"SADPEA: Structure-aware dual probability evolutionary adaptive 
algorithm for the budgeted influence maximization problem"

Written by Haosen Wang, Guangzhou University
"""

import random
import algorithm2 as al
import social_network as soc
import time
import csv
import re

def evaluate_algorithm(graph, budget_range, filename, algorithm_func):
    results = []
    results.append(["budget", "influence", "runtime_seconds"])
    
    for i in range(*budget_range):
        budget = i * 10

        start_time = time.time()
        seed_set = algorithm_func(graph, budget, filename.split('.')[0])
        algorithm_runtime = time.time() - start_time
        
        # Measure influence spread
        spread_start = time.time()
        influence = soc.spread_model(graph, seed_set)
        spread_runtime = time.time() - spread_start
        
        # Store results
        results.append([budget, influence, algorithm_runtime])
    
    # Write results to CSV file
    with open(filename, 'w', newline='') as file:
        writer = csv.writer(file)
        for row in results:
            writer.writerow(row)
            
    return results

def work_0(graph, filename):
    """
    Evaluates the two-stage evolution simulated annealing algorithm.
    Args:
        graph: The social network graph
        filename: Output CSV filename
    """
    return evaluate_algorithm(
        graph=graph,
        budget_range=(1, 11),  # From 2 to 10 inclusive
        filename=filename,
        algorithm_func=al.two_stage_evolution_sa
    )

def main():
    # Dataset file paths
    datasets = {
        'ca-GrQc': 'dataset/ca-GrQc.mtx',               # CA-GrQC
        'email-univ': 'dataset/email-univ.edges',       # Email-un
        'ca-HepTh': 'dataset/ca-HepTh.mtx',             # CA-HepTh
        'p2p-Gnutella08': 'dataset/p2p-Gnutella08.mtx', # Gnutella
        'web-EPA': 'dataset/web-EPA.edges',             # Web-EPA
        'ca-CondMat': 'dataset/ca-CondMat.mtx'          # CondMat
    }
    
    print('Creating graphs from datasets...')

    # Load all networks
    graphs = {
        'ca-GrQc': soc.create_network_(datasets['ca-GrQc']),
        'email-univ': soc.create_network_(datasets['email-univ']),
        'ca-HepTh': soc.create_network_(datasets['ca-HepTh']),
        'p2p-Gnutella08': soc.create_network_(datasets['p2p-Gnutella08']),
        'web-EPA': soc.create_network_(datasets['web-EPA']),
        'ca-CondMat': soc.create_network_(datasets['ca-CondMat'])
    }

    # work_0(graphs['ca-GrQc'], 'ca-GrQc.csv')
    # work_0(graphs['email-univ'], 'email-univ.csv')
    work_0(graphs['ca-HepTh'], 'ca-HepTh.csv')
    # work_0(graphs['p2p-Gnutella08'], 'p2p-Gnutella08.csv')
    # work_0(graphs['web-EPA'], 'web-EPA.csv')
    # work_0(graphs['ca-CondMat'], 'ca-CondMat.csv')

if __name__ == "__main__":
    main()