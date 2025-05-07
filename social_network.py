import networkx as nx
import random
import numpy as np
import copy
import queue as que
import csv

def take_second(element):
    return element[1]

def create_network_(file, ic=0, cost=0):
    g = nx.DiGraph()
    if ic == 0:
        with open(file, "r") as f:
            edges = f.read().splitlines()
            random = np.random.RandomState(1)  # Random seed for reproducibility
            weights = random.uniform(0.001, 0.2, size=len(edges))
            for i in range(2, len(edges)):
                weight = weights[i]
                if edges[i].split()[0] == edges[i].split()[1]:
                    continue
                g.add_edge(edges[i].split()[0], edges[i].split()[1], weight=weight)
    elif ic == 1:
        with open(file, "r") as f:
            edges = f.read().splitlines()
            for i in range(2, len(edges)):
                weight = 0.01
                g.add_edge(edges[i].split()[0], edges[i].split()[1], weight=weight)
    else:
        with open(file, "r") as f:
            edges = f.read().splitlines()
            random = np.random.RandomState(1)  # Random seed for reproducibility
            weight_p_s = random.randint(1, 3, size=len(edges))
            for i in range(2, len(edges)):
                weight_p = weight_p_s[i]
                weight = 0.1 ** weight_p
                g.add_edge(edges[i].split()[0], edges[i].split()[1], weight=weight)

    if cost == 0:
        random = np.random.RandomState(1)  # Random seed for reproducibility
        node_costs = random.uniform(1, 3, size=g.number_of_nodes())
        for i, node in enumerate(g.nodes()):
            node_cost = node_costs[i]
            g.nodes[node]['weight'] = node_cost
    else:
        for node in g.nodes():
            node_cost = 1 + copy.deepcopy(g.degree(node)) * 0.01
            g.nodes[node]['weight'] = node_cost

    return g


def spread_model(g, seed):
    """
    Influence diffusion model
    """
    mcs_num = 10000
    active_node_num = 0
    while mcs_num:
        active_pass_set = set()
        active_temp = que.Queue()
        for seed_node in seed:
            active_temp.put(seed_node)
            active_pass_set.add(seed_node)

        while not active_temp.empty():
            choice_node = active_temp.get()
            for choice_neighbor in g.neighbors(choice_node):
                if choice_neighbor not in active_pass_set:
                    random_probability = random.random()
                    if random_probability < g.edges[choice_node, choice_neighbor]['weight']:
                        active_temp.put(choice_neighbor)
                        active_pass_set.add(choice_neighbor)

        mcs_num -= 1
        active_node_num += len(active_pass_set)

    print("budget:", budget_detection(g, seed))
    print("influence spread:", active_node_num / 10000)
    return active_node_num / 10000


def budget_detection(g, seed):
    """Calculates total budget cost of seed set"""
    budget = 0
    for node in seed:
        budget += g.nodes[node]['weight']

    return budget


def low_bound_cost(g, pool):
    """Detects lower bound cost for nodes in pool"""
    budget_rank = list()
    for node in pool:
        budget_rank.append((node, g.nodes[node]['weight']))

    budget_rank.sort(key=take_second, reverse=False)

    if len(budget_rank) <= 20:
        return budget_rank[0][1]
    else:
        min_budget = 0
        min_len = int(0.05 * len(budget_rank))
        for i in range(min_len):
            min_budget += budget_rank[i][1]

        return min_budget / min_len


def spread_model_0(g, seed, filename):
    """
    Influence diffusion model with node labeling
    
    Args:
        g: Social network graph
        seed: Set of seed nodes
        filename: Output CSV file path
    
    Returns:
        Number of activated nodes
    """
    label_0 = seed
    label_1 = set()
    mcs_num = 1
    active_node_num = 0
    while mcs_num:
        active_pass_set = set()
        active_temp = que.Queue()
        for seed_node in seed:
            active_temp.put(seed_node)
            active_pass_set.add(seed_node)

        while not active_temp.empty():
            choice_node = active_temp.get()
            for choice_neighbor in g.neighbors(choice_node):
                if choice_neighbor not in active_pass_set:
                    random_probability = random.random()
                    if random_probability < g.edges[choice_node, choice_neighbor]['weight']:
                        active_temp.put(choice_neighbor)
                        active_pass_set.add(choice_neighbor)

        mcs_num -= 1
        active_node_num += len(active_pass_set)
        label_1 = active_pass_set

    label_2 = set()
    for node in label_1:
        label_2 = label_2.union(set(g.neighbors(node)))

    label_node = [' ']
    for node in label_0:
        label_node.append((node, 0))

    for node in label_1:
        label_node.append((node, 1))

    for node in label_2:
        if node not in label_node:
            label_node.append((node, 2))

    label_node = label_node[:201]

    result = list()
    result.append(label_node)

    for n_l in label_node[1:]:
        l1 = [n_l[0]]
        for n_l_2 in label_node[1:]:
            if g.has_edge(n_l[0], n_l_2[0]):
                l1.append(1)
            else:
                l1.append(0)
        result.append(l1)

    with open(filename, 'w', newline='') as file:
        writer = csv.writer(file)

        # Write data
        for row in result:
            writer.writerow(row)

    return active_node_num


def spread_model_1(g, seed, filename):
    """
    Influence diffusion model with node attributes
    
    Args:
        g: Social network graph
        seed: Set of seed nodes
        filename: Output CSV file path
    
    Returns:
        Set of influenced nodes (excluding seed nodes)
    """
    label_0 = seed
    label_1 = set()
    mcs_num = 1
    active_node_num = 0
    while mcs_num:
        active_pass_set = set()
        active_temp = que.Queue()
        for seed_node in seed:
            active_temp.put(seed_node)
            active_pass_set.add(seed_node)

        while not active_temp.empty():
            choice_node = active_temp.get()
            for choice_neighbor in g.neighbors(choice_node):
                if choice_neighbor not in active_pass_set:
                    random_probability = random.random()
                    if random_probability < g.edges[choice_node, choice_neighbor]['weight']:
                        active_temp.put(choice_neighbor)
                        active_pass_set.add(choice_neighbor)

        mcs_num -= 1
        active_node_num += len(active_pass_set)
        label_1 = active_pass_set

    label_1 = label_1.difference(set(label_0))

    label_2 = set()
    for node in label_1:
        label_2 = label_2.union(set(g.neighbors(node)))

    label_node = [' ']
    for node in label_0:
        label_node.append((node, 0))

    for node in label_1:
        label_node.append((node, 1))

    result = list()
    result.append(["node", "label"])

    for n_l in label_node[1:]:
        result.append([n_l[0], n_l[1], g.nodes[n_l[0]]['weight']])

    with open(filename, 'w', newline='') as file:
        writer = csv.writer(file)

        # Write data
        for row in result:
            writer.writerow(row)

    return label_1