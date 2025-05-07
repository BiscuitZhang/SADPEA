import copy
import numpy.random
import random
import math
import networkx as nx
from node2vec import Node2Vec
import gensim

def take_second(element):
    """Returns the second element of a tuple for sorting"""
    x = element[1]
    return x

def swap(list1, list2, x, y):
    """Swaps elements between two lists at given positions"""
    temp = list1[x]
    list1[x] = list2[y]
    list2[y] = temp

def fitness_function_3(g, seed, average=0.09995, variance=0.01):
    """
    Fitness function to evaluate influence spread of seed nodes
    
    Args:
        g: The network graph
        seed: Set of seed nodes
        average: Average activation probability
        variance: Variance of activation probability
    
    Returns:
        Estimated influence spread value
    """
    influence_spread = 0
    neighbor_dict = dict()
    for node in seed:
        influence_spread += 1
        for edge in g.out_edges(node):
            if edge[1] in neighbor_dict:
                neighbor_dict[edge[1]].append(g.edges[node, edge[1]]['weight'])
            else:
                neighbor_dict[edge[1]] = [g.edges[node, edge[1]]['weight']]

    for key, value in neighbor_dict.items():
        neighbor_influence_spread = 1
        for weight in value:
            neighbor_influence_spread *= 1 - weight

        neighbor_influence_spread = 1 - neighbor_influence_spread
        activate_weight = numpy.random.normal(average, variance)
        while activate_weight < 0.001 or activate_weight > 0.2:
            activate_weight = numpy.random.normal(average, variance)
        neighbor_influence_spread *= 1 + g.out_degree(key) * activate_weight
        influence_spread += neighbor_influence_spread

    return influence_spread


def low_bound_cost(g, pool):
    """
    Lower bound cost detection function for nodes
    
    Args:
        g: The network graph
        pool: Pool of candidate nodes
    
    Returns:
        Minimum activation cost estimate
    """
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


def adaptive_simulated_annealing(g, b, seed, pool):
    """
    Adaptive temperature adjustment simulated annealing algorithm
    
    Args:
        g: The network graph
        b: Budget constraint
        seed: Initial seed set
        pool: Pool of candidate nodes
    
    Returns:
        Optimized seed set
    """
    # Initialize parameters: final temperature t_f, current temperature t_t, 
    # cooling coefficient theta, iterations per temperature n
    t_f = 20
    t_t = 2000
    op1_defeat_num = 0
    op1_success_num = 0
    op2_defeat_num = 0
    op2_success_num = 0
    old_success_num = 0
    theta = 5
    n = 20
    op = 0.5
    seed = list(seed)
    pool = list(pool)
    r = 0

    # Step 1: Check seed budget and supplement with seed nodes
    budget = 0
    for node in seed:
        budget += g.nodes[node]['weight']

    min_node_budget = low_bound_cost(g, pool)
    random.shuffle(pool)
    for node in pool:
        if b - budget >= min_node_budget:
            break
        if node not in seed and g.nodes[node]['weight'] < b - budget:
            seed.append(node)
            budget += g.nodes[node]['weight']

    # Step 2: Adaptive temperature adjustment with dual probability pools
    while t_t > t_f:
        # Sort seed by cost
        seed_budget_rank = [[node, g.nodes[node]['weight']] for node in seed]
        seed_budget_rank.sort(key=lambda x: x[1], reverse=True)
        seed = [x[0] for x in seed_budget_rank]

        seed_spread = fitness_function_3(g, set(seed))
        for i in range(n):
            p_op = random.random()
            if p_op < op:
                node_x = random.randint(0, int(len(seed) / 2))
                node_y = random.randint(0, len(pool) - 1)
                while g.nodes[pool[node_y]]['weight'] - g.nodes[seed[node_x]]['weight'] >= b - budget \
                        or pool[node_y] in seed:
                    node_x = random.randint(0, int(len(seed) / 2))
                    node_y = random.randint(0, len(pool) - 1)
                swap(seed, pool, node_x, node_y)
                seed_prime_spread = fitness_function_3(g, set(seed))

                if seed_prime_spread > seed_spread:
                    r = 0
                    op1_success_num += 1
                    budget = budget - g.nodes[pool[node_y]]['weight'] + g.nodes[seed[node_x]]['weight']
                else:
                    r += 1
                    op1_defeat_num += 1
                    swap(seed, pool, node_x, node_y)
            else:
                node_x = random.randint(int(len(seed) / 2), len(seed) - 1)
                node_y = random.randint(0, len(pool) - 1)
                while g.nodes[pool[node_y]]['weight'] - g.nodes[seed[node_x]]['weight'] >= b - budget \
                        or pool[node_y] in seed:
                    node_x = random.randint(int(len(seed) / 2), len(seed) - 1)
                    node_y = random.randint(0, len(pool) - 1)
                swap(seed, pool, node_x, node_y)
                seed_prime_spread = fitness_function_3(g, set(seed))

                if seed_prime_spread > seed_spread:
                    r = 0
                    op2_success_num += 1
                    budget = budget - g.nodes[pool[node_y]]['weight'] + g.nodes[seed[node_x]]['weight']
                else:
                    r += 1
                    op2_defeat_num += 1
                    swap(seed, pool, node_x, node_y)

        # If successful search for 10 nodes, perform cost detection
        if op1_success_num + op2_success_num - old_success_num > 10:
            budget = 0
            for node in seed:
                budget += g.nodes[node]['weight']

            min_node_budget = low_bound_cost(g, pool)
            random.shuffle(pool)
            for node in pool:
                if b - budget < min_node_budget:
                    break
                if node not in seed and g.nodes[node]['weight'] < b - budget:
                    seed.append(node)
                    budget += g.nodes[node]['weight']

            old_success_num = op1_success_num + op2_success_num

        # Update temperature
        t_t -= theta * math.log(r + 1)

        # Update operation probability
        op_1 = op1_success_num / (op1_success_num + op1_defeat_num) + 0.0001
        op_2 = op2_success_num / (op2_success_num + op2_defeat_num) + 0.0001
        op = op_1 / (op_1 + op_2)

    # Final cost check and node supplementation
    budget = 0
    for node in seed:
        budget += g.nodes[node]['weight']

    min_node_budget = low_bound_cost(g, pool)
    random.shuffle(pool)
    for node in pool:
        if b - budget >= min_node_budget:
            break
        if node not in seed and g.nodes[node]['weight'] < b - budget:
            seed.append(node)
            budget += g.nodes[node]['weight']

    return seed


def two_hop_similar_degree(g, node_emd_dict, factor, average=0.09995, variance=0.01):
    """
    Heuristic algorithm for evaluating node influence
    Two-hop neighborhood mixed influence assessment algorithm
    
    Args:
        g: The network graph
        node_emd_dict: Node embedding dictionary
        factor: Weighting factor
        average: Average of random weight distribution function
        variance: Variance of random weight distribution function
    
    Returns:
        Dictionary of node influence scores
    """
    temp_g = copy.deepcopy(g)
    node_dict = dict()
    mdd_dict = dict()

    # Calculate two-hop mixed influence for all nodes
    for node in g.nodes():
        # Calculate mixed influence spread of node's two-hop neighbors
        mixed_influence_spread = 0
        node_neighbors = set()
        activate_weight = numpy.random.normal(average, variance)
        while activate_weight < 0.001 or activate_weight > 0.2:
            activate_weight = numpy.random.normal(average, variance)
        for neighbor in g.neighbors(node):
            mixed_influence_spread += 1
            mixed_influence_spread += activate_weight * node_emd_dict[neighbor]
            node_neighbors.add(neighbor)
        
        node_dict[node] = [mixed_influence_spread, node_neighbors]

    mdd_dict = {}

    for node in g.nodes():
        mixed_influence_spread = node_dict[node][0]
        one_hop_nodes = node_dict[node][1]
        for one_hop_node in one_hop_nodes:
            activate_weight = numpy.random.normal(average, variance)
            while activate_weight < 0.001 or activate_weight > 0.2:
                activate_weight = numpy.random.normal(average, variance)
            for neighbor in g.neighbors(one_hop_node):
                mixed_influence_spread += 1
                mixed_influence_spread += activate_weight * node_emd_dict[neighbor]
        
        mdd_dict[node] = mixed_influence_spread

    return mdd_dict


def cost_greedy(g, node_dict, b):
    """
    Constructs seed set based on heuristic dict and budget constraint
    
    Args:
        g: The network graph
        node_dict: Dictionary of node influence scores
        b: Budget constraint
    
    Returns:
        Seed set
    """
    node_list = list()
    for key, value in node_dict.items():
        node_list.append((key, value / g.nodes[key]['weight']))
    node_list.sort(key=lambda x: x[1], reverse=True)
    seed = set()
    for node in node_list:
        if g.nodes[node[0]]['weight'] < b:
            seed.add(node[0])
            b -= g.nodes[node[0]]['weight']

        if b < 1:
            break

    return seed


def node_embedding_sort(g, filename):
    """
    Creates and sorts node embeddings based on similarity
    
    Args:
        g: The network graph
        filename: Name of the dataset file
    
    Returns:
        Dictionary of node similarity scores
    """
    model = gensim.models.Word2Vec.load('./dataset/'+filename+'_model') 
    nodes = g.nodes
    sim_dict = {}
    for i in nodes:
        sim_dict[i] = 0
    for node in nodes:
        similar_nodes = model.wv.most_similar(node, topn=20)
        for s_node in similar_nodes:
            s_n = s_node[0]
            if s_node[1] > 0.6:
                sim_dict[s_n] += 1
        
    return sim_dict


def two_hop_mixed_degree_decomposition(g, factor, average=0.09995, variance=0.01):
    """
    Heuristic algorithm for evaluating node influence
    Two-hop mixed degree decomposition influence assessment algorithm
    
    Args:
        g: The network graph
        factor: Weighting factor
        average: Average of random weight distribution function
        variance: Variance of random weight distribution function
    
    Returns:
        Dictionary of MDD influence scores
    """
    # Copy the graph
    temp_g = copy.deepcopy(g)
    # Node calculation dictionary
    node_dict = dict()
    # MDD value dictionary
    mdd_dict = dict()

    # Add all nodes' degrees to initial dictionary
    for node in g.nodes():
        # Random activation probability for calculating two-hop neighbor influence
        activate_weight = numpy.random.normal(average, variance)
        while activate_weight < 0.001 or activate_weight > 0.2:
            activate_weight = numpy.random.normal(average, variance)

        # Calculate updated MDD value
        node_index = 0
        for neighbor in nx.neighbors(g, node):
            if neighbor in g.nodes:
                node_index += 1
                node_index += activate_weight * g.degree(neighbor)
        node_dict[node] = [node_index, set(nx.neighbors(g, node))]

    # Initialize threshold to 1
    rank = 1
    # Set of nodes less than or equal to threshold
    little_node_list = list()
    # Set of nodes whose MDD values change after node deletion
    little_node_neighbor = set()
    # Start iteration
    while len(mdd_dict) < len(g.nodes()):

        # Traverse existing nodes, add those below threshold to set and set MDD value
        # Add their neighbors to the change set
        for key, value in node_dict.items():
            if value[0] <= rank:
                mdd_dict[key] = value[0]
                little_node_list.append(key)
                little_node_neighbor = little_node_neighbor.union(value[1])
        little_node_neighbor = little_node_neighbor.difference(little_node_list)

        for del_node in little_node_list:
            node_dict.pop(del_node)

        # Check if any nodes were added to MDD dictionary
        if len(little_node_list):
            temp_g.remove_nodes_from(little_node_list)
            for new_node in little_node_neighbor:
                if new_node not in temp_g.nodes:
                    continue
                # Random activation probability for calculating two-hop neighbor influence
                activate_weight = numpy.random.normal(average, variance)
                while activate_weight < 0.001 or activate_weight > 0.2:
                    activate_weight = numpy.random.normal(average, variance)

                # Calculate updated MDD value
                mdd_influence_spread = 0
                for neighbor in nx.neighbors(g, new_node):
                    if neighbor in temp_g.nodes:
                        mdd_influence_spread += 1
                        mdd_influence_spread += activate_weight * g.degree(neighbor)
                    else:
                        mdd_influence_spread += factor
                        mdd_influence_spread += factor * activate_weight * g.degree(neighbor)

                # Update node dictionary value
                node_dict[new_node] = [mdd_influence_spread, set(nx.neighbors(temp_g, new_node))]

            little_node_list.clear()
            little_node_neighbor.clear()
        else:
            rank += 0.5

    return mdd_dict


def two_stage_evolution_sa(g, budget, filename):
    """
    Two-stage algorithm: Evolutionary Algorithm + Simulated Annealing
    
    Args:
        g: The network graph
        budget: Budget constraint
        filename: Name of the dataset file
    
    Returns:
        Optimized seed set
    """
    n = len(g.nodes)
    pop = 30
    dp = 0.6
    mp_1 = 0.2
    mp_2 = 0.1
    cp = 0.6
    g_max = 140
    k = int(budget / 2)

    node_emd_dict = node_embedding_sort(g, filename)

    # Step 1: Node influence assessment
    mdd_dict = two_hop_similar_degree(g, node_emd_dict, 0.7)
    mdd_list = [(x, y / g.nodes[x]['weight']) for x, y in mdd_dict.items()]
    mdd_list.sort(key=lambda x: x[1], reverse=True)
    pool_list = [x[0] for x in mdd_list]
    
    # Initialization
    initial_pop = list()
    for i in range(pop):
        p = random.uniform(0.1, 0.5)
        random_range = k + n * math.pow(budget / (2 * (n - k)), 1 - p) * math.sin(math.pi * p / 2)
        random_range_list = pool_list[:int(random_range)]

        initial_individuals = pool_list[:k]
        for j in range(k):
            p_div = random.random()
            if p_div < dp:
                select_node = random.choice(random_range_list)
                while select_node in initial_individuals:
                    select_node = random.choice(random_range_list)

                initial_individuals[j] = select_node

        initial_pop.append(initial_individuals)

    while g_max:
        g_max -= 1

        # Dual-probability evolution (mutation)
        mutation_pop = list()
        for i in range(pop):
            p = random.uniform(0.1, 0.5)
            random_range = k + n * math.pow(budget / (2 * (n - k)), 1 - p) * math.sin(math.pi * p / 2)
            random_range_list = pool_list[:int(random_range)]

            individuals_budget_rank = [[node, g.nodes[node]['weight']] for node in initial_pop[i]]
            individuals_budget_rank.sort(key=lambda x: x[1], reverse=True)
            initial_pop[i] = [x[0] for x in individuals_budget_rank]

            mutation_individuals = copy.deepcopy(initial_pop[i])
            for j in range(k):
                if j < k / 2:
                    # Use mutation probability 1
                    p_mutation = random.random()
                    if p_mutation < mp_1:
                        mutation_node = random.randint(0, len(random_range_list) - 1)
                        while random_range_list[mutation_node] in mutation_individuals:
                            mutation_node = random.randint(0, len(random_range_list) - 1)

                        swap(mutation_individuals, random_range_list, j, mutation_node)
                else:
                    p_mutation = random.random()
                    if p_mutation < mp_2:
                        mutation_node = random.randint(0, len(random_range_list) - 1)
                        while random_range_list[mutation_node] in mutation_individuals:
                            mutation_node = random.randint(0, len(random_range_list) - 1)

                        swap(mutation_individuals, random_range_list, j, mutation_node)

            mutation_pop.append(mutation_individuals)

        # Crossover
        crossover_pop = list()
        for i in range(pop):
            p = random.uniform(0.1, 0.5)
            random_range = k + n * math.pow(budget / (2 * (n - k)), 1 - p) * math.sin(math.pi * p / 2)
            random_range_list = pool_list[:int(random_range)]

            crossover_individuals = list()
            for j in range(k):
                p_crossover = random.random()
                if p_crossover < cp:
                    if mutation_pop[i][j] in crossover_individuals:
                        if initial_pop[i][j] in crossover_individuals:
                            crossover_node = random.choice(random_range_list)
                            while crossover_node in crossover_individuals:
                                crossover_node = random.choice(random_range_list)
                            crossover_individuals.append(crossover_node)
                        else:
                            crossover_individuals.append(initial_pop[i][j])
                    else:
                        crossover_individuals.append(mutation_pop[i][j])
                else:
                    if initial_pop[i][j] in crossover_individuals:
                        if mutation_pop[i][j] in crossover_individuals:
                            crossover_node = random.choice(random_range_list)
                            while crossover_node in crossover_individuals:
                                crossover_node = random.choice(random_range_list)
                            crossover_individuals.append(crossover_node)
                        else:
                            crossover_individuals.append(mutation_pop[i][j])
                    else:
                        crossover_individuals.append(initial_pop[i][j])

            crossover_pop.append(crossover_individuals)

        # Selection
        select_pop = list()
        for i in range(pop):
            initial_influence = fitness_function_3(g, set(initial_pop[i]))
            mutation_influence = fitness_function_3(g, set(mutation_pop[i]))
            if initial_influence > mutation_influence:
                select_pop.append(initial_pop[i])
            else:
                select_pop.append(mutation_pop[i])

        initial_pop.clear()
        initial_pop = copy.deepcopy(select_pop)

    # Step 3: Adaptive simulated annealing
    pool_2_set = set()
    for initial_individuals in initial_pop:
        pool_2_set = pool_2_set.union(set(initial_individuals))

    # degree_dict = dict(g.degree())
    degree_dict = two_hop_mixed_degree_decomposition(g, 0.7)
    seed = cost_greedy(g, degree_dict, budget)

    seed_2 = adaptive_simulated_annealing(g, budget, seed, pool_2_set)
    
    return seed_2