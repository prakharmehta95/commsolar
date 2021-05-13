#%%
import numpy as np

from scipy.sparse.csgraph import minimum_spanning_tree

def distance_between_two_agents(ag0,ag1):
    """
    This function returns the distance between two agents in meters (float)
    """
    return ((ag0.pos[0] - ag1.pos[0])**2 + (ag0.pos[1] - ag1.pos[1])**2)**0.5

def distance_matrix(agents):
    return np.array([[distance_between_two_agents(ag0,ag1) for ag1 in agents] for ag0 in agents])

def create_new_community_grid(agents):

    # Create distance matrix between agents
    d_matrix = distance_matrix(agents)

    # Compute minimum spanning tree connecting all agents
    com_grid = minimum_spanning_tree(d_matrix)

    # Translate MST into a dictionary: key is tuple of connected agents unique_id (str) and value is distance between them (float)
    com_grid_d = {(agents[k[0]].unique_id,agents[k[1]].unique_id):v for k,v in com_grid.todok().items()}

    #com_grid_length = np.sum(com_grid)

    #com_grid_edges = list(com_grid_d.keys())

    return com_grid_d

def add_agent_to_community_grid(agent,com_agents,com_grid_d):

    # Compute distance to all other points and choose shortest edge
    na_distances = {(agent.unique_id,ca.unique_id):distance_between_two_agents(agent,ca) for ca in com_agents}

    # Find connection with minimum distance from new agent to agent in com grid
    new_edge = min(na_distances, key=na_distances.get)

    # Add new connection to the com grid dictionary
    com_grid_d[new_edge] = min(na_distances.values())

    return com_grid_d