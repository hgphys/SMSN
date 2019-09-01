#==========================================
#  import packages
#==========================================
import numpy as np
import numpy.random as rd
import networkx as nx
from scipy import stats

#==========================================
#  Modules
#==========================================
def gwcc_net(G):
    """Extract giant weakly connected network in G

    Parameters
    ----------
    G : NetworkX DiGraph

    Returns
    -------
    Giant weakly connected network in G 
    and add node attribute named 'component' based on
    bowtie decomposition:
    GWCC = IN + GSCC + OUT + TE 
    Giant weakly connected component (GWCC) can be decomposed as
    giant strongly connected component (GSCC),
    its upstream (IN) and downstream (OUT) and the others (TE).
 
    """
    wcc_net = G
    gwcc_set = sorted(nx.weakly_connected_components(wcc_net), key=len, reverse=True)[0]
    list_all = list(wcc_net.nodes)
    for i in range(len(list_all)):
        if list_all[i] not in gwcc_set:
            wcc_net.remove_node(list_all[i])
    # Identify GSCC
    gscc_set = sorted(nx.strongly_connected_components(wcc_net), key=len, reverse=True)[0]
    # Identify IN and OUT components
    # IN
    in_scc = list(nx.bfs_edges(wcc_net, list(gscc_set)[0], reverse= True))
    in_gscc_net = nx.DiGraph()
    for i in range(len(in_scc)):
        in_gscc_net.add_edge(in_scc[i][0], in_scc[i][1])
    nodes_in_scc = list(in_gscc_net.nodes())
    nodes_scc = list(gscc_set)
    nodes_in_set = set(nodes_in_scc) - set(nodes_scc)
    # OUT
    out_scc = list(nx.bfs_edges(wcc_net, list(gscc_set)[0]))
    out_gscc_net = nx.DiGraph()
    for i in range(len(out_scc)):
        out_gscc_net.add_edge(out_scc[i][0], out_scc[i][1])
    nodes_out_scc = list(out_gscc_net.nodes())
    nodes_out_set = set(nodes_out_scc) - set(nodes_scc)
    # TE
    nodes_te_set = gwcc_set - set(nodes_scc) - nodes_in_set - nodes_out_set
    # Labelling components
    for i in range(len(nodes_scc)):
        wcc_net.add_node(nodes_scc[i], component = "GSCC")
    nodes_in = list(nodes_in_set)
    for i in range(len(nodes_in)):
        wcc_net.add_node(nodes_in[i], component = "IN")
    nodes_out = list(nodes_out_set)
    for i in range(len(nodes_out)):
        wcc_net.add_node(nodes_out[i], component = "OUT")
    nodes_te = list(nodes_te_set)
    for i in range(len(nodes_te)):
        wcc_net.add_node(nodes_te[i], component = "TE")
    return wcc_net


def gscc_net(G):
    """Extract giant strongly connected network in G
    Parameters
    ----------
    G : NetworkX DiGraph

    Returns
    -------
    Giant strongly connected network in G 
 
    """
    scc_net = G
    # Identify GSCC
    gscc_set = sorted(nx.strongly_connected_components(scc_net), key=len, reverse=True)[0]
    list_nodes = list(scc_net.nodes)
    for i in range(len(list_nodes)):
        if list_nodes[i] not in gscc_set:
            scc_net.remove_node(list_nodes[i])
    return scc_net


def bmSimulation(Time, Graph, J, var_a):
    """Return next X based on BM model 

    Parameters
    ----------
    Time : Number of iterations
    G : NetworkX Graph or Strongly connected graph
    J : Share of asset J<1.0 (BM model parameter)
    var_a : Variance of a (BM model parameter)

    Returns
    -------
    Return list [Asset_i, Sales_i, Costs_i] after Time iterations
    
    Asset_next = a_i * Asset_i + Sales_i - Costs_i
    where a_i is a random variable with a Gaussian distribution.
    Note that we set mean of a_i as m=1.
    
    """
    Asset = np.array([1.0]*(len(Graph.nodes)))
    J_i = np.array([J]*(len(Graph.nodes)))
    a_i = np.absolute(rd.normal(1.0, np.sqrt(var_a),len(Graph.nodes)))
    adj_mat = nx.to_scipy_sparse_matrix(Graph, weight = False, format='csr')
    #adj_mat = nx.adjacency_matrix(G, weight = False)
    for t in range(Time):
        rd.seed()
        a_i = np.absolute(rd.normal(1.0, np.sqrt(var_a), len(Graph.nodes)))
        Sales = adj_mat.T.dot(J_i * Asset)
        Costs = J_i * Asset
        Asset = a_i * Asset - Costs + Sales
        Asset = Asset/np.average(Asset)
        Sales = Sales/np.average(Sales)
        Costs = Costs/np.average(Costs)
    return Asset, Sales, Costs
