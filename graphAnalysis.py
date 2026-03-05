import os
import numpy as np
import matplotlib.pyplot as plt
from collections import Counter


# ------------------------------------------------------------------
# Helpers
# ------------------------------------------------------------------

def _ensure_dir(path):
    os.makedirs(path, exist_ok=True)


def _centrality_plot(
    graph,
    modelOrgs,
    outputFolder,
    values,
    prefix,
    title_prefix,
    xlabel
):
    _ensure_dir(outputFolder)
    nodeNames = graph.get_node_names()

    centDict = {i: [] for i in modelOrgs}
    for node in range(len(nodeNames)):
        orgType = nodeNames[node].split(':')[0]
        if orgType in modelOrgs:
            centDict[orgType].append(values[node])

    orgStats = {}
    for org, vals in centDict.items():
        if len(vals) == 0:
            continue

        plt.figure()
        plt.hist(vals)
        plt.xlabel(xlabel)
        plt.ylabel('Number of Nodes')
        plt.yscale('log')
        plt.title(f'{title_prefix} Node Distribution for {org}')
        plt.savefig(os.path.join(outputFolder, f'{prefix}{org}.png'))
        plt.close()

        counts, bins = np.histogram(vals)
        mids = 0.5 * (bins[1:] + bins[:-1])
        probs = counts / np.sum(counts)

        mean = np.sum(probs * mids)
        sd = np.sqrt(np.sum(probs * (mids - mean) ** 2))
        orgStats[org] = {'mean': mean, 'sd': sd, 'max': max(vals)}

    with open(os.path.join(outputFolder, f'{prefix}Scores.txt'), 'w') as of:
        of.write('Organism\tMean\tStandard Dev\tMax\n')
        for o, stats in orgStats.items():
            of.write(f"{o}\t{stats['mean']}\t{stats['sd']}\t{stats['max']}\n")


# ------------------------------------------------------------------
# Centrality / degree wrappers
# ------------------------------------------------------------------

def harmonicCentralityPlot(graph, modelOrgs, outputFolder):
    _centrality_plot(
        graph, modelOrgs, outputFolder,
        graph.get_harmonic_centrality(),
        prefix='HC',
        title_prefix='Harmonic Centrality',
        xlabel='Centrality Score'
    )


def centralityPlot(graph, modelOrgs, outputFolder):
    _centrality_plot(
        graph, modelOrgs, outputFolder,
        graph.get_degree_centrality(),
        prefix='DC',
        title_prefix='Degree Centrality',
        xlabel='Centrality Score'
    )


def closenessCentralityPlot(graph, modelOrgs, outputFolder):
    _centrality_plot(
        graph, modelOrgs, outputFolder,
        graph.get_closeness_centrality(),
        prefix='CC',
        title_prefix='Closeness Centrality',
        xlabel='Centrality Score'
    )


def betweenessCentralityPlot(graph, modelOrgs, outputFolder):
    _centrality_plot(
        graph, modelOrgs, outputFolder,
        graph.get_betweenness_centrality(),
        prefix='BC',
        title_prefix='Betweenness Centrality',
        xlabel='Centrality Score'
    )


def nodeDegreePlot(graph, modelOrgs, outputFolder):
    _centrality_plot(
        graph, modelOrgs, outputFolder,
        graph.get_node_degrees(),
        prefix='ND',
        title_prefix='Node Degree',
        xlabel='Node Degree'
    )


# ------------------------------------------------------------------
# Edge type analysis
# ------------------------------------------------------------------

def edgeTypes(g, nodeTypes, outputFolder):
    _ensure_dir(outputFolder)
    edgeTypesINNodeTypes = {i: {} for i in nodeTypes}
    srcDesNodeTypes = {i: {} for i in nodeTypes}

    edgeIDS = g.get_edge_node_ids(directed=False)

    for e in range(len(edgeIDS)):
        srcNode, desNode = edgeIDS[e]
        edgeType = g.get_edge_type_name_from_edge_id(e)

        srcType = g.get_node_type_names_from_node_id(srcNode)[0]
        desType = g.get_node_type_names_from_node_id(desNode)[0]

        if srcType in srcDesNodeTypes:
            srcDesNodeTypes[srcType][desType] = srcDesNodeTypes[srcType].get(desType, 0) + 1
        if desType in srcDesNodeTypes:
            srcDesNodeTypes[desType][srcType] = srcDesNodeTypes[desType].get(srcType, 0) + 1

        if srcType in edgeTypesINNodeTypes:
            edgeTypesINNodeTypes[srcType][edgeType] = edgeTypesINNodeTypes[srcType].get(edgeType, 0) + 1


def edgeTypesbyID(g, nodeTypes, outputFolder):
    _ensure_dir(outputFolder)
    edgeTypesINNodeTypes = {i: {} for i in nodeTypes}
    srcDesNodeTypes = {i: {} for i in nodeTypes}

    edgeIDS = g.get_edge_node_ids(directed=False)

    for e in range(len(edgeIDS)):
        edgeType = g.get_edge_type_name_from_edge_id(e)
        nodeNames = g.get_node_names_from_edge_id(e)

        srcType = nodeNames[0].split(':')[0]
        desType = nodeNames[1].split(':')[0]

        if srcType in srcDesNodeTypes:
            srcDesNodeTypes[srcType][desType] = srcDesNodeTypes[srcType].get(desType, 0) + 1
        if desType in srcDesNodeTypes:
            srcDesNodeTypes[desType][srcType] = srcDesNodeTypes[desType].get(srcType, 0) + 1

        if srcType in edgeTypesINNodeTypes:
            edgeTypesINNodeTypes[srcType][edgeType] = edgeTypesINNodeTypes[srcType].get(edgeType, 0) + 1

    for nodeType, edges in edgeTypesINNodeTypes.items():
        if not edges:
            continue
        labels, values = list(edges.keys()), list(edges.values())

        plt.figure().set_figwidth(15)
        plt.bar(labels, values)
        plt.xticks(rotation=15)
        plt.xlabel('Edge Types')
        plt.title(f'{nodeType.split(":")[0]} Edge Types')
        plt.savefig(os.path.join(outputFolder, f'{nodeType.split(":")[0]}EdgeTypes.png'))
        plt.close()

        with open(os.path.join(outputFolder, f'{nodeType.split(":")[0]}EdgeTypes.txt'), 'w') as of:
            for k, v in edges.items():
                of.write(f'{k}:{v}\n')


# ------------------------------------------------------------------
# Eccentricity analysis
# ------------------------------------------------------------------

def computeEccentricityAllNodes(graph):
    nodeIDs = graph.get_node_ids()
    nodeNames = graph.get_node_names()

    prefixes = {name.split(':')[0] for name in nodeNames}
    prefixCounts = {p: 0 for p in prefixes}

    for sourceNode in nodeIDs:
        ecc, distantNode = graph.get_unchecked_eccentricity_and_most_distant_node_id_from_node_id(sourceNode)
        if ecc == 0:
            continue

        path = graph.get_shortest_path_node_ids_from_node_ids(sourceNode, distantNode)
        for pathNode in path:
            prefix = nodeNames[pathNode].split(':')[0]
            prefixCounts[prefix] += 1

    return prefixCounts


def eccentricityPlot(g, outputFolder):
    _ensure_dir(outputFolder)
    counts = computeEccentricityAllNodes(g)

    labels = [k.split(':')[-1] for k in counts]
    values = list(counts.values())

    with open(os.path.join(outputFolder, 'Eccentricity.txt'), 'w') as of:
        for k, v in counts.items():
            of.write(f'{k}:{v}\n')

    plt.figure()
    plt.bar(labels, values)
    plt.xlabel('Node Type')
    plt.ylabel('Number of times in Eccentricity')
    plt.xticks(rotation=15)
    plt.title('Eccentricity Node Type Count')
    plt.savefig(os.path.join(outputFolder, 'Eccentricity.png'))
    plt.close()


# ------------------------------------------------------------------
# Singleton nodes
# ------------------------------------------------------------------
def singletonNodes(graph, modelPrefixes, outputFolder):
    _ensure_dir(outputFolder)

    nodeNames = graph.get_node_names()
    prefixes = {n.split(':')[0] for n in nodeNames}
    prefixCounts = {p: 0 for p in prefixes}

    for s in graph.get_singleton_node_ids():
        prefix = nodeNames[s].split(':')[0]
        prefixCounts[prefix] += 1

    with open(os.path.join(outputFolder, 'singletonNodes.txt'), 'w') as of:
        for p, cnt in prefixCounts.items():
            of.write(f'{p}:{cnt}\n')

    labels = [p for p in modelPrefixes if p in prefixCounts]
    values = [prefixCounts[p] for p in labels]

    plt.figure()
    plt.bar(labels, values)
    plt.ylabel('Number of Nodes')
    plt.xlabel('Node Prefix')
    plt.title('Number of Singleton Nodes per Prefix')
    plt.savefig(os.path.join(outputFolder, 'SingletonNodes.png'))
    plt.close()


# ------------------------------------------------------------------
# Connected components info
# ------------------------------------------------------------------

def getComponentsInfo(g):
    components = g.get_node_connected_component_ids()
    counts = Counter(components)

    top = dict(sorted(counts.items(), key=lambda x: x[1], reverse=True)[:5])
    print("Top components:", top)

    nodeIDs = g.get_node_ids()
    nodeNames = g.get_node_names()

    prefixComponents = {}

    for idx, compID in enumerate(components):
        if compID not in top:
            continue

        prefix = nodeNames[nodeIDs[idx]].split(':')[0]
        prefixComponents.setdefault(prefix, {})
        prefixComponents[prefix][compID] = prefixComponents[prefix].get(compID, 0) + 1

    print("Node prefixes in top components:")
    for p, comps in prefixComponents.items():
        print(p, comps)


# ------------------------------------------------------------------
# Node type counts
# ------------------------------------------------------------------

def plotNodePrefixNum(graph, modelPrefixes, outputFolder):
    _ensure_dir(outputFolder)

    nodeNames = graph.get_node_names()
    prefixCounts = Counter(n.split(':')[0] for n in nodeNames)

    labels = modelPrefixes
    values = [prefixCounts.get(p, 0) for p in labels]

    plt.figure(figsize=(15, 5))
    plt.bar(labels, values)
    plt.ylabel('Number of Nodes')
    plt.xlabel('Node Prefix')
    plt.title('Number of Nodes per Prefix')
    plt.xticks(rotation=15)
    plt.savefig(os.path.join(outputFolder, 'NodePrefix.png'))
    plt.close()

    with open(os.path.join(outputFolder, 'NodePrefixCount.txt'), 'w') as of:
        for p in labels:
            of.write(f'{p}:{prefixCounts.get(p, 0)}\n')


if __name__ == "__main__":
    from ensmallen import Graph

    # -------------------------------
    # Paths
    # -------------------------------
    outputFolder = "results/monarch_analysis/"
    kgFolder = 'monarch-kg-Sept2025/'

    os.makedirs(outputFolder, exist_ok=True)

    # -------------------------------
    # Load graph
    # -------------------------------
    print("Loading Monarch KG...")
    g = Graph.from_csv(
        directed=False,
        node_path=kgFolder + 'monarch-kg_nodes.tsv',
        edge_path=kgFolder + 'monarch-kg_edges.tsv',
        node_list_separator='\t',
        edge_list_separator='\t',
        verbose=True,
        nodes_column='id',
        node_list_node_types_column='category',
        default_node_type='biolink:NamedThing',
        sources_column='subject',
        destinations_column='object',
        edge_list_edge_types_column='predicate',
        name='Monarch KG'
    )


    print(g)
    print("Graph loaded.")
    print("----------------------------------")

    # -------------------------------
    # Define node types of interest
    # (edit to match what you care about)
    # -------------------------------

    modelOrgs = ['WB', 'ZFIN', 'RGD', 'MGI', 'FB', 'PomBase', 'dictyBase', 'SGD', 'HGNC', 'NCBIGene', 'Xenbase']
    """
    modelOrgs = [
        "HGNC",
        "MONDO",
        "HP",
        "GO"
    ]"""

    nodeTypes = modelOrgs  # for edge type analysis

    # -------------------------------
    # Run analyses
    # -------------------------------
    print("Running centrality analyses...")
    harmonicCentralityPlot(g, modelOrgs, outputFolder)
    centralityPlot(g, modelOrgs, outputFolder)
    closenessCentralityPlot(g, modelOrgs, outputFolder)
    betweenessCentralityPlot(g, modelOrgs, outputFolder) # needs to not be a multigraph
    nodeDegreePlot(g, modelOrgs, outputFolder)

    print("Running edge type analyses...")
    edgeTypes(g, nodeTypes, outputFolder)

    print("Running eccentricity analysis...")
    eccentricityPlot(g, outputFolder)

    print("Running singleton node analysis...")
    singletonNodes(g, modelOrgs, outputFolder)

    print("Plotting node type counts...")
    plotNodePrefixNum(g, modelOrgs, outputFolder)

    print("Connected components summary:")
    getComponentsInfo(g)

    print("----------------------------------")
    print("All analyses completed.")
