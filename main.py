import sys, time, argparse, os, random

from sklearn.metrics import confusion_matrix
from upsetplot import UpSet, from_memberships
from ensmallen import Graph
import editKG, embedding, rareDiseaseSubsets

parser = argparse.ArgumentParser(description='')
parser.add_argument('--allOrgsTest', metavar='allOrgsTest', type=bool, default=True, help='')
args = parser.parse_args()

random.seed(42)

def main():
    kgFolder = 'monarch-kg-Sept2025/'
    edgesF = kgFolder + 'monarch-kg_edges.filtered_60.tsv'
    nodesF = kgFolder + 'monarch-kg_nodes.filtered_60.tsv'
    out_folder = "IC_removal_60/"
    #rareDiseaseSubsetF = "subset_15_percent_each_category.csv"
    #diseaseSubsetF = "subset_15_percent_each_categoryAll.csv"
    #removedEdgesF = 'IC_removal_none/removedEdges.tsv'

    if not os.path.exists(out_folder):
        os.makedirs(out_folder)

    #diseaseGenePairsF = 'Rare Disease Gene Associations.csv'
    # Step 1: Remove edges
    #removedEdges, keptEdges = editKG.removeRandomRareGeneDiseaseEdges(edgesF, diseaseGenePairsF)
    removedEdges, keptEdges = editKG.removeRandomHumanGeneDiseaseEdges(edgesF)
    #editKG.remove_edges(edgesF, 'removedEdges.tsv', out_folder +'keptEdges.tsv')
    #removedEdges, keptEdges = editKG.removeSpecifiedGeneDiseaseEdges(rareDiseaseSubsetF, diseaseSubsetF, edgesF, out_folder)
    editKG.writeRemovedEdgeFiles(out_folder, removedEdges, keptEdges)
    embedding.negativeSampling(out_folder + 'keptEdges.tsv', out_folder+'removedEdges.tsv', out_folder, True)
    
    # Step 2: Load graphs
    
    gTest = Graph.from_csv(
        directed=False,
        node_path=nodesF,
        edge_path=out_folder + 'keptEdges.tsv',
        node_list_separator='\t',
        edge_list_separator='\t',
        verbose=True,
        nodes_column='id',
        node_list_node_types_column='category',
        default_node_type='biolink:NamedThing',
        sources_column='subject',
        destinations_column='object',
        edge_list_edge_types_column='predicate',
        name='Monarch KG Test'
    )

    # Step 3: Train embeddings
    embedding.train_embeddings(gTest, out_folder)

    # candidate genes for eval are limited to protein coding genes with no disease connections in the KG
    # Step 4: Evaluate and save results
    confM, allR = embedding.evaluate_embeddingsTOP10(gTest, out_folder+'FLOE_embeddings.csv', 'removedEdges.tsv', 'negEdges.csv', out_folder+'top100Figure.png', out_folder +'top10results.csv', out_folder+'HMI_top.csv', out_folder+'HMI10_summary.csv', out_folder)
    print(confM)
# removed 8822 has mode of inheritance edges they are common in the neg samplling meta paths

if __name__ == "__main__":
    main()