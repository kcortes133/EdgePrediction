from ensmallen import Graph

kgFolder = 'monarch-kg-Sept2025/'
nodesF = kgFolder + 'monarch-kg_nodes.tsv'
out_folder = "IC_removal_none/"
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
from embiggen.edge_prediction import (
    DecisionTreeEdgePrediction,
    PerceptronEdgePrediction,
    edge_prediction_evaluation,
)
from embiggen.embedders.ensmallen_embedders import (
    FirstOrderLINEEnsmallen,
    TransEEnsmallen,
)
from tqdm.auto import tqdm
import pandas as pd

EmbeddingMethods = [
    FirstOrderLINEEnsmallen,
    TransEEnsmallen,
]

results_embeddings = pd.concat([
    edge_prediction_evaluation(
        holdouts_kwargs=dict(train_size=0.8),
        graphs=[gTest],  # 👈 use your Monarch KG test graph
        models=[
            DecisionTreeEdgePrediction(),
            PerceptronEdgePrediction(
                edge_features=None,
                edge_embeddings="Hadamard"
            ),
        ],
        number_of_holdouts=10,
        node_features=EmbeddingMethod(),
        smoke_test=False,
        enable_cache=True
    )
    for EmbeddingMethod in tqdm(EmbeddingMethods, desc="Embedding methods")
])

# 💾 Save results
results_embeddings.to_csv("monarch_gtest_edge_embedding_results.csv", index=False)
from barplots import barplots
import matplotlib.pyplot as plt

index = [
    "graph_name",
    "evaluation_mode",
    ('node_features_parameters', 'model_name'),
    "model_name"
]

fig = barplots(
    results_embeddings[[*index,
                        "f1_score",
                        "balanced_accuracy",
                        "matthews_correlation_coefficient",
                        "auroc",
                        "auprc",
                        "fall_out"]],
    groupby=index,
    height=4,
    bar_width=.2,
    unique_minor_labels=False,
    unique_major_labels=False,
    legend_position="center right",
    minor_rotation=0,
    ncol=2,
)
plt.savefig('benchmark assessment.png')
