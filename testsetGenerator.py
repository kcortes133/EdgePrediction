import csv
import random
from pathlib import Path

# =====================
# Configuration
# =====================
"""INPUT_FILES = [
    "monarch-kg-Sept2025/monarch-kg_edges.tsv",
    "monarch-kg-Sept2025/monarch-kg_edges.filtered_40.tsv",
    "monarch-kg-Sept2025/monarch-kg_edges.filtered_60.tsv",
    "monarch-kg-Sept2025/monarch-kg_edges.filtered_80.tsv",
    "monarch-kg-Sept2025/monarch-kg_edges.filtered_100.tsv",
]"""
INPUT_FILES = [
    "robokop/rk_tsvs/rk_edges_out.tsv",
    "robokop/rk_tsvs/rk-kg_edges.filtered_40.tsv",
    "robokop/rk_tsvs/rk-kg_edges.filtered_60.tsv",
    "robokop/rk_tsvs/rk-kg_edges.filtered_80.tsv",
    "robokop/rk_tsvs/rk-kg_edges.filtered_100.tsv"
]

OUTPUT_SUFFIX = ".TestSet.tsv"
REMOVED_EDGES_FILE = "TP_hgnc_mondo_edges.tsv"
NEG_EDGES_FILE = "TN_hgnc_mondo_edges.tsv"
SUBSET_FRACTION = 0.25
RANDOM_SEED = 42

EDGE_KEY_FIELDS = ("subject", "predicate", "object")


# =====================
# Helpers
# =====================
def is_hgnc_to_mondo(row):
    return (
        row["subject"].startswith("HGNC:")
        and row["object"].startswith("MONDO:")
    )


def edge_key(row):
    return tuple(row[f] for f in EDGE_KEY_FIELDS)


def load_edge_keys(path):
    keys = set()
    with open(path, newline="", encoding="utf-8") as f:
        reader = csv.DictReader(f, delimiter="\t")
        for row in reader:
            if is_hgnc_to_mondo(row):
                keys.add(edge_key(row))
    return keys


def filter_file(input_path, output_path, keys_to_remove):
    kept = 0
    removed = 0

    with open(input_path, newline="", encoding="utf-8") as fin, \
         open(output_path, "w", newline="", encoding="utf-8") as fout:

        reader = csv.DictReader(fin, delimiter="\t")
        writer = csv.DictWriter(
            fout,
            fieldnames=reader.fieldnames,
            delimiter="\t"
        )
        writer.writeheader()

        for row in reader:
            if edge_key(row) in keys_to_remove:
                removed += 1
            else:
                writer.writerow(row)
                kept += 1

    return kept, removed

def write_removed_edges(path, keys):
    with open(path, "w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f, delimiter="\t")
        writer.writerow(EDGE_KEY_FIELDS)
        for key in sorted(keys):
            writer.writerow(key)


def negativeSampling(keptEdges, removedEdges, folder, write):
    negEdges = []
    with open(keptEdges, 'r') as f:
        trainEdges = list(csv.reader(f, delimiter='\t'))
        trainEdges.pop(0)
    with open(removedEdges, 'r') as f:
        testEdges = list(csv.reader(f, delimiter='\t'))
        header = testEdges.pop(0)
    testTriples = []
    trainTriples = []
    tripleTypes = []
    # pick random object s.t. (s,p,o)type exists in tripleTypes but specific edge does not exist in graph or removed edges
    for edge in testEdges:
        subj = edge[0]
        pred = edge[1]
        obj = edge[2]

        #predCount[pred]+=1
        testTriples.append([subj, pred, obj])
        tripleTypes.append([subj.split(':')[0], pred, obj.split(':')[0]])
    for edge in trainEdges:
        obj = edge[-1]
        subj = edge[-2]
        pred = edge[2]
        trainTriples.append([subj, pred, obj])

    while len(testEdges) > len(negEdges):
        randTriple1 = random.choice(testTriples)
        randTriple2 = random.choice(testTriples)

        sub1 = randTriple1[0].split(':')[0]
        sub2 = randTriple2[0].split(':')[0]

        pred1 = randTriple1[1]
        pred2 = randTriple2[1]

        obj1 = randTriple1[2].split(':')[0]
        obj2 = randTriple2[2].split(':')[0]

        #if sub1 == sub2 and pred1 == pred2 and obj1 == obj2:
        if [sub1, pred1, obj2] in tripleTypes:
            newNegTriple = [randTriple1[0], pred1, randTriple2[2]]
            if newNegTriple not in trainTriples and newNegTriple not in testTriples:
                #if negPredCounts[pred1] < predCount[pred1]+5:
                negEdges.append(newNegTriple)
                #negPredCounts[pred1]+=1
        if [sub2, pred2, obj1] in tripleTypes:
            newNegTriple = [randTriple2[0], pred2, randTriple1[2]]
            if newNegTriple not in trainTriples and newNegTriple not in testTriples:
                #if negPredCounts[pred1] < predCount[pred1] +5:
                negEdges.append(newNegTriple)
                #negPredCounts[pred1]+=1

    if write:
        with open(folder, 'w') as f:
            writer =csv.writer(f)
            writer.writerow(['subject', 'predicate', 'object'])
            writer.writerows(negEdges)

    return negEdges
# =====================
# Main
# =====================
def main():
    '''
    random.seed(RANDOM_SEED)

    print("Loading HGNC→MONDO edges from all files...")
    file_edge_sets = [load_edge_keys(p) for p in INPUT_FILES]

    print("Computing intersection across all four files...")
    shared_edges = set.intersection(*file_edge_sets)

    print(f"Shared HGNC→MONDO edges: {len(shared_edges)}")

    if not shared_edges:
        raise RuntimeError("No shared HGNC→MONDO edges found.")

    subset_size = int(len(shared_edges) * SUBSET_FRACTION)
    subset_edges = set(random.sample(list(shared_edges), subset_size))
    print(f"Writing removed edge list → {REMOVED_EDGES_FILE}")
    write_removed_edges(REMOVED_EDGES_FILE, subset_edges)
    print(f"Removing {subset_size} edges (20%) from each file")
    '''
    subset_edges = []
    with open(REMOVED_EDGES_FILE, 'r') as inF:
        reader = csv.reader(inF, delimiter='\t')
        for row in reader:
            subset_edges.append(row)
        subset_edges.pop(0)
    print(len(subset_edges))
    for path in INPUT_FILES:
        out_path = Path(path).with_suffix(OUTPUT_SUFFIX)
        kept, removed = filter_file(path, out_path, subset_edges)
        print(f"{path} → kept {kept}, removed {removed}")


    negativeSampling('robokop/rk_tsvs/rk-kg_edges.TestSet.tsv', REMOVED_EDGES_FILE, NEG_EDGES_FILE, True)
    print("Done.")


if __name__ == "__main__":
    main()
