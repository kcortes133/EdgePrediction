import sys

def rewrite_csv_to_tsv(in_file, out_file):
    with open(in_file, "r", encoding="utf-8") as fin, \
         open(out_file, "w", encoding="utf-8") as fout:

        for line in fin:
            line = line.strip()

            # Skip empty lines
            if not line:
                continue

            # Split by comma and strip fields
            parts = [p.strip() for p in line.split(",")]

            # Skip malformed lines
            if len(parts) != 3:
                continue

            fout.write("\t".join(parts) + "\n")


if __name__ == "__main__":
    rewrite_csv_to_tsv('TN_hgnc_mondo_edges.tsv', 'TN_hgnc_mondo_edges1.tsv')
