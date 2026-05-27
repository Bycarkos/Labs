from torch.utils.data import Dataset
from typing import List, Tuple
from pathlib import Path
import numpy as np
import re


class Graph(Dataset):


    def __init__(self, edge_index: List) -> None:
        self.raw_src, self.raw_rel, self.raw_trg = map(list, zip(*edge_index))
        self._total_nodes = list(set(list(set(self.raw_src)) + list(set(self.raw_trg))))

        self._types_of_relations = list(set(self.raw_rel))

        self.num_total_nodes = len(self._total_nodes)
        self.num_relations = len(self._types_of_relations)
        
        self._node2idx = {node: idx for idx, node in enumerate(self._total_nodes)}
        self._relation2idx = {rel: idx for idx, rel in enumerate(self._types_of_relations)}

        self._idx2node = {value:key for key, value in self._node2idx.items()}
        self._idx2relation = {value:key for key, value in self._relation2idx.items()}

        self._src_nodes = [self._node2idx[node] for node in self.raw_src]
        self._rels = [self._relation2idx[rel] for rel in self.raw_rel]
        self._trg_nodes = [self._node2idx[node] for node in self.raw_trg]

        self.adj = np.array((self._src_nodes, self._rels, self._trg_nodes), dtype=np.int32)

        self.node_embeddings = self.update_structural_embeddings()

    def check_string(s):
        if re.fullmatch(r"\d+(?:\s\d+)*", s):   # only numbers with spaces
            return 1
        elif re.fullmatch(r"[A-Za-z]+(?:\s[A-Za-z]+)*", s):   # only letters with spaces
            return 2
        elif re.fullmatch(r"[A-Za-z0-9]+(?:\s[A-Za-z0-9]+)*", s):   # letters/numbers with spaces
            return 3
        else:
            return 4    # contains special characters


    def detokenize(self, adj) -> List[Tuple]:
        src, rel, trg = adj

        src_raw = [self._idx2node[node] for node in src]
        rel_raw = [self._idx2relation[rel] for rel in rel]
        trg_raw = [self._idx2node[node] for node in trg]

        return list(zip(src_raw, rel_raw, trg_raw))

    def tokenize(self, raw_triples: List[Tuple]):
        src, rel, trg = list(zip(*raw_triples))

        src_raw = [self._node2idx[node] for node in src]
        rel_raw = [self._relation2idx[rel] for rel in rel]
        trg_raw = [self._node2idx[node] for node in trg]

        return np.vstack((src_raw, rel_raw, trg_raw), dtype=np.int32)

    def to_heterogeneous(self):

        edge_index_dict = {}
        for rel in self._types_of_relations:
            edge_index_dict[('src', rel, 'trg')] = self.adj[:, self._rels == self._relation2idx[rel]]

        return edge_index_dict

    def update_from_triples(self, triples: List[Tuple], update_embeddings: bool = True):
        """
        Incrementally add new triples/subgraphs without changing old IDs.

        Existing node and relation mappings are preserved.
        New nodes and relations are appended with new IDs.
        """

        self._count_added_nodes = 0
        self._count_added_relations = 0
        
        if len(triples) == 0:
            return

        new_raw_src = []
        new_raw_rel = []
        new_raw_trg = []

        new_src_nodes = []
        new_rels = []
        new_trg_nodes = []

        for src, rel, trg in triples:
            self._add_node_if_missing(src)
            self._add_node_if_missing(trg)
            self._add_relation_if_missing(rel)

            new_raw_src.append(src)
            new_raw_rel.append(rel)
            new_raw_trg.append(trg)

            new_src_nodes.append(self._node2idx[src])
            new_rels.append(self._relation2idx[rel])
            new_trg_nodes.append(self._node2idx[trg])

        new_src_nodes = np.array(new_src_nodes, dtype=np.int32)
        new_rels = np.array(new_rels, dtype=np.int32)
        new_trg_nodes = np.array(new_trg_nodes, dtype=np.int32)

        self.raw_src.extend(new_raw_src)
        self.raw_rel.extend(new_raw_rel)
        self.raw_trg.extend(new_raw_trg)

        self._src_nodes = np.concatenate((self._src_nodes, new_src_nodes))
        self._rels = np.concatenate((self._rels, new_rels))
        self._trg_nodes = np.concatenate((self._trg_nodes, new_trg_nodes))

        self.adj = np.vstack(
            (self._src_nodes, self._rels, self._trg_nodes)
        ).astype(np.int32)

        self.num_total_nodes = len(self._total_nodes)
        self.num_relations = len(self._types_of_relations)

        if update_embeddings:
            self.node_embeddings = self.update_structural_embeddings()

    def _add_node_if_missing(self, node):
        if node not in self._node2idx:
            self._count_added_nodes += 1
            idx = len(self._total_nodes)

            self._node2idx[node] = idx
            self._idx2node[idx] = node
            self._total_nodes.append(node)

    def _add_relation_if_missing(self, rel):
        if rel not in self._relation2idx:
            self._count_added_relations += 1
            idx = len(self._types_of_relations)

            self._relation2idx[rel] = idx
            self._idx2relation[idx] = rel
            self._types_of_relations.append(rel)


    def update_structural_embeddings(self, normalize: bool = True):
        """
        Create or update structural embeddings for all nodes.

        This recomputes the structural features from the current graph,
        but keeps the node order fixed according to self._node2idx.

        Embedding:
        [
            out_degree,
            in_degree,
            total_degree,
            unique_out_relations,
            unique_in_relations,
            outgoing_count_per_relation...,
            incoming_count_per_relation...
        ]
        """
        num_nodes = self.num_total_nodes
        num_relations = self.num_relations

        out_degree = np.zeros((num_nodes, 1), dtype=np.float32)
        in_degree = np.zeros((num_nodes, 1), dtype=np.float32)

        unique_out_relations = [set() for _ in range(num_nodes)]
        unique_in_relations = [set() for _ in range(num_nodes)]

        out_relation_counts = np.zeros((num_nodes, num_relations), dtype=np.float32)
        in_relation_counts = np.zeros((num_nodes, num_relations), dtype=np.float32)

        for src, rel, trg in zip(self._src_nodes, self._rels, self._trg_nodes):
            out_degree[src] += 1
            in_degree[trg] += 1

            unique_out_relations[src].add(rel)
            unique_in_relations[trg].add(rel)

            out_relation_counts[src, rel] += 1
            in_relation_counts[trg, rel] += 1

        total_degree = out_degree + in_degree

        unique_out_counts = np.array(
            [[len(values)] for values in unique_out_relations],
            dtype=np.float32
        )

        unique_in_counts = np.array(
            [[len(values)] for values in unique_in_relations],
            dtype=np.float32
        )

        embeddings = np.hstack(
            (
                out_degree,
                in_degree,
                total_degree,
                unique_out_counts,
                unique_in_counts,
                out_relation_counts,
                in_relation_counts,
            )
        )

        if normalize:
            embeddings = np.log1p(embeddings)

        embeddings_padded = np.zeros((self.num_total_nodes, 150 - embeddings.shape[1]), dtype=np.float32)
        embeddings_padded = np.hstack((embeddings, embeddings_padded))
        self.node_embeddings = embeddings_padded

        return self.node_embeddings

    @classmethod
    def from_path(cls, path: Path):
        triples, gt = read_data(path)

        return Graph(triples), gt

    def __len__(self):
        return self.adj.shape[1]

def read_data(path: Path) -> Tuple[List, List]:
    triplets = []
    gt = []

    with open(path, 'r', encoding='utf-8-sig') as f:
        for line in f:
            val = line.strip().split('\t')
            s, r, o = val[0], val[1], val[2]
            r = '_'.join(x for x in str(r).strip().split("_") if not x.isnumeric()).lower()

            if len(val) >= 4:
                triplets.append((s, r, o))
                gt.append(int(val[3]))
            else:
                triplets.append((s, r, o))
                gt.append(1)

    return triplets, gt