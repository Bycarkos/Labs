from torch.utils.data import Dataset
from typing import List, Tuple
from pathlib import Path
import numpy as np
import re


class Graph(Dataset):


    def __init__(self, edge_index: List) -> None:
        self.raw_src, self.raw_rel, self.raw_trg = list(zip(*edge_index))
        self._total_nodes = list(set(list(set(self.raw_src)) + list(set(self.raw_trg))))

        self._types_of_relations = list(set(self.raw_rel))

        self.num_total_nodes = len(self._total_nodes)

        self._node2idx = {node: idx for idx, node in enumerate(self._total_nodes)}
        self._relation2idx = {rel: idx for idx, rel in enumerate(self._types_of_relations)}

        self._idx2node = {value:key for key, value in self._node2idx.items()}
        self._idx2relation = {value:key for key, value in self._relation2idx.items()}

        self._src_nodes = [self._node2idx[node] for node in self.raw_src]
        self._rels = [self._relation2idx[rel] for rel in self.raw_rel]
        self._trg_nodes = [self._node2idx[node] for node in self.raw_trg]

        self.adj = np.array((self._src_nodes, self._rels, self._trg_nodes), dtype=np.int32)


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