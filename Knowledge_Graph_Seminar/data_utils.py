from typing import Tuple, List
from pathlib import Path
import pickle



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


def read_pickle(path: Path):

    with open(path, 'rb') as f:
        data = pickle.load(f)

    return data


def save_pickle(path: Path, data: any):
    with open(path, 'wb') as f:
        pickle.dump(data, f)