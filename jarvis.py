import json
import os
import random
from tqdm import tqdm
import joblib
import argparse

from jarvis.core.atoms import Atoms
from pymatgen.io.jarvis import JarvisAtomsAdaptor
from sklearn.model_selection import KFold


DATASETS_LEN = {
    'jarvis_supercon': 1058,
    'jarvis_exfoliation': 4527,
    'jarvis_magnetization': 6351,
    'jarvis_2d_gap': 3520,
    'jarvis_2d_e_tot': 3520,
    'jarvis_2d_e_fermi': 3520,
    'jarvis_qmof_energy': 20425,
    'jarvis_co2_adsp': 137652,
    'jarvis_surface': 137652,
    'jarvis_vacancy': 530,
}

def jarvis_dataset_to_mp(dataset, label, save=False):
    data = [d['atoms'] for d in dataset]
    labels = [d[label] for d in dataset]
    converter = JarvisAtomsAdaptor()
    data = [converter.get_structure(Atoms.from_dict(d)).as_dict() for d in data]
    data_matbench = [[data[i], labels[i]] for i in range(len(data))]
    index = list(range(len(data)))
    assert len(data) == len(labels)

    final = {
        "index": index,
        "columns": ['structure', label],
        "data": data_matbench
    }
    if save:
        with open(label+'.json', 'w+') as f:
            json.dump(final, f)
    return final


def make_validation(name, length, n_folds=5, random_seed=42):
    kf = KFold(n_folds, random_state=random_seed)



def mp_to_tasks(dataset, random_seed=42):
    pass


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('dataset', type=str, help='dataset to be processed, must be jarvis dataset.')
    parser.add_argument('--work', type=str, help='target directory.')
    parser.add_argument('--label', type=str, help='label name')
    parser.add_argument('--ratio', type=float, default=0.9, help='ratio of train set')
    parser.add_argument('--split', type=str, help='path to split file')
    args = parser.parse_args()

    with open(args.dataset) as f:
        dataset = json.load(f)
    make_dir(args.work)

    if args.split:
        train_idx, val_idx = joblib.load(args.split)
    else:
        train_idx, val_idx = split_dataset(dataset, args.ratio)

    process_jarvis_dataset(dataset, args.work, args.label, train_idx, val_idx)

