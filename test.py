import os
import joblib
import argparse
import shutil

import torch
from torch import nn, optim
from torch.utils.data import DataLoader

from matbench.bench import MatbenchBenchmark

from data import CrystalDataset
from train import Trainer
from model.NN import CrysToGraphNet
from model.bert_transformer import TransformerConvLayer
from model.scheduler import WarmupMultiStepLR

mb = MatbenchBenchmark(autoload=False)
mb = mb.from_preset('matbench_v0.1', 'structure')

parser = argparse.ArgumentParser(description='Run CrysToGraph on matbench.')
parser.add_argument('checkpoint', type=str)
parser.add_argument('--task', type=str, default='dielectric')
parser.add_argument('--atom_fea_len', type=int, default=156)
parser.add_argument('--nbr_fea_len', type=int, default=76)
parser.add_argument('--batch_size', type=int, default=10)
parser.add_argument('--rmtree', type=bool, default=False)
args = parser.parse_args()

for task in mb.tasks:
    task.load()
    classification = task.metadata['task_type'] == 'classification'
    name = task.dataset_name
    input = task.metadata['input_type']

    # hyperparameters
    atom_fea_len = args.atom_fea_len
    nbr_fea_len = args.nbr_fea_len
    batch_size = args.batch_size

    if atom_fea_len == 156:
        embeddings_path = 'embeddings_84_64catcgcnn.pt'
    elif atom_fea_len == 92:
        embeddings_path = 'embeddings_84_cgcnn.pt'
    else:
        embeddings_path = ''

    if args.task not in name:
        continue

    # mkdir
    try:
        os.mkdir(name)
    except FileExistsError:
        pass

    for fold in task.folds:
        train_inputs, train_outputs = task.get_train_and_val_data(fold)

        if len(train_inputs) < 2000:
            epochs = 2000
        elif len(train_inputs) < 10000:
            epochs = 1000
        elif len(train_inputs) < 20000:
            epochs = 600
            grad_accum = 2
        else:
            grad_accum = 8

        # define atom_vocab, dataset, model, trainer
        embeddings = torch.load(embeddings_path).cuda()
        atom_vocab = joblib.load('atom_vocab.jbl')
        module = nn.ModuleList([TransformerConvLayer(256, 32, 8, edge_dim=76, dropout=0.0) for _ in range(3)]), \
                 nn.ModuleList([TransformerConvLayer(76, 24, 8, edge_dim=30, dropout=0.0) for _ in range(3)])
        ctgn = CrysToGraphNet(atom_fea_len, nbr_fea_len, embeddings=embeddings, h_fea_len=256, n_conv=3, n_fc=2, module=module, norm=True)
        ctgn.load_state_dict(torch.load(args.checkpoint))
        trainer = Trainer(ctgn, name='%s_%d' % (name, fold), classification=classification)

        # predict
        test_inputs, test_outputs = task.get_test_data(fold, include_target=True)
        cd = CrystalDataset(root=name,
                            atom_vocab=atom_vocab,
                            inputs=test_inputs,
                            outputs=test_outputs)
        test_loader = DataLoader(cd, batch_size=batch_size, shuffle=False, collate_fn=cd.collate_line_graph)
        predictions = trainer.predict(test_loader=test_loader)

        if torch.cuda.is_available():
            torch.cuda.empty_cache()
    if args.rmtree:
        shutil.rmtree(name)