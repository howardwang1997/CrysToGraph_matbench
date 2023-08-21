import os
os.environ["CUDA_VISIBLE_DEVICES"]='1'

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
parser.add_argument('--task', type=str, default='')
parser.add_argument('--checkpoint', type=str, default='')
parser.add_argument('--atom_fea_len', type=int, default=156)
parser.add_argument('--nbr_fea_len', type=int, default=76)
parser.add_argument('--batch_size', type=int, default=32)
parser.add_argument('--epochs', type=int, default=-1)
parser.add_argument('--weight_decay', type=float, default=0.0)
parser.add_argument('--lr', type=float, default=0.0001)
parser.add_argument('--grad_accum', type=int, default=1)
parser.add_argument('--milestone1', type=int, default=-1)
parser.add_argument('--milestone2', type=int, default=-1)
parser.add_argument('--rmtree', type=bool, default=True)
args = parser.parse_args()

for task in mb.tasks:
    task.load()
    classification = task.metadata['task_type'] == 'classification'
    name = task.dataset_name
    input_type = task.metadata['input_type']

    # hyperparameters
    atom_fea_len = args.atom_fea_len
    nbr_fea_len = args.nbr_fea_len
    batch_size = args.batch_size
    epochs = args.epochs
    weight_decay = args.weight_decay
    lr = args.lr
    grad_accum = args.grad_accum
    pretrained = False if args.checkpoint == '' else True

    if atom_fea_len == 156:
        embeddings_path = 'embeddings_86_64catcgcnn.pt'
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

        if epochs == -1:
            if len(train_inputs) < 2000:
                epochs = 2000
            elif len(train_inputs) < 10000:
                epochs = 1000
            elif len(train_inputs) < 20000:
                epochs = 600
                grad_accum = 2
            else:
                epochs = 500
                grad_accum = 8

        milestone2 = 99999
        if args.milestone1 > 0:
            milestone1 = args.milestone1
            if args.mileston2 > milestone1:
                milestone2 = args.milestone2
        else:
            milestone1 = int(epochs/3)

        milestones = [milestone1, milestone2]

        # define atom_vocab, dataset, model, trainer
        embeddings = torch.load(embeddings_path).cuda()
        atom_vocab = joblib.load('atom_vocab.jbl')
        cd = CrystalDataset(root=name,
                            atom_vocab=atom_vocab,
                            inputs=train_inputs,
                            outputs=train_outputs)
        module = nn.ModuleList([TransformerConvLayer(256, 32, 8, edge_dim=76, dropout=0.0) for _ in range(5)]), \
                 nn.ModuleList([TransformerConvLayer(76, 24, 8, edge_dim=30, dropout=0.0) for _ in range(5)])
        drop = 0.0 if not classification else 0.2
        ctgn = CrysToGraphNet(atom_fea_len, nbr_fea_len, embeddings=embeddings, h_fea_len=256, n_conv=3, n_fc=2, module=module, norm=True, drop=drop)
        if pretrained:
            ctgn.load_state_dict(torch.load(args.checkpoint), strict=False)
        optimizer = optim.AdamW(ctgn.parameters(), lr=lr, betas=(0.9, 0.99), weight_decay=weight_decay)
        scheduler = WarmupMultiStepLR(optimizer, milestones, gamma=0.1)
        trainer = Trainer(ctgn, name='%s_%d' % (name, fold), classification=classification)

        # train
        train_loader = DataLoader(cd, batch_size=batch_size, shuffle=True, collate_fn=cd.collate_line_graph)
        trainer.train(train_loader=train_loader,
                      optimizer=optimizer,
                      epochs=epochs,
                      scheduler=scheduler,
                      grad_accum=grad_accum)

        # predict
        test_inputs, test_outputs = task.get_test_data(fold, include_target=True)
        cd = CrystalDataset(root=name,
                            atom_vocab=atom_vocab,
                            inputs=test_inputs,
                            outputs=test_outputs)
        test_loader = DataLoader(cd, batch_size=2, shuffle=False, collate_fn=cd.collate_line_graph)
        predictions = trainer.predict(test_loader=test_loader)

        # record
        task.record(fold, predictions)
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
    if args.rmtree:
        shutil.rmtree(name)

hyperparam = vars(args)
# hyperparam['grad_accum'] = grad_accum
# hyperparam['epochs'] = epochs
metadata = {
    "algorithm_version": "v5.3.7",
    "hyperparameters": hyperparam,
    "embeddings": embeddings_path
}

mb.add_metadata(metadata)

mb.to_file("CrysToGraph_benchmark%s.json.gz" % args.task)
