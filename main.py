import os
import random
import argparse
import numpy as np

import torch
import torch.nn as nn
from data.processor import DataProcessor
from utils.model_utils import make_baseline, load_config_from_file, seed_everything, get_model_cards

print('MODEL CARDS: ', get_model_cards())

parser = argparse.ArgumentParser()
parser.add_argument('--model', type=str, default='tmlp')
parser.add_argument('--dataset', type=str, default='california')
parser.add_argument('--batch_size', type=int, default=1024)
parser.add_argument('--feat_gate', type=str, default=None)
parser.add_argument('--pruning', type=str, default=None)
parser.add_argument('--n_layers', type=int, default=1)
parser.add_argument('--d_token', type=int, default=1024)
parser.add_argument('--lr', type=float, default=1e-4)
parser.add_argument('--wd', type=float, default=0)
parser.add_argument('--gpu', type=int, default=0)
parser.add_argument('--seed', type=int, default=42)
args = parser.parse_args()

device = torch.device(f'cuda:{args.gpu}')
seed_everything(args.seed)

if args.model == 'tmlp' and any([args.feat_gate, args.pruning]):
    # feature sparsity & NN sparsity
    sparsity_scheme = f'[{args.feat_gate},{args.pruning}]'
    output_dir = f'results/{args.model}{sparsity_scheme}/{args.dataset}'
else:
    output_dir = f'results/{args.model}/{args.dataset}'
# dataset
print('preparing dataset: ', args.dataset)
append_ids = args.model == 'tmlp' and args.feat_gate == 'xgb_dropout'
dataset = DataProcessor.load_preproc_default(
    output_dir, args.model, args.dataset, 
    seed=args.seed, add_ids=append_ids,
)

# model config
user_defined = False
if args.model != 'tmlp' or user_defined:
    config_file = f'configs/default/{args.model}.yaml'
    configs = load_config_from_file(config_file)
    # uniform model & training args
    configs['model']['d_ffn_factor'] = 0.66
    configs['model']['residual_dropout'] = 0.1
    configs['training']['lr'] = args.lr # 1e-4
    configs['training']['weight_decay'] = args.wd # 0
    # search spaces (can be set)
    configs['model']['n_layers'] = args.n_layers
    configs['model']['d_token'] = args.d_token
else:
    print('adaptively set T-MLP model config')
    config_file = 'configs/default/tmlp-deep.yaml' \
        if dataset.is_multiclass or dataset.size(None) >= 1e5 \
            else 'configs/default/tmlp-shallow.yaml'
    configs = load_config_from_file(config_file)

# early stop
def get_patience(dataset):
    if dataset.size(None) < 1e4: return 128
    if dataset.size(None) < 1e5 and dataset.n_features < 64: return 128
    if dataset.size(None) < 2e5 and dataset.n_features < 64: return 64
    return 32
print('adaptively set early stop') 
patience = get_patience(dataset) # 8

# batch size (based on GPU, we use GTX 3090)
def get_batch_size(dataset):
    if dataset.n_features >= 2000: return 128 + 64
    if dataset.n_features >= 1000: return 256 + 128
    if dataset.n_features >= 500: return 512 + 256
    if dataset.size('train') < 3e4: return 256
    if dataset.size('train') < 1e5: return 512
    return 1024
try:
    configs['training']['batch_size'] = args.batch_size
except:
    print('adaptively set training batch size') 
    configs['training']['batch_size'] = get_batch_size(dataset)


# meta info for evaluation metric
configs['meta'] = {
    'save_path': output_dir,
    'use_auc': dataset.is_binclass and 'openml' in args.dataset, # default is acc for non-OpenML datasets
    'use_r2': dataset.is_regression and False, # dataset in SAINT
}
print(
    'batch size: ', configs['training']['batch_size'],
    '| lr: ', args.lr,
    '| patience: ', patience,
    '| use_auc: ', configs['meta']['use_auc'],
    '| use_r2: ', configs['meta']['use_r2'],
)

# dataset args
n_num_features = dataset.n_num_features
categories = dataset.get_category_sizes('train')
categories_all = dataset.get_category_sizes(None)
if len(categories) == 0:
    categories = None
else:
    assert all(c == c2 for c, c2 in zip(categories,categories_all)), \
        "unknown categorical values appear in non-training splits"
n_labels = dataset.n_classes or 1
if dataset.is_multiclass and dataset.n_classes is None:
    n_labels = len(np.unique(dataset.y['train']))
y_std = dataset.y_info.get('std')

# model
model = make_baseline(
    args.model, configs['model'],
    n_num=n_num_features,
    cat_card=categories,
    n_labels=n_labels,
    device=device,
    feat_gate=args.feat_gate,
    pruning=args.pruning,
    dataset=dataset,
)

# convert data input to tensor
datas = DataProcessor.prepare(dataset, model)

# training
model.fit(
    X_num=datas['train'][0], X_cat=datas['train'][1], ys=datas['train'][2], ids=datas['train'][3],
    eval_set=(datas['val'], datas['test']), # add test set to report metrics each best epoch
    patience=patience, task=dataset.task_type.value,
    training_args=configs['training'],
    meta_args=configs['meta'],
)
model.load_best_dnn(output_dir)

# prediction
predictions, results = model.predict(
    X_num=datas['test'][0], X_cat=datas['test'][1], ys=datas['test'][2], y_std=y_std, ids=datas['test'][3],
    task=dataset.task_type.value,
    return_probs=True, return_metric=True, return_loss=True,
    meta_args=configs['meta'],
)
model.save_prediction(output_dir, results)

print("=== Prediction (best metric) ===")
print(results)
