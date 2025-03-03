# Sparser class: NN Pruning & Feature Gating Modules
# Reference for CoFi Pruning: https://github.com/princeton-nlp/CoFiPruning
# Reference for LSPIN Gating: https://github.com/jcyang34/lspin
# collections of NN sparse methods
import os
import gc
import pdb
import math
import logging
from typing import List, Optional
import numpy as np

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable
from torch.optim import AdamW

# Hummingbird version is strict for this implementation
from hummingbird.ml import convert
from hummingbird.ml.operator_converters._tree_implementations import PerfectTreeTraversalGBDTImpl
# For Tree Dropout
from xgboost import XGBClassifier, XGBRegressor

limit_a, limit_b, epsilon = -.1, 1.1, 1e-6

def make_sparser(model, pruning_type, feature_gate, device, dataset=None):
    return Sparser(
        model, 
        pruning_type=pruning_type, 
        feature_gate=feature_gate, 
        dataset=dataset
    ).to(device)

def log_params(param_groups, des):
    for i, grouped_parameters in enumerate(param_groups):
        print(
            f"{des}, number of params: {sum(p.nelement() for p in grouped_parameters['params'])}, \
                weight_decay: {grouped_parameters['weight_decay']}, lr: {grouped_parameters['lr']}")


def quantile_concrete(x, qz_loga, temperature):
    """Implements the quantile, aka inverse CDF, of the 'stretched' concrete distribution"""
    y = F.sigmoid((torch.log(x) - torch.log(1 - x) + qz_loga) / temperature)
    return y * (limit_b - limit_a) + limit_a

def get_eps(size, device):
    """Uniform random numbers for the concrete distribution"""
    eps = torch.FloatTensor(size, device=device).uniform_(epsilon, 1-epsilon)
    eps = Variable(eps)
    return eps

def sample_z(x, qz_loga, temperature, sample=True):
    """Sample the hard-concrete gates for training and use a deterministic value for testing"""
    batch_size, in_features = x.shape[:2]
    if sample:
        eps = get_eps(x.shape[:2], device=x.device)
        z = quantile_concrete(eps, qz_loga, temperature)
        return F.hardtanh(z, min_val=0, max_val=1)
    else:  # mode
        pi = F.sigmoid(qz_loga).view(1, in_features).expand(batch_size, in_features)
        return F.hardtanh(pi * (limit_b - limit_a) + limit_a, min_val=0, max_val=1)

class MultiHeadFeatureGate(nn.Module):
    """Some Feature Gating Mechanisms"""
    def __init__(
        self, 
        n_tokens, d_token, n_head=4, dropout_rate=0.5, 
        stype='specific', local_rep=False,
    ):
        super().__init__()
        assert stype in ['lspin', 'specific', 'shared']
        self.stype = stype
        if stype.startswith('lspin'):
            # LSPIN Feature Gating Network
            self.lspin = LSPIN(n_tokens=n_tokens, d_token=d_token)
        elif stype == 'specific': 
            # sample-wise feature gate in inference, straight through trick
            assert d_token % n_head == 0
            self.n_head = n_head
            self.W = nn.Parameter(torch.Tensor(1, 1, n_head, d_token//n_head, 1))
            nn.init.kaiming_uniform_(self.W, a=math.sqrt(5))
            self.dropout_rate = dropout_rate
        elif stype == 'shared':
            # shared feature selection, L0-norm parameterized mask
            self.local_rep = local_rep # sample-wise gate in l0 training process
            self.in_features = n_tokens
            self.qz_loga = nn.Parameter(torch.Tensor(n_tokens))
            self.temperature = 2./3.
            self.dropout_rate = dropout_rate
            self.qz_loga.data.normal_(math.log(1 - self.dropout_rate) - math.log(self.dropout_rate), 1e-2)
            
    def forward(self, x):
        if self.stype.startswith('lspin'):
            mask = self.lspin(x)
            # retain [CLS]
            cls_mask = torch.zeros_like(mask, device=mask.device)
            cls_mask[:, 0].data.fill_(1.)
            mask = mask + (1.0 - mask) * cls_mask
        elif self.stype == 'specific':
            # x: bnd, normalized
            b, n, d = x.shape
            x = x.reshape(b, n, self.n_head, d // self.n_head)
            mask = x @ self.W
            mask = mask.reshape(b, n, self.n_head)
            mask = mask.mean(-1).sigmoid()
            mask = (mask > self.dropout_rate).float() - mask.detach() + mask
        elif self.stype == 'shared':
            if self.local_rep or not self.training:
                mask = sample_z(x, self.qz_loga, self.temperature, sample=self.training)
            else:
                mask = quantile_concrete(get_eps((self.in_features,), device=x.device), self.qz_loga, self.temperature)
                mask = F.hardtanh(mask, min_val=0, max_val=1).view(1, self.in_features)
        return mask.unsqueeze(-1)

class L0CoFi(nn.Module):
    """NN param Pruning module
    
    ---
    Reference
    - [CoFi](https://aclanthology.org/2022.acl-long.107/)
    - https://github.com/princeton-nlp/CoFiPruning
    """
    def __init__(
        self,
        n_layers,
        d_token,
        d_hidden,
        droprate_init=0.5,
        lagrangian_warmup=0,
        start_sparsity=0.0,
        target_sparsity=0.5,
        temperature=2./3.,
        magical_number=0.8,
        pruning_type="mlp+sgu+layer",
    ):
        super().__init__()
        self.all_types = ['layer_z', 'intermediate_z', 'sgu_z', 'hidden_z']
        self.pruning_type = pruning_type

        self.n_layers = n_layers
        self.d_hidden = d_token
        self.d_intermediate = d_hidden
        
        self.temperature = temperature
        self.droprate_init = droprate_init if droprate_init != 0. else 0.5

        self.types = []
        self.z_logas = {}
        self.parameters_per_dim = {}
        self.sizes = {}
        self.shapes = {}

        self.params_per_mlp_layer = self.d_hidden * self.d_intermediate * 2 + self.d_hidden + self.d_intermediate # in BERT, d_intermediate = 4 * d_hidden
        self.params_per_intermediate_dim = self.params_per_mlp_layer // self.d_intermediate
        # omit dataset-specific params (embedding)
        self.params_per_sgu = self.d_hidden * self.d_intermediate + self.d_intermediate # + self.n_tokens ** 2 + self.n_tokens
        self.params_per_layer = self.params_per_mlp_layer + self.params_per_sgu

        self.hidden_loga = None
        self.hidden_type = None
        self.prunable_model_size = 0

        types = self.pruning_type.split('+')
        for type in types:
            if type != 'layer':
                self.init_one_module(type)
        if 'layer' in types:
            self.init_one_module('layer')
        # TODO: some missing shape infos in calculating model size
        if 'hidden' not in types:
            self.shapes['hidden'] = [self.d_hidden]
        if 'layer' not in types:
            self.shapes['layer'] = [self.n_layers]
        
        self.magical_number = magical_number
        self.lambda_1 = torch.nn.Parameter(torch.tensor(0.0))
        self.lambda_2 = torch.nn.Parameter(torch.tensor(0.0))

        self.lagrangian_warmup = lagrangian_warmup
        self.start_sparsity = start_sparsity
        self.target_sparsity = target_sparsity

        print("***** Init L0 Module *****")
        for type in self.types:
            print(f"*** {type} ***")
            print(f"z.shape", self.z_logas[type].shape)
            print(f"size", self.sizes[type])
        print(f"prunable model size: {self.prunable_model_size}")
    
    def set_lagrangian_warmup_steps(self, lagrangian_warmup):
        self.lagrangian_warmup = lagrangian_warmup

    def init_one_module(self, module_name):
        if module_name == 'mlp':
            self.init_pruned_mlp()
        elif module_name == 'sgu':
            self.init_pruned_sgu()
        elif module_name == 'hidden':
            self.init_hidden()
        elif module_name == 'layer':
            self.init_whole_layer()

    def init_parameters(self, size, num_layer=None):
        if num_layer is not None:
            return nn.Parameter(torch.Tensor(num_layer, size))
        else:
            return nn.Parameter(torch.Tensor(size))
    def reset_loga(self, tensor, mean=None):
        if mean is None:
            mean = math.log(1 - self.droprate_init) - math.log(self.droprate_init)
        tensor.data.normal_(mean, 1e-2)
    def add_one_module(self, z_loga, type, parameter_per_dim, size, shape):
        self.types.append(type)
        self.z_logas[type] = z_loga
        self.parameters_per_dim[type] = parameter_per_dim
        self.sizes[type] = size
        self.shapes[type] = shape
    def init_hidden(self):
        self.hidden_loga = self.init_parameters(self.d_hidden) # shared across layers, diag(z)W
        self.add_one_module(
            self.hidden_loga, type='hidden',
            parameter_per_dim=self.d_intermediate, size=self.d_hidden,
            shape=[self.d_hidden])
        self.reset_loga(self.hidden_loga, mean=0.5) # all 1s at the beginning
    def init_pruned_mlp(self):
        self.int_loga = self.init_parameters(self.d_intermediate, self.n_layers)
        self.add_one_module(
            self.int_loga, type='intermediate',
            parameter_per_dim=self.params_per_intermediate_dim, size=self.d_intermediate,
            shape=[self.n_layers, self.d_intermediate])
        self.prunable_model_size += self.n_layers * self.params_per_mlp_layer
        self.reset_loga(self.int_loga)
    def init_pruned_sgu(self):
        self.intsgu_loga = self.init_parameters(self.n_layers)
        self.add_one_module(
            self.intsgu_loga, type='sgu',
            parameter_per_dim=self.params_per_sgu, size=1, # one sgu per layer
            shape=[self.n_layers])
        self.prunable_model_size += self.n_layers * self.params_per_sgu
        self.reset_loga(self.intsgu_loga, mean=0.5)
    def init_whole_layer(self):
        self.intlayer_loga = self.init_parameters(self.n_layers)
        self.add_one_module(
            self.intlayer_loga, type='layer',
            parameter_per_dim=self.params_per_layer, size=1,
            shape=[self.n_layers])
        self.reset_loga(self.intlayer_loga, mean=0.5)
    

    def constrain_parameters(self):
        def _constrain(tensor):
            tensor.data.clamp_(min=math.log(1e-2), max=math.log(1e2))
        for key in self.z_logas:
            _constrain(self.z_logas[key])
        
    def cdf_qz(self, x, loga):
        """Implements the CDF of the 'stretched' concrete distribution"""
        xn = (x - limit_a) / (limit_b - limit_a)
        logits = math.log(xn) - math.log(1 - xn)
        return torch.sigmoid(logits * self.temperature - loga).clamp(min=epsilon, max=1 - epsilon)

    def quantile_concrete(self, x, loga):
        y = torch.sigmoid((torch.log(x) - torch.log(1 - x) + loga) / self.temperature)
        return y * (limit_b - limit_a) + limit_a
    
    def get_eps(self, size):
        """Uniform random numbers for the concrete distribution"""
        eps = torch.FloatTensor(size).uniform_(epsilon, 1-epsilon)
        eps = Variable(eps)
        return eps

    
    def get_num_parameters_for_one(self, loga, parameter_size):
        return torch.sum(1 - self.cdf_qz(0, loga)) * parameter_size
    
    def get_num_parameters_for_layer(self):
        intlayer_score = 1 - self.cdf_qz(0, self.intlayer_loga) # L
        int_score = 1 - self.cdf_qz(0, self.int_loga) # L, d_ffn
        intlayer_score = intlayer_score.unsqueeze(-1)

        num_parameters = torch.sum(intlayer_score * int_score) * self.parameters_per_dim["intermediate"]

        # spatial gate unit
        intsgu_score = 1 - self.cdf_qz(0, self.intsgu_loga) # L
        intsgu_score = intsgu_score.unsqueeze(-1)
        # per intermediate dim rather than per layer, using / not // for backward
        num_parameters += torch.sum(intlayer_score * intsgu_score * int_score) * self.parameters_per_dim["sgu"] / self.d_intermediate
        return num_parameters
    
    def get_num_parameters_and_constraint_for_hidden(self): # calculate the current sparsity
        num_parameters = 0
        
        hidden_score = 1 - self.cdf_qz(0, self.hidden_loga) # d
        intlayer_score = 1 - self.cdf_qz(0, self.intlayer_loga) # L
        int_score = 1 - self.cdf_qz(0, self.int_loga) # L, d_ffn
        intlayer_score = intlayer_score.unsqueeze(-1)

        _int_score = (intlayer_score * int_score).reshape(-1)
        num_parameters = torch.sum(torch.outer(hidden_score, _int_score)) * 2 # L * d * d_ffn * 2

        # spatial gate unit
        intsgu_score = 1 - self.cdf_qz(0, self.intsgu_loga) # L
        intsgu_score = intsgu_score.unsqueeze(-1)
        _intsgu_score = (intlayer_score * intsgu_score * int_score).reshape(-1)
        num_parameters += torch.sum(torch.outer(hidden_score, _intsgu_score))
        return num_parameters

    def get_target_sparsity(self, pruned_steps):
        target_sparsity = (self.target_sparsity - self.start_sparsity) \
            * min(1, pruned_steps / max(1e-6, self.lagrangian_warmup)) + self.start_sparsity
        return target_sparsity
    
    def lagrangian_regularization(self, pruned_steps):
        target_sparsity = self.target_sparsity
        if "hidden" in self.types:
            expected_size = self.get_num_parameters_and_constraint_for_hidden()
        else:
            expected_size = self.get_num_parameters_for_layer()
        expected_sparsity = 1 - expected_size / self.prunable_model_size
        if self.lagrangian_warmup > 0:
            target_sparsity = self.get_target_sparsity(pruned_steps)
        lagrangian_loss = (
            self.lambda_1 * (expected_sparsity - target_sparsity)
            + self.lambda_2 * (expected_sparsity - target_sparsity) ** 2
        )
        return lagrangian_loss, expected_sparsity, target_sparsity


    # during training
    def _sample_z(self, loga):
        eps = self.get_eps(torch.FloatTensor(*loga.shape)).to(loga.device)
        z = self.quantile_concrete(eps, loga)
        z = F.hardtanh(z, min_val=0, max_val=1)
        return z

    # during inference
    def _deterministic_z(self, size, loga):
        expected_num_nonzeros = torch.sum(1 - self.cdf_qz(0, loga))
        expected_num_zeros = size - expected_num_nonzeros.item()
        try:
            num_zeros = round(expected_num_zeros)
        except:
            pdb.set_trace()
        soft_mask = torch.sigmoid(loga / self.temperature * self.magical_number)
        if num_zeros > 0:
            if soft_mask.ndim == 0:
                soft_mask = torch.tensor(0).to(loga.device)
            else:
                _, indices = torch.topk(soft_mask, k=num_zeros, largest=False)
                soft_mask[indices] = 0.
        return soft_mask
    
    def get_z_from_zs(self, zs):
        numpified_zs = {}
        for type in self.all_types:
            name = type[:-2]
            z = zs.get(type, np.ones(self.shapes[name]))
            if torch.is_tensor(z):
                z = z.squeeze().detach().cpu().numpy()
            new_z = z > 0
            numpified_zs[name] = new_z
        return numpified_zs
    
    def calculate_model_size(self, zs):
        numpified_zs = self.get_z_from_zs(zs)
        hidden_z = numpified_zs['hidden']
        intermediate_z = numpified_zs['intermediate']
        sgu_z = numpified_zs['sgu'].reshape(-1, 1)
        layer_z = numpified_zs['layer'].reshape(-1, 1)

        remaining_hidden_dims = hidden_z.sum().item()
        remaining_intermediate_nums = intermediate_z.reshape(self.n_layers, self.d_intermediate).sum(-1).tolist()
        
        intermediate_nums = np.outer((intermediate_z * layer_z).reshape(-1), hidden_z).sum().item()
        sgu_nums = np.outer((intermediate_z * sgu_z * layer_z).reshape(-1), hidden_z).sum().item()

        remaining_model_size = intermediate_nums * 2 + sgu_nums
        pruned_model_size = self.prunable_model_size - remaining_model_size

        results = {}
        results['layers'] = layer_z.reshape(-1).astype(int).tolist()
        results['hidden_dims'] = remaining_hidden_dims
        results['intermediate_dims'] = remaining_intermediate_nums
        results['sgu_dims'] = sgu_nums
        results['pruned_params'] = pruned_model_size
        results['remaining_params'] = remaining_model_size
        results['pruned_model_sparsity'] = pruned_model_size / self.prunable_model_size

        print(f"remaining_layers: {layer_z}")
        print(f"remaining_sgus: {sgu_z}")
        print(f"remaining_hidden_dims: {remaining_hidden_dims}")
        print(f"remaining_intermediate_nums: {remaining_intermediate_nums}")
        print(f"pruned_model_size: {pruned_model_size}")
        print(f"remaining_model_size: {remaining_model_size}")

        return results

    
    def forward(self, training=True):
        zs = {f'{type}_z': [] for type in self.types}

        if training:
            for i, type in enumerate(self.types):
                loga = self.z_logas[type]
                z = self._sample_z(loga)
                zs[f'{type}_z'] = z.reshape(self.shapes[type])
        else:
            for i, type in enumerate(self.types):
                if type != 'hidden': # hidden is not a per layer sample
                    loga_all_layers = self.z_logas[type]
                    for layer in range(len(loga_all_layers)):
                        loga = loga_all_layers[layer]
                        size = self.sizes[type]
                        z = self._deterministic_z(size, loga)
                        zs[f'{type}_z'].append(z.reshape(self.shapes[type][1:]))
                else:
                    z = self._deterministic_z(self.sizes[type], self.hidden_loga)
                    zs[f'{type}_z'] = z
            for type in zs:
                if type != 'hidden_z':
                    zs[type] = torch.stack(zs[type])
        return zs

class Sparser(nn.Module):
    """Uniform class for Feature Gating & NN Pruning"""
    PRUNING_TYPES = ['mlp+sgu+layer', 'hidden+mlp+sgu+layer']
    FEAT_GATING_TYPES = ['xgb_dropout', 'lspin', 'specific', 'shared']
    def __init__(
        self, 
        model, 
        n_layers=None, 
        n_tokens=None, 
        d_token=None, 
        d_hidden=None,
        target_sparsity=0.33,
        lagrangian_warmup=0,
        pruning_type: Optional[str] = 'mlp+sgu+layer',
        feature_gate: Optional[str] = 'xgb_dropout',
        dataset=None, # for tree dropout (tree fitting)
    ):
        super().__init__()
        assert pruning_type is None or pruning_type in self.PRUNING_TYPES
        assert feature_gate is None or feature_gate in self.FEAT_GATING_TYPES
        # target model args
        n_layers = n_layers or len(model.layers)
        n_tokens = n_tokens or model.tokenizer.n_tokens
        d_token = d_token or model.tokenizer.weight.shape[1]
        d_hidden = d_hidden or model.d_ffn

        # learnable parameter mask (global, CoFi)
        if pruning_type != 'none':
            self.l0_module = L0CoFi(
                n_layers, d_token, d_hidden, 
                target_sparsity=target_sparsity, 
                lagrangian_warmup=lagrangian_warmup, 
                pruning_type=pruning_type)
        else:
            self.l0_module = None
        
        # feature gating
        if feature_gate != 'none':
            if feature_gate != 'xgb_dropout':
                self.feature_gate = MultiHeadFeatureGate(
                    n_tokens, d_token, stype=feature_gate
                ) # input feature mask (sample-wise or shared)
            else:
                assert dataset is not None, 'tree dropout require fitting tree first'
                self.feature_gate = XGBDropout(dataset)
        else:
            self.feature_gate = None

    def forward(self, x: torch.Tensor, x2=None, is_raw_input=True):
        zs = {}
        if not is_raw_input:
            if self.l0_module is not None:
                zs = self.l0_module(self.training)
            if self.feature_gate is not None and not isinstance(self.feature_gate, XGBDropout):
                zs['feature_z'] = self.feature_gate(x)
        else:
            if isinstance(self.feature_gate, XGBDropout):
                zs['feature_z'] = self.feature_gate.get_z(x, x2, training=self.training)
        return zs
    
    def make_optimizer(self, reg_lr):
        l0_params = [{
            "params": [p for n, p in self.named_parameters() if "lambda" not in n],
            "weight_decay": 0.0,
            "lr": reg_lr,
        }]
        log_params(l0_params, "l0 reg params")
        l0_optimizer = AdamW(l0_params)
        
        lagrangian_params = [{
            "params": [p for n, p in self.named_parameters() if "lambda" in n],
            "weight_decay": 0.0,
            "lr": -reg_lr,
        }]
        log_params(lagrangian_params, "l0 reg lagrangian params")
        lagrangian_optimizer = AdamW(lagrangian_params)
        return l0_optimizer, lagrangian_optimizer
    
    def regularization(self, step):
        loss = 0
        if self.l0_module is not None:
            loss += self.l0_module.lagrangian_regularization(step)[0]
        if self.feature_gate is not None and hasattr(self.feature_gate, 'lspin'):
            loss += self.feature_gate.lspin.regularization()
        return loss


class XGBDropout(nn.Module):
    """xgboost-based dropout
    
    sample_feature_frequency: Sample-wise GBDT Feature Frequency in the paper
    """
    def __init__(self, dataset, save_path=None, drop_rate=0.15):
        super().__init__()
        save_path = save_path or 'xgboost_cache' # model save path
        default_path = f'{save_path}/{dataset.n_num_features}-{dataset.n_cat_features}-{dataset.size(None)}.pt'
        self.drop_rate = drop_rate
        if os.path.exists(default_path):
            # no need to fit xgboost and put to the cuda
            self.cache_frequency(dataset, save_path)
        else:
            self.gbdt = self.fetch_gbdt(save_path, dataset) # fit a tree model with the dataset
            self.to_tensor() # tensorize the fitted tree model
            # self.sample_feature_frequency: torch.Tensor = None
            self.cache_frequency(dataset, save_path) # cache per-sample gbdt feature frequency
            assert len(self.sample_feature_frequency) == dataset.size(None)
            del self.operator, self.pt_gbdt, self.gbdt
            torch.cuda.empty_cache()
            gc.collect()
    
    def to_tensor(self):
        """convert sklearn-format tree models into Hummingbird class"""
        self.pt_gbdt = convert(self.gbdt, 'pytorch')
        self.pt_gbdt.to('cuda')
        self.operator: PerfectTreeTraversalGBDTImpl = self.pt_gbdt.model._operators[0]

    def fetch_gbdt(self, save_path, dataset):
        """Fit a tree model"""
        if save_path is None:
            save_path = 'xgboost_cache'
        if not os.path.isdir(save_path):
            os.makedirs(save_path)
        save_path = f'{save_path}/{dataset.n_num_features}-{dataset.n_cat_features}-{dataset.size(None)}.model'
        from model_utils import load_config_from_file
        configs = load_config_from_file('configs/default/xgboost.yaml')
        # change configs to prevent OOM for large datasets
        # if dataset.n_features < 32:
        #     pass
        # elif dataset.n_features < 64:
        #     configs['model']['n_estimators'] = 600
        #     configs['fit']['early_stopping_rounds'] = 599
        # else:
        #     configs['model']['n_estimators'] = 200
        #     configs['fit']['early_stopping_rounds'] = 199
        if dataset.is_regression:
            model = XGBRegressor(**configs['model'])
            predict = model.predict
        else:
            model = XGBClassifier(**configs['model'], disable_default_eval_metric=True)
            if dataset.is_multiclass:
                configs['fit']['eval_metric'] = 'merror'
                predict = model.predict_proba
            else:
                configs['fit']['eval_metric'] = 'error'
                predict = lambda x: model.predict_proba(x)[:, 1]

        Xs = {k: (
            dataset.X_num[k] if dataset.X_cat is None 
            else np.concatenate([dataset.X_num[k], dataset.X_cat[k]], axis=1)
        ) for k in ['train', 'val', 'test']}
        ys = {
            k: np.argmax(dataset.y[k], axis=1) if (
            dataset.y['train'].ndim == 2 and dataset.task_type.value == 'multiclass')
            else (dataset.y[k] == dataset.y[k].max()).astype(np.float32) if (
                dataset.task_type.value == 'binclass' and dataset.y[k].max() != 1.0)
            else dataset.y[k] for k in ['train', 'val', 'test']}

        if not os.path.exists(save_path):
            print('fitting xgboost with default configs')
            model.fit(
                Xs['train'],
                ys['train'],
                eval_set=[(Xs['val'], ys['val'])],
                **configs['fit'])
            model.save_model(save_path)
        else:
            print('loading saved model')
            model.load_model(save_path)
        # test XGB performance
        from utils.metrics import calculate_metrics
        prediction = {k: predict(v) for k, v in Xs.items()}
        prediction_type = None if dataset.is_regression else 'probs'
        scores = {k:  calculate_metrics(
            ys[k], prediction[k], dataset.task_type.value, prediction_type, 
            y_std=None if not dataset.is_regression else dataset.y_info['std']) for k in prediction}
        print("XGB scores")
        print(scores)
        return model
    
    def cache_frequency(self, dataset, save_path):
        """calculate and cache per-sample GBDT feature frequency"""
        if save_path is None:
            save_path = 'xgboost_cache'
        save_path = f'{save_path}/{dataset.n_num_features}-{dataset.n_cat_features}-{dataset.size(None)}.pt'
        if os.path.exists(save_path):
            print('read cached feature gbdt frequency')
            frequency = torch.load(save_path)
        else:
            print('no cached frequency, infer new cache')
            Xs = np.concatenate([
                dataset.X_num[k] if dataset.X_cat is None
                else np.concatenate([dataset.X_num[k], dataset.X_cat[k]], axis=1)
                for k in ['train', 'val', 'test']
            ], axis=0)
            # ys = np.concatenate([dataset.y[k] for k in ['train', 'val', 'test']])
            N = dataset.size(None)
            assert Xs.shape[0] == N
            batch_size = 64 if Xs.shape[1] < 32 else 16 if Xs.shape[1] < 64 else 4
            steps = math.ceil(N / batch_size)
            frequency = []
            for step in range(steps):
                X = Xs[batch_size*step:min(batch_size*(step+1), N)]
                X = torch.from_numpy(X).cuda()
                frequency.append(self.count_feature_frequency(None, X)[0])
            frequency = torch.cat(frequency).cpu()
            torch.save(frequency, save_path)
            print('cached frequency')
        self.register_buffer('sample_feature_frequency', frequency) # cache the gbdt frequency
        
    def count_feature_frequency(self, sample_ids, x: torch.Tensor):
        """Implementation: GBDT feature frequency calculation (apply once per sample)"""
        b, f = x.shape
        # fast gbdt feature frequency access using sample ids
        if sample_ids is not None:
            return self.sample_feature_frequency[sample_ids], f
        # total freqency count / sample
        # TODO: count for each tree group of one class
        tot_frequency = torch.zeros(b, f).long().cuda()
        # root amount per feature
        root_features = self.operator.root_nodes.data
        root_feature_counts = torch.bincount(root_features, minlength=f).view(1, f)
        tot_frequency += root_feature_counts

        # range tensor for comparison
        range_tensor = torch.arange(f).view(1,1,-1).cuda()

        # inference
        self.operator.eval()
        with torch.no_grad():
            prev_indices = (self.operator.decision_cond(
                torch.index_select(x, 1, self.operator.root_nodes), 
                self.operator.root_biases)).long()
            prev_indices = prev_indices + self.operator.tree_indices
            prev_indices = prev_indices.view(-1)

            factor = 2
            for nodes, biases in zip(self.operator.nodes, self.operator.biases):
                gather_indices = torch.index_select(nodes, 0, prev_indices).view(-1, self.operator.num_trees)
                # count each tree layer
                tot_frequency += (gather_indices.unsqueeze(2) == range_tensor).sum(1)
                features = torch.gather(x, 1, gather_indices).view(-1)
                prev_indices = (
                    factor * prev_indices + self.operator.decision_cond(
                        features, torch.index_select(biases, 0, prev_indices)).long())
        return tot_frequency, f
    
    def get_z(self, x_num: torch.Tensor, x_cat, stochastic=True, training=True):
        """generate feature mask using per-sample GBDT feature frequency"""
        assert self.training == training
        if isinstance(x_num, tuple):
            sample_ids, x_num = x_num
        else:
            sample_ids = None
        x = x_num if x_cat is None else torch.cat([x_num, x_cat], dim=-1) # b, f
        frequency, n_feature = self.count_feature_frequency(sample_ids, x)
        n_remain = min(math.ceil(n_feature * (1 - self.drop_rate)), n_feature - 1)
        if not stochastic:
            # deterministic feature drop for each sample
            freq_rank = frequency.argsort(-1, descending=True)
            used_features = freq_rank[:, :n_remain]
        elif training: # different? BUG: if using else, the multinomial will be called more times, it can be a hyper-param
            # stochastic feature drop during training
            norm_frequency = frequency / frequency.sum(1, keepdim=True)
            used_features = torch.multinomial(norm_frequency, n_remain, replacement=False) # False or True?
        if stochastic and not training:
            # stochastic drop during inference is invalid, keep all features in default
            feature_mask = torch.ones_like(x)
        else:
            feature_mask = torch.zeros_like(x)
            feature_mask.scatter_(1, used_features, 1) # zero out unused feature index
        feature_mask = F.pad(feature_mask, pad=(1,0,0,0), value=1.) # [CLS] token kept
        return feature_mask.unsqueeze(2)


class LSPIN(nn.Module):
    """LSPIN gating network
    
    ---
    Reference
    - https://github.com/jcyang34/lspin
    """
    def __init__(
        self,
        *,
        a = 1,
        sigma = 0.5,
        lam = 5e-3,
        activation_gating=F.tanh,
        gamma1 = 0,
        gamma2 = 0,
        n_tokens,
        d_token,
        gating_net_hidden_layers_node: List[int] = [128],
        compute_sim: bool = False,
    ):
        super().__init__()
        # hyper-param
        self.a = a
        self.sigma = sigma
        self.lam = lam
        self.activation_gating = activation_gating
        self.gamma1 = gamma1
        self.gamma2 = gamma2
    
        # gating network
        self.gatesweights = []
        self.gatesbiases = []
        prev_node = n_tokens
        for i in range(len(gating_net_hidden_layers_node)):
            weights = nn.Parameter(torch.Tensor(prev_node, gating_net_hidden_layers_node[i]))
            nn.init.kaiming_normal_(weights, a=math.sqrt(5)) # different init
            biases = nn.Parameter(torch.Tensor(gating_net_hidden_layers_node[i]))
            nn.init.zeros_(biases)
            self.gatesweights.append(weights)
            self.gatesbiases.append(biases)
            prev_node = gating_net_hidden_layers_node[i]
        self.gatesweights = nn.ParameterList(self.gatesweights)
        self.gatesbiases = nn.ParameterList(self.gatesbiases)

        self.weights2 = nn.Parameter(torch.Tensor(prev_node, n_tokens))
        nn.init.kaiming_normal_(self.weights2, a=math.sqrt(5)) # different init
        self.biases2 = nn.Parameter(torch.Tensor(n_tokens))
        nn.init.zeros_(self.biases2)
        # d-channel aggregation
        self.weightsd = nn.Parameter(torch.Tensor(d_token, 1))
        nn.init.kaiming_normal_(self.weightsd, a=math.sqrt(5))
        # self.biasesd = nn.Parameter(torch.Tensor(1, n_tokens, 1))
        # nn.init.zeros_(self.biasesd)

        self.alpha = None # tmp results
        self.stochastic_gate = None # tmp results
        self.sim_matrix = None

        # self.train_gates = 1.0 # 1.0 if self.training else 0.0
        # compute sim
        self.compute_sim = compute_sim
        self.normalization = nn.LayerNorm(d_token) # to compute sim
        self.n_tokens = n_tokens
        self.d_token = d_token
    
    def hard_sigmoid(self, x: torch.Tensor, a):
        x = a * x + 0.5
        x.data.clamp_(min=0., max=1.)
        return x
    
    def get_stochastic_gate_train(self, prev_x: torch.Tensor):
        train_gates = 1.0 if self.training else 0.0
        base_noise = torch.randn((prev_x.size(0), self.n_tokens), device=prev_x.device) # * 0.5 / 0.1, small std may be better
        self.alpha = self.activation_gating(prev_x @ self.weights2 + self.biases2)

        z = self.alpha \
            + self.sigma * base_noise * train_gates
        stochastic_gate = self.hard_sigmoid(z, self.a)
        self.stochastic_gate = stochastic_gate
        return stochastic_gate
    
    def compute_similarity(self, x: torch.Tensor):
        # x: normalized
        batch_size = x.size(0)
        x = x.reshape(-1, self.d_token)
        sim = x @ x.T
        sim = sim.reshape(batch_size, self.n_tokens, self.n_tokens, batch_size)
        mask = torch.eye(self.n_tokens, device=x.device).unsqueeze(0).unsqueeze(-1)
        sim = torch.mean(torch.sum(sim * mask, dim=2), dim=1)
        return sim
    
    @staticmethod
    def squared_distance(X):
        r = torch.sum(X*X, dim=1) # b,
        r = r.reshape(-1, 1) # b, 1
        D = r - 2*(X @ X.T) + r.transpose()
        return D
    
    def forward(self, x: torch.Tensor):
        # x: b, f, d
        x = self.normalization(x)
        if self.compute_sim: # using similarity affinity
            self.sim_matrix = self.compute_similarity(x.detach())
        x = x.transpose(1, 2)
        for i in range(len(self.gatesweights)):
            x = x @ self.gatesweights[i] + self.gatesbiases[i]
        x = x.transpose(1, 2)
        x = x @ self.weightsd # + self.biasesd
        stochastic_gate = self.get_stochastic_gate_train(x.squeeze(2))
        return stochastic_gate
    
    def regularization(self):
        input2cdf = self.alpha
        reg = 0.5 - 0.5 * torch.erf((-1/(2*self.a) - input2cdf) / (self.sigma*math.sqrt(2)))
        reg_gates = self.lam*torch.mean(torch.mean(reg, dim=-1))
        if self.compute_sim:
            gate_sd = LSPIN.squared_distance(self.stochastic_gate)
            reg_sim = self.gamma1*torch.mean(torch.mean((1.0 - self.sim_matrix/2.0) * gate_sd, dim=-1)) \
                + self.gamma2*torch.mean(torch.mean(self.sim_matrix/2.0 * -gate_sd, dim=-1))
            return reg_gates + reg_sim
        return reg_gates
