# %%
import os
import json
import math
import time
import typing as ty
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader

from models.abstract import TabModel, check_dir
# %%
class _MLP(nn.Module):
    def __init__(
        self,
        d_in: int,
        d_layers: ty.List[int],
        dropout: float,
        d_out: int,
        categories: ty.Optional[ty.List[int]],
        d_embedding: int,
    ) -> None:
        super().__init__()

        self.n_categories = 0 if categories is None else len(categories)
        if categories is not None:
            d_in += len(categories) * d_embedding
            category_offsets = torch.tensor([0] + categories[:-1]).cumsum(0)
            self.register_buffer('category_offsets', category_offsets)
            self.category_embeddings = nn.Embedding(sum(categories), d_embedding)
            nn.init.kaiming_uniform_(self.category_embeddings.weight, a=math.sqrt(5))
            print(f'{self.category_embeddings.weight.shape}')

        self.layers = nn.ModuleList(
            [
                nn.Linear(d_layers[i - 1] if i else d_in, x)
                for i, x in enumerate(d_layers)
            ]
        )
        self.dropout = dropout
        self.head = nn.Linear(d_layers[-1] if d_layers else d_in, d_out)

    def forward(self, x_num, x_cat):
        x = []
        if x_num is not None:
            x.append(x_num)
        if x_cat is not None:
            x.append(
                self.category_embeddings(x_cat + self.category_offsets[None]).view(
                    x_cat.size(0), -1
                )
            )
        x = torch.cat(x, dim=-1)

        for layer in self.layers:
            x = layer(x)
            x = F.relu(x)
            if self.dropout:
                x = F.dropout(x, self.dropout, self.training)
        x = self.head(x)
        x = x.squeeze(-1)

        return x


# %%
class MLP(TabModel):
    def __init__(
        self,
        model_config: dict,
        n_num_features: int,
        categories: ty.Optional[ty.List[int]],
        n_labels: int,
        device: ty.Union[str, torch.device] = 'cuda',
    ):
        super().__init__()
        model_config = self.preproc_config(model_config)
        self.model = _MLP(
            d_in=n_num_features,
            categories=categories,
            d_out=n_labels,
            **model_config
        ).to(device)
        self.base_name = 'mlp'
        self.device = torch.device(device)
    
    def preproc_config(self, model_config: dict):
        """MLP config preprocessing"""
        # process mlp configs
        self.saved_model_config = model_config.copy()
        d_layers = []
        n_layers, first_dim, mid_dim, last_dim = \
            (
                model_config.pop('n_layers'), model_config.pop('first_dim'),
                model_config.pop('mid_dim'), model_config.pop('last_dim')
            )
        for i in range(n_layers):
            if i == 0:
                d_layers.append(first_dim)
            elif i == n_layers - 1 and n_layers > 1:
                d_layers.append(last_dim)
            else:
                d_layers.append(mid_dim)
        model_config['d_layers'] = d_layers
        return model_config

    def fit(
        self,
        # API for specical sampler like curriculum learning
        train_loader: ty.Optional[ty.Tuple[DataLoader, int]] = None, # (loader, missing_idx)
        # using normal sampler if is None
        X_num: ty.Optional[torch.Tensor] = None, 
        X_cat: ty.Optional[torch.Tensor] = None, 
        ys: ty.Optional[torch.Tensor] = None,
        y_std: ty.Optional[float] = None, # for RMSE
        eval_set: ty.Tuple[torch.Tensor, np.ndarray] = None,
        patience: int = 0,
        task: str = None,
        training_args: dict = None,
        meta_args: ty.Optional[dict] = None,
    ):
        def train_step(model, x_num, x_cat, y): # input is X and y
            # process input (model-specific)
            # define your running time calculation
            start_time = time.time()
            # define your model API
            logits = model(x_num, x_cat)
            used_time = time.time() - start_time # don't forget backward time, calculate in outer loop
            return logits, used_time
        
        # to custom other training paradigm
        # 1. add self.dnn_fit2(...) in abstract class for special training process
        # 2. (recommended) override self.dnn_fit in abstract class
        self.dnn_fit( # uniform training paradigm
            dnn_fit_func=train_step,
            # training data
            train_loader=train_loader,
            X_num=X_num, X_cat=X_cat, ys=ys, y_std=y_std,
            # dev data
            eval_set=eval_set, patience=patience, task=task,
            # args
            training_args=training_args,
            meta_args=meta_args,
        )
          
    def predict(
        self,
        dev_loader: ty.Optional[ty.Tuple[DataLoader, int]] = None, # reuse, (loader, missing_idx)
        X_num: ty.Optional[torch.Tensor] = None, 
        X_cat: ty.Optional[torch.Tensor] = None, 
        ys: ty.Optional[torch.Tensor] = None, 
        y_std: ty.Optional[float] = None, # for RMSE
        task: str = None,
        return_probs: bool = True,
        return_metric: bool = False,
        return_loss: bool = False,
        meta_args: ty.Optional[dict] = None,
    ):
        def inference_step(model, x_num, x_cat): # input only X (y inaccessible)
            """
            Inference Process
            `no_grad` will be applied in `dnn_predict'
            """
            # process input (model-specific)
            # define your running time calculation
            start_time = time.time()
            # define your model API
            logits = model(x_num, x_cat)
            used_time = time.time() - start_time
            return logits, used_time
        
        # to custom other inference paradigm
        # 1. add self.dnn_predict2(...) in abstract class for special training process
        # 2. (recommended) override self.dnn_predict in abstract class
        return self.dnn_predict( # uniform training paradigm
            dnn_predict_func=inference_step,
            dev_loader=dev_loader,
            X_num=X_num, X_cat=X_cat, ys=ys, y_std=y_std, task=task,
            return_probs=return_probs, return_metric=return_metric, return_loss=return_loss,
            meta_args=meta_args
        )
    
    def save(self, output_dir):
        check_dir(output_dir)
        self.save_pt_model(output_dir)
        self.save_history(output_dir)
        self.save_config(output_dir)