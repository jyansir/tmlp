# %%
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
class CrossLayer(nn.Module):
    def __init__(self, d, dropout):
        super().__init__()
        self.linear = nn.Linear(d, d)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x0, x):
        return self.dropout(x0 * self.linear(x)) + x


class _DCNv2(nn.Module):
    def __init__(
        self,
        *,
        d_in: int,
        d: int,
        n_hidden_layers: int,
        n_cross_layers: int,
        hidden_dropout: float,
        cross_dropout: float,
        d_out: int,
        stacked: bool,
        categories: ty.Optional[ty.List[int]],
        d_embedding: int = None,
    ) -> None:
        super().__init__()

        if categories is not None:
            assert d_embedding is not None
            d_in += len(categories) * d_embedding
            category_offsets = torch.tensor([0] + categories[:-1]).cumsum(0)
            self.register_buffer('category_offsets', category_offsets)
            self.category_embeddings = nn.Embedding(sum(categories), d_embedding)
            nn.init.kaiming_uniform_(self.category_embeddings.weight, a=math.sqrt(5))
            print(f'{self.category_embeddings.weight.shape}')

        self.first_linear = nn.Linear(d_in, d)
        self.last_linear = nn.Linear(d if stacked else 2 * d, d_out)

        deep_layers = sum(
            [
                [nn.Linear(d, d), nn.ReLU(True), nn.Dropout(hidden_dropout)]
                for _ in range(n_hidden_layers)
            ],
            [],
        )
        cross_layers = [CrossLayer(d, cross_dropout) for _ in range(n_cross_layers)]

        self.deep_layers = nn.Sequential(*deep_layers)
        self.cross_layers = nn.ModuleList(cross_layers)
        self.stacked = stacked

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

        x = self.first_linear(x)

        x_cross = x
        for cross_layer in self.cross_layers:
            x_cross = cross_layer(x, x_cross)

        if self.stacked:
            return self.last_linear(self.deep_layers(x_cross)).squeeze(1)
        else:
            return self.last_linear(
                torch.cat([x_cross, self.deep_layers(x)], dim=1)
            ).squeeze(1)


# %%
class DCNv2(TabModel):
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
        self.model = _DCNv2(
            d_in=n_num_features,
            categories=categories,
            d_out=n_labels,
            **model_config
        ).to(device)
        self.base_name = 'dcnv2'
        self.device = torch.device(device)
    
    def preproc_config(self, model_config: dict):
        # process autoint configs
        self.saved_model_config = model_config.copy()
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
            # define your model API
            start_time = time.time()
            # define your model API
            logits = model(x_num, x_cat)
            used_time = time.time() - start_time
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
            # define your model API
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