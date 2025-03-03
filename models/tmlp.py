# %%
import math
import time
import typing as ty
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.functional import einsum
import torch.nn.init as nn_init
from torch.utils.data import DataLoader
from torch import Tensor

from .abstract import TabModel, check_dir
from .sparser import Sparser, make_sparser

ATTN_DIM = 1 # 64 default
# %%
class Tokenizer(nn.Module):
    category_offsets: ty.Optional[Tensor]

    def __init__(
        self,
        d_numerical: int,
        categories: ty.Optional[ty.List[int]],
        d_token: int,
        bias: bool,
    ) -> None:
        super().__init__()
        if categories is None:
            d_bias = d_numerical
            self.category_offsets = None
            self.category_embeddings = None
        else:
            d_bias = d_numerical + len(categories)
            category_offsets = torch.tensor([0] + categories[:-1]).cumsum(0)
            self.register_buffer('category_offsets', category_offsets)
            self.category_embeddings = nn.Embedding(sum(categories), d_token)
            nn_init.kaiming_uniform_(self.category_embeddings.weight, a=math.sqrt(5))
            print(f'{self.category_embeddings.weight.shape}')

        # take [CLS] token into account
        self.weight = nn.Parameter(Tensor(d_numerical + 1, d_token))
        self.bias = nn.Parameter(Tensor(d_bias, d_token)) if bias else None
        # The initialization is inspired by nn.Linear
        nn_init.kaiming_uniform_(self.weight, a=math.sqrt(5))
        if self.bias is not None:
            nn_init.kaiming_uniform_(self.bias, a=math.sqrt(5))

    @property
    def n_tokens(self) -> int:
        return len(self.weight) + (
            0 if self.category_offsets is None else len(self.category_offsets)
        )

    def forward(self, x_num: Tensor, x_cat: ty.Optional[Tensor]) -> Tensor:
        x_some = x_num if x_cat is None else x_cat
        assert x_some is not None
        x_num = torch.cat(
            [torch.ones(len(x_some), 1, device=x_some.device)]  # [CLS]
            + ([] if x_num is None else [x_num]),
            dim=1,
        )
        x = self.weight[None] * x_num[:, :, None]
        if x_cat is not None:
            x = torch.cat(
                [x, self.category_embeddings(x_cat + self.category_offsets[None])],
                dim=1,
            )
        if self.bias is not None:
            bias = torch.cat(
                [
                    torch.zeros(1, self.bias.shape[1], device=x.device),
                    self.bias,
                ]
            )
            x = x + bias[None]
        return x


class _tMLP(nn.Module):
    """Adapted gMLP for Tabular Data Prediction with Feature Embedding & NN Sparsity

    References:
    - [Pay Attention to MLPs](https://openreview.net/forum?id=KBnXrODoBW)
    - [gMLP Pytorch] https://github.com/lucidrains/g-mlp-pytorch
    """

    def __init__(
        self,
        *,
        # tokenizer
        d_numerical: int,
        categories: ty.Optional[ty.List[int]],
        token_bias: bool,
        # transformer
        n_layers: int = 1,
        d_token: int = 1024,
        d_ffn_factor: float = 0.66,
        ffn_dropout: float | None = None,
        residual_dropout: float | None = 0.1,
        #
        d_out: int,
    ) -> None:

        super().__init__()
        self.tokenizer = Tokenizer(d_numerical, categories, d_token, token_bias)
        self.n_categories = 0 if categories is None else len(categories)
        n_tokens = self.tokenizer.n_tokens

        def make_normalization(d=d_token):
            return nn.LayerNorm(d)
        
        d_hidden = int(d_token * d_ffn_factor)
        self.d_ffn = d_hidden
        class SGU(nn.Module):
            def __init__(self, n_token):
                super().__init__()
                self.proj = nn.Linear(n_token, n_token)
                self.norm = make_normalization(d=d_hidden)
                
            def forward(self, x, z=None):
                u, v = torch.chunk(x, 2, -1)
                v = self.norm(v).transpose(1,2)
                v = self.proj(v).transpose(1,2)
                if z is not None:
                    # prune SGU
                    return u * (
                        z * v 
                        + (1 - z) 
                        * torch.ones_like(v, device=v.device))
                return u * v

        self.layers = nn.ModuleList([])
        for layer_idx in range(n_layers):
            layer = nn.ModuleDict(
                {
                    'linear0': nn.Linear(
                        d_token, d_hidden * 2 # * (2 if activation.endswith('glu') else 1) # using F.gelu, then sgu
                    ),
                    'sgu': SGU(n_tokens),
                    'linear1': nn.Linear(d_hidden, d_token),
                }
            )
            self.layers.append(layer)

        self.activation = F.gelu
        self.normalization = make_normalization() # pre_normalization, last_normalization
        self.ffn_dropout = ffn_dropout
        self.residual_dropout = residual_dropout
        self.head = nn.Linear(d_token, d_out)
    
    def _sp_linear(self, lin: nn.Linear, x, z=None):
        """sparse calculation for linear"""
        if z is None:
            x = x @ lin.weight.T
        else:
            x = x @ (torch.diag(z) @ lin.weight.T)
        if lin.bias is not None:
            x = x + lin.bias
        return x
    
    def _sp_residual(self, x, x_residual, z=None):
        """sparse calculation for residual"""
        if z is None:
            return x + x_residual
        return z * x + x_residual
    
    def _sp_forward(self, x_num: Tensor, x_cat: ty.Optional[Tensor], sparser: Sparser) -> Tensor:
        """sparse forward: feature sparsity + param sparsity"""
        # feature gating (tree-based)
        zs = sparser(x_num, x_cat)
        if isinstance(x_num, tuple):
            x_num = x_num[1]
        x = self.tokenizer(x_num, x_cat) # b, f, d
        # feature gating (nn-based) & NN pruning
        zs.update(sparser(x, is_raw_input=False))
        if 'feature_z' in zs:
            # apply feature (spatial) gating
            x = x * zs['feature_z']
        
        for layer_idx, layer in enumerate(self.layers):
            layer = ty.cast(ty.Dict[str, nn.Module], layer)
            x_residual = x
            x = self.normalization(x)
            hidden_z = zs.get('hidden_z') # shared across layers, prune hidden size
            x = self._sp_linear(layer['linear0'], x, hidden_z)
            x = self.activation(x)
            sgu_z = zs['sgu_z'][layer_idx] if 'sgu_z' in zs else None # prune sgu
            x = layer['sgu'](x, sgu_z)
            if self.ffn_dropout:
                x = F.dropout(x, self.ffn_dropout, self.training)
            int_z = zs['intermediate_z'][layer_idx] if 'intermediate_z' in zs else None
            x = self._sp_linear(layer['linear1'], x, int_z) # prune lin1 params
            if self.residual_dropout:
                x = F.dropout(x, self.residual_dropout, self.training)
            layer_z = zs['layer_z'][layer_idx] if 'layer_z' in zs else None # prune gmlp layer
            x = self._sp_residual(x, x_residual, layer_z)
        
        x = x[:, 0] # [CLS]
        x = self.normalization(x)
        x = self.activation(x)
        x = self.head(x)
        x = x.squeeze(-1)

        return x

    def forward(self, x_num: Tensor, x_cat: ty.Optional[Tensor]) -> Tensor:
        x = self.tokenizer(x_num, x_cat) # b, f, d

        for layer_idx, layer in enumerate(self.layers):
            layer = ty.cast(ty.Dict[str, nn.Module], layer)
            # start residual
            x_residual = x
            x = self.normalization(x) # pre_nrom
            x = layer['linear0'](x)
            x = self.activation(x) # gelu
            x = layer['sgu'](x)
            if self.ffn_dropout:
                x = F.dropout(x, self.ffn_dropout, self.training)
            x = layer['linear1'](x)
            if self.residual_dropout:
                x = F.dropout(x, self.residual_dropout, self.training)
            x = x + x_residual

        x = x[:, 0] # [CLS]
        x = self.normalization(x)
        x = self.activation(x)
        x = self.head(x)
        x = x.squeeze(-1)

        return x

# %%
class tMLP(TabModel):
    """Tree MLPs
    
    Adapted gMLP + Tree-based Dropout + Sparse Params
    """
    def __init__(
        self,
        model_config: dict,
        n_num_features: int,
        categories: ty.Optional[ty.List[int]],
        n_labels: int,
        device: ty.Union[str, torch.device] = 'cuda',
        feat_gate: ty.Optional[str] = 'xgb_dropout',
        pruning: ty.Optional[str] = 'mlp+sgu+layer',
        dataset = None,
    ):
        super().__init__()
        model_config = self.preproc_config(model_config)
        self.model = _tMLP(
            d_numerical=n_num_features,
            categories=categories,
            d_out=n_labels,
            **model_config
        ).to(device)
        if any([feat_gate, pruning]):
            self.sparser = make_sparser(
                self.model, pruning, feat_gate, device, dataset)
        self.base_name = 'tmlp'
        self.device = torch.device(device)

    def fit(
        self,
        # API for specical sampler like curriculum learning
        train_loader: ty.Optional[ty.Tuple[DataLoader, int]] = None, # (loader, missing_idx)
        # using normal sampler if is None
        X_num: ty.Optional[torch.Tensor] = None, 
        X_cat: ty.Optional[torch.Tensor] = None, 
        ys: ty.Optional[torch.Tensor] = None,
        ids: ty.Optional[torch.Tensor] = None,
        y_std: ty.Optional[float] = None, # for RMSE
        eval_set: ty.Tuple[torch.Tensor, np.ndarray] = None,
        patience: int = 0,
        task: str = None,
        training_args: dict = None,
        meta_args: ty.Optional[dict] = None,
    ):
        def train_step(model, x_num, x_cat, y, sparser=None): # input is X and y
            # process input (model-specific)
            # define your running time calculation
            start_time = time.time()
            # define your model API
            if sparser is None:
                logits = model(x_num, x_cat)
            else:
                logits = model._sp_forward(x_num, x_cat, sparser)
            used_time = time.time() - start_time # don't forget backward time, calculate in outer loop
            return logits, used_time
        
        # to custom other training paradigm
        # 1. add self.dnn_fit2(...) in abstract class for special training process
        # 2. (recommended) override self.dnn_fit in abstract class
        self.dnn_fit( # uniform training paradigm
            dnn_fit_func=train_step,
            # training data
            train_loader=train_loader,
            X_num=X_num, X_cat=X_cat, ys=ys, y_std=y_std, ids=ids,
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
        ids: ty.Optional[torch.Tensor] = None, 
        y_std: ty.Optional[float] = None, # for RMSE
        task: str = None,
        return_probs: bool = True,
        return_metric: bool = False,
        return_loss: bool = False,
        meta_args: ty.Optional[dict] = None,
    ):
        def inference_step(model, x_num, x_cat, sparser=None): # input only X (y inaccessible)
            """
            Inference Process
            `no_grad` will be applied in `dnn_predict'
            """
            # process input (model-specific)
            # define your running time calculation
            start_time = time.time()
            # define your model API
            if sparser is None:
                logits = model(x_num, x_cat)
            else:
                logits = model._sp_forward(x_num, x_cat, sparser)
            used_time = time.time() - start_time
            return logits, used_time
        
        # to custom other inference paradigm
        # 1. add self.dnn_predict2(...) in abstract class for special training process
        # 2. (recommended) override self.dnn_predict in abstract class
        return self.dnn_predict( # uniform training paradigm
            dnn_predict_func=inference_step,
            dev_loader=dev_loader,
            X_num=X_num, X_cat=X_cat, ys=ys, y_std=y_std, ids=ids, task=task,
            return_probs=return_probs, return_metric=return_metric, return_loss=return_loss,
            meta_args=meta_args
        )
    
    def save(self, output_dir):
        check_dir(output_dir)
        self.save_pt_model(output_dir)
        self.save_history(output_dir)
        self.save_config(output_dir)