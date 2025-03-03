# abstract class for all tabular models
# TODO: model soup ensembling and ensemble predict
from abc import ABC, abstractmethod
from collections import OrderedDict
from typing import Optional, Tuple, Union, Dict, Any, Callable
from pathlib import Path
import warnings
warnings.filterwarnings("ignore")
import os
import json
import yaml
import time
import numpy as np

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import TensorDataset, DataLoader

from .sparser import Sparser
from utils.metrics import calculate_metrics
from sklearn.metrics import log_loss, mean_squared_error

DNN_FIT_API = Callable[
    [nn.Module, torch.Tensor, torch.Tensor, torch.Tensor],
    Tuple[torch.Tensor, float]
] # input X, y and return logits, used time
DNN_PREDICT_API = Callable[
    [nn.Module, torch.Tensor, torch.Tensor],
    Tuple[torch.Tensor, float]
] # input X and return logits, used time

def default_dnn_fit(model, x_num, x_cat, y):
    """
    Training Process
    """
    start_time = time.time()
    logits = model(x_num, x_cat)
    used_time = time.time() - start_time # omit backward time, add in outer loop
    return logits, used_time

def default_dnn_predict(model, x_num, x_cat):
    """
    Inference Process
    `no_grad` will be applied in `dnn_predict'
    """
    start_time = time.time()
    logits = model(x_num, x_cat)
    used_time = time.time() - start_time
    return logits, used_time

def check_dir(dir):
    if not os.path.exists(dir):
        os.makedirs(dir)

def make_optimizer(
    optimizer: str,
    parameter_groups,
    lr: float,
    weight_decay: float,
) -> optim.Optimizer:
    Optimizer = {
        'adam': optim.Adam,
        'adamw': optim.AdamW,
        'sgd': optim.SGD,
    }[optimizer]
    momentum = (0.9,) if Optimizer is optim.SGD else ()
    return Optimizer(parameter_groups, lr, *momentum, weight_decay=weight_decay)

def make_lr_scheduler(
    optimizer: optim.Optimizer,
    lr: float,
    epoch_size: int,
    lr_schedule: Optional[Dict[str, Any]],
) -> Tuple[
    Optional[optim.lr_scheduler._LRScheduler],
    Dict[str, Any],
    Optional[int],
]:
    if lr_schedule is None:
        lr_schedule = {'type': 'constant'}
    lr_scheduler = None
    n_warmup_steps = None
    if lr_schedule['type'] in ['transformer', 'linear_warmup']:
        n_warmup_steps = (
            lr_schedule['n_warmup_steps']
            if 'n_warmup_steps' in lr_schedule
            else lr_schedule['n_warmup_epochs'] * epoch_size
        )
    elif lr_schedule['type'] == 'cyclic':
        lr_scheduler = optim.lr_scheduler.CyclicLR(
            optimizer,
            base_lr=lr,
            max_lr=lr_schedule['max_lr'],
            step_size_up=lr_schedule['n_epochs_up'] * epoch_size,
            step_size_down=lr_schedule['n_epochs_down'] * epoch_size,
            mode=lr_schedule['mode'],
            gamma=lr_schedule.get('gamma', 1.0),
            cycle_momentum=False,
        )
    return lr_scheduler, lr_schedule, n_warmup_steps

class TabModel(ABC):
    def __init__(self):
        self.model: Optional[nn.Module] = None # true model
        self.sparser: Optional[Sparser] = None
        self.base_name = None # model type name
        self.device = None
        self.saved_model_config = None
        self.training_config = None
        self.meta_config = None
        self.post_init()

    def post_init(self):
        self.history = {
            'train': {'loss': [], 'tot_time': 0, 'avg_step_time': 0, 'avg_epoch_time': 0}, 
            'val': {
                'metric_name': None, 'metric': [], 'best_metric': None,
                'best_metrics': [], 'test_metrics': [],
                'log_loss': [], 'best_log_loss': None,
                'best_epoch': None, 'best_step': None,
                'best_epochs': [], 'best_steps': [],
                'tot_time': 0, 'avg_step_time': 0, 'avg_epoch_time': 0
            }, 
            # 'test': {'loss': [], 'metric': [], 'final_metric': None},
            'device': torch.cuda.get_device_name(),
        } # save metrics
        self.no_improvement = 0 # for dnn early stop
    
    def preproc_config(self, model_config: dict):
        """default preprocessing for model configurations"""
        self.saved_model_config = model_config
        return model_config
    
    @abstractmethod
    def fit(
        self,
        X_num: Union[torch.Tensor, np.ndarray], 
        X_cat: Union[torch.Tensor, np.ndarray], 
        ys: Union[torch.Tensor, np.ndarray],
        ids: Optional[torch.Tensor], # sample ids
        y_std: Optional[float],
        eval_set: Optional[Tuple[Union[torch.Tensor, np.ndarray]]],
        patience: int,
        task: str,
        training_args: dict,
        meta_args: Optional[dict],
        # sparsity
        # sparser: Optional[Sparser] = None,
    ):
        """
        Training Model with Early Stop(optional)
        load best weights at the end
        """
        pass
    
    def dnn_fit(
        self,
        *,
        dnn_fit_func: Optional[DNN_FIT_API] = None,
        # API for specical sampler like curriculum learning
        train_loader: Optional[Tuple[DataLoader, int]] = None, # (loader, missing_idx)
        # using normal dataloader sampler if is None
        X_num: Optional[torch.Tensor] = None, 
        X_cat: Optional[torch.Tensor] = None, 
        ys: Optional[torch.Tensor] = None,
        ids: Optional[torch.Tensor] = None, # sample ids
        y_std: Optional[float] = None, # for RMSE
        eval_set: Tuple[torch.Tensor, np.ndarray] = None, # similar API as sk-learn
        patience: int = 0, # <= 0 without early stop
        task: str,
        training_args: dict,
        meta_args: Optional[dict] = None,
        # sparsity
        # sparser: Optional[Sparser] = None,
    ):
        # DONE: move to abstract class (dnn_fit)
        use_mixup = False # TODO: add mixup techniques
        if dnn_fit_func is None:
            dnn_fit_func = default_dnn_fit
        # meta args
        if meta_args is None:
            meta_args = {}
        meta_args.setdefault('save_path', f'results/{self.base_name}')
        if not os.path.exists(meta_args['save_path']):
            print('create new results dir: ', meta_args['save_path'])
            os.makedirs(meta_args['save_path'])
        self.meta_config = meta_args
        # optimzier and scheduler
        training_args.setdefault('optimizer', 'adamw')
        optimizer, scheduler = TabModel.make_optimizer(self.model, training_args)
        if self.sparser is not None:
            l0_optimizer, lagrangian_optimizer = self.sparser.make_optimizer(1e-3)
        # data loader
        training_args.setdefault('batch_size', 64)
        training_args.setdefault('ghost_batch_size', None)
        if train_loader is not None:
            train_loader, placeholders = train_loader
            training_args['batch_size'] = train_loader.batch_size
        else:
            train_loader, placeholders = TabModel.prepare_tensor_loader(
                X_num=X_num, X_cat=X_cat, ys=ys, ids=ids,
                batch_size=training_args['batch_size'],
                shuffle=True,
            )
        if eval_set is not None:
            dev_eval_set = eval_set[0] # only use the first dev set
            dev_loader = TabModel.prepare_tensor_loader(
            X_num=dev_eval_set[0], X_cat=dev_eval_set[1], ys=dev_eval_set[2], ids=dev_eval_set[3],
            batch_size=training_args['batch_size'],
            )
            if len(eval_set) == 2:
                test_eval_set = eval_set[1]
                test_loader = TabModel.prepare_tensor_loader(
                X_num=test_eval_set[0], X_cat=test_eval_set[1], ys=test_eval_set[2], ids=test_eval_set[3],
                batch_size=training_args['batch_size'],
                )
            else:
                print('[info] provide test set if you want real-time report on test score')
                test_loader = None
        else:
            print('[warning] no eval set is provided for early stopping')
            dev_loader, test_loader = None, None
        # training loops
        training_args.setdefault('max_epochs', 10000)
        # training_args.setdefault('report_frequency', 100) # same as save_freq
        # training_args.setdefault('save_frequency', 100) # save per 100 steps
        training_args.setdefault('patience', patience)
        training_args.setdefault('save_frequency', 'epoch') # save per epoch
        self.training_config = training_args

        steps_per_backward = 1 if training_args['ghost_batch_size'] is None \
            else training_args['batch_size'] // training_args['ghost_batch_size'] # for ghost batch size
        steps_per_epoch = len(train_loader)
        tot_step, tot_time = 0, 0
        for e in range(training_args['max_epochs']):
            self.model.train()
            if self.sparser is not None:
                self.sparser.train()
            tot_loss = 0
            for step, batch in enumerate(train_loader):
                optimizer.zero_grad()
                if self.sparser is not None:
                    l0_optimizer.zero_grad()
                    lagrangian_optimizer.zero_grad()
                x_num, x_cat, y = TabModel.parse_batch(batch, placeholders, self.device)
                if self.sparser is None:
                    logits, forward_time = dnn_fit_func(self.model, x_num, x_cat, y)
                    loss = TabModel.compute_loss(logits, y, task)
                else:
                    logits, forward_time = dnn_fit_func(self.model, x_num, x_cat, y, self.sparser)
                    loss = TabModel.compute_loss(logits, y, task)
                
                    lagrangian_loss = self.sparser.regularization(tot_step)
                    loss += lagrangian_loss
                # backward
                start_time = time.time()
                loss.backward()
                backward_time = time.time() - start_time
                self.gradient_policy()
                tot_time += forward_time + backward_time
                optimizer.step()
                if self.sparser is not None:
                    l0_optimizer.step()
                    lagrangian_optimizer.step()
                if scheduler is not None:
                    scheduler.step()
                if self.sparser is not None and self.sparser.l0_module is not None:
                    self.sparser.l0_module.constrain_parameters()
                # print or save infos
                tot_step += 1
                tot_loss += loss.cpu().item()
                if isinstance(training_args['save_frequency'], int) \
                    and tot_step % training_args['save_frequency'] == 0:
                    is_early_stop = self.save_evaluate_dnn(
                        tot_step, steps_per_epoch, 
                        tot_loss, tot_time,
                        task, training_args['patience'], meta_args['save_path'],
                        dev_loader, y_std, test_loader=test_loader,
                    )
                    if is_early_stop:
                        self.save(meta_args['save_path'])
                        self.load_best_dnn(meta_args['save_path'])
                        return
            if training_args['save_frequency'] == 'epoch':
                if hasattr(self.model, 'layer_masks'):
                    print('layer_mask: ', self.model.layer_masks > 0)
                is_early_stop = self.save_evaluate_dnn(
                    tot_step, steps_per_epoch, 
                    tot_loss, tot_time,
                    task, training_args['patience'], meta_args['save_path'],
                    dev_loader, y_std, test_loader=test_loader,
                )
                if is_early_stop:
                    self.save(meta_args['save_path'])
                    self.load_best_dnn(meta_args['save_path'])
                    return
        self.save(meta_args['save_path'])
        self.load_best_dnn(meta_args['save_path'])
    
    @abstractmethod
    def predict(
        self,
        dev_loader: Optional[DataLoader],
        X_num: Union[torch.Tensor, np.ndarray], 
        X_cat: Union[torch.Tensor, np.ndarray], 
        ys: Union[torch.Tensor, np.ndarray],
        ids: Optional[torch.Tensor], # sample ids
        y_std: Optional[float],
        task: str,
        return_probs: bool = True,
        return_metric: bool = True,
        return_loss: bool = True,
        meta_args: Optional[dict] = None,
    ):
        """
        Prediction
        """
        pass
    
    def dnn_predict(
        self,
        *,
        dnn_predict_func: Optional[DNN_PREDICT_API] = None,
        dev_loader: Optional[Tuple[DataLoader, int]] = None, # reuse, (loader, missing_idx)
        X_num: Optional[torch.Tensor] = None, 
        X_cat: Optional[torch.Tensor] = None, 
        ys: Optional[torch.Tensor] = None, 
        ids: Optional[torch.Tensor] = None, # sample ids
        y_std: Optional[float] = None, # for RMSE
        task: str,
        return_probs: bool = True,
        return_metric: bool = False,
        return_loss: bool = False,
        meta_args: Optional[dict] = None,
    ):
        # DONE: move to abstract class (dnn_predict)
        if dnn_predict_func is None:
            dnn_predict_func = default_dnn_predict
        if dev_loader is None:
            dev_loader, placeholders = TabModel.prepare_tensor_loader(
            X_num=X_num, X_cat=X_cat, ys=ys, ids=ids,
            batch_size=128,
        )
        else:
            dev_loader, placeholders = dev_loader
        # print("Evaluate...")
        predictions, golds = [], []
        tot_time = 0
        self.model.eval()
        if self.sparser is not None:
            self.sparser.eval()
        for batch in dev_loader:
            x_num, x_cat, y = TabModel.parse_batch(batch, placeholders, self.device)
            with torch.no_grad():
                if self.sparser is None:
                    logits, used_time = dnn_predict_func(self.model, x_num, x_cat)
                else:
                    logits, used_time = dnn_predict_func(self.model, x_num, x_cat, self.sparser)
            tot_time += used_time
            predictions.append(logits)
            golds.append(y)
        self.model.train()
        predictions = torch.cat(predictions).squeeze(-1)
        golds = torch.cat(golds)
        if return_loss:
            loss = TabModel.compute_loss(predictions, golds, task).cpu().item()
        else:
            loss = None
        if return_probs and task != 'regression':
            predictions = (
                predictions.sigmoid()
                if task == 'binclass'
                else predictions.softmax(-1)
            )
            prediction_type = 'probs'
        elif task == 'regression':
            prediction_type = None
        else:
            prediction_type = 'logits'
        predictions = predictions.cpu().numpy()
        golds = golds.cpu().numpy()
        if return_metric:
            # multi-class with one-hot labels
            if task == 'multiclass' and golds.ndim == 2:
                golds = np.argmax(golds, axis=1)
            # binclass with label smoothing
            elif task == 'binclass' and golds.max() != 1.0:
                golds = (golds == golds.max()).astype(np.float32)
            metric = TabModel.calculate_metric(
                golds, predictions,
                task, prediction_type, y_std
            )
            logloss = (
                log_loss(golds, np.stack([1-predictions, predictions], axis=1), labels=[0,1])
                if task == 'binclass'
                else log_loss(golds, predictions, labels=list(range(len(set(golds)))))
                if task == 'multiclass'
                else None
            )
        else:
            metric, logloss = None, None
        results = {'loss': loss, 'metric': metric, 'time': tot_time, 'log_loss': logloss}
        if meta_args is not None:
            self.save_prediction(meta_args['save_path'], results)
        return predictions, results
    
    def gradient_policy(self):
        """For post porcess model gradient"""
        pass
    
    @abstractmethod
    def save(self, output_dir):
        """
        Save model weights and configs,
        the following default save functions
        can be combined to override this function
        """
        pass

    def save_pt_model(self, output_dir):
        print('saving pt model weights...')
        # save model params
        torch.save(self.model.state_dict(), Path(output_dir) / 'final.bin')
        if self.sparser is not None:
            torch.save(self.sparser.state_dict(), Path(output_dir) / 'final-sparser.bin')
    
    def save_tree_model(self, output_dir):
        print('saving tree model...')
        pass

    def save_history(self, output_dir):
        # save metrics
        with open(Path(output_dir) / 'results.json', 'w') as f:
            json.dump(self.history, f, indent=4)
    
    def save_prediction(self, output_dir, results, file='prediction'):
        check_dir(output_dir)
        # save test results
        print("saving prediction results")
        saved_results = {
            'loss': results['loss'], 
            'metric_name': results['metric'][1], 
            'metric': results['metric'][0], 
            'time': results['time'],
            'log_loss': results['log_loss'],
        }
        with open(Path(output_dir) / f'{file}.json', 'w') as f:
            json.dump(saved_results, f, indent=4)
    
    def save_config(self, output_dir):
        def serialize(config: dict):
            for key in config:
                # serialized object to store yaml or json files 
                if any(isinstance(config[key], obj) for obj in [Path, ]):
                    config[key] = str(config[key])
            return config
        # save all configs
        with open(Path(output_dir) / 'configs.yaml', 'w') as f:
            configs = {
                'model': self.saved_model_config, 
                'training': self.training_config,
                'meta': serialize(self.meta_config)
            }
            yaml.dump(configs, f, indent=2)

    @staticmethod
    def make_optimizer(
        model: nn.Module,
        training_args: dict,
    ) -> Tuple[optim.Optimizer, optim.lr_scheduler._LRScheduler]:
        training_args.setdefault('optimizer', 'adamw')
        training_args.setdefault('no_wd_group', None)
        training_args.setdefault('scheduler', None)
        # optimizer
        if training_args['no_wd_group'] is not None:
            assert isinstance(training_args['no_wd_group'], list)
            def needs_wd(name):
                return all(x not in name for x in training_args['no_wd_group'])
            parameters_with_wd = [v for k, v in model.named_parameters() if needs_wd(k)]
            parameters_without_wd = [v for k, v in model.named_parameters() if not needs_wd(k)]
            model_params = [
                {'params': parameters_with_wd},
                {'params': parameters_without_wd, 'weight_decay': 0.0},
            ]
        else:
            model_params = model.parameters()
        optimizer = make_optimizer(
            training_args['optimizer'],
            model_params,
            training_args['lr'],
            training_args['weight_decay'],
        )
        # scheduler
        if training_args['scheduler'] is not None:
            scheduler = None
        else:
            scheduler = None

        return optimizer, scheduler
    
    @staticmethod
    def prepare_tensor_loader(
        X_num: Optional[torch.Tensor],
        X_cat: Optional[torch.Tensor],
        ys: torch.Tensor,
        ids: Optional[torch.Tensor] = None,
        batch_size: int = 64,
        shuffle: bool = False,
    ):
        assert not all(x is None for x in [X_num, X_cat])
        # missing_placeholder = 0 if X_num is None else 1 if X_cat is None else -1
        placeholders, datas = [], []
        if ids is not None:
            datas.append(ids)
            placeholders.append('id')
        if X_num is not None:
            datas.append(X_num)
            placeholders.append('X_num')
        if X_cat is not None:
            datas.append(X_cat)
            placeholders.append('X_cat')
        datas.append(ys)
        placeholders.append('y')        
        # datas = [x for x in [X_num, X_cat, ys] if x is not None]
        tensor_dataset = TensorDataset(*datas)
        tensor_loader = DataLoader(
            tensor_dataset,
            batch_size=batch_size,
            shuffle=shuffle,
        )
        # return tensor_loader, missing_placeholder
        return tensor_loader, placeholders
    
    @staticmethod
    def parse_batch(batch: Tuple[torch.Tensor], placeholders, device: torch.device):
        if batch[0].device.type != device.type:
        # if batch[0].device != device: # initialize self.device with model.device rather than torch.device()
            # batch = (x.to(device) for x in batch) # generator
            batch = tuple([x.to(device) for x in batch]) # list
        # if missing_idx == -1:
        #     return batch
        # else:
        #     return batch[:missing_idx] + [None,] + batch[missing_idx:]
        ids, X_num, X_cat, ys = None, None, None, None
        for i, ph in enumerate(placeholders):
            if ph == 'id':
                ids = batch[i]
            elif ph == 'X_num':
                X_num = batch[i]
            elif ph == 'X_cat':
                X_cat = batch[i]
            else:
                ys = batch[i]
        if ids is None:
            return X_num, X_cat, ys
        return (ids, X_num), X_cat, ys
    
    @staticmethod
    def compute_loss(logits: torch.Tensor, targets: torch.Tensor, task: str, reduction: str = 'mean'):
        loss_fn = {
            'binclass': F.binary_cross_entropy_with_logits,
            'multiclass': F.cross_entropy,
            'regression': F.mse_loss,
        }[task]
        if targets.ndim == 2: # multi-class one-hot
            loss_fn = F.binary_cross_entropy_with_logits
        return loss_fn(logits.squeeze(-1), targets, reduction=reduction)
    
    @staticmethod
    def calculate_metric(
        golds,
        predictions,
        task: str,
        prediction_type: Optional[str] = None,
        y_std: Optional[float] = None,
    ):
        """Calculate metrics"""
        metric = {
            'regression': 'rmse', 
            'binclass': 'accuracy', # 'roc_auc', # depends on benchmark
            'multiclass': 'accuracy'
        }[task]
        
        return calculate_metrics(
            golds, predictions,
            task, prediction_type, y_std
        )[metric], metric
    
    def better_result(self, dev_metric, task, is_loss=False):
        if is_loss: # logloss
            best_dev_metric = self.history['val']['best_log_loss']
            if best_dev_metric is None or best_dev_metric > dev_metric:
                self.history['val']['best_log_loss'] = dev_metric
                return True
            else:
                return False
        best_dev_metric = self.history['val']['best_metric']
        if best_dev_metric is None:
            self.history['val']['best_metric'] = dev_metric
            self.history['val']['best_metrics'].append(dev_metric)
            return True
        elif task == 'regression': # rmse
            if best_dev_metric > dev_metric:
                self.history['val']['best_metric'] = dev_metric
                self.history['val']['best_metrics'].append(dev_metric)
                return True
            else:
                return False
        else:
            if best_dev_metric < dev_metric:
                self.history['val']['best_metric'] = dev_metric
                self.history['val']['best_metrics'].append(dev_metric)
                return True
            else:
                return False
    
    def early_stop_handler(self, epoch, tot_step, dev_metric, task, patience, save_path, y_std, test_loader=None):
        if task != 'regression' and self.better_result(dev_metric['log_loss'], task, is_loss=True):
            # record best logloss
            torch.save(self.model.state_dict(), Path(save_path) / 'best-logloss.bin')
            if self.sparser is not None:
                torch.save(self.sparser.state_dict(), Path(save_path) / 'best-logloss-sparser.bin')
        if self.better_result(dev_metric['metric'], task):
            print('<<< Best Dev Result', end='')
            torch.save(self.model.state_dict(), Path(save_path) / 'best.bin')
            if self.sparser is not None:
                torch.save(self.sparser.state_dict(), Path(save_path) / 'best-sparser.bin')
            self.no_improvement = 0
            self.history['val']['best_epoch'] = epoch
            self.history['val']['best_step'] = tot_step
            self.history['val']['best_epochs'].append(epoch)
            self.history['val']['best_steps'].append(tot_step)
            if test_loader is not None:
                _, results = self.predict(dev_loader=test_loader, y_std=y_std, task=task, return_metric=True)
                test_metric, metric_name = results['metric']
                print(f" | test-[{metric_name}] {test_metric:.4g} ", end='')
                self.history['val']['test_metrics'].append(test_metric)                
        else:
            self.no_improvement += 1
            print(f'| [no improvement] {self.no_improvement}', end='')
        if patience <= 0:
            return False
        else:
            return self.no_improvement >= patience
    
    def save_evaluate_dnn(
        self, 
        # print and saved infos
        tot_step, steps_per_epoch, 
        tot_loss, tot_time,
        # evaluate infos
        task, patience, save_path,
        dev_loader, y_std,
        # other kits
        test_loader=None, # if you want to print test score each best dev results
    ):
        """For DNN models"""
        epoch, step = tot_step // steps_per_epoch, (tot_step - 1) % steps_per_epoch + 1
        avg_loss = tot_loss / step
        self.history['train']['loss'].append(avg_loss)
        self.history['train']['tot_time'] = tot_time
        self.history['train']['avg_step_time'] = tot_time / tot_step
        self.history['train']['avg_epoch_time'] = self.history['train']['avg_step_time'] * steps_per_epoch
        print(f"[epoch] {epoch} | [step] {step} | [tot_step] {tot_step} | [used time] {tot_time:.4g} | [train_loss] {avg_loss:.4g} ", end='')
        if dev_loader is not None:
            _, results = self.predict(dev_loader=dev_loader, y_std=y_std, task=task, return_metric=True)
            dev_metric, metric_name = results['metric']
            print(f"| [{metric_name}] {dev_metric:.4g} ", end='')
            if task != 'regression':
                print(f"| [log-loss] {results['log_loss']:.4g} ", end='')
                self.history['val']['log_loss'].append(results['log_loss'])
            self.history['val']['metric_name'] = metric_name
            self.history['val']['metric'].append(dev_metric)
            self.history['val']['tot_time'] += results['time']
            self.history['val']['avg_step_time'] = self.history['val']['tot_time'] / tot_step
            self.history['val']['avg_epoch_time'] = self.history['val']['avg_step_time'] * steps_per_epoch
            dev_metric = {'metric': dev_metric, 'log_loss': results['log_loss']}
            if self.early_stop_handler(epoch, tot_step, dev_metric, task, patience, save_path, y_std, test_loader):
                print(' <<< Early Stop')
                return True
        print()
        return False
    
    def load_best_dnn(self, save_path, file='best'):
        model_file = Path(save_path) / f"{file}.bin"
        sparser_file = Path(save_path) / f"{file}-sparser.bin"
        if not os.path.exists(model_file):
            print(f'There is no {file} checkpoint, loading the last one...')
            model_file = Path(save_path) / 'final.bin'
            sparser_file = Path(save_path) / "final-sparser.bin"
        else:
            print(f'Loading {file} model...')
        self.model.load_state_dict(torch.load(model_file))
        if os.path.exists(sparser_file):
            self.sparser.load_state_dict(torch.load(sparser_file))
        print('successfully')
