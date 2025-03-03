
from typing import List, Optional, Union, Literal
from pathlib import Path
import os
import yaml
import json
import shutil
import warnings

import numpy as np
import pandas as pd
from dataclasses import replace
from collections import Counter
from sklearn.preprocessing import OrdinalEncoder
from sklearn.model_selection import train_test_split

import torch.nn as nn
from .env import (
    BENCHMARKS, DATASETS, CUSTOM_DATASETS, 
    push_custom_datasets, read_custom_infos, write_custom_infos
)
from utils.data_utils import (
    Normalization, NumNanPolicy, CatNanPolicy, CatEncoding, YPolicy,
    CAT_MISSING_VALUE, ArrayDict, TensorDict, TaskType,
    Dataset, Transformations, prepare_tensors, build_dataset, transform_dataset
)
from models.abstract import TabModel

DataFileType = Literal['csv', 'excel', 'npy', 'arff']

class DataProcessor:
    """Base class to process a single dataset"""
    def __init__(
        self, 
        normalization: Optional[Normalization] = None,
        num_nan_policy: Optional[NumNanPolicy] = None,
        cat_nan_policy: Optional[CatNanPolicy] = None,
        cat_min_frequency: Optional[float] = None,
        cat_encoding: Optional[CatEncoding] = None,
        y_policy: Optional[YPolicy] = 'default',
        seed: int = 42,
        cache_dir: Optional[str] = None,
    ):
        self.transformation = Transformations(
            seed=seed, 
            normalization=normalization, 
            num_nan_policy=num_nan_policy,
            cat_nan_policy=cat_nan_policy,
            cat_min_frequency=cat_min_frequency,
            cat_encoding=cat_encoding,
            y_policy=y_policy
        )
        self.cache_dir = cache_dir
    
    def apply(self, dataset: Dataset):
        return transform_dataset(dataset, self.transformation, self.cache_dir)
    
    def save(self, file, **kwargs):
        data_config = {
            'transformation': vars(self.transformation),
            'cache_dir': str(self.cache_dir),
            'meta': kwargs,
        }
        with open(file, 'w') as f:
            yaml.dump(data_config, f, indent=2)
    
    @staticmethod
    def check_splits(dataset: Dataset):
        valid_splits = True
        if 'train' in dataset.y:
            if 'test' not in dataset.y:
                warnings.warn("Missing test split, unable to prediction")
                valid_splits = False
            if 'val' not in dataset.y:
                warnings.warn("Missing dev split, unable to early stop, or ignore this message if no early stop needed.")
                valid_splits = False
            if valid_splits:
                print("ready for training!")
        else:
            raise ValueError("Missing training split in the dataset")
    
    @staticmethod
    def prepare(dataset: Dataset, model: Optional[TabModel] = None, device: str = 'cuda'):
        assert model is not None or device is not None
        def get_spl(X: Optional[Union[ArrayDict, TensorDict]], spl):
            return None if X is None else X[spl]
        if device is not None or isinstance(model.model, nn.Module):
            device = device or model.model.device
            X_num, X_cat, ys, ids = prepare_tensors(dataset, device)
            return {spl: (
                get_spl(X_num, spl), 
                get_spl(X_cat, spl), 
                get_spl(ys, spl),
                get_spl(ids, spl)
            ) for spl in ys}
        else:
            return {spl: (
                get_spl(dataset.X_num, spl), 
                get_spl(dataset.X_cat, spl), 
                get_spl(dataset.y, spl),
                get_spl(dataset.ids, spl)
            ) for spl in dataset.y}
    
    @staticmethod
    def load_preproc_default(
        output_dir, # output preprocessing infos
        model_name, 
        dataset_name, 
        benchmark_name: Optional[str] = None, 
        seed: int = 42, 
        cache_dir: Optional[str] = None,
        add_ids: bool = False, # add sample ids
        one_hot_label: bool = False, # for multi-class labels
        label_smoothing: float = 0.0,
    ):
        global DATASETS, CUSTOM_DATASETS
        """default data preprocessing pipeline"""
        if dataset_name in DATASETS or dataset_name in CUSTOM_DATASETS:
            data_src = DATASETS if dataset_name in DATASETS else CUSTOM_DATASETS
            data_config = data_src[dataset_name]
            data_path = Path(data_config['path'])
            data_config.setdefault('normalization', 'quantile')
            normalization = data_config['normalization']
        elif benchmark_name is not None:
            assert benchmark_name in BENCHMARKS, f"Benchmark '{benchmark_name}' is not included, \
                please choose one of '{list(BENCHMARKS.keys())}', for include your benchmark manually."
            benchmark_info = BENCHMARKS[benchmark_name]
            assert dataset_name in benchmark_info['datasets'], f"dataset '{dataset_name}' not in benchmark '{benchmark_name}'"
            data_path = Path(benchmark_info['path']) / dataset_name
            normalization = benchmark_info['normalization'].get(dataset_name, 'quantile')
        else:
            raise ValueError(f"No dataset '{dataset_name}' is available, \
                if you want to use a custom dataset (from csv file), using `add_custom_dataset`")
        
        dataset = Dataset.from_dir(data_path)
        # default preprocess settings
        num_nan_policy = 'mean' if dataset.X_num is not None and \
            any(np.isnan(dataset.X_num[spl]).any() for spl in dataset.X_num) else None
        cat_nan_policy = None
        if model_name in ['xgboost', 'catboost', 'lightgbm']: # for tree models or other sklearn algorithms
            normalization = None
            cat_min_frequency = None
            cat_encoding = 'one-hot'
            if model_name in ['catboost']:
                cat_encoding = None
        else: # for dnns
            # BUG: (dataset.X_cat[spl] == CAT_MISSING_VALUE).any() has different action
            # dtype: int -> bool, dtype: string -> array[bool], dtype: object -> np.load error
            # CURRENT: uniformly using string type to store catgorical features
            if dataset.X_cat is not None and \
                any((dataset.X_cat[spl] == CAT_MISSING_VALUE).any() for spl in dataset.X_cat):
                cat_nan_policy = 'most_frequent'
            cat_min_frequency = None
            cat_encoding = None
        cache_dir = cache_dir or data_path
        processor = DataProcessor(
            normalization=normalization,
            num_nan_policy=num_nan_policy,
            cat_nan_policy=cat_nan_policy,
            cat_min_frequency=cat_min_frequency,
            cat_encoding=cat_encoding,
            seed=seed,
            cache_dir=Path(cache_dir),
        )
        dataset = processor.apply(dataset)
        # check train, val, test splits
        DataProcessor.check_splits(dataset)
        # save preprocessing infos
        if not os.path.exists(output_dir):
            os.makedirs(output_dir)
        processor.save(
            Path(output_dir) / 'data_config.yaml',
            benchmark=str(benchmark_name),
            dataset=dataset_name
        )
        if add_ids:
            ids = {
                'train': np.arange(dataset.size('train')),
                'val': np.arange(dataset.size('val')) + dataset.size('train'),
                'test': np.arange(dataset.size('test')) + dataset.size('train') + dataset.size('val'),
            }
            dataset = replace(dataset, ids=ids)
        else:
            dataset = replace(dataset, ids=None)
        if one_hot_label or label_smoothing > 0:
            if dataset.task_type.value == 'multiclass':
                # convert to one-hot and use BCE loss function
                new_ys = {k: np.zeros((dataset.size(k), dataset.n_classes), dtype=np.float32) for k in ['train', 'val', 'test']}
                if label_smoothing > 0:
                    sm_label = label_smoothing / (dataset.n_classes - 1)
                    new_ys['train'] = np.ones((dataset.size('train'), dataset.n_classes), dtype=np.float32) * sm_label
                for k in ['train', 'val', 'test']:
                    for i in range(dataset.size(k)):
                        if label_smoothing > 0 and k == 'train':
                            new_ys[k][i, dataset.y[k][i]] = 1 - label_smoothing
                        else:
                            new_ys[k][i, dataset.y[k][i]] = 1.0
            elif dataset.task_type.value == 'binclass':
                new_ys = {k: np.abs(dataset.y[k] - label_smoothing) if k =='train' else dataset.y[k] for k in ['train', 'val', 'test']}
            else:
                new_ys = dataset.y
            dataset = replace(dataset, y=new_ys)
        return dataset

    @staticmethod
    def split(
        X_num: Optional[np.ndarray] = None, 
        X_cat: Optional[np.ndarray] = None,  
        ys: np.ndarray = None,  
        train_ratio: float = 0.8,
        stratify: bool = True,
        seed: int = 42,
    ):
        assert 0 < train_ratio < 1
        assert ys is not None
        sample_idx = np.arange(len(ys))
        test_ratio = 1 - train_ratio
        _stratify = None if not stratify else ys
        train_idx, test_idx = train_test_split(sample_idx, test_size=test_ratio, random_state=seed, stratify=_stratify)
        _stratify = None if not stratify else ys[train_idx]
        train_idx, val_idx = train_test_split(train_idx, test_size=test_ratio, random_state=seed, stratify=_stratify)
        if X_num is not None:
            X_num = {'train': X_num[train_idx], 'val': X_num[val_idx], 'test': X_num[test_idx]}
        if X_cat is not None:
            X_cat = {'train': X_cat[train_idx], 'val': X_cat[val_idx], 'test': X_cat[test_idx]}
        ys = {'train': ys[train_idx], 'val': ys[val_idx], 'test': ys[test_idx]}
        idx = {'train': train_idx, 'val': val_idx, 'test': test_idx}
        return X_num, X_cat, ys, idx
    
    @staticmethod
    def del_custom_dataset(
        dataset_names: Union[str, List[str]]
    ):
        global DATASETS, CUSTOM_DATASETS
        all_infos = read_custom_infos()
        if isinstance(dataset_names, str):
            dataset_names = [dataset_names]
        for dataset_name in dataset_names:
            if dataset_name not in CUSTOM_DATASETS:
                print(f"custom dataset: {dataset_name} not exist")
                continue
            elif dataset_name in DATASETS:
                print(f"can not delete an in-built dataset: {dataset_name}")
                continue
            data_info = CUSTOM_DATASETS[dataset_name]
            task = data_info['task_type']
            data_path = data_info['path']
            data_idx = [info['name'] for info in all_infos['data_list']].index(dataset_name)
            all_infos['data_list'].pop(data_idx)
            all_infos['n_datasets'] -= 1
            all_infos[task] -= 1
            shutil.rmtree(data_path)
            print(f"delete dataset: {dataset_name} successfully")
        write_custom_infos(all_infos)
        from .env import CUSTOM_DATASETS # BUG: refresh the global variable

    @staticmethod
    def add_custom_dataset(
        file: Union[str, Path],
        format: DataFileType = 'csv',
        dataset_name: Optional[str] = None,
        task: Optional[str] = None,
        num_cols: Optional[List[int]] = None,
        cat_cols: Optional[List[int]] = None,
        label_index: int = -1, # label column index
        header: Optional[int] = 0, # header row
        max_cat_num: int = 16,
        train_ratio: float = 0.8, # split train / test, train / val
        seed: float = 42, # random split seed
    ):
        """
        Support for adding a custom dataset from a single data file
        ---
        read a raw csv file, process into 3 splits (train, val, test), and add to custom_datasets

        TODO: adding a dataset from prepared data split files 
        TODO: support no validation split
        """
        global DATASETS, CUSTOM_DATASETS
        file_name = Path(file).name
        assert file_name.endswith(format), f'please check if the file \
            is in {format} format, or add the suffix manually'
        dataset_name = dataset_name or file_name[:-len(format)-1]
        assert dataset_name not in DATASETS, f'same dataset name as an in-built dataset: {dataset_name}'
        assert dataset_name not in CUSTOM_DATASETS, f"existing custom dataset '{dataset_name}' found"
        
        if format == 'csv':
            datas: pd.DataFrame = pd.read_csv(file, header=header)
            columns = datas.columns if header is not None else None
        elif format == 'npy':
            header = None # numpy file has no headers
            columns = None
            datas = np.load(file)
            raise NotImplementedError("only support load csv file now")
        else:
            raise ValueError("other support format to be add further")
        
        X_idx = list(range(datas.shape[1]))
        y_idx = X_idx.pop(label_index)
        label_name = columns[y_idx] if columns is not None else None
        # numerical and categorical feature detection
        if num_cols is None or cat_cols is None:
            print('automatically detect column type...')
            print('max category amount: ', max_cat_num)
            num_cols, cat_cols = [], []
            num_names, cat_names = [], []
            for i in X_idx:
                if datas.iloc[:, i].values.dtype == float:
                    num_cols.append(i)
                    if columns is not None:
                        num_names.append(columns[i])
                else: # int or object (str)
                    if len(set(datas.iloc[:, i].values)) <= max_cat_num:
                        cat_cols.append(i)
                        if columns is not None:
                            cat_names.append(columns[i])
                    elif datas.iloc[:, i].values.dtype == int:
                        num_cols.append(i)
                        if columns is not None:
                            num_names.append(columns[i])
            if not num_names and not cat_names:
                num_names, cat_names = None, None
        elif columns:
            num_names = [columns[i] for i in num_cols]
            cat_names = [columns[i] for i in cat_cols]
        else:
            num_names, cat_names = None, None
        n_num_features = len(num_cols)
        n_cat_features = len(cat_cols)
        # build X_num and X_cat
        X_num, ys = None, datas.iloc[:, y_idx].values
        if len(num_cols) > 0:
            X_num = datas.iloc[:, num_cols].values.astype(np.float32)
        # check data type
        X_cat = []
        for i in cat_cols:
            if datas.iloc[:, i].values.dtype == int:
                x = datas.iloc[:, i].values.astype(np.int64)
                # ordered by value
                # x = OrdinalEncoder(categories=[sorted(list(set(x)))]).fit_transform(x.reshape(-1, 1))
            else: # string object
                x = datas.iloc[:, i].values.astype(object)
                # most_common = [item[0] for item in Counter(x).most_common()]
                # ordered by frequency
                # x = OrdinalEncoder(categories=[most_common]).fit_transform(x.reshape(-1, 1))
            X_cat.append(x.astype(np.str0)) # Encoder Later, compatible with Line 140
        X_cat = np.stack(X_cat, axis=1) if len(X_cat) > 0 else None # if using OrdinalEncoder, np.concatenate
        # detect task type
        def process_non_regression_labels(ys: np.ndarray, task):
            if ys.dtype in [int, float]:
                ys = OrdinalEncoder(categories=[sorted(list(set(ys)))]).fit_transform(ys.reshape(-1, 1))
            else:
                most_common = [item[0] for item in Counter(ys).most_common()]
                ys = OrdinalEncoder(categories=most_common).fit_transform(ys.reshape(-1, 1))
            ys = ys[:, 0]
            return ys.astype(np.float32) if task == 'binclass' else ys.astype(np.int64)
        
        if task is None:
            if ys.dtype in [int, object]:
                task = 'binclass' if len(set(ys)) == 2 else 'multiclass'
                ys = process_non_regression_labels(ys, task)
            elif ys.dtype == float:
                if len(set(ys)) == 2:
                    task = 'binclass'
                    ys = process_non_regression_labels(ys, task)
                else:
                    task = 'regression'
                    ys = ys.astype(np.float32)
        else:
            if task == 'regression':
                ys = ys.astype(np.float32)
            else:
                ys = process_non_regression_labels(ys, task)

        # split datasets
        stratify = task != 'regression'
        X_num, X_cat, ys, idx = DataProcessor.split(X_num, X_cat, ys, train_ratio, stratify, seed)
        # push to CUSTOM_DATASETS
        data_info = {
            'name': dataset_name,
            'id': f'{dataset_name.lower()}--custom',
            'task_type': task,
            'label_name': label_name,
            'n_num_features': n_num_features,
            'num_feature_names': num_names,
            'n_cat_features': n_cat_features,
            'cat_feature_names': cat_names,
            'test_size': len(ys['test']),
            'train_size': len(ys['train']),
            'val_size': len(ys['val'])}
        push_custom_datasets(X_num, X_cat, ys, idx, data_info)
        from .env import CUSTOM_DATASETS # refresh global variable
        print(f'finish, now you can load your dataset with `load_preproc_default({dataset_name})`')

class BenchmarkProcessor:
    """Prepare datasets in the Literatures"""
    def __init__(self) -> None:
        pass