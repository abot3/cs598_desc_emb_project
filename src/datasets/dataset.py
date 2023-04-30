import logging
import hashlib
import pickle
import tqdm
import os
# Typing includes.
from typing import Dict, List, Optional, Any, Tuple, Callable, Iterable

import torch
from torch.utils.data import Dataset, DataLoader, random_split

from pyhealth.datasets.utils import MODULE_CACHE_PATH

logger = logging.getLogger(__name__)


# The collate function idea doesn't work because collating requires the batch size.
# The batch size can change so it doesn't really work.
# Unless we can use https://pytorch.org/docs/stable/data.html#disable-automatic-batching
# But then the train/val/test datasets & batch sizes are fixed after the initial save to file.
# Should probably incorporate the batch size into the file hash.
# That would also mean we should return 

def load_pickle(filename):
    with open(filename, "rb") as f:
        return pickle.load(f)


def save_pickle(data, filename):
    with open(filename, "wb") as f:
        pickle.dump(data, f)

        
class SimpleDataset(Dataset):
    '''Wraps a list. For post cache datasets.
    '''
    def __init__(self, samples: Iterable):
        self.samples = samples
        
    def __len__(self):
        return len(self.samples)
    
    def __getitem__(self, index):
        return self.samples[index]
    
    
class StructuredDataset(Dataset):
    '''Wraps a list. For post cache datasets.
    '''
    def __init__(self, samples: Iterable, metadata: Dict):
        self.samples = samples
        self.metadata = metadata
        assert('extra_data' in self.metadata.keys())
        self.keywords = self.metadata['extra_data']['keywords']
        
    def __len__(self):
        return len(self.samples)
    
    def __getitem__(self, index):
        sample: tuple = self.samples[index]
        return {k:v for k,v in zip(self.keywords, sample)}

    
class DatasetCacher:
    '''Caches a dataset to file. Returns dataset from file with loader if desired.
        Datasets are defined by 3 params:
            1. name
            2. batch_size
            3. length
    '''
    def __init__(self):
        pass
   

    def _HashStr(self, s) -> str:
        return hashlib.md5(s.encode()).hexdigest() 


    def _MakeMetadata(self, name: str, batch_size: int, length: int) -> Dict:
        metadata = {
            'auto_batch': batch_size == 0,
            'batch_size': batch_size,
            'length': length,
            'dataset_name': name
        }
        return metadata

    
    def _GetHashString(self, args) -> str:
        '''Compute filename for cache.
        '''
        args_to_hash = [
            args['dataset_name'],
            args['auto_batch'],
            args['batch_size'],
            args['length']
        ]
        filename = '_'.join([
            self._HashStr("+".join([str(arg) for arg in args_to_hash])),
            args['dataset_name'],
            'bs_' + str(args['batch_size']),
            'len_' + str(args['length']),
            '.pkl',
        ])
        return os.path.join(MODULE_CACHE_PATH, filename)

    
    def DatasetToCacheFromLoader(self, loader: DataLoader,
                                 task_fn: Callable,
                                 batch_size: int=0,
                                 overwrite: bool=False,
                                 extra_data: Dict=None):
        to_pickle = {
            'dataset': [x for x in tqdm.tqdm(loader)],
            'metadata': self._MakeMetadata(task_fn.__name__, batch_size, len(loader.dataset))
        }
        if extra_data:
            to_pickle['metadata']['extra_data'] = extra_data
        fname = self._GetHashString(to_pickle['metadata'])
        logging.info(f'Caching dataset {to_pickle["metadata"]} to file `{fname}`.')
        if not os.path.exists(fname) or overwrite:
            save_pickle(to_pickle, fname)
        return to_pickle['metadata']
        
    
    def DatasetToCache(self, dataset: Dataset,
                           task_fn: Callable,
                           batch_size: int=0,
                           overwrite: bool=False,
                           extra_data: Dict=None) -> None:
        '''Write all items in the dataset to file.
        
          If the dataset already exists on disk this function is a no-op.
          The dataset is hashed by iterating over the whole dataset and writing
          all items to disk using pickle.
        '''
        to_pickle = {
            'dataset': [x for x in tqdm.tqdm(dataset)],
            'metadata': self._MakeMetadata(task_fn.__name__, batch_size, len(dataset))
        }
        if extra_data:
            to_pickle['metadata']['extra_data'] = extra_data
        fname = self._GetHashString(to_pickle['metadata'])
        logging.info(f'Caching dataset {to_pickle["metadata"]} to file `{fname}`.')
        if not os.path.exists(fname) or overwrite:
            save_pickle(to_pickle, fname)
        return to_pickle['metadata']
    
    
    def SimpleDatasetFromCache(self, task_fn: Callable,
                         batch_size: int,
                         length: int) -> Dataset:
        '''Retrieve all items in the dataset from file.
        '''
        metadata = self._MakeMetadata(task_fn.__name__, batch_size, length)
        fname = self._GetHashString(metadata)
        logging.info(f'Loading cached dataset {metadata} from file `{fname}`.')
        d = load_pickle(fname)
        return SimpleDataset(d['dataset']), d['metadata'] 
    
    def StructuredDatasetFromCache(self, name: str,
                         batch_size: int,
                         length: int) -> Dataset:
        '''Retrieve all items in the dataset from file.
        '''
        metadata = self._MakeMetadata(name, batch_size, length)
        fname = self._GetHashString(metadata)
        logging.info(f'Loading cached dataset {metadata} from file `{fname}`.')
        d = load_pickle(fname)
        return StructuredDataset(d['dataset'], d['metadata']), d['metadata']

    def DataloaderFromCache(self, *args, **kwargs) -> DataLoader:
        dataset, metadata = self.SimpleDatasetFromCache(*args, **kwargs)
        batch_size = metadata['batch_size']
        # Disable auto batch if batch_size=0. Good for already collated data.
        loader = DataLoader(dataset,
                   batch_size=None if batch_size < 2 else batch_size,
                   shuffle=False)
        return loader, metadata
    