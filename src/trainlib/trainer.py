# Typing includes.
from typing import Dict, List, Optional, Any, Tuple, Callable, Iterable

import os
import time
import logging
import pprint
import tqdm
import pickle

import torch
import torch.nn as nn

from sklearn.metrics import roc_auc_score, average_precision_score, accuracy_score
from sklearn.metrics import precision_recall_fscore_support, roc_auc_score, roc_curve, precision_recall_curve
from torch.nn.parallel.data_parallel import DataParallel
from torch.utils.data import DataLoader, random_split

from models import EHRModel
from datasets import StructuredDataset, DatasetCacher 

# from utils.trainer_utils import (
#     rename_logger,
#     should_stop_early
# )
# from datasets import (
#   Dataset,
#   TokenizedDataset,
#   MLMTokenizedDataset
# )

logger = logging.getLogger(__name__)


def should_stop_early():
    return False

def save_pickle(data, filename):
    with open(filename, "wb") as f:
        pickle.dump(data, f)

class Trainer:
    def __init__(self, args):
        self.args = args
        self.save_dir = args.save_dir
        self.save_prefix = args.save_prefix
        self.embed_model_type = args.embed_model_type
        
        
        # Load either a code_emb embed model or a desc_emb embed model.
        self.model = EHRModel(self.args)
        self.n_epochs = self.args.n_epochs
        self.criterion = nn.BCELoss()
        self.optimizer = torch.optim.Adam(self.model.parameters(), lr=self.args.learning_rate)
      
    
        # Load train + validation splits.
        self.data_loaders = {}
        # for subset in ['train'] + self.args.valid_subsets:
        self.load_dataset()
        
        
    def train(self):
        # def train(model, train_loader, val_loader, n_epochs, criterion, optimizer):
        """
        Train the model.

        Arguments:
            model: the RNN model
            train_loader: training dataloder
            val_loader: validation dataloader
            n_epochs: total number of epochs

        You need to call `eval_model()` at the end of each training epoch to see how well the model performs 
        on validation data.

        Note that please pass all four arguments to the model so that we can use this function for both 
        models. (Use `model(x, masks, rev_x, rev_masks)`.)
        """
        epoch_times = []
        eval_results_val = []
        eval_results_test = []
        for epoch in range(self.n_epochs):
            start_epoch = time.time()
            self.model.train()
            train_loss = 0
            # for x, masks, rev_x, rev_masks, y in tqdm.tqdm(self.data_loaders['train']):
            print(f'Training epoch {epoch + 1}')
            for sample_dict in tqdm.tqdm(self.data_loaders['train']):
                """
                    1. zero grad
                    2. model forward
                    3. calculate loss
                    4. loss backward
                    5. optimizer step
                """
                self.optimizer.zero_grad()
                outputs = self.model(**sample_dict)
                y = sample_dict['y']
                loss = self.criterion(outputs, y)
                loss.backward()
                self.optimizer.step()
                train_loss += loss.item()
            # train_loss = train_loss / len(train_loader)
            train_loss = train_loss / len(self.data_loaders['train'])
            print('Epoch: {} \t Training Loss: {:.6f}'.format(epoch+1, train_loss))
            eval_out = self.eval_and_save(epoch+1, self.data_loaders['val'], self.data_loaders['test'])
            should_stop = eval_out[0]
            p, r, f, roc_auc, rcurve, prec_curve, rec_curve, acc = eval_out[1]
            print('Epoch: {} \t Validation p: {:.2f}, r:{:.2f}, acc:{:.2f}, f: {:.2f}, roc_auc: {:.2f}'
                  .format(epoch+1, p, r, acc, f, roc_auc))
            p, r, f, roc_auc, rcurve, prec_curve, rec_curve, acc = eval_out[2]
            print('Epoch: {} \t Test p: {:.2f}, r:{:.2f}, acc:{:.2f}, f: {:.2f}, roc_auc: {:.2f}'
                  .format(epoch+1, p, r, acc, f, roc_auc))
            eval_results_val.append(eval_out[1])
            eval_results_test.append(eval_out[2])
            end_epoch = time.time()
            epoch_times.append(end_epoch - start_epoch)
        
        # Save the test results.
        fname = os.path.join(
            self.save_dir,
            f'train_results_{self.args.embed_model_type}_task_{self.args.task}_isdev{self.args.is_dev}.pkl'
        )
        d = {'eval_results_val': eval_results_val,
             'eval_results_test': eval_results_test,
             'epoch_times': epoch_times}
        save_pickle(d, fname)
        
            
        # p, r, f, roc_auc, rcurve, precision_curve, acc
        return eval_results_val, eval_results_test, epoch_times
            

    def load_dataset(self):
        dataset = None
        metadata = None
        name = ''
        batch_size = 0
        length = 0
        if self.embed_model_type == 'desc_emb_ft':
            return self.load_desc_emb_ft_dataset()
        
        if self.embed_model_type.startswith('code'):
            if self.args.task == 'mort':
                if self.args.is_dev:
                    name, batch_size, length = ('mortality_pred_task_cemb', 0, 631)
                else:
                    name, batch_size, length = ('mortality_pred_task_cemb', 0, 33426)
            elif self.args.task == 'readm':
                if self.args.is_dev:
                    name, batch_size, length = ('readmission_pred_task_cemb', 0, 631)
                else:
                    name, batch_size, length = ('readmission_pred_task_cemb', 0, 33426)
            else:
                raise ValueError(f'Invalid task type: {self.args.task}')
        elif self.embed_model_type.startswith('desc'):
            if self.args.task == 'mort':
                if self.args.is_dev:
                    name, batch_size, length = ('mortality_pred_task_demb', 0, 880)
                else:
                    name, batch_size, length = ('mortality_pred_task_demb', 0, 10000)
            elif self.args.task == 'readm':
                if self.args.is_dev:
                    name, batch_size, length = ('readmission_pred_task_demb', 0, 880)
                else:
                    name, batch_size, length = ('readmission_pred_task_demb', 0, 10000)
            else:
                raise ValueError(f'Invalid task type: {self.args.task}')
        else:
            raise NotImplementedError(self.embed_model_type)
         
        cacher = DatasetCacher()
        if self.embed_model_type.startswith('code'):
            dataset, metadata = cacher.StructuredDatasetFromCache(name, batch_size, length)
        elif self.embed_model_type.startswith('desc'):
            dataset, metadata = cacher.SimpleDatasetFromCache(name, batch_size, length)
        assert(dataset is not None)
        assert(metadata is not None)
        
        generator = torch.Generator().manual_seed(self.args.random_seed)
        datsets = random_split(dataset, [0.8, 0.1, 0.1], generator=generator)
        splits = ['train', 'val', 'test']
        for split, ds in zip(splits, datsets):
            # https://www.kaggle.com/general/159828
            if self.args.override_batch_size:
                # If we have a collate function, we need batching.
                self.data_loaders[split] = DataLoader(ds,
                    batch_size=self.args.override_batch_size,
                    collate_fn=self.args.collate_fn,
                    shuffle=True)
            else:
                # Disable auto batch if batch_size=0. Good for already collated data.
                self.data_loaders[split] = DataLoader(ds,
                    batch_size=None if batch_size < 2 else batch_size,
                    collate_fn=self.args.collate_fn,
                    shuffle=True)
                

    def load_desc_emb_ft_dataset(self):
        generator = torch.Generator().manual_seed(self.args.random_seed)
        datsets = random_split(self.args.no_use_cached_dataset, [0.8, 0.1, 0.1], generator=generator)
        splits = ['train', 'val', 'test']
        for split, ds in zip(splits, datsets):
            # https://www.kaggle.com/general/159828
            assert(self.args.override_batch_size)
            # If we have a collate function, we need batching.
            self.data_loaders[split] = DataLoader(ds,
                batch_size=self.args.override_batch_size,
                collate_fn=self.args.collate_fn,
                shuffle=True)
    
                                          
    def eval_and_save(self, epoch, val_loader, test_loader):
        stop = False
        stop |= should_stop_early()
        
        # args.save_prefix
        fname = f'model_{self.args.embed_model_type}_task_{self.args.task}_isdev{self.args.is_dev}'
        path = os.path.join(self.save_dir, fname + "_best.pt")
        logger.info(
            "Saving checkpoint to {}".format(path)
        )
        # print(f'model_state_dict {self.model.state_dict() is None}')
        # print(f'optimizer_state_dict {self.optimizer.state_dict() is None}')
        # print(f'model_state_dict \n{self.model.state_dict()}')
        # print(f'optimizer_state_dict \n{self.optimizer.state_dict()}')
        # print(f'epoch {epoch}')
        # print(f'args {self.args}')

        #https://stackoverflow.com/questions/50888391/pickle-of-object-with-getattr-method-in-python-returns-typeerror-object-no
        # if not self.embed_model_type == 'desc_emb_ft':
        torch.save(
            {
                # 'model_state_dict': self.model.module.state_dict() if (
                #     isinstance(self.model, DataParallel)
                # ) else self.model.state_dict(),
                'model_state_dict': self.model.state_dict(),
                'optimizer_state_dict': self.optimizer.state_dict(),
                'epochs': epoch,
                'args': self.args,
            },
            path
        )
        logger.info(
            "Finished saving checkpoint to {}".format(path)
        )
        
        # p, r, f, roc_auc, rcurve, prec_curve, rec_curve 
        val_results = self.eval_model(epoch, val_loader)
        test_results = self.eval_model(epoch, test_loader)
        return stop, val_results, test_results


    def eval_model(self, epoch, data_loader):
        self.model.eval()
        y_pred = torch.LongTensor()
        y_score = torch.Tensor()
        y_true = torch.LongTensor()
        # for x, masks, rev_x, rev_masks, y in val_loader:
        for sample_dict in data_loader:
            y = sample_dict['y']
            y_hat = self.model(**sample_dict)
            y_score = torch.cat((y_score,  y_hat.detach().to('cpu')), dim=0)
            y_hat = (y_hat > 0.5).int()
            y_pred = torch.cat((y_pred,  y_hat.detach().to('cpu')), dim=0)
            y_true = torch.cat((y_true, y.detach().to('cpu')), dim=0)
        """
        TODO:
            Calculate precision, recall, f1, and roc auc scores.
            Use `average='binary'` for calculating precision, recall, and fscore.
        """
        p, r, f, _ = precision_recall_fscore_support(y_true, y_pred, average='binary')
        roc_auc = roc_auc_score(y_true, y_score)
        rcurve = roc_curve(y_true, y_score)
        precision_curve, recall_curve, thresholds = precision_recall_curve(y_true, y_score)
        accuracy = accuracy_score(y_true, y_pred)
        return p, r, f, roc_auc, rcurve, precision_curve, recall_curve, accuracy

