# General includes.
import os
import re

# Typing includes.
from typing import Dict, List, Optional, Any, Tuple, Callable, Iterable

# Numerical includes.
import numpy as np
import pandas as pd
import torch

# Model imports 
from transformers import BertTokenizer, BertModel, BertForMaskedLM, BertConfig, BertTokenizerFast
from transformers import TensorType
from transformers import AutoTokenizer, AutoConfig, AutoModel

# pyHealth includes.
from pyhealth.datasets import BaseDataset, MIMIC3Dataset, eICUDataset, SampleDataset

"""
BertFineTuneTransform
"""
class BertFineTuneTransform(object):
    """Transform a sample's (a single visit's) text into 1 embedding vector.
    
    The embeddings of each text field are combined by embedding
    each separately then summing.

    Args:
        output_size (tuple or int): Desired output size. If tuple, output is
            matched to output_size. If int, smaller of image edges is matched
            to output_size keeping aspect ratio the same.
    """
    def __init__(self):
        self.tokenizer = BertTokenizerFast.from_pretrained('bert-base-uncased', model_max_length=100)
        self.tokenzier = AutoTokenizer.from_pretrained("prajjwal1/bert-tiny")
        print(f'Tokenizer: {self.tokenizer}')
        # max length of sentence not real emb size.
        self.emb_size = 20  # 15 worked here.
    
    def _get_embeddings_of_sentences_with_mask(self, field, pad) -> torch.tensor:
        return self._get_embeddings_of_sentences(field[:pad])
        
    def _get_embeddings_of_sentences(self, sentences: List[str]) -> torch.tensor:
        batch_enc = self.tokenizer.batch_encode_plus(sentences, padding=True,
                                return_attention_mask=True, return_length=True,
                                truncation='longest_first', max_length=self.emb_size)
        batch_enc_tensor = batch_enc.convert_to_tensors(tensor_type=TensorType.PYTORCH)
        # decoded = self.tokenizer.decode(batch_enc['input_ids'][0])
        # print(f'decoded {decoded}')
        # assert(decoded[0:5] == '[CLS]')
        return batch_enc_tensor
    
    def __call__(self, sample):
        len_embeddings = (len(sample['conditions_text']) +
                          len(sample['procedures_text']) +
                          len(sample['drugs_text']))
        sample_text = []
        sample_masks = []
        if sample.get('conditions_text_pad'):
            pad = sample['conditions_text_pad']
            sample_text.extend(sample['conditions_text'][:pad])
        if sample.get('procedures_text_pad'):
            pad = sample['procedures_text_pad']
            sample_text.extend(sample['procedures_text'][:pad])
        if sample.get('drugs_text_pad'):
            pad = sample['drugs_text_pad']
            sample_text.extend(sample['drugs_text'][:pad])
            
        batch_enc_tensor = self._get_embeddings_of_sentences(sample_text)
        assert(batch_enc_tensor['input_ids'].device == torch.device('cpu'))
        assert(batch_enc_tensor['token_type_ids'].device == torch.device('cpu'))
        # print(f'FT transform shape\n'
        #       f"iids {batch_enc_tensor['input_ids'].shape}\n"
        #       f"ttids {batch_enc_tensor['token_type_ids'].shape}")
        # Return the ((model_inputs), label) 
        d = {
            'input_ids': batch_enc_tensor['input_ids'],
            'attention_masks': batch_enc_tensor['attention_mask'],
            'token_type_ids': batch_enc_tensor['token_type_ids'],
        }
        return (d, sample['label'])

    
"""
BertTextEmbedTransform
"""
# Could use DistilBERT to decrease embedding time.
# https://huggingface.co/docs/transformers/model_doc/distilbert?highlight=distilberttokenizerfast#distilbert
class BertTextEmbedTransform(object):
    """Transform a sample's (a single visit's) text into 1 embedding vector.
    
    The embeddings of each text field are combined by embedding
    each separately then summing.

    Args:
        output_size (tuple or int): Desired output size. If tuple, output is
            matched to output_size. If int, smaller of image edges is matched
            to output_size keeping aspect ratio the same.
    """
    # https://stackoverflow.com/questions/69517460/bert-get-sentence-embedding
    # https://huggingface.co/docs/transformers/internal/tokenization_utils#transformers.tokenization_utils_base.PreTrainedTokenizerBase
    # https://gmihaila.github.io/tutorial_notebooks/bert_inner_workings/

    def __init__(self, bert_model: Any, embedding_size: int, use_tokenizer_fast: bool,
                 use_gpu: bool = True, use_bert_tiny: bool = False):
        assert isinstance(embedding_size, (int, tuple))
        self.use_gpu = use_gpu
        self.cuda = 'cuda'
        self.bert_config = BertConfig()
        # self.bert_config = AutoConfig.from_pretrained("bert-base-uncased")
        # self.bert_model = BertModel(self.bert_config).from_pretrained('bert-base-uncased', config=self.bert_config)
        # self.bert_model = AutoModel.from_pretrained('bert-base-uncased')
        if use_bert_tiny:
            with torch.no_grad():
                self.bert_model = AutoModel.from_pretrained('prajjwal1/bert-tiny')
                self.bert_emb_size = self.bert_model.config.hidden_size 
        else:
            with torch.no_grad():
                self.bert_model = BertModel(self.bert_config).from_pretrained('bert-base-uncased')
                self.bert_emb_size = self.bert_model.config.hidden_size 
        if self.use_gpu:
            self.bert_model.to(self.cuda)
        self.bert_config = self.bert_model.config
        self.bert_model.eval()
        # We unfortunately can't put the tokenizer on the GPU.
        # https://stackoverflow.com/questions/66096703/running-huggingface-bert-tokenizer-on-gpu
        if use_tokenizer_fast:
            self.tokenizer = BertTokenizerFast.from_pretrained('bert-base-uncased', model_max_length=100)
        else:
            self.tokenizer = BertTokenizer.from_pretrained('bert-base-uncased', model_max_length=100)
    
    def _tokenize_text(self, text: str) -> str:
        tokenized_text = self.tokenizer.tokenize(text)
        return tokenized_text
    
    def _get_embeddings_of_sentences_with_mask(self, field, pad) -> torch.tensor:
        return self._get_embeddings_of_sentences(field[:pad])
        
    
    def _get_embeddings_of_sentences(self, sentences: List[str]) -> torch.tensor:
        # tokenized_sentences = [self.tokenizer.tokenize(t, padding=True) for t in sentences]
        # Tokenize the input sentence with attention masks.
        batch_enc = self.tokenizer.batch_encode_plus(sentences, padding=True,
                                return_attention_mask=True, return_length=True)
        # print(f'input sentence: {sentences[0]}\n'
        #       f'token sentence: {self.tokenizer.decode(batch_enc["input_ids"][0])}\n')
        batch_enc_tensor = batch_enc.convert_to_tensors(tensor_type=TensorType.PYTORCH)
        if self.use_gpu:
            batch_enc_tensor.to(self.cuda)
      
        # Run BERT model forward pass on tokenized input.
        with torch.no_grad():
            embeddings = self.bert_model(input_ids=batch_enc_tensor['input_ids'],
                                         attention_mask=batch_enc_tensor['attention_mask'],
                                         token_type_ids=batch_enc_tensor['token_type_ids'],
                                         # could turn off so we're faster
                                         output_attentions=True)
        # embeddings, _ = self.bert_model(**batch_enc)
        # print(f'embeddings:\n {dir(embeddings)}')
        #attention = encoded['attention_mask'].reshape((lhs.size()[0], lhs.size()[1], -1)).expand(-1, -1, 768)
       
        # These may be on GPU.
        return embeddings.last_hidden_state, embeddings.attentions 
    
    def __call__(self, sample):
        len_embeddings = (len(sample['conditions_text']) +
                          len(sample['procedures_text']) +
                          len(sample['drugs_text']))
      
        sample_text = []
        sample_masks = []
        if sample.get('conditions_text_pad'):
            pad = sample['conditions_text_pad']
            sample_text.extend(sample['conditions_text'][:pad])
        if sample.get('procedures_text_pad'):
            pad = sample['procedures_text_pad']
            sample_text.extend(sample['procedures_text'][:pad])
        if sample.get('drugs_text_pad'):
            pad = sample['drugs_text_pad']
            sample_text.extend(sample['drugs_text'][:pad])
           
        sample_embeddings, sample_attentions = self._get_embeddings_of_sentences(sample_text)
        
        # https://stackoverflow.com/questions/61323621/how-to-understand-hidden-states-of-the-returns-in-bertmodelhuggingface-transfo
        # https://datascience.stackexchange.com/questions/66207/what-is-purpose-of-the-cls-token-and-why-is-its-encoding-output-important
        # Take only the last hidden state embeddings from BERT.
        # We need to take index 0, because it's the CLS token prepended to our sentence.
        # The [CLS] represents the meaning of the whole sentence.
        # embeddings = torch.squeeze(sample_embeddings[:, -1, :], dim=1)
        embeddings = torch.squeeze(sample_embeddings[:, 0, :], dim=1)
        if self.use_gpu:
            assert(embeddings.device == torch.device('cuda:0'))
        else:
            assert(embeddings.device == torch.device('cpu'))
        if self.use_gpu:
            embeddings = embeddings.to('cpu')
        
        # We could multiply by attentions here:
        # attentions = torch.squeeze(sample_attentions[:, -1, :], dim=1)
        # embeddings = embeddings * attentions

        # The 1st dimension is seq length. The second dimension is embedding length of each sentence.
        assert(embeddings.shape[-1] == self.bert_config.hidden_size)
        assert(len(embeddings.shape) == 2)
        assert(embeddings.dtype == torch.float)
        
        # Return the ((model_inputs), label) 
        return (embeddings, sample['label'])
    
    
"""
TextEmbedDataset
"""
class TextEmbedDataset(SampleDataset):
    '''The BERT text embedding process is very slow. We want to avoid it.
   
    To prevent re-processing of the same input cache the sample locally.
    Some suggestions here:
        https://stackoverflow.com/questions/61393613/pytorch-speed-up-data-loading.
        https://discuss.pytorch.org/t/best-practice-to-cache-the-entire-dataset-during-first-epoch/19608
        
    1. Preprocess and write the preprocessed text back out to disk.
    2. Cache the transform output in a hashtable. See functools.lru_cache().
    3. https://pytorch.org/data/main/ ?
    
    Some concerns related to num_workers > 1, i.e. multiprocessing enabled.
    See torch.save() to cache a tensor.
    '''
    
    def __init__(self, dataset: SampleDataset, transform=None, should_cache=True):
        """Wraps a SampleEHRDataset with transforms.
        Arguments:
            dataset: dataset to transform
            transform (callable, optional): Optional transform to be applied
                on a sample.
        """
        # self.dataset = dataset
        self.transform = transform
        self.transformed = [False for x in dataset]
        self.should_cache = should_cache
        super().__init__([x for x in dataset])

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        if torch.is_tensor(idx):
            assert(False)
            idx = idx.tolist()

        # Cache the transformed version of the data.
        sample = None
        if self.should_cache and self.transform and not self.transformed[idx]:
            self.samples[idx] = self.transform(self.samples[idx])
            self.transformed[idx] = True
            sample = self.samples[idx]
        elif not self.should_cache:
            sample = self.transform(self.samples[idx])

        return sample