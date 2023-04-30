import logging
# Typing includes.
from typing import Dict, List, Optional, Any, Tuple, Callable, Iterable

import torch
import torch.nn as nn
from torch.nn.utils.rnn import pack_padded_sequence, pad_packed_sequence
from transformers import BertTokenizer, BertModel, BertForMaskedLM, BertConfig, BertTokenizerFast
from transformers import TensorType
from transformers import AutoConfig, AutoModel

from transformers import AutoConfig, AutoModel

logger = logging.getLogger(__name__)
_BERT_EMBEDDING_SIZE=768


def demb_sum_embeddings_with_mask(x, masks):
    '''
    Inputs:
        x: the embeddings of diagnosis sequence of shape (batch_size, # events(diag+proc+presc), embedding_dim)
        masks: the padding masks of shape (batch_size, # events(diag+proc+presc))
    Outputs:
        sum_embeddings: the sum of embeddings of shape (batch_size, embeddings_dim)
    '''
    # tmp = torch.zeros(x.shape)
    # m = torch.sum(masks, dim=1) > 0
    # tmp[m, :] = 1
    # tmp = x * tmp
    # tmp = torch.sum(tmp, dim=1)
    return x
    
    a = x
    a[~masks, :] = 0
    # tmp = torch.sum(a, dim=1)
    tmp = a
    return tmp


def demb_get_last_visit(hidden_states, masks):
    """
    TODO: obtain the hidden state for the last true visit (not padding visits)

    Arguments:
        hidden_states: the hidden states of each visit of shape
                       (batch_size, # events(diag+proc+presc), embedding_dim)
        masks: the padding masks of shape (batch_size, # events(diag+proc+presc))

    Outputs:
        last_hidden_state: the hidden state for the last true visit of shape (batch_size, embedding_dim)
        
    NOTE: DO NOT use for loop.
    
    HINT: First convert the mask to a vector of shape (batch_size,) containing the true visit length; 
          and then use this length vector as index to select the last visit.
    """
    m = torch.sum(masks, dim=1)
    # m[m > 0] = 1
    # m = torch.sum(m, dim=1)
    m[m > 0] = m[m > 0] - 1
    # print(f'selecting {m}')
    
    tmp1 = m
    tmp = torch.reshape(m, (-1,1,1))
    tmp = tmp.expand(-1, -1, hidden_states.shape[2])
    last_hidden_state = torch.gather(hidden_states, axis=1, index=tmp)
    last_hidden_state = torch.squeeze(last_hidden_state)
    # print(f'tmp.shape {tmp.shape}\n'
    #       f'last_hidden_state {last_hidden_state.shape}\n{last_hidden_state}')
    assert(torch.equal(last_hidden_state[0, :], torch.squeeze(hidden_states[0, tmp1[0], :])))
    
    return last_hidden_state


class DembFtEmbed(nn.Module):
    '''This model runs BERT with torch.grad enabled. This allows fine-tuning the model's
       weights.
       
        This takes significantly longer than DembRNN because the model is evaluated twice.
        1. The forward pass is run to generate embeddings for the prediction RNN.
        2. The BERT model weights (large) are updated during backprop to fine-tune.
    '''
    def __init__(self, args):
        super().__init__()
    
    
    def forward(self, x, masks, rev_x, rev_masks):
         return x, masks, rev_x, rev_masks
    

class DembFtRNN(nn.Module):
    ''' Bidirectional RNN model accepting text string inputs.
    
    This model runs BERT with torch.grad enabled. This allows fine-tuning the model's
    weights.
    This takes significantly longer than DembRNN because the model is evaluated twice.
    1. The forward pass is run to generate embeddings for 
    '''
    
    def __init__(self, args, bert_emb_size:int=_BERT_EMBEDDING_SIZE):
        super().__init__()
        """
        TODO: 
            2. Define the RNN using `nn.GRU()`; Set `hidden_size` to 128. Set `batch_first` to True.
            2. Define the RNN for the reverse direction using `nn.GRU()`;
               Set `hidden_size` to 128. Set `batch_first` to True.
            3. Define the linear layers using `nn.Linear()`; Set `in_features` to 256, and `out_features` to 1.
            4. Define the final activation layer using `nn.Sigmoid().

        Arguments:
            num_codes: total number of diagnosis codes
        """
        self.bert_emb_size = bert_emb_size
        # self.embedding = nn.Embedding(num_embeddings=self.bert_emb_size, embedding_dim=128) 
        self.rnn = nn.GRU(input_size=self.bert_emb_size, hidden_size=128, batch_first=True)
        self.rev_rnn = nn.GRU(input_size=self.bert_emb_size, hidden_size=128, batch_first=True)
        self.fc = nn.Linear(in_features=256,out_features=1)
        self.sigmoid = nn.Sigmoid()
        # Contiguous memory.
        self.rnn.flatten_parameters()
        self.rev_rnn.flatten_parameters()
        
        
    def forward(self, x, masks, rev_x, rev_masks):
        """
        Arguments:
            x: the diagnosis sequence of shape (batch_size, #events(diag+proc+presc), embedding_dim)
            masks: the padding masks of shape (batch_size, #events(diag+proc+presc))

        Outputs:
            probs: probabilities of shape (batch_size)
        """
        
        batch_size = x.shape[0]
        
        # 2. Sum the embeddings for each diagnosis code up for a visit of a patient.
        x = demb_sum_embeddings_with_mask(x, masks)
        
        # 3. Pass the embeddings through the RNN layer;
        output, _ = self.rnn(x)
        # 4. Obtain the hidden state at the last visit.
        true_h_n = demb_get_last_visit(output, masks)
        
        """
        TODO:
            5. Do the step 1-4 again for the reverse order (rev_x), and concatenate the hidden
               states for both directions;
        """
        # xr = self.embedding(rev_x)
        xr = demb_sum_embeddings_with_mask(rev_x, rev_masks)
        routput, _ = self.rev_rnn(xr)
        true_h_n_rev = demb_get_last_visit(routput, rev_masks)
        
        # 6. Pass the hidden state through the linear and activation layers.
        logits =  self.fc(torch.cat([true_h_n, true_h_n_rev], 1))
        probs = self.sigmoid(logits)
        assert(probs.shape == (batch_size, 1))
        return probs.view(batch_size)
    