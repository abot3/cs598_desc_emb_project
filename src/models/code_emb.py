import logging
# Typing includes.
from typing import Dict, List, Optional, Any, Tuple, Callable, Iterable

import torch
import torch.nn as nn

logger = logging.getLogger(__name__)



def cemb_sum_embeddings_with_mask(x, masks):
    '''
    Inputs:
        x: the embeddings of diagnosis sequence of shape
           (batch_size, # visits, # diagnosis codes, embedding_dim)
        masks: the padding masks of shape (batch_size, # visits, # diagnosis codes)
    
    Outputs:
        sum_embeddings: the sum of embeddings of shape (batch_size, # visits, embedding_dim)
    '''
    tmp = torch.zeros(x.shape)
    m = torch.sum(masks, dim=2) > 0
    tmp[m, :, :] = 1
    tmp = x * tmp
    tmp = torch.sum(tmp, dim=2)
    
    a = x
    a[~masks] = 0
    tmp = torch.sum(a, dim=2)
    return tmp


def cemb_get_last_visit(hidden_states, masks):
    """
    TODO: obtain the hidden state for the last true visit (not padding visits)

    Arguments:
        hidden_states: the hidden states of each visit of shape (batch_size, # visits, embedding_dim=128)
        masks: the padding masks of shape (batch_size, # visits, # diagnosis codes)

    Outputs:
        last_hidden_state: the hidden state for the last true visit of shape (batch_size, embedding_dim=128)
        
    NOTE: DO NOT use for loop.
    
    HINT: First convert the mask to a vector of shape (batch_size,) containing the true visit length; 
          and then use this length vector as index to select the last visit.
    """
    m = torch.sum(masks, dim=2)
    m[m > 0] = 1
    m = torch.sum(m, dim=1)
    m[m > 0] = m[m > 0] - 1
   
    # (batch_size,)
    tmp1 = m
    # (batch_size,1,1)
    tmp = torch.reshape(m, (-1,1,1))
    # (batch_size,1,embedding_dim)
    tmp = tmp.expand(-1, -1, hidden_states.shape[2])
    # print(tmp)
    # print(tmp.shape)
    last_hidden_state = torch.gather(hidden_states, axis=1, index=tmp)
    # print(last_hidden_state.shape)
    last_hidden_state = torch.squeeze(last_hidden_state)
    
    # print(last_hidden_state[3, :])
    # print(torch.squeeze(hidden_states[3, tmp1[3], :]))
    assert(torch.equal(last_hidden_state[0, :], torch.squeeze(hidden_states[0, tmp1[0], :])))
    
    return last_hidden_state


class CembEmbed(nn.Module):
    def __init__(self, args):
        super().__init__()
        num_codes = args.embed_index_size
        emb_dim_size = args.pred_embed_size  #128
        self.embedding = nn.Embedding(num_embeddings=num_codes, embedding_dim=emb_dim_size)
    
    def forward(self, x, rev_x, **kwargs):
        return self.embedding(x), self.embedding(rev_x)

    
class CembRNN(nn.Module):
    ''' Bidirectional RNN model accepting a sequence of patient visits.
    '''
        
    def __init__(self, args):
        super().__init__()
        """
        TODO: 
            1. Define the embedding layer using `nn.Embedding`. Set `embDimSize` to 128.
            2. Define the RNN using `nn.GRU()`; Set `hidden_size` to 128. Set `batch_first` to True.
            2. Define the RNN for the reverse direction using `nn.GRU()`;
               Set `hidden_size` to 128. Set `batch_first` to True.
            3. Define the linear layers using `nn.Linear()`; Set `in_features` to 256, and `out_features` to 1.
            4. Define the final activation layer using `nn.Sigmoid().

        Arguments:
            num_codes: total number of diagnosis codes
        """
        # self.emb_dim_size = args.pred_embed_size
        # self.embedding = nn.Embedding(num_embeddings=num_codes, embedding_dim=128) 
        self.rnn = nn.GRU(input_size=128, hidden_size=128, batch_first=True)
        self.rev_rnn = nn.GRU(input_size=128, hidden_size=128, batch_first=True)
        self.fc = nn.Linear(in_features=256, out_features=1)
        self.sigmoid = nn.Sigmoid()
    
    def forward(self, x, masks, rev_x, rev_masks, **kwargs):
        """
        Arguments:
            x: the diagnosis sequence of shape (batch_size, # visits, # diagnosis codes)
            masks: the padding masks of shape (batch_size, # visits, # diagnosis codes)

        Outputs:
            probs: probabilities of shape (batch_size)
        """
        
        batch_size = x.shape[0]
        
        # 1. Pass the sequence through the embedding layer;
        # print(f"pre embedding: {x.shape}")
        # e = self.embedding(x)
        e = x
        # print(f"post embedding: {e.shape}")
        # 2. Sum the embeddings for each diagnosis code up for a visit of a patient.
        e = cemb_sum_embeddings_with_mask(e, masks)
        # print(f"post sum_embeddings_with_mask: {e.shape}")
        
        # 3. Pass the embeddings through the RNN layer;
        output, _ = self.rnn(e)
        # 4. Obtain the hidden state at the last visit.
        true_h_n = cemb_get_last_visit(output, masks)
        
        """
        TODO:
            5. Do the step 1-4 again for the reverse order (rev_x), and concatenate the hidden
               states for both directions;
        """
        # 1. Pass the sequence through the embedding layer;
        # rev_e = self.embedding(rev_x)
        rev_e = rev_x
        # 2. Sum the embeddings for each diagnosis code up for a visit of a patient.
        rev_e = cemb_sum_embeddings_with_mask(rev_e, rev_masks)
        
        # 3. Pass the embegginds through the RNN layer;
        output, _ = self.rnn(rev_e)
        # 4. Obtain the hidden state at the last visit.
        true_h_n_rev = cemb_get_last_visit(output, rev_masks)
        
        
        # 6. Pass the hidden state through the linear and activation layers.
        logits = self.fc(torch.cat([true_h_n, true_h_n_rev], 1))        
        probs = self.sigmoid(logits)
        return probs.view(batch_size)

