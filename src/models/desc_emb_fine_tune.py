import logging
# Typing includes.
from typing import Dict, List, Optional, Any, Tuple, Callable, Iterable

import torch
import torch.nn as nn
# from GPUtil import showUtilization as gpu_usage
def gpu_usage():
    pass
def tensor_bytes(t):
    return t.element_size() * t.nelement()
from torch.nn.utils.rnn import pack_padded_sequence, pad_packed_sequence
from transformers import BertTokenizer, BertModel, BertForMaskedLM, BertConfig, BertTokenizerFast
from transformers import TensorType
from transformers import AutoConfig, AutoModel

from transformers import AutoConfig, AutoModel

logger = logging.getLogger(__name__)
_BERT_EMBEDDING_SIZE=768
_BERT_TINY_EMBEDDING_SIZE=128


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
        self.use_gpu = True 
        if self.use_gpu:
            self.cuda = 'cuda' 
            self.device = torch.device('cuda:0')
        else:
            self.cuda = 'cpu'
            self.device = torch.device('cpu')
        self.bert_config = BertConfig()
        # self.bert_config = AutoConfig.from_pretrained("bert-base-uncased")
        # self.bert_model = BertModel(self.bert_config).from_pretrained('bert-base-uncased', config=self.bert_config)
        # self.bert_model = AutoModel.from_pretrained('bert-base-uncased')
        # with torch.no_grad():
        #     self.bert_model = BertModel(self.bert_config).from_pretrained('bert-base-uncased')
        # self.bert_model = BertModel(self.bert_config).from_pretrained('bert-base-uncased')
        self.bert_model = AutoModel.from_pretrained('prajjwal1/bert-tiny')
        self.bert_emb_size = self.bert_model.config.hidden_size 
        print(f'In DembFtEmbed constructor, before BERT.')
        gpu_usage()
        print(torch.cuda.memory_summary(device=None, abbreviated=False))
        if self.use_gpu:
            self.bert_model.to(self.cuda)
        self.bert_config = self.bert_model.config
        print(f'BERT model \n{self.bert_model}')
        print(f'BERT Config \n{self.bert_config}')
        # print(f'BERT named params \n{list(self.bert_model.named_parameters())}')
        print(f'In DembFtEmbed constructor, after BERT.')
        gpu_usage()
        print(torch.cuda.memory_summary(device=None, abbreviated=False))
        
        # for name, param in self.bert_model.named_parameters():
        #     if 'embeddings' in name: # embeddings layer
        #         param.requires_grad = False
        #     if 'encoder' in name and (('attention' in name or 'intermediate' in name) and (not 'dense' in name)):
        #         param.requires_grad = False
        #     if 'encoder': # encoder layer
        #         param.requires_grad = False
    
    
    def forward(self, x, rev_x, **kwargs):
        # if self.use_gpu:
        #     batch_enc_tensor.to(self.cuda)
     
        # TODO - I think we need to flatten the (B, S, word_max_len) -> (BxS, word_max_len)
        # then convert back to (B,S,768) after BERT.
        bsz, seq_len, word_max_len = x['input_ids'].shape
        # print(f'DembFtEmbed forward input x shape {x["input_ids"].shape}')
          
        # Copy to GPU. Flatten first two dimensions batch and sequence into one so BERT can run
        # (B, S, word_max_len) -> (BxS, word_max_len)
        gpu_x = {'output_attentions': False}
        gpu_rev_x = {'output_attentions': False}
        # print(f'In DembFtEmbed, before GPU copy.')
        gpu_usage()
        # print(torch.cuda.memory_summary(device=None, abbreviated=False))
        for k,v in x.items():
            if k in ['input_ids', 'attention_mask', 'token_type_ids']:
                gpu_x[k] = v.to(self.cuda).view(-1, word_max_len)
            else:
                gpu_x[k] = v
        assert(x['input_ids'].device == torch.device('cpu'))
        assert(gpu_x['input_ids'].device == self.device)
            
        # print(f'iids  bytes: {tensor_bytes(gpu_x["input_ids"])}\n'
        #       f'attn  bytes: {tensor_bytes(gpu_x["attention_mask"])}\n'
        #       f'ttids bytes: {tensor_bytes(gpu_x["token_type_ids"])}')
        # print(f'In DembFtEmbed, after GPU copy.')
        # print(torch.cuda.memory_summary(device=None, abbreviated=False))
        # Run BERT model forward pass on tokenized input.
        # print(f'gpu_x.keys {gpu_x.keys()}')
        # print(f'gpu_x type {type(gpu_x["input_ids"])}')
        # print(f'gpu_x shape {gpu_x["input_ids"].shape}')
        # with torch.no_grad():
        fwd_embeddings = self.bert_model(**gpu_x)
        fwd_embeddings = fwd_embeddings.last_hidden_state
        gpu_usage()
        assert(x['input_ids'].device == torch.device('cpu'))
        assert(fwd_embeddings.device == self.device)
        if self.use_gpu:
            fwd_embeddings = fwd_embeddings.to('cpu')
        
        for k,v in rev_x.items():
            if k in ['input_ids', 'attention_mask', 'token_type_ids']:
                gpu_rev_x[k] = v.to(self.cuda).view(-1, word_max_len)
            else:
                gpu_rev_x[k] = v
            
        assert(rev_x['token_type_ids'].device == torch.device('cpu'))
        assert(gpu_rev_x['token_type_ids'].device == self.device)
        
        # bert_args_fwd = {
        #     'input_ids': x['input_ids'].view(-1, word_max_len),
        #     'attention_mask': x['attention_masks'].view(-1, word_max_len),
        #     'token_type_ids': x['token_type_ids'].view(-1, word_max_len),
        #     'output_attentions': False,
        # }
        # bert_args_rev = {
        #     'input_ids': rev_x['input_ids'].view(-1, word_max_len),
        #     'attention_mask': rev_x['attention_masks'].view(-1, word_max_len),
        #     'token_type_ids': rev_x['token_type_ids'].view(-1, word_max_len),
        #     'output_attentions': False,
        # }
       
        # with torch.no_grad():
        rev_embeddings = self.bert_model(**gpu_rev_x)
        rev_embeddings = rev_embeddings.last_hidden_state
        gpu_usage()
        
        assert(rev_x['token_type_ids'].device == torch.device('cpu'))
        assert(rev_embeddings.device == self.device)
        
        if self.use_gpu:
            rev_embeddings = rev_embeddings.to('cpu')
        
        assert(x['input_ids'].device == torch.device('cpu'))
        assert(rev_x['token_type_ids'].device == torch.device('cpu'))
# https://stackoverflow.com/questions/61323621/how-to-understand-hidden-states-of-the-returns-in-bertmodelhuggingface-transfo
# https://datascience.stackexchange.com/questions/66207/what-is-purpose-of-the-cls-token-and-why-is-its-encoding-output-important
        # Take only the last hidden state embeddings from BERT.
        # We need to take index 0, because it's the CLS token prepended to our sentence.
        # The [CLS] represents the meaning of the whole sentence.
        # TODO - I think we need to un-flatten the (BxS, 768) -> (B, S, 768) here.
       
        # (BxS, max_word_len, 768) -> (BxS, 1, 768)
        fwd_embeddings = torch.squeeze(fwd_embeddings[:, 0, :], dim=1)
        rev_embeddings = torch.squeeze(rev_embeddings[:, 0, :], dim=1)

        # The 1st dimension is seq length. The second dimension is embedding length of each sentence.
        assert(fwd_embeddings.shape[-1] == self.bert_config.hidden_size)
        assert(rev_embeddings.shape[-1] == self.bert_config.hidden_size)
        assert(len(fwd_embeddings.shape) == 2)
        assert(fwd_embeddings.dtype == torch.float)
        
        # (BxS, 1, 768) -> (B, S, 768)
        assert(fwd_embeddings.view(bsz, -1, self.bert_emb_size).shape[1] == seq_len)
        assert(rev_embeddings.view(bsz, -1, self.bert_emb_size).shape[1] == seq_len)
        fwd_embeddings = fwd_embeddings.view(bsz, -1, self.bert_emb_size)
        rev_embeddings = rev_embeddings.view(bsz, -1, self.bert_emb_size)
        
        # print(f'DembFtEmbed forward output x shape {fwd_embeddings.shape}')
        # # Return the ((model_inputs), label) 
        # return (embeddings, sample['label'])
        
        return fwd_embeddings, rev_embeddings
    

class DembFtRNN(nn.Module):
    ''' Bidirectional RNN model accepting text string inputs.
    
    This model runs BERT with torch.grad enabled. This allows fine-tuning the model's
    weights.
    This takes significantly longer than DembRNN because the model is evaluated twice.
    1. The forward pass is run to generate embeddings for 
    '''
    
    def __init__(self, args, bert_emb_size:int=_BERT_TINY_EMBEDDING_SIZE):
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
        self.rnn = nn.GRU(input_size=self.bert_emb_size, hidden_size=128, batch_first=True)
        self.rev_rnn = nn.GRU(input_size=self.bert_emb_size, hidden_size=128, batch_first=True)
        self.fc = nn.Linear(in_features=256,out_features=1)
        self.sigmoid = nn.Sigmoid()
        # Contiguous memory.
        self.rnn.flatten_parameters()
        self.rev_rnn.flatten_parameters()
        
        
    def forward(self, x, masks, rev_x, rev_masks, **kwargs):
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
    