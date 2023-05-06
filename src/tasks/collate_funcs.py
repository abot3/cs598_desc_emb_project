# Typing includes.
from typing import Dict, List, Optional, Any, Tuple, Callable, Iterable

# Numerical includes.
import numpy as np
import torch

"""
BERT Collate
"""
def bert_per_patient_collate_function_new_trainer(data):
    """
    Note returns a dict instead of tuple.
    
    Collates a tensor of (batch_size, seq_len, bert_emb_len) i.e. (32, <events per patient>, 768)
    Need to pad the second dimension to max(<variable_per_patient>).
    
    TODO: Collate the the list of samples into batches. For each patient, you need to pad the diagnosis
        sequences to the sample shape (max # visits, max # diagnosis codes). The padding infomation
        is stored in `mask`.
    
    Arguments:
        data: a list of samples fetched from `CustomDataset`
        
    Outputs:
        x: a tensor of shape (batch_size, max #conditions, max embedding size) of type torch.float
        masks: a tensor of shape (batch_size, max #conditions, max embedding_size) of type torch.bool
        rev_x: same as x but in reversed time. This will be used in our RNN model for masking 
        rev_masks: same as mask but in reversed time. This will be used in our RNN model for masking
        y: a tensor of shape (batch_size) of type torch.float
        
    Note that you can obtains the list of diagnosis codes and the list of hf labels
        using: `sequences, labels = zip(*data)`
    """
    # print(f"bert_per_patient_collate_function data[0] {data[0]}")
    tmp, labels = zip(*data)
    sequences = tmp
    if not torch.is_tensor(tmp[0]):
        sequences = [torch.from_numpy(x) for x in tmp]
    
    # Quick stats on the amount of memory in each batch.
    sizes = [t.element_size() * t.nelement() for t in sequences]
    
    # Convert output labels for each sample in batch to tensor.
    # shape: (batch_size, 1)
    y = torch.tensor(labels, dtype=torch.float)
   
    num_patients = len(sequences)
    num_events = [patient.shape[0] for patient in sequences]
    embedding_length = [patient.shape[1] for patient in sequences]

    max_num_events = max(num_events)
    max_embedding_length = max(embedding_length)
    
    x = torch.zeros((num_patients, max_num_events, max_embedding_length), dtype=torch.float)
    rev_x = torch.zeros((num_patients, max_num_events, max_embedding_length), dtype=torch.float)
    # Mask dimensions are 1 less than inputs.
    masks = torch.zeros((num_patients, max_num_events), dtype=torch.bool)
    rev_masks = torch.zeros((num_patients, max_num_events), dtype=torch.bool)
    for i_patient, patient in enumerate(sequences):
        # Patient (#events, 768)
        j_visits = patient.shape[0]
        # for j_visit, visit in enumerate(patient):
        """
        TODO: update `x`, `rev_x`, `masks`, and `rev_masks`
        """
        # l = len(visit)
        x[i_patient, :j_visits, :] = patient[:, :].unsqueeze(0)
        # The tensor is (seq_length, emb_size). Leave embeddings,
        # flip temporal order of code/event sequence.
        rev_x[i_patient, :j_visits, :] = torch.flip(patient, dims=[0]).unsqueeze(0)
        masks[i_patient, :j_visits] = 1
        rev_masks[i_patient, :j_visits] = 1
      
        # TODO(botelho3) - comment this out to reduce spew.
        # if i_patient == 0:
        #     print(f"------ p: {i_patient} ------")
        #     print(x[i_patient, :, :25])
        #     print(rev_x[i_patient, :, :25])
        #     print(masks[i_patient, :, :25])
        #     print(rev_masks[i_patient, :, :25])
    return {
        'x': x,
        'masks': masks,
        'rev_x': rev_x,
        'rev_masks': rev_masks,
        'y': y,
    }


'''
BERT Fine-Tune Collate
'''
def bert_fine_tune_collate(data):
    """
    Collates a tensor of (batch_size, seq_len, bert_emb_len) i.e. (32, <events per patient>, 768)
    Need to pad the second dimension to max(<variable_per_patient>).
    
    TODO: Collate the the list of samples into batches. For each patient, you need to pad the diagnosis
        sequences to the sample shape (max # visits, max # diagnosis codes). The padding infomation
        is stored in `mask`.
    
    Arguments:
        data: list of ({'inputs_ids':tensor, 'attention_mask':tensor, 'token_type_ids':tensor}, y)
        
    Outputs:
        x: a tensor of shape (batch_size, max #conditions, max embedding size) of type torch.float
        masks: a tensor of shape (batch_size, max #conditions, max embedding_size) of type torch.bool
        rev_x: same as x but in reversed time. This will be used in our RNN model for masking 
        rev_masks: same as mask but in reversed time. This will be used in our RNN model for masking
        y: a tensor of shape (batch_size) of type torch.float
        
    Note that you can obtains the list of diagnosis codes and the list of hf labels
        using: `sequences, labels = zip(*data)`
    """
    kMaxEvents = 25
    kMaxWordLength = 20
    sequences, labels = zip(*data)
    
    # Convert output labels for each sample in batch to tensor.
    # shape: (batch_size, 1)
    y = torch.tensor(labels, dtype=torch.float)
   
    num_patients = len(sequences)
    num_events = [patient['input_ids'].shape[0] for patient in sequences]
    embedding_length = [patient['input_ids'].shape[1] for patient in sequences]
    # if len(sequences) > 1:
    #     assert(sequences[0]['input_ids'].shape[1] == sequences[1]['input_ids'].shape[1])

    max_num_events = min(kMaxEvents, max(num_events))
    max_embedding_length = min(kMaxWordLength, max(embedding_length))
    
    input_ids = torch.zeros((num_patients, max_num_events, max_embedding_length), dtype=torch.long)
    rev_input_ids = torch.zeros((num_patients, max_num_events, max_embedding_length), dtype=torch.long)
    attention_masks = torch.zeros((num_patients, max_num_events, max_embedding_length), dtype=torch.long)
    rev_attention_masks = torch.zeros((num_patients, max_num_events, max_embedding_length), dtype=torch.long)
    token_type_ids = torch.zeros((num_patients, max_num_events, max_embedding_length), dtype=torch.long)
    rev_token_type_ids = torch.zeros((num_patients, max_num_events, max_embedding_length), dtype=torch.long)
    masks = torch.zeros((num_patients, max_num_events), dtype=torch.bool)
    rev_masks = torch.zeros((num_patients, max_num_events), dtype=torch.bool)
    for i_patient, patient in enumerate(sequences):
        # Patient 3 tensors (#events, sentence_length+pad)
        j_visits = min(patient['input_ids'].shape[0], max_num_events)
        w_len = min(patient['input_ids'].shape[1], max_embedding_length)
        iids = patient['input_ids']
        ams = patient['attention_masks']
        ttids = patient['token_type_ids']
        assert(iids.shape == ams.shape)
        assert(iids.shape == ttids.shape)
        """
        TODO: update `x`, `rev_x`, `masks`, and `rev_masks`
        """
        input_ids[i_patient, :j_visits, :w_len] = iids[:j_visits, :w_len].unsqueeze(0)
        attention_masks[i_patient, :j_visits, :w_len] = ams[:j_visits, :w_len].unsqueeze(0)
        token_type_ids[i_patient, :j_visits, :w_len] = ttids[:j_visits, :w_len].unsqueeze(0)
        # The tensor is (seq_length, sentence_length+pad). Leave embeddings,
        # flip temporal order of code/event sequence.
        rev_input_ids[i_patient, :j_visits, :iids.shape[-1]] = torch.flip(iids, dims=[0]).unsqueeze(0)[:,:j_visits, :w_len]
        rev_attention_masks[i_patient, :j_visits, :ams.shape[-1]] = torch.flip(ams, dims=[0]).unsqueeze(0)[:,:j_visits, :w_len]
        rev_token_type_ids[i_patient, :j_visits, :ttids.shape[-1]] = torch.flip(ttids, dims=[0]).unsqueeze(0)[:,:j_visits, :w_len]
        masks[i_patient, :j_visits] = 1
        rev_masks[i_patient, :j_visits] = 1
        
    # print(f'Bert Fine Tune Collate\n'
    #       f'seq {sequences[0]["input_ids"].shape}, '
    #       f'seq {sequences[-1]["input_ids"].shape}\n'
    #       f'input_ids {input_ids.shape}, '
    #       f'rev_attention_masks {rev_attention_masks.shape}')
    x = {
        'input_ids': input_ids,
        'attention_mask': attention_masks,
        'token_type_ids': token_type_ids,
    }
    rev_x = {
        'input_ids': rev_input_ids,
        'attention_mask': rev_attention_masks,
        'token_type_ids': rev_token_type_ids,
    }
    return {
        'x': x,
        'rev_x': rev_x,
        'masks': masks,
        'rev_masks': rev_masks,
        'y': y,
    }


'''
CodeEMb Collate
'''
def code_emb_per_visit_collate_function(code2idx: Dict[str, int], data: List[Any]) -> Tuple[Any]:
    """
    TODO: Collate the the list of samples into batches. For each patient, you need to pad the diagnosis
    sequences to the sample shape (max # visits, max # diagnosis codes). The padding infomation
    is stored in `mask`.
                                   
    
    Arguments:
        data: a list of samples fetched from `CustomDataset`
        
    Outputs:
    x: a tensor of shape (# patients, max # visits, max # diagnosis codes) of type torch.long
    masks: a tensor of shape (# patients, max # visits, max # diagnosis codes) of type torch.bool
    rev_x: same as x but in reversed time. This will be used in our RNN model for masking
    rev_masks: same as mask but in reversed time. This will be used in our RNN model for masking
    y: a tensor of shape (# patiens) of type torch.float
        
    Note that you can obtains the list of diagnosis codes and the list of hf labels\n",
    using: `sequences, labels = zip(*data)`\n",
    """
    
    samples = data
    
    y = torch.tensor([s['label'] for s in samples], dtype=torch.float)
    
    num_patients = len(samples)
    num_visits = [patient['num_visits'] for patient in samples]
    num_codes = []
    for patient_idx, _ in enumerate(num_visits):
        num_codes.extend([len(visit) for visit in samples[patient_idx]['conditions']])
        
    # print(f'num samples: {len(samples)}')
    # print(f'num visits: {num_visits}')
    # print(f'num codes: {num_codes}')
    # print(f'code_emb_per_patient_collate_function data[0]:\n{data[0]}')
 
    max_num_visits = max(num_visits)
    max_num_codes = max(num_codes)
    # max_num_codes = len(MORTALITY_PER_VISIT_ICD_9_CODE2IDX_)
    assert(max_num_codes > 0)
    
    x = torch.zeros((num_patients, max_num_visits, max_num_codes), dtype=torch.long)
    rev_x = torch.zeros((num_patients, max_num_visits, max_num_codes), dtype=torch.long)
    masks = torch.zeros((num_patients, max_num_visits, max_num_codes), dtype=torch.bool)
    rev_masks = torch.zeros((num_patients, max_num_visits, max_num_codes), dtype=torch.bool)
    for i_patient, patient in enumerate(samples):
        nvisits = patient['num_visits']
        for j_visit in range(nvisits):
            """
            TODO: update `x`, `rev_x`, `masks`, and `rev_masks`
            """
            codes = patient['conditions'][j_visit] 
            indices = [code2idx[code] for code in codes]
            l = len(codes)
            if len(indices) < 1:
                print(f'No code indices after lookup')
                print(f'codes {codes}'+'\n'+f'indices{indices}')
                print(f'patient\n{patient}')
                assert(len(indices) >= 1)
            # for idx in indices:
            x[i_patient, j_visit, 0:l] = torch.tensor(indices)
            rev_x[i_patient, nvisits-1-j_visit, 0:l] = torch.tensor(indices)
            masks[i_patient, j_visit, 0:l] = 1
            rev_masks[i_patient, nvisits-1-j_visit, 0:l] = 1
        # if i_patient == 0:
        #     print(f"------ code_emb_per_visit_collate_function p: {i_patient} ------")
        #     print(x[i_patient, :, ])
        #     print(rev_x[i_patient, :, ])
        #     print(masks[i_patient, :, ])
        #     print(rev_masks[i_patient, :, ])
    
    return x, masks, rev_x, rev_masks, y