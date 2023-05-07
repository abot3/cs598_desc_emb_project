# General includes.
import itertools
import os
import re

# Typing includes.
from typing import Dict, List, Optional, Any, Tuple, Callable, Iterable

from pyhealth.data import Patient, Visit, Event

re_drug_prefix = re.compile('Event with eICU_DRUGNAME code{.*}from table medication')
re_bar = re.compile('\|')
ICD_9_LUT_ = {}
ICD_10_LUT_ = {}
fname = os.path.expanduser('~/sw/icd10cm-code descriptions- April 1 2023/icd10cm-codes- April 1 2023.txt')
with open(fname, 'r') as f:
    for l in f:
        code, desc = l.split(sep=' ', maxsplit=1)
        ICD_10_LUT_[code] = desc
        
def eicu_mortality_pred_task_demb(CODE_COUNT, eicubase, patient):
    """
    patient is a <pyhealth.data.Patient> object
    """
    samples = []
    visits = []
    kMaxListSize = 40
    
    global_mortality_label = 0
    # loop over all visits but the last one
    for i in range(len(patient)):

        # visit and next_visit are both <pyhealth.data.Visit> objects
        # there are no vists.attr_dict keys
        visit: Visit = patient[i]
        mortality_label = 0 if visit.discharge_status == 'Alive' else 1
        global_mortality_label |= mortality_label
        
    # loop over all visits but the last one
    for i, visit in enumerate(patient):
        # visit: Visit.
        
        # step 2: get code-based feature information
        conditions = visit.get_code_list(table="diagnosis")
        procedures = visit.get_code_list(table="treatment")
        # drugs = [x.code for x in visit.get_event_list(table="medication")]
        drugs_full = visit.get_event_list(table="medication")
        drugs_full = [d.code for d in drugs_full]
        # if i == 0: print([d.attr_dict for d in drugs_full])
        # if i == 0: print(conditions)
        # if i == 0: print(procedures)
        # if i == 0: print(drugs)
        # TODO(botelho3) - add this datasource back in once we have full MIMIC-III dataset.
        # labevents = visit.get_code_list(table="LABEVENTS")

        # step 3: exclusion criteria: visits without condition, procedure, or drug
        if len(conditions) * len(procedures) == 0 * len(drugs_full) == 0:
            # print(f'Excluded something 0 {len(conditions)}, {len(procedures)}, {len(drugs_full)}')
            # print(f'conditions {conditions}')
            # print(f'procedures {procedures}')
            # print(f'drugs_full {drugs_full}')
            continue
        if len(conditions) + len(procedures) + len(drugs_full) < 5:
            # Exclude stays with less than 5 procedures.
            continue
        
        # step 3.5: build text lists from the ICD codes
        # diag_lut = mimic3base.get_text_lut("D_ICD_DIAGNOSES")
        # proc_lut = mimic3base.get_text_lut("D_ICD_PROCEDURES")
        
        # if i == 0: print(d_diag)
        # if i == 0: print(d_proc)
        # Index 0 is shortname, index 1 is longname.
        # print([str(cond) + ' ' + str(d_diag.get(cond)) for cond in conditions])
        # print(d_proc.get(procedures[0]))
        # print(f'condition {conditions}')
        # print(f'proc {procedures}')
        # print(f'drugs {drugs_full}')
        # conditions_text = [diag_lut.get(cond,("", ""))[1] for cond in conditions]
        # procedures_text = [proc_lut.get(proc,("", ""))[1] for proc in procedures]
        conditions = filter(lambda x: True if x[0].isalpha() else False, conditions)
        conditions = [cond.replace('.', '') for cond in conditions]
        conditions_text = [ICD_10_LUT_.get(cond, '') for cond in conditions] 
        procedures_text = [re_bar.sub(' ', proc) for proc in procedures]
        drugs_text = [re_drug_prefix.sub('\1', str(d)) for d in drugs_full]
        # TODO(botelho3) - add the labevents data source back in once we have full MIMIC-III dataset.
        # labevents_text =
        
        # step 4: assemble the samples into a pyHealth Visit.
        visits.append(
            {
                "visit_id": visit.visit_id,
                "patient_id": patient.patient_id,
                # the following keys can be the "feature_keys" or "label_key" for initializing downstream ML model
                "conditions": conditions,
                "procedures": procedures,
                "conditions_text": conditions_text,
                "procedures_text": procedures_text,
                "drugs_text": drugs_text,
                # "labevents": labevents,
                # "labevents_text": labevents_text
                "label": global_mortality_label,
            }
        )
   
    
    # Return empty list, didn't meet exclusion criteria.
    num_visits = len(visits)
    if num_visits < 1:
        return [] 
    
   
    # pyHealth requires that all list fields in sample are equal size.
    def pad_field(field, visits, empty_val: Any):
        l = [empty_val for x in range(kMaxListSize)]
        data = [x[field] for x in visits]
        data = list(itertools.chain.from_iterable(data))
        slice_size = min(kMaxListSize, len(data))
        l[:slice_size] = data[:slice_size]
        return l, slice_size
    
    conditions, conditions_pad = pad_field("conditions", visits, '0')
    conditions_text, conditions_text_pad = pad_field("conditions_text", visits, '')
    procedures, procedures_pad = pad_field("procedures", visits, '0')
    procedures_text, procedures_text_pad = pad_field("procedures_text", visits, '')
    drugs_text, drugs_text_pad = pad_field("drugs_text", visits, '')
    sample = {
        "patient_id": patient.patient_id,
        # TODO(botelho3) Why does pyhealth require a visit id in the keys if we're combining vists?
        "visit_id": visits[0]["visit_id"],
        "num_visits": num_visits,
        # the following keys can be the "feature_keys" or "label_key" for initializing downstream ML model
        "conditions": conditions,
        "conditions_text": conditions_text,
        # "procedures": procedures,
        "procedures_text": procedures_text,
        "drugs_text": drugs_text,
        
        "conditions_pad": conditions_pad,
        "procedures_pad": procedures_pad,
        "conditions_text_pad": conditions_text_pad,
        "procedures_text_pad": procedures_text_pad,
        "drugs_text_pad": drugs_text_pad,
        # "labevents": labevents,
        # "labevents_text": labevents_text
        "label": global_mortality_label,
    }
   
    # For every condition in the sample (all visits). Record frequency.
    # Will be used to build code->index LUT.
    for code in sample['conditions']:
        CODE_COUNT[code] = CODE_COUNT.get(code, 0) + 1
       
    # if len(CODE_COUNT) in [10,11,12]:
    #     print(sample)
    samples.append(sample)
    return samples


def eicu_mortality_pred_task_cemb(CODE_COUNT, eicubase, patient):
    """
    patient is a <pyhealth.data.Patient> object
    each sample is a list of vists for 1 patient.
    """
   
    # TODO(botelho4) - stupid hack around the limitations of SampleDataset validator.
    kMaxVisits = 5
    if len(patient) < 1 or len(patient) > kMaxVisits:
        return []
        
    samples = [{
        "visit_id": patient[0].visit_id,
        "patient_id": patient.patient_id,
        # the following keys can be the "feature_keys" or "label_key" for initializing downstream ML model
        "num_visits": 0,
        "conditions": [[] for v in range(kMaxVisits)],
        "procedures": [[] for v in range(kMaxVisits)],
        "conditions_text": [[] for v in range(kMaxVisits)],
        "procedures_text": [[] for v in range(kMaxVisits)],
        "drugs": [[] for v in range(kMaxVisits)],
        "drugs_text": [[] for v in range(kMaxVisits)],
        # "labevents": labevents,
        # "labevents_text": labevents_text
        "label": 0,
    }]
    
    global_mortality_label = 0
    # loop over all visits but the last one
    for i in range(len(patient)):

        # visit and next_visit are both <pyhealth.data.Visit> objects
        # there are no vists.attr_dict keys
        visit: Visit = patient[i]
        # next_visit: Visit = patient[i + 1]
        # if i == 0: print(visit)

        # step 1: define the mortality_label
        # if next_visit.discharge_status not in [0, 1]:
        #     mortality_label = 0
        # else:
        #     mortality_label = int(next_visit.discharge_status)
        mortality_label = int(visit.discharge_status)
        global_mortality_label |= mortality_label

    # loop over all visits
    out_idx = 0
    for i, visit in enumerate(patient):
        # visit and next_visit are both <pyhealth.data.Visit> objects
        # there are no vists.attr_dict keys

        # step 2: get code-based feature information
        conditions = visit.get_code_list(table="diagnosis")
        procedures = visit.get_code_list(table="treatment")
        drugs = visit.get_code_list(table="medication")
        drugs_full = visit.get_event_list(table="medication")
        # if i == 0: print([d.attr_dict for d in drugs_full])
        # if i == 0: print(conditions)
        # if i == 0: print(procedures)
        # if i == 0: print(drugs)
        # TODO(botelho3) - add this datasource back in once we have full MIMIC-III dataset.
        # labevents = visit.get_code_list(table="LABEVENTS")

        # step 3: exclusion criteria: visits without condition, procedure, or drug
        if len(conditions) * len(procedures) * len(drugs_full) == 0:
            continue
        if len(conditions) + len(procedures) + len(drugs_full) < 5:
            # Exclude stays with less than 5 procedures.
            continue
        
        # step 3.5: build text lists from the ICD codes
        # diag_lut = mimic3base.get_text_lut("D_ICD_DIAGNOSES")
        # proc_lut = mimic3base.get_text_lut("D_ICD_PROCEDURES")
        # if i == 0: print(diag_lut)
        # if i == 0: print(proc_lut)
        # Index 0 is shortname, index 1 is longname.
        # print([str(cond) + ' ' + str(diag_lut.get(cond)) for cond in conditions])
        # print(proc_lut.get(procedures[0]))
        print(f'condition {conditions}')
        print(f'proc {procedures}')
        print(f'drugs {drugs_full}')
        conditions_text = [diag_lut.get(cond,("", ""))[1] for cond in conditions]
        procedures_text = [proc_lut.get(proc,("", ""))[1] for proc in procedures]
        drugs_text = [' '.join([d['dname'], d['dtype'], str(d['dose']), d['route'], str(d['duration'])])
                      for d in [d.attr_dict for d in drugs_full]]
        # TODO(botelho3) - add the labevents data source back in once we have full MIMIC-III dataset.
        # labevents_text =
        
        # step 4: assemble the samples
        # Each field is a list of visits.
        samples[0]['num_visits'] = samples[0]['num_visits'] + 1
        samples[0]['conditions'][out_idx]=conditions
        samples[0]['procedures'][out_idx]=procedures
        samples[0]['conditions_text'][out_idx]= conditions_text
        samples[0]['procedures_text'][out_idx] = procedures_text
        samples[0]['drugs'][out_idx] = drugs
        samples[0]['drugs_text'][out_idx] = drugs_text
        out_idx = out_idx + 1
    samples[0]['label'] = global_mortality_label
   
    # Record all unique codes and their frequency for LUT.
    for visit in samples[0]['conditions']:
        for code in visit:
            CODE_COUNT[code] = CODE_COUNT.get(code, 0) + 1
           
    # If none of the samples met the criteria return an empty list.
    if samples[0]['num_visits'] == 0:
        return []
        
    return samples