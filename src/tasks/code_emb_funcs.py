# General includes.
import os
import itertools

# Typing includes.
from typing import Dict, List, Optional, Any, Tuple, Callable, Iterable

from pyhealth.data import Patient, Visit, Event

def readmission_pred_task_cemb(CODE_COUNT, mimic3base, patient):
    """
    patient is a <pyhealth.data.Patient> object
    each sample is a list of vists for 1 patient.
    """
   
    # TODO(botelho3) - stupid hack around the limitations of SampleDataset validator.
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

    # They were admitted > 1 time.
    global_readmission_label = 1 if len(patient) > 1 else 0

    # loop over all visits
    out_idx = 0
    for i, visit in enumerate(patient):
        # visit and next_visit are both <pyhealth.data.Visit> objects
        # there are no vists.attr_dict keys

        # step 2: get code-based feature information
        conditions = visit.get_code_list(table="DIAGNOSES_ICD")
        procedures = visit.get_code_list(table="PROCEDURES_ICD")
        drugs = visit.get_code_list(table="PRESCRIPTIONS")
        drugs_full = visit.get_event_list(table="PRESCRIPTIONS")
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
        diag_lut = mimic3base.get_text_lut("D_ICD_DIAGNOSES")
        proc_lut = mimic3base.get_text_lut("D_ICD_PROCEDURES")
        # if i == 0: print(diag_lut)
        # if i == 0: print(proc_lut)
        # Index 0 is shortname, index 1 is longname.
        # print([str(cond) + ' ' + str(diag_lut.get(cond)) for cond in conditions])
        # print(proc_lut.get(procedures[0]))
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
    samples[0]['label'] = global_readmission_label
   
    # Record all unique codes and their frequency for LUT.
    for visit in samples[0]['conditions']:
        for code in visit:
            CODE_COUNT[code] = CODE_COUNT.get(code, 0) + 1
           
    # If none of the samples met the criteria return an empty list.
    if samples[0]['num_visits'] == 0:
        return []
        
    return samples
    
    
def mortality_pred_task_cemb(CODE_COUNT, mimic3base, patient):
    """
    patient is a <pyhealth.data.Patient> object
    each sample is a list of vists for 1 patient.
    """
   
    # TODO(botelho3) - stupid hack around the limitations of SampleDataset validator.
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
        conditions = visit.get_code_list(table="DIAGNOSES_ICD")
        procedures = visit.get_code_list(table="PROCEDURES_ICD")
        drugs = visit.get_code_list(table="PRESCRIPTIONS")
        drugs_full = visit.get_event_list(table="PRESCRIPTIONS")
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
        diag_lut = mimic3base.get_text_lut("D_ICD_DIAGNOSES")
        proc_lut = mimic3base.get_text_lut("D_ICD_PROCEDURES")
        # if i == 0: print(diag_lut)
        # if i == 0: print(proc_lut)
        # Index 0 is shortname, index 1 is longname.
        # print([str(cond) + ' ' + str(diag_lut.get(cond)) for cond in conditions])
        # print(proc_lut.get(procedures[0]))
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