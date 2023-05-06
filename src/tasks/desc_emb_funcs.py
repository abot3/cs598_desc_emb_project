# General includes.
import os

# Typing includes.
from typing import Dict, List, Optional, Any, Tuple, Callable, Iterable

from pyhealth.data import Patient, Visit, Event

# The original authors tackled 5 tasks
#   1. readmission
#   2. mortality
#   3. an ICU stay exceeding three days
#   4. an ICU stay exceeding seven days
#   5. diagnosis prediction

#{code: idx for idx, code in enumerate(READMISSION_PER_PATIENT_ICD_9_CODE_COUNT_.keys())}
def readmission_pred_task_demb(CODE_COUNT, patient, time_window=3):
    """
    patient is a <pyhealth.data.Patient> object
    """
    samples = []
    visits = []
    kMaxListSize = 40
   
    # Length 1 patients by defn are not readmitted.
    if len(patient) < 1:
        return samples

    # we will drop the last visit
    global_readmission_label = 0
    global_readmission_label = 1 if len(patient) > 1 else 0
#     for i in range(len(patient) - 1):
#         visit: Visit = patient[i]
#         next_visit: Visit = patient[i + 1]

#         # step 1: define the readmission label 
#         # get time difference between current visit and next visit
#         time_diff = (next_visit.encounter_time - visit.encounter_time).days
#         readmission_label = 1 if time_diff < time_window else 0
#         global_readmission_label |= readmission_label  
       
    for i, visit in enumerate(patient):
        # visit: Visit = patient[i]
        # step 2: get code-based feature information
        conditions = visit.get_code_list(table="DIAGNOSES_ICD")
        procedures = visit.get_code_list(table="PROCEDURES_ICD")
        # drugs = visit.get_code_list(table="PRESCRIPTIONS")
        drugs = [x.code for x in visit.get_event_list(table="PRESCRIPTIONS")]
        drugs_full = visit.get_event_list(table="PRESCRIPTIONS")
        
        # step 3: exclusion criteria: visits without condition, procedure, or drug
        if len(conditions) * len(procedures) == 0 * len(drugs_full) == 0:
            continue
        if len(conditions) + len(procedures) + len(drugs_full) < 5:
            # Exclude stays with less than 5 procedures.
            continue
        
        # step 3.5: build text lists from the ICD codes
        d_diag = mimic3base.get_text_lut("D_ICD_DIAGNOSES")
        d_proc = mimic3base.get_text_lut("D_ICD_PROCEDURES")
        conditions_text = [d_diag.get(cond,("", ""))[1] for cond in conditions]
        procedures_text = [d_proc.get(proc,("", ""))[1] for proc in procedures]
        drugs_text = [' '.join([d['dname'], d['dtype'], d['dose'], d['route'], str(d['duration'])])
                      for d in [d.attr_dict for d in drugs_full]]
            
        # step 4: assemble the samples
        visits.append(
            {
                "visit_id": visit.visit_id,
                "patient_id": patient.patient_id,
                # the following keys can be the "feature_keys" or "label_key" for initializing downstream ML model
                "conditions": conditions,
                "procedures": procedures,
                "conditions_text": conditions_text,
                "procedures_text": procedures_text,
                "drugs": drugs,
                "drugs_text": drugs_text,
                # "labevents": labevents,
                # "labevents_text": labevents_text
                "label": global_readmission_label,
            }
        )
        

    # Return empty list, didn't meet exclusion criteria.
    num_visits = len(visits)
    if num_visits < 1:
        return samples
    
    def pad_field(field, visits, empty_val: Any):
        l = [empty_val for x in range(kMaxListSize)]
        data = [x[field] for x in visits]
        data = list(itertools.chain.from_iterable(data))
        slice_size = min(kMaxListSize, len(data)) - 1
        l[:slice_size] = data[:slice_size]
        return l, slice_size 
    
    conditions, conditions_pad = pad_field("conditions", visits, '0')
    conditions_text, conditions_text_pad = pad_field("conditions_text", visits, '')
    procedures, procedures_pad = pad_field("procedures", visits, '0')
    procedures_text, procedures_text_pad = pad_field("procedures_text", visits, '')
    drugs, drugs_pad = pad_field("drugs", visits, '0')
    drugs_text, drugs_text_pad = pad_field("drugs_text", visits, '')
    assert(drugs_pad == drugs_text_pad)
    sample = {
        "patient_id": patient.patient_id,
        # TODO(botelho3) Why does pyhealth require a visit id in the keys if we're combining vists?
        "visit_id": visits[0]["visit_id"],
        "num_visits": num_visits,
        # the following keys can be the "feature_keys" or "label_key" for initializing downstream ML model
        "conditions": conditions,
        "conditions_text": conditions_text,
        "procedures": procedures,
        "procedures_text": procedures_text,
        "drugs": drugs,
        "drugs_text": drugs_text,
        
        "conditions_pad": conditions_pad,
        "procedures_pad": procedures_pad,
        "conditions_text_pad": conditions_text_pad,
        "procedures_text_pad": procedures_text_pad,
        "drugs_pad": drugs_pad,
        "drugs_text_pad": drugs_text_pad,
        # "labevents": labevents,
        # "labevents_text": labevents_text
        "label": global_readmission_label,
    }
    # sample = {
    #     "patient_id": patient.patient_id,
    #     # TODO(botelho3) Why does pyhealth require a visit id in the keys if we're combining vists?
    #     "visit_id": visits[0]["visit_id"],
    #     "num_visits": num_visits,
    #     # the following keys can be the "feature_keys" or "label_key" for initializing downstream ML model
    #     "conditions": [[x["conditions"] for x in visits]],
    #     "procedures": [[x["procedures"] for x in visits]],
    #     "conditions_text": [[x["conditions_text"] for x in visits]],
    #     "procedures_text": [[x["procedures_text"] for x in visits]],
    #     "drugs": [[[x["drugs"] for x in visits]]],
    #     "drugs_text": [[x["drugs_text"] for x in visits]],
    #     # "labevents": labevents,
    #     # "labevents_text": labevents_text
    #     "label": global_readmission_label,
    # }
    for code in sample['conditions']:
        CODE_COUNT[code] = CODE_COUNT.get(code, 0) + 1

    samples.extend(
        list(deepcopy(sample) for s in range(SAMPLE_MULTIPLIER_))
    )
    return samples


def mortality_pred_task_demb(CODE_COUNT, patient):
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
        # next_visit: Visit = patient[i + 1]
        # if i == 0: print(visit)

        # step 1: define the mortality_label
        # if next_visit.discharge_status not in [0, 1]:
        #     mortality_label = 0
        # else:
        #     mortality_label = int(next_visit.discharge_status)
        mortality_label = int(visit.discharge_status)
        global_mortality_label |= mortality_label
        
    # loop over all visits but the last one
    for i, visit in enumerate(patient):
        # visit: Visit.
        
        # step 2: get code-based feature information
        conditions = visit.get_code_list(table="DIAGNOSES_ICD")
        procedures = visit.get_code_list(table="PROCEDURES_ICD")
        # drugs = visit.get_code_list(table="PRESCRIPTIONS")
        drugs = [x.code for x in visit.get_event_list(table="PRESCRIPTIONS")]
        drugs_full = visit.get_event_list(table="PRESCRIPTIONS")
        # if i == 0: print([d.attr_dict for d in drugs_full])
        # if i == 0: print(conditions)
        # if i == 0: print(procedures)
        # if i == 0: print(drugs)
        # TODO(botelho3) - add this datasource back in once we have full MIMIC-III dataset.
        # labevents = visit.get_code_list(table="LABEVENTS")

        # step 3: exclusion criteria: visits without condition, procedure, or drug
        if len(conditions) * len(procedures) == 0 * len(drugs_full) == 0:
            continue
        if len(conditions) + len(procedures) + len(drugs_full) < 5:
            # Exclude stays with less than 5 procedures.
            continue
        
        # step 3.5: build text lists from the ICD codes
        diag_lut = mimic3base.get_text_lut("D_ICD_DIAGNOSES")
        proc_lut = mimic3base.get_text_lut("D_ICD_PROCEDURES")
        # if i == 0: print(d_diag)
        # if i == 0: print(d_proc)
        # Index 0 is shortname, index 1 is longname.
        # print([str(cond) + ' ' + str(d_diag.get(cond)) for cond in conditions])
        # print(d_proc.get(procedures[0]))
        conditions_text = [diag_lut.get(cond,("", ""))[1] for cond in conditions]
        procedures_text = [proc_lut.get(proc,("", ""))[1] for proc in procedures]
        drugs_text = [' '.join([d['dname'], d['dtype'], d['dose'], d['route'], str(d['duration'])])
                      for d in [d.attr_dict for d in drugs_full]]
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
                "drugs": drugs,
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
    drugs, drugs_pad = pad_field("drugs", visits, '0')
    drugs_text, drugs_text_pad = pad_field("drugs_text", visits, '')
    assert(drugs_pad == drugs_text_pad)
    sample = {
        "patient_id": patient.patient_id,
        # TODO(botelho3) Why does pyhealth require a visit id in the keys if we're combining vists?
        "visit_id": visits[0]["visit_id"],
        "num_visits": num_visits,
        # the following keys can be the "feature_keys" or "label_key" for initializing downstream ML model
        "conditions": conditions,
        "conditions_text": conditions_text,
        "procedures": procedures,
        "procedures_text": procedures_text,
        "drugs": drugs,
        "drugs_text": drugs_text,
        
        "conditions_pad": conditions_pad,
        "procedures_pad": procedures_pad,
        "conditions_text_pad": conditions_text_pad,
        "procedures_text_pad": procedures_text_pad,
        "drugs_pad": drugs_pad,
        "drugs_text_pad": drugs_text_pad,
        # "labevents": labevents,
        # "labevents_text": labevents_text
        "label": global_mortality_label,
    }
    # sample = {
    #     "patient_id": patient.patient_id,
    #     # TODO(botelho3) Why does pyhealth require a visit id in the keys if we're combining vists?
    #     "visit_id": visits[0]["visit_id"],
    #     "num_visits": num_visits,
    #     # the following keys can be the "feature_keys" or "label_key" for initializing downstream ML model
    #     "conditions": [[x["conditions"] for x in visits]],
    #     "procedures": [[x["procedures"] for x in visits]],
    #     "conditions_text": [[x["conditions_text"] for x in visits]],
    #     "procedures_text": [[x["procedures_text"] for x in visits]],
    #     "drugs": [[[x["drugs"] for x in visits]]],
    #     "drugs_text": [[x["drugs_text"] for x in visits]],
    #     # "labevents": labevents,
    #     # "labevents_text": labevents_text
    #     "label": global_mortality_label,
    # }
   
   
    # For every condition in the sample (all visits). Record frequency.
    # Will be used to build code->index LUT.
    for code in sample['conditions']:
        CODE_COUNT[code] = CODE_COUNT.get(code, 0) + 1

    if SAMPLE_MULTIPLIER_:
        samples.extend(
            list(deepcopy(sample) for s in range(SAMPLE_MULTIPLIER_))
        )
        
    return samples