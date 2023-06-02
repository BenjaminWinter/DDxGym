from typing import List, Tuple, Dict
import glob
import xmltodict
import os
import csv
import json
from .constants import PROCEDURE_TYPE, HAZARD_LEVEL, PROBABILITY, ONSET
from .medical import Procedure, Disease, Symptom 


# TODO: symptoms with different procedures depending on disease vs aggregating
# TODO: diseases with not properly split symptoms (e.g. Womb uterus cancer)
def process_csv(path: str, max_diseases: int = None) -> Dict:
    
    procedures = {}
    symptoms = {}
    diseases = {}
    necessary_procedures = {}
    hazard_level = ["Low", "Mid", "High"] # labeled differently than HAZARD_LEVEL
    cleanup_dicts = {
        "symptoms": json.load(open(os.path.join(path, "symptoms_cleanup.json"), "r")),
        "diagnostics": json.load(open(os.path.join(path, "diagnostics_cleanup.json"), "r")),
        "treatments": json.load(open(os.path.join(path, "treatments_cleanup.json"), "r"))
    }
    
    with open(os.path.join(path, "data.csv"), "r", newline='') as csvfile:
        reader = csv.reader(csvfile)
        next(reader)
        
        for row in reader:
            disease, _, _, symptom, severity, probability, onset, diagnostics, treatments, is_main, *_ = row
            disease = disease.lower()
            
            if symptom == "": #for diseases where the disease itself is the main symptom
                symptom = disease + " discovered"
                is_main = "yes"
                probability = "always"
                onset = "initial"
            
            if symptom in cleanup_dicts["symptoms"]:
                symptom = cleanup_dicts["symptoms"][symptom]
            symptom = symptom.lower()
            
            severity    = severity or "Low"
            probability = probability or "always"
            onset       =  onset or "initial"
            
            is_main         = is_main == "yes"  
            temp_procedures = {"Examination": [], "Treatment": []}
            
            
            
            for ptype, ps in {"Examination": diagnostics, "Treatment": treatments}.items():
                
                if len(ps) == 0: #some symptoms dont have treatments
                    ps_list = []
                elif ps[0] == '{': #if consists of multiple
                    ps_list = json.loads(ps.replace("\'", "\""))['text']
                else:
                    ps_list = [ps]

                for p in ps_list:
                    p = p.rstrip()
                    if p in cleanup_dicts["diagnostics"]:
                        p = cleanup_dicts["diagnostics"][p]
                    elif p in cleanup_dicts["treatments"]:
                        p = cleanup_dicts["treatments"][p]
                    p = p.lower()

                    if not p in procedures.keys():
                        procedures[p] = Procedure(p, p, "", PROCEDURE_TYPE.index(ptype))
                    temp_procedures[ptype].append(procedures[p])


            if not symptom in symptoms:
                symptoms[symptom] = Symptom(symptom,
                                            symptom, 
                                            "", 
                                            hazard_level.index(severity), 
                                            temp_procedures["Examination"].copy(), 
                                            temp_procedures["Treatment"].copy(), 
                                            is_main)
            else:
                symptoms[symptom].add_examinations(temp_procedures["Examination"])
                symptoms[symptom].add_treatments(temp_procedures["Treatment"])
            
            dsymptom = {"symptom": symptoms[symptom], 
                        "probability": PROBABILITY[probability], 
                        "firstDay": ONSET[onset.lower()]}
            if not disease in diseases:
                if (max_diseases == None or len(diseases) < max_diseases):
                    diseases[disease] = Disease(disease, 
                                                disease,
                                                "",  
                                                3, 
                                                {dsymptom["symptom"].id: dsymptom}, 
                                                dsymptom["symptom"] if is_main else None, 
                                                temp_procedures["Examination"].copy(), 
                                                temp_procedures["Treatment"].copy())
            else:
                diseases[disease].add_symptom(dsymptom)

        diseases  = {dname: disease for dname, disease in diseases.items() if len(disease.symptoms) > 1}
        
        if max_diseases is not None:
            necessary_procedures = {}
            for disease in diseases.values():
                for p in disease.treatments + disease.examinations:
                    necessary_procedures[p.id] = p
        
    #    with open(os.path.join(path, "actions_csv.txt"), "w") as f:
    #        f.write("\n".join([p.id for p in procedures.values()]))
    return diseases, symptoms, necessary_procedures if max_diseases is not None else procedures