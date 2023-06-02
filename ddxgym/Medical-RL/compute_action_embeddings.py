import torch
import argparse
import glob
import os
import xmltodict
from transformers import AutoModel, AutoTokenizer

def prepare_actions(args):
    localizations = {}
    procedures = {}

    localizations_xml = glob.glob(os.path.join(args.data_path, "Localization/*"))
    for f in localizations_xml:
        parse = xmltodict.parse(open(f, "rb"))
        for loc in parse["Database"]["GameDBStringTable"]["LocalizedStrings"]["GameDBLocalizedString"]:
            localizations[loc["LocID"]] = loc["Text"]


    exam_xml = os.path.join(args.data_path, "Procedures/Examinations.xml")
    treatment_xml = [
        os.path.join(args.data_path, "Procedures/Surgery.xml"),
        os.path.join(args.data_path, "Procedures/Treatments.xml"),
        os.path.join(args.data_path, "Procedures/TreatmentsHospitalization.xml"),
    ]

    exams = xmltodict.parse(open(exam_xml, "rb"))
    for e in exams["Database"]["GameDBExamination"]:
        procedures[e["@ID"]] = f"Examination: {localizations[e['@ID']]}. {localizations[e['AbbreviationLocID']]}"
    
    for xml in treatment_xml:
        treatments = xmltodict.parse(open(xml, "rb"))
        for t in treatments["Database"]["GameDBTreatment"]:
            procedures[t["@ID"]] = f"Treatment: {localizations[t['@ID']]}. {localizations[t['AbbreviationLocID']]}"

    return procedures
    

if __name__ == "__main__":
    parser = argparse.ArgumentParser("Compute Action Embeddings for Medical Reinforcement Learning")

    parser.add_argument("--model_name_or_path", type=str, default="bert-base-uncased", help="Transformer Encoder to use.")
    parser.add_argument("--data_path", type=str, default="../Medical-Gym/data/project_hospital", help="Path to Environment data")
    parser.add_argument("--output_path", type=str, default="../Medical-Gym/data/project_hospital/action_embeddings/", help="Path to Environment data")

    args = parser.parse_known_args()[0]  


    actions = prepare_actions(args)

    tokenizer = AutoTokenizer.from_pretrained(args.model_name_or_path)

    action_list = open(os.path.join(args.data_path, "actions.txt")).read().splitlines()

    ordered_actions = [ actions[key] for key in action_list ]

    encoding = tokenizer(ordered_actions, padding="longest", return_tensors="pt")
    
    model = AutoModel.from_pretrained(args.model_name_or_path)
    
    with torch.no_grad():
        embeddings = model(encoding["input_ids"], attention_mask=encoding["attention_mask"])[1]

    print(embeddings.size())  
    save_path = os.path.join(args.output_path, args.model_name_or_path)
    os.makedirs(save_path, exist_ok=True)
    torch.save(embeddings, os.path.join(save_path, "embeddings.pt"))
