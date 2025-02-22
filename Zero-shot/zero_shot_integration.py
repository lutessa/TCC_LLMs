import csv
import re
from transformers import pipeline

'Hardware accelerator e.g. GPU is available in the environment, but no `device` argument is passed to the `Pipeline` object. Model will be on CPU.'

# input_file = r'C:\Users\Raissa\Documents\dev\TCC\OpenAI\Full_4o_mini_clean.txt'
# output_file = './results_Full_4o_mini_clean.csv'
# candidate_labels = ["sidewalks","tree-lined streets","porches", "fenced front yards", "attached garages", "cul-de-sacs", "hills", "private front entrances"]



input_file = r'C:\Users\Raissa\Documents\dev\TCC\OpenAI\var_Full_4o_mini_clean.txt'
output_file = './results_var_Full_4o_mini_clean.csv'
candidate_labels = ["Yes"]

classifier = pipeline("zero-shot-classification", model="MoritzLaurer/DeBERTa-v3-base-mnli-fever-anli")



def process_file(input_file, output_file):
    with open(input_file, 'r', encoding='utf-8') as infile, open(output_file, 'w', newline='', encoding='utf-8') as outfile:
        writer = csv.writer(outfile)
        
        # Escrever o cabeçalho do CSV
        writer.writerow(["id", "geoid"] + candidate_labels)
        
        for line in infile:
            match = re.match(r'^(\d+):(\d+):\s*"(.+)"$', line.strip())
            if not match:
                continue  
            
            record_id, geoid, description = match.groups()
            print(record_id)
            
            # Classificação
            result = classifier(description, candidate_labels, multi_label=True)
            labels_scores = dict(zip(result['labels'], result['scores']))
            
            # Criar linha com 1 se o score for maior que 0.9 e 0 caso contrário
            row = [record_id, geoid] + [1 if labels_scores.get(label, 0) >= 0.9 else 0 for label in candidate_labels]
            writer.writerow(row)

process_file(input_file, output_file)
