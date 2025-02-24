from openai import OpenAI
import os

import base64
import requests
import time
import json
import argparse
import csv
import ast
# PROMPT_1 = "Describe this image."
# PROMPT_2 = "Answer, yes or no. The region favors conversations between neighbors? "
# PROMPT_3 = ""
# PROMPT_4 = "Describe the atmosphere of this area."
# PROMPT_5 = "You will be provided with a satellite image. List one or more built environment features in the image only if the feature is in this list: [1.Multi-lane highway or freeway 2.Local or residential road 3.Railroad track 4.Other transportation infrastructure 5.Physical barrier (includes guardrail, bollard, fencing, wall, or other physical barrier) 6.Street sign indicating no passage 7.Vegetation 8.Residential buildings and property 9.Community buildings and property 10.Industrial buildings and property 11.Other buildings and property 12.Recreational areas 13. Parking facility 14. Cemeteries 15. Industrial area that is not a building 16. Undeveloped land 17. Water body or waterway 18. Topographical feature]"
# PROMPT_6 = "You will be provided with a satellite image. List one or more built environment features in the image only if the feature is in this list:[1. sidewalks 2.tree-lined streets 3.porches 4.fenced front yards 5.attached garages 6.cul-de-sacs 7.hills 8.private front entrances.]"
# PROMPT_7 = "You will be provided with a satellite image. List one or more built environment features in the image only if the feature is in this list:[1. sidewalks 2.tree-lined streets 3.porches 4.fenced front yards 5.attached garages 6.cul-de-sacs 7.hills 8.private front entrances.]. Please just provide the output  as a list  in the following format: [feature1, feature2,feature3...]"
# PROMPT_8 = "You will be provided with a satellite image. List one or more built environment features in the image only if the feature is in this list:[1. sidewalks 2.tree-lined streets 3.porches 4.fenced front yards 5.attached garages 6.cul-de-sacs 7.hills 8.private front entrances.]. List only the features you are really sure. Please just provide the output  as a list  in the following format: [feature1, feature2,feature3...]"
# PROMPT_9 = 'Answer, yes or no. Does the region favor conversations between neighbors? Explain your answer. Please just provide the output  as a list  in the following format: [Yes/No],[*Explanation*]'




# prompt = PROMPT_8

def parse_args():
    parser = argparse.ArgumentParser(description="Demo")
    parser.add_argument("--prompt", type=int, default=1, help="prompt number.")
    parser.add_argument("--model", type=str, default="gpt-4o-mini", help="model gpt-4o-mini or gpt-4o.")
    parser.add_argument("--batch", type=int, default=1, help="images processed.")

    args = parser.parse_args()
    return args
args = parse_args()
prompts_csv = 'prompts.csv'
with open(prompts_csv, 'r') as f:
    reader = csv.reader(f)
    for l in reader:
        if l[0] == str(args.prompt):
            prompt = l[1].strip("'")

# print(prompt)
# print(args.prompt)     
model = args.model    
batch = args.batch
             
api_key=os.environ["OPEN_AI_KEY"]
client = OpenAI(api_key=api_key)

def encode_image(image_path):

    with open(image_path, "rb") as img_file:
        return base64.b64encode(img_file.read()).decode('utf-8')

def process_images(root_folder, pictures_txt, result_file, prompt, api_key, batch_size=10):

    with open(pictures_txt, 'r') as p_txt:
        image_names = p_txt.read().splitlines()
    

    processed_images = set()
    if os.path.exists(result_file):
        with open(result_file, 'r') as r:
            processed_images = {line.split(":")[0].strip() for line in r}
    
    processed_count = 0

    with open(result_file, 'a') as r:

        for file_name in image_names:
            if processed_count >= batch_size:
                break


            if file_name in processed_images:
                print(f"{file_name}: Pulando.")
                continue
            
            file_path = os.path.join(root_folder, file_name)
            

            if os.path.exists(file_path) and file_name.lower().endswith(('.png', '.jpg', '.jpeg', '.gif', '.bmp')):
                start_time = time.time()  

                base64_image = encode_image(file_path)

                headers = {
                    "Content-Type": "application/json",
                    "Authorization": f"Bearer {api_key}"
                }

                payload = {
                    "model": model,#"gpt-4o-mini", #"model":  "gpt-4o",
                    "messages": [
                        {
                            "role": "user",
                            "content": [
                                {
                                    "type": "text",
                                    "text": prompt

                                },
                                {
                                    "type": "image_url",
                                    "image_url": {
                                        "url": f"data:image/jpeg;base64,{base64_image}",
                                        "detail": "low"
                                    }
                                }
                            ]
                        }
                    ],
                    "max_tokens": 100
                }

                response = requests.post("https://api.openai.com/v1/chat/completions", headers=headers, json=payload)

                end_time = time.time()  
                process_time = end_time - start_time

 
                if response.status_code == 200:
                    result = response.json()
                    print(f"{file_name}: Processado em {process_time:.2f} segundos")
                    r.write(f"{file_name}: {json.dumps(result)} - Tempo de processamento: {process_time:.2f} segundos\n")
                    processed_count += 1  
                else:
                    print(f"{file_name}: Falha: {response.status_code}")

            else:
                print(f"{file_name}: Not found\n")

def clear_result_file(filename):
    base_name = os.path.splitext(filename)[0]  
    output_file = f"{base_name}_clean.txt"
    with open(filename, 'r') as infile, open(output_file, 'w') as outfile:
        for line in infile:

            name, dictionary_str = line.strip().split(': ', 1)

            try:
                dictionary = ast.literal_eval(dictionary_str)
            except Exception as e:
                print(f"Erro na linha: {line.strip()} - {e}")
                continue

            try:
                message_content = dictionary["choices"][0]["message"]["content"]
            except KeyError:
                print(f"Erro 'content' não encontrado em {line.strip()}")
                continue
            
            outfile.write(f"{name}: {message_content}\n")
#root = r'C:\Users\Raissa\Documents\dev\TCC\City_Street_View_Dataset'
#txt = r'C:\Users\Raissa\Documents\dev\TCC\AnaliseExploratoria\Chicago_pictures.txt'

root = r'C:\Users\Raissa\Documents\dev\TCC\GoogleMaps\chicago_satellite_images_large'
#txt = r"C:\Users\Raissa\Documents\dev\TCC\OpenAI\chicago_satellite_images.txt"
txt = r'C:\Users\Raissa\Documents\dev\TCC\AnaliseExploratoria\Chicago_satellite_test2.txt'
#result_file = f'prompt_6_OpenAI_mini_Chicago_satellitetest2.txt'


txt = r'C:\Users\Raissa\Documents\dev\TCC\random_satellite_selection.txt'
#result_file = f'prompt_8_OpenAI_mini_random_satellite_selection.txt'

result_file = f'prompt_{args.prompt}_{model}_random_satellite_selection.txt'

txt = r'C:\Users\Raissa\Documents\dev\TCC\hand_picked.txt'

result_file = f'prompt_{args.prompt}_{model}_hand_picked.txt'

process_images(root, txt, result_file, prompt, api_key, batch)

clear_result_file(result_file)
#4oMINI PROMPT1
# The image shows a residential neighborhood with a clear blue sky. In the foreground, there\u2019s a landscaped yard with green grass and some flower beds.
# There are two houses visible; one is a two-story home with a distinctive roof, and the other is a single-story house next to it. 
# Trees surround the area, providing some shade, and the street appears to be quiet with well-maintained sidewalks. The overall atmosphere is peaceful and suburban. 

#4o PROMPT1
# "The image depicts a residential suburban neighborhood with a focus on two houses. The house on the left has a sloped brown roof and beige brick exterior. 
# The house on the right has a blue-grey exterior with white trim and a gabled roof. Both houses have front lawns with well-maintained grass and trees. 
# There's a clear blue sky and some sunlight casting shadows on the ground. A sidewalk runs along the front, bordered by a low hedge near the driveway of the left house."

#4o "Describe the atmosphere of this area." PROMT 4
# "The area appears to be a quiet, residential neighborhood with a peaceful atmosphere. The well-maintained houses and neatly trimmed lawns suggest a sense of community pride.
# The presence of mature trees provides ample shade and adds to the picturesque, suburban feel. It's likely a serene and family-friendly environment.",