import json
import base64
import requests
import os
from openai import OpenAI
import csv
def load_images_as_base64(image_paths):
    images_base64 = []
    for path in image_paths:
        with open(path, "rb") as image_file:
            images_base64.append(base64.b64encode(image_file.read()).decode("utf-8"))
    return images_base64

api_key=os.environ["OPEN_AI_KEY"]
client = OpenAI(api_key=api_key)
prompt = """You will be provided with three images of the same location. The first image is a satellite 
image and the second and third images are street-level images. Follow these instructions in order:
1. For the satellite image, list one or more built environment features in the image only if the feature is in this list:
[1. sidewalks 2.tree-lined streets 3.porches 4.fenced front yards 5.attached garages 6.cul-de-sacs 7.hills 8.private front entrances.]:
2. For the street-level images, identify any features from the list that you did not
identify in the satellite image, if there are any.
Please just provide the output  as a list  in the following format: [feature1, feature2,feature3...]"""

prompt ="You will be provided with three images of the same location. Answer, yes or no. Does the region favor conversations between neighbors?"
def save_list_to_txt(data_list, output_file):
    with open(output_file, mode="w", encoding="utf-8") as file:
        for item in data_list:
            file.write(f"{item}\n")
def load_processed_ids(result_file):
    """Carrega os IDs já processados a partir do arquivo de resultados."""
    if not os.path.exists(result_file):
        return set()
    
    processed_ids = set()
    with open(result_file, "r", encoding="utf-8") as file:
        for line in file:
            processed_id = line.split(":")[0]
            processed_ids.add(processed_id)
    return processed_ids

def save_partial_result(id_, geo_id,result, result_file):
    """Salva o resultado parcial no arquivo."""
    with open(result_file, "a", encoding="utf-8") as file:
        file.write(f"{id_}:{geo_id}: {json.dumps(result)}\n")

def process_images(root_satellite, root_street_view, pictures_txt, result_file, result_file_clean, prompt, api_key):
    processed_ids = load_processed_ids(result_file)
    
    image_list = []
    with open(pictures_txt, mode="r", encoding="utf-8") as file:
        reader = csv.reader(file)
        next(reader)  
        for row in reader:
            if len(row) >= 3:  
                geoid = row[1]
                image_name = row[2]
                image_list.append([row[0], geoid, image_name, row[3], row[4]])

    for l in image_list:
        id_ = l[0]    
        geo_id = l[1]

        if id_ in processed_ids:
            print(f"{id_} já processado, pulando...")
            continue
        
        sat = l[2]     
        st1 = l[3]     
        st2 = l[4]     
        
        satellite = f"{root_satellite}/{sat}"
        street_1 =  f"{root_street_view}/{st1}"
        street_2 = f"{root_street_view}/{st2}"

        image_paths = [satellite, street_1, street_2]  
        base64_images = load_images_as_base64(image_paths)
        
        headers = {
            "Content-Type": "application/json",
            "Authorization": f"Bearer {api_key}"
        }
        
        messages = [
            {
                "role": "user",
                "content": [
                    {"type": "text", "text": prompt}
                ] + [
                    {
                        "type": "image_url",
                        "image_url": {
                            "url": f"data:image/jpeg;base64,{base64_image}",
                            "detail": "low"
                        }
                    }
                    for base64_image in base64_images
                ]
            }
        ]

        payload = {
            "model": "gpt-4o-mini",
            "messages": messages,
            "max_tokens": 100
        }

        response = requests.post(
            "https://api.openai.com/v1/chat/completions", 
            headers=headers, 
            json=payload
        )

        if response.status_code == 200:
            result = response.json()
            result_clean = result['choices'][0]['message']['content']
            
            save_partial_result(id_,geo_id , result, result_file)
            save_partial_result(id_,geo_id , result_clean, result_file_clean)
        else:
            print(f"{id_}: Falha: {response.status_code}")

# root_satellite = r'C:\Users\Raissa\Documents\dev\TCC\GoogleMaps\chicago_satellite_images_large'
# root_street_view = r'C:\Users\Raissa\Documents\dev\TCC\City_Street_View_Dataset'
# images_csv = r'C:\Users\Raissa\Documents\dev\TCC\AnaliseExploratoria\chicago_random_50_tracts_images.csv'
# result_file = '50_multi_4o.txt'
# result_file_clean = '50_multi_4o_clean.txt'


root_satellite = r'C:\Users\Raissa\Documents\dev\TCC\GoogleMaps\chicago_satellite_tracts_images_large'
root_street_view = r'C:\Users\Raissa\Documents\dev\TCC\City_Street_View_Dataset'
images_csv = r'C:\Users\Raissa\Documents\dev\TCC\AnaliseExploratoria\final_Chicago_tracts_pic_sat_grouped_util_id.csv'
result_file = 'var_Full_4o_mini.txt'
result_file_clean = 'var_Full_4o_mini_clean.txt'
process_images(root_satellite, root_street_view, images_csv, result_file, result_file_clean, prompt, api_key)
