import argparse
import os
import random

import numpy as np
import torch
import torch.backends.cudnn as cudnn
import gradio as gr

from transformers import StoppingCriteriaList

from minigpt4.common.config import Config
from minigpt4.common.dist_utils import get_rank
from minigpt4.common.registry import registry
from minigpt4.conversation.conversation import Chat, CONV_VISION_Vicuna0, CONV_VISION_LLama2, StoppingCriteriaSub

# imports modules for registration
from minigpt4.datasets.builders import *
from minigpt4.models import *
from minigpt4.processors import *
from minigpt4.runners import *
from minigpt4.tasks import *

from PIL import Image
import time
import pandas as pd

def parse_args():
    parser = argparse.ArgumentParser(description="Demo")
    parser.add_argument("--cfg-path", required=True, help="path to configuration file.")
    parser.add_argument("--gpu-id", type=int, default=0, help="specify the gpu to load the model.")
    parser.add_argument(
        "--options",
        nargs="+",
        help="override some settings in the used config, the key-value pair "
        "in xxx=yyy format will be merged into config file (deprecate), "
        "change to --cfg-options instead.",
    )
    args = parser.parse_args()
    return args


def setup_seeds(config):
    seed = config.run_cfg.seed + get_rank()

    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)

    cudnn.benchmark = False
    cudnn.deterministic = True


# ========================================
#             Model Initialization
# ========================================

conv_dict = {'pretrain_vicuna0': CONV_VISION_Vicuna0,
             'pretrain_llama2': CONV_VISION_LLama2}

print('Initializing Chat')
args = parse_args()
cfg = Config(args)

model_config = cfg.model_cfg
model_config.device_8bit = args.gpu_id
model_cls = registry.get_model_class(model_config.arch)
model = model_cls.from_config(model_config).to('cuda:{}'.format(args.gpu_id))

CONV_VISION = conv_dict[model_config.model_type]

vis_processor_cfg = cfg.datasets_cfg.cc_sbu_align.vis_processor.train
vis_processor = registry.get_processor_class(vis_processor_cfg.name).from_config(vis_processor_cfg)

stop_words_ids = [[835], [2277, 29937]]
stop_words_ids = [torch.tensor(ids).to(device='cuda:{}'.format(args.gpu_id)) for ids in stop_words_ids]
stopping_criteria = StoppingCriteriaList([StoppingCriteriaSub(stops=stop_words_ids)])

chat = Chat(model, vis_processor, device='cuda:{}'.format(args.gpu_id), stopping_criteria=stopping_criteria)
print('Initialization Finished')
chat_state = CONV_VISION.copy()
img_list = []


#image_file = r"C:\Users\Raissa\Documents\dev\TCC\INPUTGPT.jpg"
# image_file = r"C:\Users\Raissa\Documents\dev\TCC\City_Street_View_Dataset\street_view_1.jpg"

# # open method used to open different extension image file
# im = Image.open(image_file)

# llm_message = chat.upload_img(im, chat_state, img_list)
# chat.encode_img(img_list)
# user_message = 'describe the image'
# chat.ask(user_message, chat_state)

# llm_message = chat.answer(conv=chat_state,
#                               img_list=img_list,
#                               num_beams=1,
#                               temperature=1,
#                               max_new_tokens=300,
#                               max_length=2000)[0]

# print(llm_message)

def get_description(image_path):
    print(image_path)
    
    with Image.open(image_path) as im:
        #im = Image.open(image_path)
        img_list = []
        chat_state = CONV_VISION.copy()
        llm_message = chat.upload_img(im, chat_state, img_list)
        chat.encode_img(img_list)
        prompt = """You will be provided with three images of the same location. The first image is a satellite 
image and the second and third images are street-level images. Follow these instructions in order:
1. For the satellite image, list one or more built environment features in the image only if the feature is in this list:
[1. sidewalks 2.tree-lined streets 3.porches 4.fenced front yards 5.attached garages 6.cul-de-sacs 7.hills 8.private front entrances.]:
2. For the street-level images, identify any features from the list that you did not
identify in the satellite image, if there are any.
Please just provide the output  as a list  in the following format: [feature1, feature2,feature3...]"""  
        user_message = (prompt)
        chat.ask(user_message, chat_state)

        llm_message = chat.answer(conv=chat_state,
                                    img_list=img_list,
                                    num_beams=1,
                                    temperature=0.1,
                                    max_new_tokens=300,
                                    max_length=2000)[0]

        print(llm_message)
        chat_state.messages = []

        return llm_message

def process(image_folder, result_file):
    count = 0
    with open(result_file, 'w') as r:

        for root, dir, files in os.walk(image_folder):
            for file_name in files:

                if file_name.lower().endswith(('.png', '.jpg', '.jpeg', '.gif', '.bmp')):

                    file_path = os.path.join(root, file_name)
                    

                    result = get_description(file_path)
                    

                    r.write(f"{file_name}: {result}\n")
                count +=1
                if count==15:
                    break

def process_from_csv(root_folder, pictures_csv,result_file, batch_size=50):

    df = pd.read_csv(pictures_csv)
    image_names = df['merged_image']
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
                print(f"{file_name}: Já processado, pulando.")
                continue
            file_path = os.path.join(root_folder, file_name)
            

            if os.path.exists(file_path) and file_name.lower().endswith(('.png', '.jpg', '.jpeg', '.gif', '.bmp')):

                start_time = time.time()  

                result = get_description(file_path)
                print(result)
                end_time = time.time()  
                process_time = end_time - start_time
                print(f"{file_name}: Processado em {process_time:.2f} segundos")
                
                r.write(f"{file_name}: {result}\n")
                processed_count += 1

            else:

                print(f"{file_name}: Arquivo não encontrado ou extensão inválida\n")
    

# image_folder = r'C:\Users\Raissa\Documents\dev\TCC\City_Street_View_Dataset'
# result_file = 'results_miniGPT4_building_01.txt'


# process(image_folder, result_file)
#root = r'C:\Users\Raissa\Documents\dev\TCC\City_Street_View_Dataset'
#txt = r'C:\Users\Raissa\Documents\dev\TCC\AnaliseExploratoria\Chicago_pictures.txt'
#result_file = 'results_Chicago.txt'

root = r'merged_images'
csv_file = r'final_chicago_random_50_tracts_images_JPG.csv'
result_file = 'results_Chicago_random_50_tracts.txt'



process_from_csv(root, csv_file, result_file)