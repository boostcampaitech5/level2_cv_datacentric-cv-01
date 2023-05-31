import pandas as pd
import streamlit as st
import json
from pandas import DataFrame
import os

train_json_path = "../../data/medical/ufo/train.json"
def load_all_annotations() -> pd.DataFrame:
    with open(train_json_path,"r") as f:
        json_data = json.load(f)
    return json.dumps(json_data)

def json_to_df(json_file=train_json_path):
    with open(json_file,'r') as f:
        json_data = json.load(f)
        json_img = json_data['images']
    img_df = pd.DataFrame(json_img)
    return img_df



train_img_path = "../../data/medical/img/train"



def find_list_of_imgs(img_path=train_img_path):
    return sorted(os.listdir(img_path))

def count_imgs(img_path=train_img_path)->int:
    return len(os.listdir(img_path)) 


