import cv2
import pandas as pd
import streamlit as st
import json
from pandas import DataFrame
import os
from visualization_tool import *
from img_bbox_tools import *
from PIL import Image
train_json_path = "../../data/medical/ufo/train.json"

img_path = "../../data/medical/img/train"


img_df = json_to_df()
img_id_list = find_list_of_imgs() # 이미지들 이름 저장
# print(count_imgs())
total_imgs = count_imgs()

if 'image_index' not in st.session_state:
    st.session_state.image_index = 0
    
col1,col6,col2 = st.columns([2,2,1])
col3,col4,col5 = st.columns([2,1,1])

prev_btn = col1.button('prev')
next_btn = col2.button('next')
target_idx = col3.text_input('index or id')
move_idx = col4.button('move with index')
move_img_id = col5.button('move with id')
button_saving_all = col6.button("saving all images")

if prev_btn:
    st.session_state.image_index = (st.session_state.image_index -1)%total_imgs
if next_btn:
    st.session_state.image_index = (st.session_state.image_index +1)%total_imgs 
if move_idx:
    target_idx = int(target_idx)
    if target_idx>=0 and target_idx< total_imgs:
        st.session_state.image_index = target_idx
if move_img_id:
    for idx,id in enumerate(img_id_list):
        if target_idx in id:
            st.session_state.image_index = idx
            break
if button_saving_all:
    saving_all(img_path)

saved_folder = "./saved_img_with_bbox"
target_img = img_id_list[st.session_state.image_index]
target_img_points = img_df[target_img]

if os.path.exists(os.path.join(saved_folder,target_img)):
    pass
else:
    save_image_with_bbox(img_path, target_img,target_img_points)

fig1 = Image.open(os.path.join(saved_folder,target_img))

st.image(fig1)
