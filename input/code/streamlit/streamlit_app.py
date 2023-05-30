import cv2
import pandas as pd
import streamlit as st
import json
from pandas import DataFrame
import os
from visualization_tool import *
from img_bbox_tools import *
from PIL import Image
import sys
import numpy as np
sys.path.insert(0,"..")
from dataset import SceneTextDataset

from augmentation.augmentation import aug_config,change_config

root_dir = "../../data/medical"
train_json_path = "../../data/medical/ufo/train.json"
img_path = "../../data/medical/img"
saved_root = "./saved_img_with_bbox"
saved_folder = "./saved_img_with_bbox/img"
img_df = json_to_df()
img_id_list = find_list_of_imgs() # 이미지들 이름 저장
total_imgs = count_imgs()
# data_dir = "../../data/medical/img"
aug_list = ["Resize",
    "AdjustHeight",
    "Rotate",
    "Crop",
    "ToNumpy",
    "RandomShadow",
    "ColorJitter",
    "Normalize"]
st.session_state.dataset = SceneTextDataset(
        root_dir=root_dir,
        split="train",
        image_size=2048,
        crop_size=1024,
        # ignore_tags=ignore_tags,
        aug_list=aug_list,
    )

def main():
    if 'image_index' not in st.session_state:
        st.session_state.image_index = 0
        
        

    col_prev_btn,col_saving_btn,col_next_btn = st.columns([2,2,1])
    col_text_input,col_move_with_index_btn,col_move_with_img_id = st.columns([2,1,1])

    prev_btn = col_prev_btn.button('prev',key="1")
    next_btn = col_next_btn.button('next',key="2")
    target_idx = col_text_input.text_input('index or id')
    move_idx = col_move_with_index_btn.button('move with index',key="4")
    move_img_id = col_move_with_img_id.button('move with id',key="move_with_img_id")
    button_saving_all = col_saving_btn.button("saving all images",key="6")

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

    target_img = img_id_list[st.session_state.image_index]
    target_img_points = img_df[target_img]

    #TODO augmentation을 checkbox로 적용 전과 후를 확인할 수 있는 기능 구현
    # dict형식인데 False면 참고 안하게?

    st.session_state.aug_dict = {i : False for i in aug_list}
    # Not False로 하면 될 것
    
    # with open('../augmentation/aug_config.json','r')as f:
    #     aug_config = json.load(f)
    
    st.session_state.new_aug_list = []
    bounding_box = st.sidebar.checkbox("bounding_box",key="bounding_box")   # bounding box를 표시할지 말지 여부를 나타내는 checkbox
    aug_checkbox_Resize = st.sidebar.checkbox("Resize",key="Resize")
    if aug_checkbox_Resize:
        rs = st.sidebar.slider("Resize Scale",100,2500,2048)
        aug_config["Resize"]["size"] = rs
        change_config(aug_config,"Resize")
        st.session_state.new_aug_list.append("Resize")
        # st.write(rs)
        
    aug_checkbox_AdjustHeight = st.sidebar.checkbox("AdjustHeight",key="AdjustHeight")
    if aug_checkbox_AdjustHeight:
        hr = st.sidebar.slider("Height Ratio * 10",0,10,2)
        st.session_state.new_aug_list.append("AdjustHeight")
        change_config(aug_config,"AdjustHeight")
        aug_config["AdjustHeight"]["ratio"]=hr/10
        
    aug_rotate = st.sidebar.checkbox("Rotate",key = "Rotate")
    if aug_rotate:
        rotate_s = st.sidebar.slider("Angle Range",0,360,10)
        aug_config["Rotate"]["angle_range"] = rotate_s
        change_config(aug_config,"Rotate")
        st.session_state.new_aug_list.append("Rotate")
        
    aug_crop = st.sidebar.checkbox("Crop",key="Crop")
    if aug_crop:
        cs = st.sidebar.slider("Crop Scale",0,2500,1024)
        aug_config["Crop"]["length"]=cs
        change_config(aug_config,"Crop")
        st.session_state.new_aug_list.append("Crop")
        
    # 꼭 저장돼야함
    # st.session_state.aug_dict["ToNumpy"] = True
    st.session_state.new_aug_list.append("ToNumpy")
    
    aug_random_shadow = st.sidebar.checkbox("RandomShadow",key="RandomShadow")
    if aug_random_shadow:
        st.session_state.aug_dict["RandomShadow"] = True
        s_roi1 = st.sidebar.slider("x_min",0,10,0)/10
        s_roi2 = st.sidebar.slider("y_min",0,10,5)/10
        s_roi3 = st.sidebar.slider("x_max",0,10,5)/10
        s_roi4 = st.sidebar.slider("y_max",0,10,10)/10
        aug_config["RandomShadow"]['shadow_roi'] = [s_roi1,s_roi2,s_roi3,s_roi4]
        # st.write(aug_config["RandomShadow"]['shadow_roi'])
        # st.write(aug_config["RandomShadow"]['shadow_roi'])
        num_shadows_lower = st.sidebar.slider("num_shadows_lower",0,10,1)
        aug_config["RandomShadow"]['num_shadows_lower'] = num_shadows_lower
        num_shadows_upper = st.sidebar.slider("num_shadows_upper",0,10,2)
        aug_config["RandomShadow"]['num_shadows_upper'] = num_shadows_upper
        shadow_dimension = st.sidebar.slider("shadow_dimension",0,10,5)
        aug_config["RandomShadow"]['shadow_dimension'] = shadow_dimension
        p = st.sidebar.slider("p",0,10,5) / 10
        aug_config["RandomShadow"]['p'] = p
        # st.write(aug_config["RandomShadow"])
        change_config(aug_config,"RandomShadow")
        st.session_state.new_aug_list.append("RandomShadow")
        
        
    aug_color_jitter = st.sidebar.checkbox("ColorJitter",key="ColorJitter")
    if aug_color_jitter:
        brightness = st.sidebar.slider("brightness * 10",0,10,2)/10
        contrast = st.sidebar.slider("contrast * 10",0,10,2)/10
        saturation = st.sidebar.slider("saturation * 10",0,10,2)/10
        hue = st.sidebar.slider("hue * 10",0,10,2)/10
        p = st.sidebar.slider("p * 10",0,10,5)/10
        aug_config["ColorJitter"]["brightness"] = brightness
        aug_config["ColorJitter"]["contrast"] = contrast
        aug_config["ColorJitter"]["saturation"] = saturation
        aug_config["ColorJitter"]["hue"] = hue
        aug_config["ColorJitter"]["p"] = p
        change_config(aug_config,"ColorJitter")
        st.session_state.new_aug_list.append("ColorJitter")
        
    
    add_augmentation("Emboss")
    add_augmentation("OnlyBlack")
    add_augmentation("Sharpen")
    add_augmentation("CLAHE")
    
    # TODO normalize 에러 수정
    aug_normalize = st.sidebar.checkbox("Normalize",key="Normalize")
    if aug_normalize:
        mean = st.sidebar.slider("mean",0,10,5)/10
        std = st.sidebar.slider("std",0,10,5)/10
        aug_config["Normalize"]["mean"] = mean
        aug_config["Normalize"]["std"] = std
        change_config(aug_config,"Normalize")
        st.session_state.new_aug_list.append("Normalize")
    st.session_state.dataset.aug_list = st.session_state.new_aug_list  
      
    if bounding_box:    # bounding box 보이기
        if os.path.exists(os.path.join(saved_folder,"train",target_img)):
            pass
        else:
            save_image_with_bbox(img_path,"train", target_img,target_img_points)
        st.session_state.dataset.image_dir = os.path.join(saved_root, 'img', "train")
    else:   # bounding box 없는 원본 이미지 보기
        st.session_state.dataset.image_dir = os.path.join(root_dir, 'img', "train")
    fig,_,_ = st.session_state.dataset[st.session_state.image_index]
    std = np.max(fig)-np.min(fig)
    if "Normalize" in aug_list:
        fig = (fig-np.min(fig))/std
    #TODO Normalize 적용 에러 수정
    # fig=np.clip(0,1,fig/255)
    
    st.write(st.session_state.dataset.aug_list)
    # st.write(aug_config)
    st.image(fig)

# #TODO 서버간 데이터 이동 구현
# #TODO 데이터가 들어올때마다 inference 적용하고 bbox 확인해 볼 수 있는 코드
#     # -> 진행상황을 tqdm등으로 실시간 확인 가능하면 좋을 것 같음
# #TODO
    
# page_names_to_funcs = {
#     "—": intro,
#     "Augmentation": Augmentation,
#     "Mapping Demo": mapping_demo,
#     "DataFrame Demo": data_frame_demo
# }

# demo_name = st.sidebar.selectbox("Choose a demo", page_names_to_funcs.keys())
# page_names_to_funcs[demo_name]()

def add_augmentation(augmentation:str):
    aug = st.sidebar.checkbox(augmentation,key=augmentation)
    if aug:
        st.session_state.new_aug_list.append(augmentation)

if __name__ == "__main__":
    main()
