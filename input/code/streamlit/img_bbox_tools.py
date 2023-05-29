import matplotlib.pyplot as plt
import matplotlib.patches as patches
from PIL import Image,ImageDraw
import os

import cv2

from streamlit_app import img_df

saved_image_folder = "./saved_img_with_bbox"
def save_image_with_bbox(img_path:os.path,dataset:str,image_id,img_bbox,saving_path=saved_image_folder) -> None:
    '''
    이미지를 bbox를 포함하여 저장하는 함수
    '''
    saving_path = os.path.join(saving_path,dataset)
    if not os.path.exists(saving_path): # 존재하지 않는 경로인 경우 생성
        os.mkdir(saving_path)
    im = Image.open(os.path.join(img_path,image_id))
    draw = ImageDraw.Draw(im)
    for bbox in img_bbox["words"]:
        points = img_bbox["words"][bbox]["points"]
        vertices = list(map(tuple,points))
        draw.polygon(vertices,fill=None,outline=(255,0,0),width=3)
    im.save(os.path.join(saving_path,image_id),"PNG")
    

def saving_all(img_path,saving_path=saved_image_folder,dataset="train")->None:
    '''
    한번에 img_path 안의 모든 이미지들을 bbox를 포함하여 저장하는 함수
    '''
    # 순서대로 저장
    img_list = sorted(os.listdir(img_path))
    # saving_path = os.path.join(saving_path,dataset)
    for img_id in img_list:
        if os.path.exists(os.path.join(saving_path,dataset,img_id)): #이미 저장된 이미지가 있으면 pass
            pass
        else:
            print(f"saving {img_id}..")
            img_bbox = img_df[img_id]
            save_image_with_bbox(img_path,dataset,img_id,img_bbox)
            print(f"done")
