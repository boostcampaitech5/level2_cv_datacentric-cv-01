# LV2 data-centric

## 제약조건
### git ignore
수정하면 안되는 파일 + dataset 추가

### 수정하면 안되는 파일(git ignore 추가)
model.py<br>
loss.py<br>
east_dataset.py<br>
detect.py<br>

### Requirements
python==3.8.5
pandas==2.0.1
torch==1.7.1
lanms==1.0.2
numpy==1.21.3
opencv-python==4.5.4.58
shapely==1.7.1
tqdm==4.62.3
matplotlib
albumentations==1.1.0

### Team Role
- 추가 하고 싶은 기능 부담 갖지말고 건의하기
- wandb 개인 repo에서 팀프로젝트 repo로 옮기기
- 매일 피어세션 진행 과정 공유
- 기본적으로 2개의 제출 기회 소모, `초과 제출시` 카톡으로 물어보고 사용
- 구현이 끝나면 PR 카톡 알림 후,  `Pull Request` 올린 후 `merge`한 뒤, 실험해 보고 카톡하기
- 카톡으로 Merge 알림오면 무조건 git pull origin dev


### Code Convention
- Black formatter사용
- Class - 대문자 Camel case (Pascal case)
- def - Snake case(동사_명사)
- 변수 - Snake case(명사)

### Git Flow
>main
>├── dev

>│   ├── kbh

>│   ├── kdk

>│   ├── ltg

>│   ├── jhj

>│   ├── lhm
