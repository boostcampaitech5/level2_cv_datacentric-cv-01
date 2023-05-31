python train_fixed_cosAnn.py --max_epoch=130 --save_interval=130
python inference.py --pretrain='130Epoch.pth' --output_name='130Epoch_AdamW.csv' --output_dir='../../output/fixed_cosAnn'