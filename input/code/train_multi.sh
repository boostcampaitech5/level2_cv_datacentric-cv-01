python train_cosAnn.py --max_epoch=120 --save_interval=120
python inference.py --pretrain='120Epoch.pth' --output_name='120Epoch_cosAnn.csv' --output_dir='../../output/Ann'
python train_cosAnn.py --max_epoch=130 --save_interval=130
python inference.py --pretrain='130Epoch.pth' --output_name='130Epoch_cosAnn.csv' --output_dir='../../output/Ann'
python train_cosRes.py --max_epoch=120 --save_interval=120
python inference.py --pretrain='120Epoch.pth' --output_name='120Epoch_cosRes.csv' --output_dir='../../output/Res'
python train_cosRes.py --max_epoch=130 --save_interval=130
python inference.py --pretrain='130Epoch.pth' --output_name='130Epoch_cosRes.csv' --output_dir='../../output/Res'