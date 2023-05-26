python train_cosAnn.py --max_epoch=60 --save_interval=60
python inference.py --pretrain='60Epoch.pth' --output_name='60Epoch_cosAnn.csv' --output_dir='../../output'
python train_cosAnn.py --max_epoch=70 --save_interval=70
python inference.py --pretrain='70Epoch.pth' --output_name='70Epoch_cosAnn.csv' --output_dir='../../output'
python train_cosAnn.py --max_epoch=80 --save_interval=80
python inference.py --pretrain='80Epoch.pth' --output_name='80Epoch_cosAnn.csv' --output_dir='../../output'
python train_cosAnn.py --max_epoch=90 --save_interval=90
python inference.py --pretrain='90Epoch.pth' --output_name='90Epoch_cosAnn.csv' --output_dir='../../output'
python train_cosAnn.py --max_epoch=100 --save_interval=100
python inference.py --pretrain='100Epoch.pth' --output_name='100Epoch_cosAnn.csv' --output_dir='../../output'