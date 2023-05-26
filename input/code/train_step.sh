python train.py --max_epoch=30 --save_interval=30
python inference.py --pretrain='30Epoch.pth' --output_name='30Epoch_step.csv'
python train.py --max_epoch=40 --save_interval=40
python inference.py --pretrain='40Epoch.pth' --output_name='40Epoch_step.csv'
python train.py --max_epoch=50 --save_interval=50
python inference.py --pretrain='50Epoch.pth' --output_name='50Epoch_step.csv'
python train.py --max_epoch=60 --save_interval=60
python inference.py --pretrain='60Epoch.pth' --output_name='60Epoch_step.csv'
python train.py --max_epoch=70 --save_interval=70
python inference.py --pretrain='70Epoch.pth' --output_name='70Epoch_step.csv'
python train.py --max_epoch=80 --save_interval=80
python inference.py --pretrain='80Epoch.pth' --output_name='80Epoch_step.csv' --output_dir='../../output'
python train.py --max_epoch=90 --save_interval=90
python inference.py --pretrain='90Epoch.pth' --output_name='90Epoch_step.csv' --output_dir='../../output'
python train.py --max_epoch=100 --save_interval=100
python inference.py --pretrain='100Epoch.pth' --output_name='100Epoch_step.csv' --output_dir='../../output'
