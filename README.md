# Protonet

### Requirements:
1. python==3.6 <br>
2. pytorch 0.4.0
```
conda create -n protonet-env python=3.6
source activate protonet-env
pip install -r requirements.txt
```

### miniImageNet
[[Google Drive](https://drive.google.com/uc?id=1HkgrkAwukzEZA0TpO7010PkAOREb2Nuk&export=download)]  (2.9 GB)
```
# Download and place "mini-imagenet.zip" in "PATH/TO/data/raw/mini-imagenet".
mkdir -p PATH/TO/data/raw/mini-imagenet
cd data/mini-imagenet
mv PATH/TO/mini-imagenet.zip  PATH/TO/data/raw/
unzip PATH/TO/data/raw/mini-imagenet.zip 
rm -f PATH/TO/data/raw/mini-imagenet.zip 

# Download and place Ravi split in "PATH/TO/data/splits/mini_imagenet_split"
git clone https://github.com/renmengye/few-shot-ssl-public
mkdir -p PATH/TO/data/splits/mini_imagenet_split
mv PATH/TO/few-shot-ssl-public/fewshot/data/mini_imagenet_split/Ravi PATH/TO/data/splits/mini_imagenet_split

# Prepare data
python prepro/gen_miniImageNet.py
```

### Run:
```
python train_protonet.py --hparams=protonet_5way_1shot
python train_protonet.py --hparams=protonet_5way_5shot
...
```
