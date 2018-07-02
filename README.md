# Protonet

### miniImageNet
[[Google Drive](https://drive.google.com/open?id=16V_ZlkW4SsnNDtnGmaBRq2OoPmUOc5mY)]  (1.1 GB)
```
# Download and place "mini-imagenet.tar.gz" in "data/raw/mini-imagenet".
mkdir -p PATH/TO/data/raw/mini-imagenet
cd data/mini-imagenet
mv PATH/TO/mini-imagenet.tar.gz PATH/TO/data/raw/
tar -xzvf PATH/TO/data/raw/mini-imagenet.tar.gz
rm -f PATH/TO/data/raw/mini-imagenet.tar.gz

# Download and place "[[Ravi](https://github.com/renmengye/few-shot-ssl-public/tree/master/fewshot/data/mini_imagenet_split)]" in "data/splits/mini_imagenet_split"
git clone https://github.com/renmengye/few-shot-ssl-public
mkdir -p PATH/TO/data/splits/mini_imagenet_split
mv PATH/TO/few-shot-ssl-public/fewshot/data/mini_imagenet_split/Ravi PATH/TO/data/splits/mini_imagenet_split
```

### Run:
```
python train_protonet.py --hparams=protonet_5way_1shot
python train_protonet.py --hparams=protonet_5way_5shot
...
```
