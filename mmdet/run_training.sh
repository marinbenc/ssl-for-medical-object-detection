#!/bin/sh

cd pretraining &&
python pretrain.py --experiment-name pretrain10 --batch-size 8 --lr 0.001 --epochs 100 --labeled-dataset-percent 0.9 --replace &&
cd .. &&
python train.py --experiment-name pretrain10 --batch-size 8 --lr 0.001 --epochs 15 --labeled-dataset-percent 0.9 &&

cd pretraining &&
python pretrain.py --experiment-name pretrain20 --batch-size 8 --lr 0.001 --epochs 100 --labeled-dataset-percent 0.8 --replace &&
cd .. &&
python train.py --experiment-name pretrain20 --batch-size 8 --lr 0.001 --epochs 15 --labeled-dataset-percent 0.8 &&

cd pretraining &&
python pretrain.py --experiment-name pretrain30 --batch-size 8 --lr 0.001 --epochs 100 --labeled-dataset-percent 0.7 --replace &&
cd .. &&
python train.py --experiment-name pretrain30 --batch-size 8 --lr 0.001 --epochs 15 --labeled-dataset-percent 0.7 &&

cd pretraining &&
python pretrain.py --experiment-name pretrain40 --batch-size 8 --lr 0.001 --epochs 100 --labeled-dataset-percent 0.6 --replace &&
cd .. &&
python train.py --experiment-name pretrain40 --batch-size 8 --lr 0.001 --epochs 15 --labeled-dataset-percent 0.6 &&

cd pretraining &&
python pretrain.py --experiment-name pretrain50 --batch-size 8 --lr 0.001 --epochs 100 --labeled-dataset-percent 0.5 --replace &&
cd .. &&
python train.py --experiment-name pretrain50 --batch-size 8 --lr 0.001 --epochs 15 --labeled-dataset-percent 0.5 &&

cd pretraining &&
python pretrain.py --experiment-name pretrain60 --batch-size 8 --lr 0.001 --epochs 100 --labeled-dataset-percent 0.4 --replace &&
cd .. &&
python train.py --experiment-name pretrain60 --batch-size 8 --lr 0.001 --epochs 15 --labeled-dataset-percent 0.4 &&

cd pretraining &&
python pretrain.py --experiment-name pretrain70 --batch-size 8 --lr 0.001 --epochs 100 --labeled-dataset-percent 0.3 --replace &&
cd .. &&
python train.py --experiment-name pretrain70 --batch-size 8 --lr 0.001 --epochs 15 --labeled-dataset-percent 0.3 &&

cd pretraining &&
python pretrain.py --experiment-name pretrain80 --batch-size 8 --lr 0.001 --epochs 100 --labeled-dataset-percent 0.2 --replace &&
cd .. &&
python train.py --experiment-name pretrain80 --batch-size 8 --lr 0.001 --epochs 15 --labeled-dataset-percent 0.2 &&

cd pretraining &&
python pretrain.py --experiment-name pretrain90 --batch-size 8 --lr 0.001 --epochs 100 --labeled-dataset-percent 0.1 --replace &&
cd .. &&
python train.py --experiment-name pretrain90 --batch-size 8 --lr 0.001 --epochs 15 --labeled-dataset-percent 0.1