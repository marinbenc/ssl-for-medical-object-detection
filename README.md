# Using self-supervised pretraining to reduce the need for labeled data for medical object detection

This is code from the paper TODO.

Paper link: TODO

BibTex: TODO

---

### Requirements:

 - Python 3.8
 - PyTorch 1.10
 - MMDetection 2.20 
 - Lightly SSL 1.2
 - Check `environment.yml` for more packages.

## Data used

**Ha Q. Nguyen** *et al.* *“VinDr-CXR: An open dataset of chest X-rays with radiologist’s annotations”* – A preprint is available on [ArXiv](https://arxiv.org/abs/2012.15029) 

https://vindr.ai/datasets/cxr

## Usage

`source/` contains all the code. `source/pretraining` contains the code for self-supervised pretraining. Use `train.py` and `pretrain.py` for training and pre-training, and `test.py` to test on a test dataset.

To prepare the data, download the VinBigData dataset from here:

https://www.kaggle.com/awsaf49/vinbigdata-512-image-dataset

This is a 512x512 .png version of the original VinDr-CXR dataset.

Store it in a folder named `vinbigdata` at the root of the repository. Then run `source/convert_to_coco.py` to convert the dataset and store it into `source/data`.

Check `source/run_training.sh` for details on how to run pre-training and training. If you want to pre-train the models, it's important that you use the same experiment name for both pre-training and fine-tuning. Pre-training stores the backbone checkpoint in `vinbig_output/<experiment-name>`, which is then loaded before fine-tuning begins in `train.py`.