## Implementation of Deep Learning-based DIP

This repository is TianYu Li's implementation of Assignment_02(deep learning-based DIP) of DIP.  

## Requirements

To install requirements:

```setup
pip install -r requirements.txt
```

To download datasets,run:

```bash
bash download_facades_dataset.sh
```

## Training

To train the model, run:

```train
python train.py
```

## Pre-trained Models

Download pix2pix_model_epoch_800.pth:

## Results

Our model achieves the following performance on :

### [Facades Datatset]((https://cmp.felk.cvut.cz/~tylecr1/facade/))

| Model name         | Training Loss  | Validation Loss |
| ------------------ |---------------- | -------------- |
| My awesome model   |     0.1021         |      0.4122       |


## Acknowledgement

>📋 Thanks for the algorithms proposed by [Paper: Fully Convolutional Networks for Semantic Segmentation](https://arxiv.org/abs/1411.4038).
