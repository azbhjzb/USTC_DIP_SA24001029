
## Implementation of Deep Learning-based DIP

This repository is TianYu Li's implementation of Assignment_03(incremental work) of DIP.  

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

## Results

Our model achieves the following performance on :

### [Facades Dataset]((https://cmp.felk.cvut.cz/~tylecr1/facade/))

| Model name         | Training Loss  | Validation Loss |
| ------------------ |---------------- | -------------- |
| model 1  |     0.1         |      0.3       |

Train result:

<img src="comparison_0.png" alt="alt text" width="800">

Validation result:

<img src="comparison_3.png" alt="alt text" width="800">

## Acknowledgement

>📋 Thanks for the algorithms proposed by [Paper: Fully Convolutional Networks for Semantic Segmentation](https://arxiv.org/abs/1411.4038).

