
## Implementation of Draggan+facial_alignment

This repository is TianYu Li's implementation of Assignment_03(combination work） of DIP. 

## Requirements

To install draggan:

```setup
git clone https://github.com/XingangPan/DragGAN.git
```

To install facial_alignment:

```setup
pip install face-alignment
```

To setup environment

```setup
conda env create -f environment.yml
conda activate stylegan3
```

To download pretrained model

```setup
python scripts/download_model.py
```

## Running

To run Draggan+facial_alignment, run:

```basic
python basic_transformation.py
```

To run point guided deformation, run:

```point
scripts/gui.bat
```

or

```point
scripts/gui.sh
```

## Results
### Basic Transformation
<img src="basic_transformation.gif" alt="alt text" width="800">

### Point Guided Deformation:
<img src="point_guided_deformation.gif" alt="alt text" width="800">

## Acknowledgement

>📋 Thanks for the algorithms proposed by [Paper: Image Warping by Radial Basis Functions](https://www.sci.utah.edu/~gerig/CS6640-F2010/Project3/Arad-1995.pdf).

