
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

Note: You need to replace relative files(drag_widget.py and visualizer_drag.py) in Draggan folder with these in this folder before running. 

To run Draggan+facial_alignment, run:

```setup
scripts/gui.bat
```

or

```setup
scripts/gui.sh
```

## Results

<img src="tinywow_Video_2024-11-29_141754_70301211" alt="alt text" width="800">
see  also "Video_2024-11-29_141754"
