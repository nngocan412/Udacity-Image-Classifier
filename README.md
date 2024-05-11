# AI Programming with Python Project

Project code for Udacity's AI Programming with Python Nanodegree program. In this project, students first develop code for an image classifier built with PyTorch, then convert it into a command line application.

# NGUYEN NGOC AN

# How to run

## 1. Install Anacoda

Go to [anaconda download page](https://www.anaconda.com/download/) and download the installer to install.

## 2. Create a python environment

```script
  conda create -n python_3 python=3.6
```

## 3. Active pyhton environment

```script
  conda activate python_3
```

## 4. Install packages

```script
  pip install torchvision numpy pandas matplotlib seaborn argparse torch
```

## 5. Download flowers training data

Go to [this link](https://s3.amazonaws.com/content.udacity-data.com/nd089/flower_data.tar.gz) to download the flowers training data.

Extract the data to the flowers folder in the root folder

The folder structure will be like this

```
root-folder/
|- assets/
  |- Flowers.png
  |- inference_example.png
|- flowers/
  |- test/
  |- train/
  |- valid/
|- .gitignore
|- cat_to_name.json
|- Image Classifier Project.ipynb
|- LICENSE
|- predict.py
|- README.md
|- train.py
|- workspace-utils.py
```

## 6. Run notebook

Open the notebook file `Image Classifier Project.ipynb` and click on the `Run All` button

## 7. Training model

```script
  python train.py ./flowers
```

## 8. Run predict

```script
  python predict.py
```
