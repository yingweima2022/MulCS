# MulCS: Towards a unified Deep Representation for Multilingual Code Search

<p align="center">   
    <a href="https://pytorch.org/" alt="PyTorch">
      <img src="https://img.shields.io/badge/PyTorch-%23EE4C2C.svg?e&logo=PyTorch&logoColor=white" /></a>
<p/>



An official source code for paper MulCS: Towards a unified Deep Representation for Multilingual Code Search.

-------------

### Requirements

The proposed MulCS is implemented with python 3.7.11 on a NVIDIA Tesla V100 GPU. 


- torch==1.5.0
- tqdm==4.62.2
- numpy==1.19.5
- scikit_learn==0.24.2



### Quick Start

- Step1: train

  ```
  python train.py --data_path ./data/ --model IREmbeder
  ```

- Step2: test
  
  ```
  python test.py --data_path ./data/ --model IREmbeder  --reload_from 100
  ```


Parameter setting

- data_path: the path of dataset
- model: the name of the model.
- reload_from: checkpoint for testing
- lr: learning rate(1e-3)
