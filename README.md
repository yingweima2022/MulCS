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



- Step1: data preparation 

data link
> https://drive.google.com/file/d/1_-BcLEerRFA8Ms7d9xUmWr8ai-EC4ZPp/view?usp=sharing

checkpoint link
> https://drive.google.com/file/d/1aacga6uakq_PNVFiwA49maovb9lYaSA9/view?usp=sharing

Download the data and checkpoint folders, unzip them, and put them directly in the home directory.


- Step2: train

  ```
  python train.py --data_path ./data/ --model IREmbeder
  ```

- Step3: test

  ```
  python test.py --data_path ./data/ --model IREmbeder  --reload_from 100
  ```


Parameter setting

- data_path: the path of dataset
- model: the name of the model.
- reload_from: checkpoint for testing
- lr: learning rate(1e-3)
