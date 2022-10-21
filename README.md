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

- Step1: data preparation 

data link (mixed dataset):
> NL Code Search:https://drive.google.com/file/d/1pV9-qQuSCk2OeA7efBuVd4EIRIhz_NAE/view?usp=sharing


> XL Code Search:coming soon...

other single data link:

https://github.com/reddy-lab-code-research/XLCoST

Download the data folders, unzip them, and put them directly in the 'nl2codesearch/dataset/program_level/' or 'code2codesearch/dataset/program_level/' directory.

- Step2: train

  ```
  bash run_code_search.sh 0 all nl2code program codebert train
  ```

- Step3: test
  
  ```
  bash run_code_search.sh 0 java code2code program codebert eval
  ```


Parameter setting

- GPU: 0
- Language: all
- task: nl2code
- model: codebert
- type: train
