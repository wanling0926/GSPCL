# GSPCL
## Installation
Clone the repository and run
```
conda env create --name GSPCL --file env.yml
conda activate GSPCL
```

## Pretrain and Evaluation
```
python pretrain.py
```
We measure the quality of the learned representations by linear separability.
```
python evaluation.py
```
