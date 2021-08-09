# Hypernym-and-Hyponym-Detection-Based-on-Auxiliary-Sentences-and-the-BERT-Model
# Introduction 
We utilize English BERT pre-trained model and auxiliary sentences for Hypernym and Hyponym Detection.
# Run
## Training Task1
first, go to the path
```
cd code
cd train
```
use
```
$python task1_BERT.py
```
or 
```
$python task1_Q.py
```
or
```
$python task1_PosNeg.py
```
to train task1 model with default parameters, or use
```

$python task1_BERT.py -dataset        <path to training set, default to WordNet training set> 
                      -label          <training set with label or not> 
                      -b              <batch size, default 64> 
                      -epoch          <number of epochs, default 19> 
                      -lr             <learning rate, default 1e-5> 
                      -model          <name of model default save to'output/task1_BERT_epoch.pth'> 
```                      
to train with your own parameters.
  
## Training Task2
first, go to the path
```
cd code
cd train
```
use
```
$python task2_BERT.py
```
or 
```
$python task2_Q.py
```
or
```
$python task2_PosNeg.py
```
to train task2 model with default parameters, or use
```
$python task2_BERT.py -dataset        <path to training set, default to WordNet training set> 
                      -label          <training set with label or not> 
                      -b              <batch size, default 64> 
                      -epoch          <number of epochs, default 19> 
                      -lr             <learning rate, default 1e-5> 
                      -model          <name of model default save to'output/task1_BERT_epoch.pth'> 
```                      
to train with your own parameters.
  
# Evaluation
first, go to the path
```
cd code
cd evaluate
```
## Task1 Evaluation
use
```
$python eval_task1_BERT.py -model_path MODEL_PATH    <path to model>
```
```
$python eval_task1_Q.py -model_path MODEL_PATH    <path to model>
```
```
$python eval_task1_PosNeg.py -model_path MODEL_PATH    <path to model>
```
to evaluate task1 model.

## Eask2 and Eask1&2 Evaluation
use
```
$python eval_task1_2_BERT.py -model1_path <MODEL1_PATH>
                             -model2_path <MODEL2_PATH>
```
```
$python eval_task1_2_Q.py -model1_path <MODEL1_PATH>
                          -model2_path <MODEL2_PATH>
```
```
$python eval_task1_2_PosNeg.py -model1_path <MODEL1_PATH>
                               -model2_path <MODEL2_PATH>
```
to evaluate task2 and task1&2.
# Datasets
see more in [training dataset README](https://github.com/ncu-dart/Hypernym-and-Hyponym-Detection-Based-on-Auxiliary-Sentences-and-the-BERT-Model/blob/main/data/README.md)
