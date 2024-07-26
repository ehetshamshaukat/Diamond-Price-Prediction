# Project: Diamond Price Prediction
## Problem statement
```
To predict the price of diamond 
```
## Description
### 1. Dataset
```
Dataset available on kaggle 
```

### 2. Features
``` 
Input features = [carat,depth,table,cut,color,clarity]
Target feature = [price]
```
### 3. Pipeline Structure 
```requirements
Google define pipeline 
```
# Requirements
### 1. Language
```
Python 3.10
```
### 2. Libraries
```
1. numpy
2. pandas
3. scikit-learn
4. pickle
5. os 
6. streamlit 
 ```
# code
### 1. Enviroment
```requirements
conda create -p venv python==3.10 -y 
```
### 2.setup
```
The setup.py is a Python script typically included with Python-written libraries or apps. Its objective is to ensure that the program is installed correctly. 
```
### 3. Components
- Data ingestion
```
reading data from different source and splitting data into train and test
```
- Data transformation
```
  reading train and test dataset and apply different transformation and save transformation setting in pickle format
```
- Model training
```requirements
transformed dataset and using different machine learning model and save the best model in pickle format
```
### 4. Pipeline
- Training pipeline
```
using components and creating pipeline for model training
```
- Prediction pipeline
```
taking data from user transform for model and predict 
```

## Run
#### 1. Download repository
```
git clone https://github.com/ehetshamshaukat/Diamond-Price-Prediction.git
```
#### 2. Install dependences
```requirements
pip install -r requirements.txt
```
#### 3. Streamlit
```
streamlit run application.py
```
## Deployment
```
Deploy on AWS using Github actions which is CI CD technique
```