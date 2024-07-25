import pandas as pd
from sklearn.model_selection import train_test_split
from dataclasses import dataclass
import os

@dataclass
class DataIngestionConfig:
    train_dataset_path=os.path.join('artifacts/train_test_dataset','train_dataset.csv')
    test_dataset_path=os.path.join('artifacts/train_test_dataset','test_dataset.csv')

class DataIngestion:
    def __init__(self):
        self.data_ingestion_config=DataIngestionConfig()
    def initiate_data_ingestion(self):
        try:
            df = pd.read_csv("dataset/diamond_price.csv")
            os.makedirs(os.path.dirname(self.data_ingestion_config.train_dataset_path),exist_ok=True)
            train_dataset,test_dataset = train_test_split(df,test_size=0.3,random_state=69)
            train_dataset.to_csv(self.data_ingestion_config.train_dataset_path,index=False,header=True)
            test_dataset.to_csv(self.data_ingestion_config.test_dataset_path,index=False,header=True)
            return (self.data_ingestion_config.train_dataset_path,self.data_ingestion_config.test_dataset_path)
        except Exception as e:
            print(e)
