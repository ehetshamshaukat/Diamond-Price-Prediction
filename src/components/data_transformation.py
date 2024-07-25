import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler,OrdinalEncoder
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from dataclasses import dataclass
import os
from src.utils import save_file_as_pickle

@dataclass
class DataTransformationConfig:
    data_transformation_pickle_file=os.path.join('artifacts/pickle','data_transformation.pkl')

class DataTransformation:
    def __init__(self):
        self.data_transformation_config=DataTransformationConfig()
    def transforming_data(self):
        try:
            numerical_columns=['carat','table','depth']
            categorical_columns=['cut','color','clarity']

            cut_categories = ['Fair', 'Good', 'Very Good','Premium','Ideal']
            color_categories = ['D', 'E', 'F', 'G', 'H', 'I', 'J']
            clarity_categories = ['I1','SI2','SI1','VS2','VS1','VVS2','VVS1','IF']

            numerical_columns_pipeline = Pipeline(steps=[
                ('impute',SimpleImputer(strategy='median')),
                ('standardscaler',StandardScaler())
            ])
            categorical_columns_pipeline = Pipeline(steps=[
                ('impute',SimpleImputer(strategy='most_frequent')),
                ('ordinalencoder',OrdinalEncoder(categories=[cut_categories,color_categories,clarity_categories])),
                ('standardscaler',StandardScaler())
            ])
            transform_data=ColumnTransformer(transformers=[
                ('numerical_column_pipeline',numerical_columns_pipeline,numerical_columns),
                ('categorical_columns_pipeline',categorical_columns_pipeline,categorical_columns)
            ])
            return transform_data
        except Exception as e:
            print(e)


    def initiate_data_transformation(self,train_dataset_path,test_dataset_path):
        try:
            train_dataset=pd.read_csv(train_dataset_path)
            test_dataset=pd.read_csv(test_dataset_path)

            transform_data = self.transforming_data()

            target_column = 'price'
            columns_to_drop = ["x","y","z",'price']

            xtrain = train_dataset.drop(columns=columns_to_drop,axis=1)
            ytrain = train_dataset[target_column]

            xtest = test_dataset.drop(columns=columns_to_drop,axis=1)
            ytest = test_dataset[target_column]

            xtrain_processed_data_arr = transform_data.fit_transform(xtrain)
            xtest_processed_data_arr = transform_data.transform(xtest)

            train_processed_dataset_arr = np.c_[xtrain_processed_data_arr , np.array(ytrain)]
            test_processed_dataset_arr = np.c_[xtest_processed_data_arr , np.array(ytest)]

            save_file_as_pickle(self.data_transformation_config.data_transformation_pickle_file,transform_data)

            return (train_processed_dataset_arr, test_processed_dataset_arr)

        except Exception as e:
            print(e)