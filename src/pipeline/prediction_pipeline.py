import os
from src.utils import load_pickle_file
import pandas as pd


class PredictionPipeline:
    def __init__(self):
        pass
    def prediction(self, feature):
        try:
            preprocessing_path = os.path.join('artifacts/pickle', 'data_transformation.pkl')
            model_path = os.path.join('artifacts/pickle', 'model.pkl')

            preprocessing = load_pickle_file(preprocessing_path)
            model = load_pickle_file(model_path)

            scaled_data = preprocessing.transform(feature)

            result = model.predict(scaled_data)

            return result
        except Exception as e:
            raise e


class GetFeature:
    def __init__(self, carat:float, table:float, color:str, cut:str, x:float, y:float, z:float, depth:float, clarity:str):
        self.carat = carat
        self.table = table
        self.depth = depth
        self.x = x
        self.y = y
        self.z = z
        self.color = color
        self.clarity = clarity
        self.cut = cut

    def to_dataframe(self):
        features_as_dict= {
            "carat": [self.carat],
            "depth": [self.depth],
            "table": [self.table],
            "x": [self.x],
            "y": [self.y],
            "z": [self.z],
            "cut": [self.cut],
            "color": [self.color],
            "clarity": [self.clarity]
        }
        df = pd.DataFrame(features_as_dict)
        print(df)
        return df
