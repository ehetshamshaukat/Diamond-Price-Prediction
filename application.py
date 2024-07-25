import streamlit as st
from src.pipeline.prediction_pipeline import GetFeature,PredictionPipeline


cut_categories = ['Fair', 'Good', 'Very Good','Premium','Ideal']
color_categories = ['D', 'E', 'F', 'G', 'H', 'I', 'J']
clarity_categories = ['I1','SI2','SI1','VS2','VS1','VVS2','VVS1','IF']

carat=st.number_input("please enter carat",value=0)
table=st.number_input("please enter table value",value=0)
depth=st.number_input("please enter depth",value=0)
cut:str=st.selectbox("select cut",cut_categories)
color:str=st.selectbox("select color",color_categories)
clarity:str=st.selectbox("select clarity",clarity_categories)


ok=st.button("predict")

if ok:
    features=GetFeature(carat=carat,depth=depth,table=table,cut=cut,color=color,clarity=clarity)
    feature=features.to_dataframe()

    pp=PredictionPipeline()
    output=pp.prediction(feature)
    st.subheader(float(output))


