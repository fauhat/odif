from fastapi import FastAPI
from dif_invoker import invoke_dif
#import main_2
from forest_builder_main import data_preprocessing_local
from forest_builder import ForestBuilder
import pandas as pd
import numpy as np

api = FastAPI()

@api.get("/")
def root():
    return {"info": "DIF Instance"}

@api.get("/dif-run-time/")
def get_dif_runtime(ensemblesize: int = 20):
    runtimes = invoke_dif(ensemblesize)
    return {"runtimes" : [{"runtime": runtimes[0]}, {"runtime": runtimes[1]}]}

@api.get("/create-forest/")
def get_dif_runtime(ensemblesize: int = 3):
    #runtimes = invoke_dif(ensemblesize)
    df = pd.read_csv("data\\tabular\\cricket.csv")
    x, y = data_preprocessing_local(df) 
    ForestBuilder(n_estimators=3,max_samples=16,random_state=49,x=x)
    return {"status": "success"}