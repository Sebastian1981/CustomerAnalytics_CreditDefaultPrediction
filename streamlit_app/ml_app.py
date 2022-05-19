import joblib

def load_pipeline(PATH, pipeline_name:str):
    with open(PATH / pipeline_name,'rb') as f:
        pipeline = joblib.load(f)
    return pipeline