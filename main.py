import os
from contextlib import asynccontextmanager
from fastapi import FastAPI, File, UploadFile, HTTPException
import apiConfig as config
import apiLogic

# Singleton vars
ort_session = None
pred_num = int(os.environ.get(config.PRED_NUM, config.DEFAULT_PRED_NUM))
imagenet_labels = []

@asynccontextmanager
async def lifespan(app: FastAPI):
    # Load the ONNX model at startup
    global ort_session
    ort_session = apiLogic.loadModel()
        
    # Load ImageNet labels
    global imagenet_labels
    imagenet_labels = apiLogic.loadLabels()
    print(f"Loaded {len(imagenet_labels)} labels.")

    yield
    print("Stopping API")

app = FastAPI(lifespan=lifespan)

@app.get("/")
async def root():
    return {"message": "Running ONNX API here"}

@app.get("/health")
async def health():
    return {
        "status": "ok",
        "model_loaded": ort_session is not None
    }

@app.post("/infer")
def infer(file: UploadFile = File(...)):
    if not ort_session:
        raise HTTPException(status_code=503, detail="Model not loaded on server.")
    
    return apiLogic.inferenceEndpoint(file, ort_session, imagenet_labels, pred_num)
