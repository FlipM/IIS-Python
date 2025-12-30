import os, sys, threading
from contextlib import asynccontextmanager
from fastapi import FastAPI, File, UploadFile, HTTPException

sys.path.append(os.path.join(os.path.dirname(__file__), "methods"))
import apiConfig as config
import apiLogic as logic

# Singleton vars
ort_session = None
model_lock = threading.Lock()
pred_num = int(os.environ.get(config.PRED_NUM, config.DEFAULT_PRED_NUM))
imagenet_labels = []

@asynccontextmanager
async def lifespan(app: FastAPI):
    # Load the ONNX model at startup
    global ort_session
    with model_lock:
        ort_session = logic.loadModel()
        
    # Load ImageNet labels
    global imagenet_labels
    imagenet_labels = logic.loadLabels()
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

@app.post("/reload")
async def reload_model():
    global ort_session
    try:
        new_session = logic.loadModel(exit_on_error=False)
        with model_lock:
            ort_session = new_session
        return {"message": "Model reloaded successfully"}
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to reload model: {str(e)}")

@app.post("/infer")
def infer(file: UploadFile = File(...)):
    # Use a local reference to the session to stay thread-safe during reload
    current_session = ort_session
    if not current_session:
        raise HTTPException(status_code=503, detail="Model not loaded on server.")
    
    return logic.inferenceEndpoint(file, current_session, imagenet_labels, pred_num)
