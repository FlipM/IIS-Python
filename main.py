import os, time, sys
from contextlib import asynccontextmanager
from fastapi import FastAPI, File, UploadFile, HTTPException
import onnxruntime as ort
import apiConfig as config
import image, inference

# Global vars
ort_session = None
pred_num = int(os.environ.get(config.PRED_NUM, config.DEFAULT_PRED_NUM))
imagenet_labels = []

def load_labels(path):
    with open(path, "r") as f:
        return [line.strip() for line in f.readlines()]

@asynccontextmanager
async def lifespan(app: FastAPI):
    # Load the ONNX model at startup
    global ort_session
    model_path = os.environ.get(config.MODEL_PATH)
    print(f"Loading model from environment variable {config.MODEL_PATH}: {model_path}")
    
    if model_path and os.path.exists(model_path):
        try:
            sess_options = ort.SessionOptions()
            num_threads = os.environ.get(config.NUM_THREADS)
            if num_threads:
                sess_options.intra_op_num_threads = int(num_threads)
            
            ort_session = ort.InferenceSession(model_path, sess_options)
            print("Model loaded successfully.")
        except Exception as e:
            print(f"Error loading model: {e}")
            ort_session = None
    else:
        print(f"Model file not found at {model_path}. Stopping initialization.")
        ort_session = None
        sys.exit(1)
        
    # Load ImageNet labels
    global imagenet_labels
    labels_path = os.path.join(os.path.dirname(__file__), "classes", "imagenet_classes.txt")
    if os.path.exists(labels_path):
        imagenet_labels = load_labels(labels_path)
        print(f"Loaded {len(imagenet_labels)} labels.")
    else:
        print(f"Labels file not found at {labels_path}")
        
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
    
    start_preprocess = time.perf_counter()
    try:
        contents = file.file.read()
        input_data = image.PreprocessImage(contents)
    except Exception as e:
        raise HTTPException(status_code=400, detail=f"Error processing image: {str(e)}")
   
    end_preprocess = time.perf_counter()
    preprocess_time_ms = (end_preprocess - start_preprocess) * 1000

    try:
        start_inference = time.perf_counter()
        results = inference.infer(input_data, ort_session, imagenet_labels, pred_num)
        end_inference = time.perf_counter()
        inference_time_ms = (end_inference - start_inference) * 1000
        
        return {
            "predictions": results,
            "preprocess_time_ms": preprocess_time_ms,
            "inference_time_ms": inference_time_ms
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Inference failed: {str(e)}")
