import os, sys, time
import apiConfig as config
import onnxruntime as ort
import image
import inference
from fastapi import HTTPException, UploadFile


def loadModel(exit_on_error=True):
    ort_session = None
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
            if exit_on_error:
                sys.exit(1)
            raise e
    else:
        print(f"Model file not found at {model_path}.")
        if exit_on_error:
            print("Stopping initialization.")
            sys.exit(1)
        raise FileNotFoundError(f"Model file not found at {model_path}")
    
    return ort_session

def loadLabels():
    
    labels_path = os.path.join(os.path.dirname(__file__), "..","classes", "imagenet_classes.txt")
    if os.path.exists(labels_path):
        with open(labels_path, "r") as f:
            return [line.strip() for line in f.readlines()]
    else:
        print(f"Labels file not found at {labels_path}")

def inferenceEndpoint(file : UploadFile, ort_session, imagenet_labels, pred_num):
    start_time = time.perf_counter()
    try:
        contents = file.file.read()
        input_data = image.PreprocessImage(contents)
    except Exception as e:
        raise HTTPException(status_code=400, detail=f"Error processing image: {str(e)}")
   

    try:
        results = inference.infer(input_data, ort_session, imagenet_labels, pred_num)
        end_time = time.perf_counter()
        inference_time_ms = (end_time - start_time) * 1000
        
        return {
            "predictions": results,
            "inference_time_ms": inference_time_ms
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Inference failed: {str(e)}")