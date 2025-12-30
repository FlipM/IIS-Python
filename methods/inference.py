import numpy as np
from onnxruntime import InferenceSession

def softmax(x):
    x = x.reshape(-1)
    e_x = np.exp(x - np.max(x))
    return e_x / e_x.sum(axis=0)

def infer(input_data, ort_session : InferenceSession , imagenet_labels, pred_num):  
    # Run inference and organize the top predictions
    inputs = ort_session.get_inputs()
    input_name = inputs[0].name
    outputs = ort_session.run(None, {input_name: input_data})
    res = softmax(np.array(outputs)).tolist()
    top_indices = np.flip(np.squeeze(np.argsort(res)))[:pred_num]

    # Match predictions with labels from the ImageNet dataset
    results = []
    for idx in top_indices:
        label = imagenet_labels[idx] if idx < len(imagenet_labels) else "unknown"
        results.append({
            "label": label,
            "confidence": float(res[idx])
        })
        
    return results