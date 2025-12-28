import os, numpy as np

def infer(input_data, ort_session, imagenet_labels, pred_num):  
    # Run inference and organize the top predictions
    input_name = ort_session.get_inputs()[0].name
    outputs = ort_session.run(None, {input_name: input_data})
    probs = outputs[0][0]
    top_indices = np.argsort(probs)[-pred_num:][::-1]

    # Match predictions with labels from the ImageNet dataset
    results = []
    for idx in top_indices:
        label = imagenet_labels[idx] if idx < len(imagenet_labels) else "unknown"
        results.append({
            "label": label,
            "confidence": float(probs[idx])
        })
        
    return results