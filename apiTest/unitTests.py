import os, sys
sys.path.append(os.path.join(os.path.dirname(__file__), "..", "methods"))
import apiConfig as config

def testConversion():
    from image import PreprocessImage, DEFAULT_IMAGE_SIZE
    # Test the goldfish
    with open(os.path.join(os.path.dirname(__file__), "../examples", "gf.jpg"), "rb") as img_GF:

        gf = img_GF.read()
        processed_gf = PreprocessImage(gf, target_size=DEFAULT_IMAGE_SIZE)
        assert processed_gf.shape == (1, 3, DEFAULT_IMAGE_SIZE[1], DEFAULT_IMAGE_SIZE[0]), "Processed image has incorrect shape"
        yield processed_gf

    # Test the car
    with open(os.path.join(os.path.dirname(__file__), "../examples", "car.jpg"), "rb") as img_GF:

        car = img_GF.read()
        processed_car = PreprocessImage(car, target_size=DEFAULT_IMAGE_SIZE)
        assert processed_car.shape == (1, 3, DEFAULT_IMAGE_SIZE[1], DEFAULT_IMAGE_SIZE[0]), "Processed image has incorrect shape"
        yield processed_car
    


def testInference(inputs):
    from apiLogic import loadModel
    from inference import infer
    import onnxruntime as ort
    
    # Mock labels
    imagenet_labels = []   
    labels_path = os.path.join(os.path.dirname(__file__), "../classes/imagenet_classes.txt")
    if os.path.exists(labels_path):
        print("Labels file found.")
        with open(labels_path, "r") as f:
            imagenet_labels = [line.strip() for line in f.readlines()]
    
    # Load a simple ONNX model for testing (assuming a model file 'test_model.onnx' exists)
    ort_session = ort.InferenceSession(os.environ.get(config.MODEL_PATH))
    print("Loaded model successfully.")

    # Run inference
    pred_num = 5
    resList = []
    for input_data in inputs:
        results = infer(input_data, ort_session, imagenet_labels, pred_num)
        # Check the number of predictions returned
        assert len(results) == pred_num, "Number of predictions returned is incorrect"
        
        # Check the structure of each prediction
        for res in results:
            assert "label" in res and "confidence" in res, "Prediction result structure is incorrect"
            assert isinstance(res["label"], str), "Label should be a string"
            assert isinstance(res["confidence"], float), "Confidence should be a float"

        resList.append(results)

    n = len(resList)
    if resList[0][0]["label"] != "goldfish":
        raise AssertionError("Goldfish was not identified correctly.")
    
    if resList[1][0]["label"] != "sports car":
        raise AssertionError("Sports car was not identified correctly.")

if __name__ == "__main__":
    inputs = testConversion()
    print("Image conversion tests passed.")

    testInference(inputs)   
    print("All tests passed.")