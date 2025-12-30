# IIS-Python
This repository provides a Image Inference Service (IIS) developed in Python using Fast API. It classifies images using ONNX Runtime and can return the top k predictions of the pretrained model. Unity tests and a deployable Docker are available.

# Setup

Linux setup:

    The repository comes with a virtual environment ready to use for linux systems. The user only needs to activate it:

    source venv/bin/activate

    Once inside the virtual environment, the launch commands can be used normally. The environemnt variables are added to the 'activate' script. If they need to be changed, it can be directly done in the script and, after reloading the virtual environment, changes will take place. The variables can also be manually changed via the 'export' command. The relevant environment variables for the API are:

    MODEL_PATH=model/resnet50-v2-7.onnx
    TOP_K=3
    NUM_THREADS=4

Windows setup: 

    The virtual environment from the repository will not work in windows. The user needs to create a new virtual environment and install the required libraris using the requirements.txt file, with the command:

    pip install -r requirements.txt

    As an alternative, the user can install all the requirements outside the virtual environment, but that is not recommended.

    Regarding the environment variables, they must be added directly into the system. To learn how to add environment variables to the system, please refer to https://imatest.atlassian.net/wiki/spaces/KB/pages/12049809418/Editing+System+Environment+Variables or any other trusted guide.

Running the API:

    The API can be started using the following command:

    uvicorn main:app 

    The API will then be available at http://localhost:8000 until the server is closed.
    To use any endpoint, user has multiple options:
    - Use the interactive API documentation at http://localhost:8000/docs
    - Use the interactive API documentation at http://localhost:8000/redoc
    - Use the curl command line tool
    - Use any HTTP client, such as Postman

Tests:

    With the API running, the unit tests for the image and model inference methods can be run using the following command:

    python3 apiTests/unitTests.py

    On the other hand, the API endpoint can be tested using the following command:

    python3 apiTests/testCurrency.py

# API examples:

    The images in examples can be used to test the API. The infer endpoint can be tested using the following command on the linux terminal:

    curl -X POST http://localhost:8000/infer -F "file=@examples/car.jpg"

    Which should yield the following response:

    {"predictions":[{"label":"sports car","confidence":0.964098334312439},{"label":"convertible","confidence":0.01698175258934498},{"label":"racer","confidence":0.01084537710994482}],"inference_time_ms":379.8530270032643}

    Other images can be tested using the same command, just replace the file path with the path to the image.

    The other relevant endpoints that can be accessed are:

    - GET /health

        Returns if the model is properly loaded.

    - POST /reload

        Reload the model without needing to shutdown the API.

    For additional access options, please refer to the "Running the API" section above.

# Design Decisions

Image preprocessing library:

    The Pillow library was chosen for image preprocessing. Altough using OpenCv is possible and maybe faster, it is not a library designed primarily for image resizing as Pillow is. Preliminary tests shown that the difference in performance is minimal. It would decrease slightly the preprocessing time and also the model confidence.

Image resize method:

    The resize method used in the image preprocessing stage is BILINEAR. When compared to a more complex method (LANCZOS), it decreases the average preprocessing time by 25%, and the confidence levels are similar. This could be parametrized if needed, but I consider this choice to be optimal for a general use case.

Async vs sync calls:

    The main endpoint, infer, was made sync so that the CPU-bound operations (image processing, model inference) do not block the main thread. This allows the API to handle multiple requests concurrently and improve the overall performance.

Extra task:

    I added the "reload" endpoint to the API, which allows the model to be reloaded without needing to shutdown the API. This allows the user to change models without shutting down the API and interrupting the users. I believe it is a useful feature for production environments.

# Future Improvements

    If more time is available, I would like to implement the following improvements:
    - Implement security measures, like authentication and autorizathion, which were not mentioned in the requirements.
    - Add a docker container to the repository, which would allow the API to be run with minimal setup in any environment. 
    - Add database support to the API, which would allow the API to store useful information such as logs, metrics, etc.
