# IIS-Python
This repository provides a Image Inference Service (IIS) developed in Python using Fast API. It classifies images using ONNX Runtime and can return the top k predictions of the pretrained model. Unity tests and a deployable Docker are available.

# Design Decisions

Image preprocessing library:

The Pillow library was chosen for image preprocessing. Altough using OpenCv is possible and maybe faster, it is not a library designed primarily for image resizing as Pillow is. Preliminary tests shown that the difference in performance is minimal. It would decrease slightly the preprocessing time and also the model confidence.

Image resize method:

The resize method used in the image preprocessing stage is BILINEAR. When compared to a more complex method (LANCZOS), it decreases the average preprocessing time by 25%, and the confidence levels are similar. This could be parametrized if needed, but I consider this choice to be optimal for a general use case.

Async vs sync calls:

The main endpoint, infer, was made sync so that the CPU-bound operations (image processing, model inference) do not block the main thread. This allows the API to handle multiple requests concurrently and improve the overall performance.