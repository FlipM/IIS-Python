import threading
import requests
import time
import os

BASE_URL = "http://localhost:8000"
EXAMPLES_DIR = "/home/felipe/Prog/IIS-Python/examples"
IMAGES = ["car.jpg", "tree.jpg"]

def call_infer(image_name):
    # print(f"Starting inference for {image_name}")
    path = os.path.join(EXAMPLES_DIR, image_name)
    with open(path, "rb") as f:
        files = {"file": f}
        start = time.perf_counter()
        response = requests.post(f"{BASE_URL}/infer", files=files)
        end = time.perf_counter()
        print(f"Inference for {image_name} took {end-start:.3f}s, status: {response.status_code}")

def call_health(label):
    time.sleep(0.1)  # small delay to hit it while inference is running
    # print(f"Calling health {label}")
    start = time.perf_counter()
    response = requests.get(f"{BASE_URL}/health")
    end = time.perf_counter()
    print(f"Health check {label} took {end-start:.3f}s, status: {response.status_code}")

if __name__ == "__main__":
    threads = []
    
    # 2 inference calls
    for img in IMAGES:
        t = threading.Thread(target=call_infer, args=(img,))
        threads.append(t)
    
    # 2 health calls in between
    for i in range(2):
        t = threading.Thread(target=call_health, args=(f"#{i+1}",))
        threads.append(t)

    for t in threads:
        t.start()
    
    for t in threads:
        t.join()
