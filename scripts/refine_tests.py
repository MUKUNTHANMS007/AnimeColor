import base64
import json
import os

def refine_tests_json():
    base_dir = "d:/Animation-Project/AnimeColor-Worker"
    ref_path = os.path.join(base_dir, "inputs/ref.png")
    lineart_path = os.path.join(base_dir, "inputs/lineart/frame_0000.png")
    
    if not os.path.exists(ref_path) or not os.path.exists(lineart_path):
        print("Error: Reference or lineart images missing in inputs/")
        return

    with open(ref_path, "rb") as f:
        ref_b64 = base64.b64encode(f.read()).decode("utf-8")
    
    with open(lineart_path, "rb") as f:
        lineart_b64 = base64.b64encode(f.read()).decode("utf-8")

    tests_data = {
        "tests": [
            {
                "name": "basic-colorization-test",
                "input": {
                    "ref_image": ref_b64,
                    "lineart_frames": [lineart_b64],
                    "width": 512,
                    "height": 320,
                    "inference_steps": 10
                },
                "timeout": 1200000 
            }
        ],
        "config": {
            "gpuTypeId": "NVIDIA L4",
            "gpuCount": 1,
            "allowedCudaVersions": ["12.1", "12.2", "12.3", "12.4", "12.5", "12.6", "12.7"]
        }
    }

    output_path = os.path.join(base_dir, ".runpod/tests.json")
    with open(output_path, "w") as f:
        json.dump(tests_data, f, indent=2)
    
    print(f"Successfully refined {output_path}")

if __name__ == "__main__":
    refine_tests_json()
