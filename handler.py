import runpod
import os
import uuid
import json
import base64
import shutil
import subprocess
from run_animecolor import run_animecolor

def decode_base64_to_file(base64_str, target_path):
    with open(target_path, "wb") as f:
        f.write(base64.b64decode(base64_str))

def handler(job):
    job_id = str(uuid.uuid4())
    base_path = "/workspace"
    input_dir = f"{base_path}/inputs/{job_id}"
    output_dir = f"{base_path}/outputs/{job_id}"
    
    try:
        job_input = job["input"]

        # =============================
        # SETUP CHECK (Download weights if missing)
        # =============================
        weights_check = os.path.join(base_path, "pretrained_weights/cogvideox-fun-base")
        if not os.path.exists(weights_check) or not os.listdir(weights_check):
            print("Weights missing. Running setup.py...")
            subprocess.run(["python3", "setup.py"], check=True)

        os.makedirs(input_dir + "/lineart", exist_ok=True)
        os.makedirs(output_dir, exist_ok=True)

        # =============================
        # INPUT HANDLING (Base64)
        # =============================
        
        # 1. Reference Image
        ref_path = os.path.join(input_dir, "ref.png")
        if "ref_image" in job_input:
            decode_base64_to_file(job_input["ref_image"], ref_path)
        else:
            raise ValueError("ref_image (base64) is required")

        # 2. Lineart Frames
        if "lineart_frames" in job_input and isinstance(job_input["lineart_frames"], list):
            for i, frame_b64 in enumerate(job_input["lineart_frames"]):
                frame_path = os.path.join(input_dir, f"lineart/frame_{i:05d}.png")
                decode_base64_to_file(frame_b64, frame_path)
        else:
            raise ValueError("lineart_frames (list of base64) is required")

        config = {
            "base_path": base_path,
            "src_dir": "AnimeColor_Code",
            "ckpt_dir": "pretrained_weights/animecolor-weights",
            "base_model_dir": "pretrained_weights/cogvideox-fun-base",
            "radio_dir": "pretrained_weights/radio-model",

            "lineart_dir": os.path.join(input_dir, "lineart"),
            "ref_image": ref_path,
            "output_dir": output_dir,

            "start_frame": job_input.get("start_frame", 0),
            "num_frames": len(job_input["lineart_frames"]),
            "width": job_input.get("width", 512),
            "height": job_input.get("height", 320),
            "output_fps": job_input.get("output_fps", 24),
            "guidance_scale": job_input.get("guidance_scale", 8.5),
            "inference_steps": job_input.get("inference_steps", 25)
        }

        # =============================
        # RUN MODEL
        # =============================
        result = run_animecolor(config)

        # Encode output video to base64
        output_video_path = result["output_video"]
        with open(output_video_path, "rb") as f:
            video_base64 = base64.b64encode(f.read()).decode("utf-8")

        return {
            "status": "success",
            "job_id": job_id,
            "video_base64": video_base64,
            "video_size": os.path.getsize(output_video_path),
            "metadata": {
                "frames": result["frames"],
                "duration": result["duration"]
            }
        }

    except Exception as e:
        return {
            "status": "error",
            "message": str(e)
        }
    finally:
        # CLEANUP
        if os.path.exists(input_dir):
            shutil.rmtree(input_dir)
        if os.path.exists(output_dir):
            shutil.rmtree(output_dir)

# Start worker
runpod.serverless.start({"handler": handler})