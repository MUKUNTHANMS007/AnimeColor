# AnimeColor: Reference-based Animation Colorization

[![Runpod](https://api.runpod.io/badge/compilershader-svg/AnimeColor)](https://console.runpod.io/hub/compilershader-svg/AnimeColor)

This is the RunPod serverless worker for AnimeColor. It uses Diffusion Transformers to colorize animation lineart based on a reference image.

## Setup
The worker automatically handles cloning the required model code and downloading pre-trained weights on its first run.

## Repository Structure
- `.runpod/`: Configuration for RunPod Hub.
- `handler.py`: Entry point for RunPod Serverless.
- `setup.py`: Downloads models and weights.
- `run_animecolor.py`: Main inference logic.
