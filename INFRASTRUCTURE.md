## General workflow

Training runs on RunPod compute (Pods) with persistent storage on a RunPod network volume. Resulting model is served from the network volume (or huggingface private repository) on a serverless REST API endpoint. 