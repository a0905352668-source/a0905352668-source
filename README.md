# JianKong Surveillance Anti-Filming Project

This repository tracks the code and configuration for the JianKong surveillance anti-filming project.

Tracked content:

- `01_algorithms/`: inference, TensorRT/C++ pipeline, dashboard, pre-labeling, training utilities, and tests.
- `02_configs/`: camera/screen calibration configuration.
- `00_README/`: project notes.
- `07_models/MODEL_REGISTRY.md`: model registry and provenance notes only.

Not tracked:

- raw videos and extracted frames;
- LabelMe/YOLO datasets;
- training runs and dashboard outputs;
- model weights, TensorRT engines, ONNX files, compiled binaries, and cache files.

The runtime model files and data remain on the Ubuntu server under `/media/boshi/Data/JianKong/`.
