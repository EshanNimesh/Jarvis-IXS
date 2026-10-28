# Kokoro ONNX model files

Place these files in this directory before starting the client:

- `kokoro-v1.0.onnx` — [download from the Kokoro ONNX release](https://github.com/thewh1teagle/kokoro-onnx/releases/download/model-files-v1.1/kokoro-v1.0.onnx)
- `voices-v1.0.bin` — [download the voice embeddings](https://github.com/thewh1teagle/kokoro-onnx/releases/download/model-files-v1.1/voices-v1.0.bin)

The client loads both files once when `TTSService` starts. Override their locations with `KOKORO_MODEL_PATH` and `KOKORO_VOICES_PATH` if needed. Model files are kept outside source control because the ONNX weights are large.
