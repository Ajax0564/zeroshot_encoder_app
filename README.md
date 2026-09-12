

https://github.com/user-attachments/assets/e7ed0134-f885-45df-8f0f-105a2cd9a9d7



https://github.com/user-attachments/assets/0ebdc2b3-9b32-4568-b887-59b839f5f559

# GLiNER 2.5 Inference App

This project provides a local text-processing service built around
[GLiNER 2.5](https://huggingface.co/fastino/gliner2.5-base-v1). It exposes a
FastAPI backend and a Gradio web interface for:

- Zero-shot single-label and multi-label text classification
- Zero-shot named entity extraction with confidence scores and spans
- Redis-backed caching for repeated requests
- Interactive inspection of formatted results and raw JSON responses

The model included in this repository is loaded locally at startup, so the
application does not need to download model weights when the checked-in model
directory is present.

## Requirements

- Python 3.12 or newer
- Memurai running on `localhost:6379` (or a custom `REDIS_URL`)
- The repository's model files under
	`src/gliner_app/model/fastino_gliner2.5-base-v1/`
- A CUDA-enabled PyTorch installation to use an NVIDIA GPU; otherwise the
	application automatically uses the CPU

The first API startup loads the model and may take a little time. Keep the API
process running while using the web interface.

## Run From a Clone

Clone the repository and enter the project directory:

```bash
git clone <repository-url>
cd glinear_app
```

Create and activate a virtual environment:

```bash
python -m venv .venv
```

On macOS/Linux:

```bash
source .venv/bin/activate
```

On Windows PowerShell:

```powershell
.\.venv\Scripts\Activate.ps1
```

Install the project and its dependencies:

```bash
python -m pip install --upgrade pip
python -m pip install -e .
```

Install [Memurai](https://www.memurai.com/get-memurai) on Windows and make
sure the Memurai service is running before starting the application. Memurai
uses the Redis protocol and listens on `localhost:6379` by default, which
matches this project's default configuration.

If Memurai is running on another host or port, set the connection URL before
starting the API:

```bash
# macOS/Linux
export REDIS_URL=redis://localhost:6379/0

# Windows PowerShell
$env:REDIS_URL = "redis://localhost:6379/0"
```

The cache lifetime defaults to 3600 seconds. Override it with `CACHE_TTL` if
needed:

```powershell
$env:CACHE_TTL = "1800"
```

### GPU support

At startup, the API checks `torch.cuda.is_available()`. If it returns `True`,
GLiNER loads on CUDA; otherwise it loads on the CPU. No application setting is
needed to switch between the two.

The default PyTorch package may be CPU-only. To enable NVIDIA CUDA support,
install the CUDA-enabled PyTorch build appropriate for your driver from the
[official PyTorch installation selector](https://pytorch.org/get-started/locally/).
Verify it before starting the API:

```powershell
python -c "import torch; print(torch.cuda.is_available()); print(torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'CPU')"
```

The API logs `Using device: cuda` or `Using device: cpu` during startup.

## Start the API

From the project root, run:

```bash
python -m uvicorn gliner_app.main:app --reload
```

The API is available at `http://127.0.0.1:8000`.

- OpenAPI documentation: `http://127.0.0.1:8000/docs`
- Health check: `GET /`
- Classification: `POST /classify`
- Entity extraction: `POST /entities`
- Clear application cache: `DELETE /cache`

### Classification example

```bash
curl -X POST http://127.0.0.1:8000/classify \
	-H "Content-Type: application/json" \
	-d '{
		"texts": ["The phone has an excellent camera and battery life."],
		"labels": ["camera", "battery", "price"],
		"task_name": "product_features",
		"is_multilabel": true,
		"include_confidence": true
	}'
```

### Entity extraction example

```bash
curl -X POST http://127.0.0.1:8000/entities \
	-H "Content-Type: application/json" \
	-d '{
		"texts": ["Apple released the new iPhone in California."],
		"labels": ["company", "product", "location"],
		"task_name": "named_entities",
		"include_confidence": true,
		"include_spans": true,
		"threshold": 0.5
	}'
```

Both endpoints accept either one string or a list of strings in `texts`. Labels
can be a list of names or a dictionary of label names to descriptions.

## Start the Web App

Leave the API running, open a second terminal, activate the same virtual
environment, and run:

```bash
python -m gliner_app.ui
```

Open the Gradio playground at `http://127.0.0.1:7860`.

The playground includes three tabs:

- **Classification**: submit one or more texts and custom labels, with optional
	multi-label predictions and confidence scores.
- **Entity Extraction**: submit entity labels and view detected spans highlighted
	in the original text. Adjust the confidence threshold to control results.
- **System Configuration**: check API connectivity and clear the Redis cache.

The UI sends requests to `http://127.0.0.1:8000` by default. If the API is
running on another host or port, update `API_URL` in
`src/gliner_app/ui.py` before launching the UI.

## Run Tests

With the virtual environment active:

```bash
python -m pytest
```

## Project Layout

```text
src/gliner_app/
├── main.py                 # FastAPI application and HTTP endpoints
├── ui.py                   # Gradio playground
├── config.py               # Model and Redis configuration
├── cache/redis_cache.py    # Async Redis cache implementation
├── models/model.py         # GLiNER loading, classification, and NER logic
└── model/                  # Local GLiNER 2.5 model files
tests/models/               # Model tests
```

## Configuration

| Variable | Default | Description |
| --- | --- | --- |
| `REDIS_URL` | `redis://localhost:6379/0` | Redis connection URL |
| `CACHE_TTL` | `3600` | Cache lifetime in seconds |

The API loads the model once during application startup and limits concurrent
inference requests to four. Cached classification and entity-extraction
responses are keyed by their complete request payload.
