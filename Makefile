.PHONY: env test lint help

help:
	@echo "Available targets:"
	@echo "  env   : Set up conda environment and install dependencies"
	@echo "  test  : Run unit tests with pytest"
	@echo "  lint  : Check code style and linting"

env:
	conda create -y -n bevlite python=3.10
	conda run -n bevlite pip install torch torchvision --index-url https://download.pytorch.org/whl/cu121
	conda run -n bevlite pip install numpy scipy opencv-python pillow pyyaml easydict tensorboard tqdm pytest fvcore scikit-image numba matplotlib pandas open3d kornia

test:
	conda run -n bevlite pytest tests/ -v

lint:
	conda run -n bevlite flake8 bevfusionlite/ tests/ scripts/ --max-line-length=120 --ignore=E203,W503 || true
