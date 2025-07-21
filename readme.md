python -m venv gemma-env
source gemma-env/bin/activate
pip install packaging wheel torch
pip install flash-attn --no-build-isolation
pip install -r requirements.txt

