# KeyChunk

GPU page reuse and selective recomputation with vLLM.

Includes [serial GPU page performance experiments](examples/gpu_page_serial.py) and QA quality experiments.

## Installation

Requires Linux, Python 3.12, the CUDA 12.6 toolkit, and local Llama/Mistral model weights.

```bash
python3.12 -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
bash scripts/install_runtime.sh
```

## Performance experiments

Download the [TopiOCQA development set](https://zenodo.org/records/7709644/files/topiocqa_dev.json) separately. For the script below, use the [official AllHistory development set with retrieved passages](https://zenodo.org/records/6153434/files/results/retriever/dpr/topiocqa_all_history_dev.json), saved as `data/topiocqa_all_history_dev.json`.

Prepare KV, then run one request at a time on a single GPU in input order. Inputs can be a JSON array or JSONL, with `question` and `chunks` or `ctxs` in each record. Temporary KV files are removed after execution; results are saved in the output directory.

```bash
python examples/gpu_page_serial.py --model /path/to/model \
  --input data/topiocqa_all_history_dev.json --kv-cache-dir data/gpu_page_kv \
  --max-model-len 32768 --output-dir results/gpu_page_serial
```

Check actual GPU page reads:

```bash
python examples/gpu_page_serial.py --model /path/to/model \
  --input data/topiocqa_all_history_dev.json --kv-cache-dir data/gpu_page_kv \
  --max-model-len 32768 --verify-reuse --output-dir results/gpu_page_check
```

## Quality experiments

### Data and retrieval

Download the QA data, Wikipedia corpus, and [E5-base-v2](https://huggingface.co/intfloat/e5-base-v2) weights. Replace the model paths with local directories and enter the number of passages to retrieve when prompted.

```bash
python -m pip install -e '.[retrieval]'
python scripts/prepare_qa.py download --dataset all --output-dir data/raw
python scripts/prepare_qa.py download-corpus --output data/wiki18_100w.jsonl
python scripts/prepare_qa.py build-index --corpus data/wiki18_100w.jsonl \
  --retriever-model /path/to/e5-base-v2 --output-dir data/index --fp16
read -r -p "Number of retrieved passages: " CHUNK_COUNT
python scripts/prepare_qa.py retrieve --input data/raw/nq/test.jsonl \
  --index-dir data/index --top-k "$CHUNK_COUNT" --output data/qa/nq.jsonl --fp16
```

### Run

NQ example; use the same entrypoint for the other datasets.

```bash
python examples/run_qa.py --model /path/to/model \
  --input data/qa/nq.jsonl --output-dir results/nq
```

### Single-request reuse check

```bash
python examples/gpu_reuse.py --model /path/to/model \
  --input data/qa/nq.jsonl --output-dir results/gpu_reuse
```

Third-party sources and licenses: [NOTICE](NOTICE).
