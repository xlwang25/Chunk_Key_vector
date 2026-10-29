# KeyChunk

基于 vLLM 的 GPU page 复用与选择性重算。

提供 [GPU page 串行性能实验](examples/gpu_page_serial.py)和 QA 质量实验。

## 安装

需要 Linux、Python 3.12、CUDA 12.6 工具链和本地 Llama/Mistral 模型权重。

```bash
python3.12 -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
bash scripts/install_runtime.sh
```

## 性能实验

自行下载 [TopiOCQA 开发集](https://zenodo.org/records/7709644/files/topiocqa_dev.json)。以下入口可直接读取[官方 AllHistory 已检索开发集](https://zenodo.org/records/6153434/files/results/retriever/dpr/topiocqa_all_history_dev.json)，保存为 `data/topiocqa_all_history_dev.json`。

单 GPU、单请求串行执行，保持输入顺序。先准备 KV，再运行。支持 JSON 数组或 JSONL，每条请求包含 `question` 和 `chunks` 或 `ctxs`。临时 KV 在结束后清理，结果保存在输出目录。

```bash
python examples/gpu_page_serial.py --model /path/to/model \
  --input data/topiocqa_all_history_dev.json --kv-cache-dir data/gpu_page_kv \
  --max-model-len 32768 --output-dir results/gpu_page_serial
```

检查实际 GPU 页读取：

```bash
python examples/gpu_page_serial.py --model /path/to/model \
  --input data/topiocqa_all_history_dev.json --kv-cache-dir data/gpu_page_kv \
  --max-model-len 32768 --verify-reuse --output-dir results/gpu_page_check
```

## 质量实验

### 数据与检索

自行下载 QA 数据、Wikipedia 语料和 [E5-base-v2](https://huggingface.co/intfloat/e5-base-v2) 权重。替换命令中的本地模型路径；执行时输入检索段落数。

```bash
python -m pip install -e '.[retrieval]'
python scripts/prepare_qa.py download --dataset all --output-dir data/raw
python scripts/prepare_qa.py download-corpus --output data/wiki18_100w.jsonl
python scripts/prepare_qa.py build-index --corpus data/wiki18_100w.jsonl \
  --retriever-model /path/to/e5-base-v2 --output-dir data/index --fp16
read -r -p "检索段落数: " CHUNK_COUNT
python scripts/prepare_qa.py retrieve --input data/raw/nq/test.jsonl \
  --index-dir data/index --top-k "$CHUNK_COUNT" --output data/qa/nq.jsonl --fp16
```

### 运行

以下以 NQ 为例，其他数据集使用相同入口。

```bash
python examples/run_qa.py --model /path/to/model \
  --input data/qa/nq.jsonl --output-dir results/nq
```

### 单条请求复用检查

```bash
python examples/gpu_reuse.py --model /path/to/model \
  --input data/qa/nq.jsonl --output-dir results/gpu_reuse
```

第三方来源和许可见 [NOTICE](NOTICE)。
