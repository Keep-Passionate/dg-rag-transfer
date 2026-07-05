# DocBench RAG-Anything + Multimodal Grounding

This folder contains a server runner for testing the portable multimodal
grounding operator on DocBench subsets with the existing RAG-Anything indexes.

The runner writes new result files into the original DocBench document folders:

- `qa_results_mmground_docbench.json`: MM grounding only.
- `qa_results_dg_mm_docbench.json`: DG text grounding plus MM grounding.
- `qa_results_mm_base_docbench.json`: optional fresh base when `RUN_BASE=1`.

It evaluates through a symlinked subset root so the LLM evaluator does not scan
the whole DocBench tree by accident.

Recommended smoke:

```bash
cd /root/autodl-tmp/dg-rag-transfer
export MAIN=/root/autodl-tmp/rag-L1
export DOCBENCH=/root/autodl-tmp/DocBench_subset
export WORK=/root/autodl-tmp/RAG-Anything-thesis/rag_storage_baseline
export PARSE_OUT=/root/autodl-tmp/RAG-Anything-thesis/output
export PY=/root/miniconda3/envs/rag/bin/python

bash docbench/run_raganything_mm.sh smoke
```

Then run a larger subset:

```bash
bash docbench/run_raganything_mm.sh 25
```

The default is conservative and non-overriding:
`MM_ENABLE_VISUAL_ROUTING=false`, `MM_BROAD_VISUAL_FALLBACK=false`, and
`MM_FORCE_VLM=false`.

Visual routing is only an opt-in probe. Enable it explicitly when running a
controlled ablation against a VLM-only baseline:

```bash
export MM_ENABLE_VISUAL_ROUTING=true
export MM_ALLOW_RANKED_VISUAL=true
export MM_FORCE_VLM=false
```
