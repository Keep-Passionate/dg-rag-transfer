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

DocBench is not primarily visual, so the default is conservative:
`MM_BROAD_VISUAL_FALLBACK=false`. MMLongBench-Doc should usually use
`MM_BROAD_VISUAL_FALLBACK=true`.
