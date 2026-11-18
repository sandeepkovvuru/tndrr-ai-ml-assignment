# tndrr AI/ML Internship Assignment — Sandeep Kovvuru

Complete, reproducible implementations and empirical T4 GPU benchmarks for all three assignment tasks.

## Repository Structure

```text
tndrr-ai-ml-assignment/
├── README.md
├── task1_lora_finetuning/
│   ├── train_and_eval.py
│   ├── eval_results.json
│   └── requirements.txt
├── task2_rag_from_scratch/
│   ├── rag_pipeline.py
│   ├── test_questions.json
│   ├── benchmark_results.json
│   └── requirements.txt
└── task3_local_inference_optimization/
    ├── benchmark_inference.py
    ├── prompts.json
    ├── benchmark_summary.json
    └── requirements.txt
```

## Summary of Empirical Results (NVIDIA Tesla T4 GPU)

### Task 1: QLoRA Fine-Tuning (`Qwen/Qwen2.5-0.5B-Instruct` -> Strict JSON Extraction)
- **LoRA Config:** `r=16`, `lora_alpha=32`, target modules: `q_proj, k_proj, v_proj, o_proj, gate_proj, up_proj, down_proj`, `lr=2e-4` (cosine), `3 epochs`, 4-bit NF4 base.
- **Base Model (60 test examples):** `0.00%` valid JSON (wrapped output in markdown fences) | `0.00%` exact match | `0.00%` field accuracy.
- **Fine-Tuned QLoRA Model (60 test examples):** **`100.00%` valid JSON** | **`96.67%` exact match** | **`98.81%` field accuracy**.

### Task 2: From-Scratch RAG (`all-MiniLM-L6-v2` + `FAISS IndexFlatIP`, zero LangChain/LlamaIndex)
- **Corpus:** NIST SP 800-63B Digital Identity Guidelines (80 extracted PDF pages).
- **150-Word Chunks (30w overlap, 253 chunks):** **`80.00%` Recall@5 (`12/15`)** | **`0.5489` MRR@5**.
- **400-Word Chunks (80w overlap, 129 chunks):** **`66.67%` Recall@5 (`10/15`)** | **`0.5022` MRR@5**.

### Task 3: Local Inference & Quantization (`Qwen2.5-1.5B-Instruct-GGUF` via `llama.cpp`)
- **FP16 (16-bit):** `3,316.0 MB` VRAM | `59.16 tokens/sec` average generation throughput.
- **Q4_K_M (4-bit):** **`1,306.0 MB` VRAM (-60.6%)** | **`124.50 tokens/sec` (+2.10x speedup)**.
