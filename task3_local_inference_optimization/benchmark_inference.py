import gc
import json
import time
import torch
from huggingface_hub import hf_hub_download
from llama_cpp import Llama

REPO_ID = "Qwen/Qwen2.5-1.5B-Instruct-GGUF"
MODELS = {
    "FP16_16bit": "qwen2.5-1.5b-instruct-fp16.gguf",
    "Q4_K_M_4bit": "qwen2.5-1.5b-instruct-q4_k_m.gguf",
}

def main():
    with open("prompts.json") as f:
        prompts = json.load(f)

    summary = {}
    for label, fname in MODELS.items():
        path = hf_hub_download(repo_id=REPO_ID, filename=fname)
        torch.cuda.empty_cache()
        free_before, _ = torch.cuda.mem_get_info()
        llm = Llama(model_path=path, n_ctx=2048, n_gpu_layers=-1, verbose=False)
        free_after, _ = torch.cuda.mem_get_info()
        vram_mb = round((free_before - free_after) / (1024 ** 2), 1)

        tot_tok, tot_sec, outputs = 0, 0.0, []
        for p in prompts:
            t0 = time.perf_counter()
            res = llm.create_chat_completion(messages=[{"role": "user", "content": p}], max_tokens=160, temperature=0.0)
            dt = time.perf_counter() - t0
            ntok = res["usage"]["completion_tokens"]
            tot_tok += ntok
            tot_sec += dt
            outputs.append({
                "prompt": p,
                "tokens": ntok,
                "tps": round(ntok / dt, 2),
                "response": res["choices"][0]["message"]["content"].strip()
            })
        summary[label] = {
            "weights_file": fname,
            "vram_used_mb": vram_mb,
            "avg_tokens_per_sec": round(tot_tok / tot_sec, 2),
            "outputs": outputs
        }
        del llm
        gc.collect()

    with open("benchmark_summary.json", "w") as f:
        json.dump(summary, f, indent=2)

if __name__ == "__main__":
    main()
