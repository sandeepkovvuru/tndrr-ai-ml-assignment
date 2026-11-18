import json
import random
import torch
from datasets import Dataset
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    BitsAndBytesConfig,
    TrainingArguments,
    Trainer,
    DataCollatorForSeq2Seq,
)
from peft import LoraConfig, get_peft_model, prepare_model_for_kbit_training

MODEL_ID = "Qwen/Qwen2.5-0.5B-Instruct"
REQUIRED_KEYS = {"intent", "title", "date", "time", "duration_mins", "participants", "priority"}

SYSTEM_PROMPT = (
    "Extract scheduling parameters into strict minified JSON with keys: "
    "intent, title, date, time, duration_mins, participants, priority. Output JSON only."
)

TEMPLATES = [
    ("Schedule a {dur}-minute {prio} {title} with {p1} and {p2} on {date} at {time}.", "create_event"),
    ("Set up {title} for {date} starting {time} ({dur} mins) — {prio} priority, invite {p1}.", "create_event"),
    ("Can we book a {title} with {p1}, {p2} next {date} around {time} for {dur} minutes? Mark it {prio}.", "create_event"),
    ("Move our {title} with {p1} to {date} at {time} for {dur} mins, keep priority {prio}.", "reschedule_event"),
    ("Reschedule {title} ({dur}m) with {p1} and {p2} to {date} {time} [{prio}].", "reschedule_event"),
]

HARD_TEST_CASES = [
    {
        "input": "Push the Q4 Budget Review with Neha and Vikram to 2026-11-28 at half past two in the afternoon for an hour and a half, urgent!",
        "output": json.dumps({"intent":"reschedule_event","title":"Q4 Budget Review","date":"2026-11-28","time":"14:30","duration_mins":90,"participants":["Neha","Vikram"],"priority":"high"}, separators=(",", ":"))
    },
    {
        "input": "Cancel tomorrow's Sprint Planning with Aarav on 2026-11-20 at 10:00 (30m) low priority.",
        "output": json.dumps({"intent":"cancel_event","title":"Sprint Planning","date":"2026-11-20","time":"10:00","duration_mins":30,"participants":["Aarav"],"priority":"low"}, separators=(",", ":"))
    },
    {
        "input": "Book Client Onboarding Call with Priya, Rohan, and Divya on 2026-11-25 at 11:15 for 45 mins [medium].",
        "output": json.dumps({"intent":"create_event","title":"Client Onboarding Call","date":"2026-11-25","time":"11:15","duration_mins":45,"participants":["Priya","Rohan","Divya"],"priority":"medium"}, separators=(",", ":"))
    }
]

TITLES = ["Sprint Planning", "API Architecture Sync", "Q4 Budget Review", "ML Pipeline Triage", "Client Onboarding Call", "Security Audit Walkthrough"]
PEOPLE = ["Aarav", "Priya", "Rohan", "Neha", "Vikram", "Ananya", "Karthik", "Divya"]
DATES = ["2026-11-20", "2026-11-21", "2026-11-23", "2026-11-24", "2026-11-25", "2026-11-27"]
TIMES = ["09:30", "10:00", "11:15", "14:00", "15:30", "16:45"]
DURATIONS = [15, 30, 45, 60, 90]
PRIORITIES = ["low", "medium", "high"]

def generate_splits(seed=42, n_train=300, n_test=57):
    rng = random.Random(seed)
    records = []
    for _ in range(n_train + n_test):
        tmpl, intent = rng.choice(TEMPLATES)
        title = rng.choice(TITLES)
        p1, p2 = rng.sample(PEOPLE, 2)
        date, t_str, dur, prio = rng.choice(DATES), rng.choice(TIMES), rng.choice(DURATIONS), rng.choice(PRIORITIES)
        text = tmpl.format(dur=dur, prio=prio, title=title, p1=p1, p2=p2, date=date, time=t_str)
        participants = [p1] if "{p2}" not in tmpl else [p1, p2]
        target = {"intent": intent, "title": title, "date": date, "time": t_str, "duration_mins": dur, "participants": participants, "priority": prio}
        records.append({"input": text, "output": json.dumps(target, separators=(",", ":"))})
    return records[:n_train], records[n_train:] + HARD_TEST_CASES

def format_prompt(tokenizer, user_text, target_json=None):
    msgs = [{"role": "system", "content": SYSTEM_PROMPT}, {"role": "user", "content": user_text}]
    if target_json is not None:
        msgs.append({"role": "assistant", "content": target_json})
        return tokenizer.apply_chat_template(msgs, tokenize=False, add_generation_prompt=False)
    return tokenizer.apply_chat_template(msgs, tokenize=False, add_generation_prompt=True)

def evaluate_model(model, tokenizer, test_records):
    model.eval()
    valid, exact, f_hits, failures = 0, 0, 0, []
    for rec in test_records:
        inputs = tokenizer(format_prompt(tokenizer, rec["input"]), return_tensors="pt").to(model.device)
        with torch.no_grad():
            out = model.generate(**inputs, max_new_tokens=160, do_sample=False, pad_token_id=tokenizer.eos_token_id)
        gen = tokenizer.decode(out[0][inputs["input_ids"].shape[1]:], skip_special_tokens=True).strip()
        gt = json.loads(rec["output"])
        try:
            pred = json.loads(gen)
            if isinstance(pred, dict) and REQUIRED_KEYS.issubset(pred.keys()):
                valid += 1
                if pred == gt:
                    exact += 1
                else:
                    failures.append({"input": rec["input"], "expected": gt, "predicted": pred})
                for k in REQUIRED_KEYS:
                    if pred.get(k) == gt.get(k):
                        f_hits += 1
            else:
                failures.append({"input": rec["input"], "expected": gt, "predicted_raw": gen})
        except Exception:
            failures.append({"input": rec["input"], "expected": gt, "predicted_raw": gen})
    n = len(test_records)
    return {
        "total_examples": n,
        "schema_valid_json_pct": round(valid / n * 100, 2),
        "exact_json_match_pct": round(exact / n * 100, 2),
        "field_level_accuracy_pct": round(f_hits / (n * len(REQUIRED_KEYS)) * 100, 2),
        "failures": failures[:3],
    }

def main():
    train_records, test_records = generate_splits()
    tokenizer = AutoTokenizer.from_pretrained(MODEL_ID)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    bnb_config = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_compute_dtype=torch.float16,
        bnb_4bit_use_double_quant=True,
    )
    base_model = AutoModelForCausalLM.from_pretrained(MODEL_ID, quantization_config=bnb_config, device_map="auto")
    t1_base = evaluate_model(base_model, tokenizer, test_records)

    model = prepare_model_for_kbit_training(base_model)
    model = get_peft_model(model, LoraConfig(
        r=16, lora_alpha=32,
        target_modules=["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"],
        lora_dropout=0.05, bias="none", task_type="CAUSAL_LM"
    ))

    def tok_fn(row):
        f_ids = tokenizer(format_prompt(tokenizer, row["input"], row["output"]), truncation=True, max_length=256)["input_ids"]
        p_len = len(tokenizer(format_prompt(tokenizer, row["input"]), truncation=True, max_length=256)["input_ids"])
        return {"input_ids": f_ids, "attention_mask": [1]*len(f_ids), "labels": [-100]*p_len + f_ids[p_len:]}

    train_ds = Dataset.from_list(train_records).map(tok_fn, remove_columns=["input", "output"])
    Trainer(
        model=model,
        args=TrainingArguments(
            output_dir="./tmp_lora", num_train_epochs=3, per_device_train_batch_size=8,
            gradient_accumulation_steps=2, learning_rate=2e-4, lr_scheduler_type="cosine",
            warmup_steps=4, logging_steps=20, save_strategy="no", fp16=True,
            optim="paged_adamw_8bit", report_to="none"
        ),
        train_dataset=train_ds,
        data_collator=DataCollatorForSeq2Seq(tokenizer=tokenizer, padding=True)
    ).train()

    t1_ft = evaluate_model(model, tokenizer, test_records)
    with open("eval_results.json", "w") as f:
        json.dump({"base": t1_base, "fine_tuned": t1_ft}, f, indent=2)

if __name__ == "__main__":
    main()
