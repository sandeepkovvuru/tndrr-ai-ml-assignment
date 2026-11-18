import json
import random

# Deterministic synthetic + realistic dataset generator for structured scheduling extraction
# Schema: {"intent": str, "title": str, "date": str, "time": str, "duration_mins": int, "participants": list[str], "priority": str}

TEMPLATES = [
    ("Schedule a {dur}-minute {prio} {title} with {p1} and {p2} on {date} at {time}.", "create_event"),
    ("Set up {title} for {date} starting {time} ({dur} mins) — {prio} priority, invite {p1}.", "create_event"),
    ("Can we book a {title} with {p1}, {p2} next {date} around {time} for {dur} minutes? Mark it {prio}.", "create_event"),
    ("Move our {title} with {p1} to {date} at {time} for {dur} mins, keep priority {prio}.", "reschedule_event"),
    ("Reschedule {title} ({dur}m) with {p1} and {p2} to {date} {time} [{prio}].", "reschedule_event"),
]

TITLES = ["Sprint Planning", "API Architecture Sync", "Q4 Budget Review", "ML Pipeline Triage", "Client Onboarding Call", "Security Audit Walkthrough"]
PEOPLE = ["Aarav", "Priya", "Rohan", "Neha", "Vikram", "Ananya", "Karthik", "Divya"]
DATES = ["2026-11-20", "2026-11-21", "2026-11-23", "2026-11-24", "2026-11-25", "2026-11-27"]
TIMES = ["09:30", "10:00", "11:15", "14:00", "15:30", "16:45"]
DURATIONS = [15, 30, 45, 60, 90]
PRIORITIES = ["low", "medium", "high"]

def generate_splits(seed=42, n_train=300, n_test=60):
    rng = random.Random(seed)
    records = []
    for i in range(n_train + n_test):
        tmpl, intent = rng.choice(TEMPLATES)
        title = rng.choice(TITLES)
        p1, p2 = rng.sample(PEOPLE, 2)
        date = rng.choice(DATES)
        time = rng.choice(TIMES)
        dur = rng.choice(DURATIONS)
        prio = rng.choice(PRIORITIES)
        text = tmpl.format(dur=dur, prio=prio, title=title, p1=p1, p2=p2, date=date, time=time)
        participants = [p1] if "{p2}" not in tmpl else [p1, p2]
        target = {
            "intent": intent,
            "title": title,
            "date": date,
            "time": time,
            "duration_mins": dur,
            "participants": participants,
            "priority": prio
        }
        records.append({"input": text, "output": json.dumps(target, separators=(",", ":"))})
    return records[:n_train], records[n_train:]

if __name__ == "__main__":
    train_data, test_data = generate_splits()
    with open("train.jsonl", "w") as f:
        for r in train_data:
            f.write(json.dumps(r) + "\n")
    with open("test.jsonl", "w") as f:
        for r in test_data:
            f.write(json.dumps(r) + "\n")
    print(f"Generated {len(train_data)} train and {len(test_data)} test examples.")
