import os
import re

def get_solved_ids(log_dir):
    solved_ids = set()
    if not os.path.exists(log_dir):
        return solved_ids
    for filename in os.listdir(log_dir):
        if filename.endswith(".log"):
            puzzle_id = filename.replace(".log", "")
            with open(os.path.join(log_dir, filename), "r", encoding="utf-8", errors="ignore") as f:
                content = f.read()
                if "RESULT: SUCCESS" in content or ">>> SOLVED at step" in content:
                    solved_ids.add(puzzle_id)
                else:
                    match = re.search(r"Accuracy:? (\d+)/(\d+)", content)
                    if match and match.group(1) == match.group(2):
                        solved_ids.add(puzzle_id)
                    else:
                        match2 = re.search(r"Final match:? (\d+)/(\d+)", content)
                        if match2 and match2.group(1) == match2.group(2):
                             solved_ids.add(puzzle_id)
    return solved_ids

base_dir = r"c:\Users\10755\Documents\GitHub\CSE291A-AI-Agent\llm_new_env_and_agent\logs"
dirs = {
    "Kimi-N0": os.path.join(base_dir, "moonshotai_kimi-k2_5"),
    "Kimi-N5": os.path.join(base_dir, "distractor_5_kimi"),
    "Kimi-N10": os.path.join(base_dir, "distractor_10_kimi")
}

results = {name: get_solved_ids(d) for name, d in dirs.items()}

print("=== KIMI SOLVED COUNTS ===")
for name, solved in results.items():
    print(f"{name:15}: {len(solved)}/85")

n0 = results["Kimi-N0"]
n5 = results["Kimi-N5"]
n10 = results["Kimi-N10"]

lost_in_n10 = n0 - n10
print(f"\nLost in N10 (Solved in N0, Failed in N10) [{len(lost_in_n10)}]: {sorted(list(lost_in_n10))[:10]}...")

gained_in_n10 = n10 - n0
print(f"Gained in N10 (Failed in N0, Solved in N10) [{len(gained_in_n10)}]: {sorted(list(gained_in_n10))[:10]}...")
