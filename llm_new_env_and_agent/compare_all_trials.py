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
                # Check for RESULT: SUCCESS or Accuracy: 100/100 or final match: total
                if "RESULT: SUCCESS" in content or ">>> SOLVED at step" in content:
                    solved_ids.add(puzzle_id)
                else:
                    # Fallback for some log formats
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
    "Unrestricted": os.path.join(base_dir, "api-gpt-oss-120b"),
    "DSL-Only": os.path.join(base_dir, "oss_strictDSLonly"),
    "No-Hardcode": os.path.join(base_dir, "oss_noHardcode"),
    "Combined": os.path.join(base_dir, "oss_restrictCombined")
}

results = {name: get_solved_ids(d) for name, d in dirs.items()}

print("=== SOLVED COUNTS ===")
for name, solved in results.items():
    print(f"{name:15}: {len(solved)}/85")

print("\n=== COMBINED vs UNRESTRICTED ===")
combined = results["Combined"]
unrestricted = results["Unrestricted"]

regressions = unrestricted - combined
improvements = combined - unrestricted

print(f"Regressions (Unrestricted solved, Combined failed) [{len(regressions)}]: {sorted(list(regressions))}")
print(f"Improvements (Combined solved, Unrestricted failed) [{len(improvements)}]: {sorted(list(improvements))}")

print("\n=== COMBINED vs COMPONENTS ===")
only_in_combined = combined - (results["DSL-Only"] | results["No-Hardcode"])
failed_in_combined_but_solved_in_both_components = (results["DSL-Only"] & results["No-Hardcode"]) - combined

print(f"Solved ONLY by Combined (and not by DSL-Only or No-Hardcode) [{len(only_in_combined)}]: {sorted(list(only_in_combined))}")
print(f"Failed in Combined but solved in BOTH components [{len(failed_in_combined_but_solved_in_both_components)}]: {sorted(list(failed_in_combined_but_solved_in_both_components))}")
