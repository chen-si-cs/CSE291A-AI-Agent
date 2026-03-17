import os
import glob

def get_solved(dir_path):
    solved = set()
    for filepath in glob.glob(os.path.join(dir_path, "*.log")):
        with open(filepath, 'r', encoding='utf-8') as f:
            content = f.read()
            # api-gpt-oss-120b uses ">>> SOLVED at step"
            # others use "RESULT: SOLVED"
            if ">>> SOLVED at step" in content or "RESULT: SOLVED" in content:
                puzzle_id = os.path.basename(filepath).split('.')[0]
                solved.add(puzzle_id)
    return solved

dir_unrestricted = r"c:\Users\10755\Documents\GitHub\CSE291A-AI-Agent\llm_new_env_and_agent\logs\api-gpt-oss-120b"
dir_no_hardcode = r"c:\Users\10755\Documents\GitHub\CSE291A-AI-Agent\llm_new_env_and_agent\logs\oss_noHardcode"

unrestricted_solved = get_solved(dir_unrestricted)
no_hardcode_solved = get_solved(dir_no_hardcode)

print(f"Total Unrestricted Solved: {len(unrestricted_solved)}")
print(f"Total No Hardcode Solved: {len(no_hardcode_solved)}")

print(f"\nSolved ONLY in Unrestricted ({len(unrestricted_solved - no_hardcode_solved)}):")
for p in sorted(unrestricted_solved - no_hardcode_solved):
    print(p)

print(f"\nSolved ONLY in No Hardcode ({len(no_hardcode_solved - unrestricted_solved)}):")
for p in sorted(no_hardcode_solved - unrestricted_solved):
    print(p)
