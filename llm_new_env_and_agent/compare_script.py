import os
import glob

def get_solved(dir_path):
    solved = set()
    for filepath in glob.glob(os.path.join(dir_path, "*.log")):
        with open(filepath, 'r', encoding='utf-8') as f:
            content = f.read()
            if ">>> SOLVED at step" in content or "RESULT: SOLVED" in content:
                puzzle_id = os.path.basename(filepath).split('.')[0]
                solved.add(puzzle_id)
    return solved

dir1 = r"c:\Users\10755\Documents\GitHub\CSE291A-AI-Agent\llm_new_env_and_agent\logs\oss_strictDSLonly"
dir2 = r"c:\Users\10755\Documents\GitHub\CSE291A-AI-Agent\llm_new_env_and_agent\logs\api-gpt-oss-120b"

strict_solved = get_solved(dir1)
unrestricted_solved = get_solved(dir2)

print(f"Total Strict Solved: {len(strict_solved)}")
print(f"Total Unrestricted Solved: {len(unrestricted_solved)}")

print(f"\nSolved ONLY in Unrestricted ({len(unrestricted_solved - strict_solved)}):")
for p in sorted(unrestricted_solved - strict_solved):
    print(p)

print(f"\nSolved ONLY in Strict DSL ({len(strict_solved - unrestricted_solved)}):")
for p in sorted(strict_solved - unrestricted_solved):
    print(p)
