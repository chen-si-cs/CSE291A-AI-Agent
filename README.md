# CSE291A-AI-Agent (Team 1)

The main branch is the template environment for the ARC simulator.

## LLM API Agent

Switch to branches `zejia` (for general LLM API agent tests) and `jiarui` (for ablations on LLM API agents).

### Ablations (`jiarui` branch)

The notebook `arc_agi_agent_distractor.ipynb` contains the ablation trial setups, including:
- restrictions on no hardcoding and DSL-only execution (for example, no lambdas),
- ablations on the number of distractor DSL functions in the agent inventory.

Prompt-level restrictions are implemented in `build_system_prompt()`.

Code-level restrictions are implemented in `execute_line()` inside the `DSLExecutor` class.

For the reported experiments, only prompt-level restriction trials were run and logged. The current implementation keeps both prompt-level and code-level restrictions enabled.

Distractor variables are configured in the notebook configuration cell through `NUM_DISTRACTOR_DSLS`.

If you run `play_puzzle()` to conduct a trial, logs will be saved to the folder specified by `LOG_DIR`.

### Logs

Recorded logs are stored in `/logs`.

- `/api-gpt-oss-120b` and `/moonshotai_kimi-k2_5` are copied from `zejia`’s branch for analysis convenience.
- `/distractor_N` contains distractor trials for `GPT-OSS-120B` with `N` distractor functions.
- `/distractor_N_kimi` contains distractor trials for `moonshotai_kimi-k2.5`.
- `/oss_*` contains prompt-level restriction trials:
  - no hardcode restriction,
  - DSL-only restriction,
  - combined restriction.

---

## SFT Agent

Switch to branch `shuhao-SFT-modified`.

All commands below are relative to the `finalPJ/` directory.

### 1. Environment setup

```bash
cd finalPJ
pip install -r requirements.txt
pip install torch transformers datasets accelerate
```
If using GPU, install a CUDA-enabled PyTorch build that matches your machine.

### 2. Prepare offline trajectories

SFT training expects:
```
data/offline_trajectories.json
```
Use the existing file if it is already in the repo, or convert your own traces:
```
cd finalPJ
python -m scripts.build_offline_data \
  --in data/your_raw_traces.json \
  --out data/offline_trajectories.json
```

### 3. Train the SFT model

```
cd finalPJ
python -m scripts.train_bc_model \
  --data data/offline_trajectories.json \
  --save_dir checkpoints/bc_qwen \
  --model_name Qwen/Qwen3-1.7B \
  --epochs 2 \
  --batch_size 4 \
  --lr 2e-5
```
### 4. Evaluate the SFT agent

Pure SFT evaluation without lookup fallback:

```
cd finalPJ
python -m scripts.evaluate \
  --agent learning \
  --model-path checkpoints/bc_qwen \
  --offline-data data/offline_trajectories.json \
  --data data/evaluation \
  --budget 50 \
  --disable-learning-lookup
```
To save evaluation outputs:
```
cd finalPJ
python -m scripts.evaluate \
  --agent learning \
  --model-path checkpoints/bc_qwen \
  --offline-data data/offline_trajectories.json \
  --data data/evaluation \
  --budget 50 \
  --disable-learning-lookup \
  --output results_sft_learning_eval.json
```

### 5. Optional: Run in Colab

Use colab_sft_training_and_evaluation.ipynb.

It covers:  
cloning the repo and switching branch,  
installing dependencies,  
checking or uploading offline_trajectories.json,  
training the SFT model,  
evaluating the trained checkpoint.

## RL Agent

Switch to branch `feiyang`.

### Setup
 
**Dependencies:** Python 3.8+, PyTorch. No additional packages required beyond the base project environment.
 
**Data layout expected:**
```
agents/
  rl_agent.py                  # Policy network + action space (203 actions)
  dfs_agent.py              # Pure DFS baseline to compare with
  rl_beam_agent.py            # RL with Beam Search at inference time
scripts/
  train_rl_agent.py            # Pure GRPO training from scratch
  evaluate.py                  # Evaluation script (supports dfs, rl_beam agents)
data/
  train/          # 104 training puzzles (.json)
  eval/           # 27 evaluation puzzles (.json)
checkpoints/      # created automatically on training
env.py                       # ARC simulator environment
dsl.py / dsl_engine.py       # DSL function definitions and execution
inventory.py                 # Named variable scratchpad
reward.py                    # Reward computation
```

---
 
### Step 1 — Pure RL from scratch (GRPO)
 
Trains the MLP policy end-to-end with GRPO. Expect slow convergence due to cold-start.
 
```bash
python -m scripts.train_rl_agent \
  --data data/train \
  --save_dir checkpoints/rl_agent \
  --episodes 5000 \
  --verbose
```

---
 
### Step 2 — Evaluate
  
**DFS baseline** (ceiling reference — no learned policy):
```bash
python -m scripts.evaluate \
  --agent dfs \
  --data data/eval \
  --budget 50 \
  --verbose
```

**Pure RL policy** (greedy rollout):
```bash
python -m scripts.evaluate \
  --agent rl \
  --rl-checkpoint checkpoints/rl_agent/ckpt_final.pt \
  --data data/eval \
  --verbose
```
 
**RL-guided beam search** (recommended inference):
```bash
python -m scripts.evaluate \
  --agent rl_beam \
  --rl-checkpoint checkpoints/rl_agent/ckpt_final.pt \
  --data data/eval \
  --beam-width 8 \
  --beam-depth 4 \
  --top-b 10 \
  --budget 50 \
  --verbose
```
