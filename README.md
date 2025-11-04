# Accelerating LLM Reasoning via Early Rejection with Partial Reward Modeling

This directory contains the official implementation of the paper:

> **Accelerating LLM Reasoning via Early Rejection with Partial Reward Modeling**  (EMNLP 2025)
> Seyyed Saeid Cheshmi\*, Azal Ahmad Khan\*, Xinran Wang, Zirui Liu, Ali Anwar  
> University of Minnesota  
> [Paper](https://aclanthology.org/2025.findings-emnlp.551.pdf)

---

# Abstract
Large Language Models (LLMs) are increasingly relied upon for solving complex reasoning tasks in domains such as mathematics, logic, and multi-step question answering. A growing line of work seeks to improve reasoning quality by scaling inference time compute particularly through Process Reward Models (PRMs), used to reward the reasoning at intermediate steps. While effective, these methods introduce substantial computational overhead, especially when generating large numbers of solutions in parallel. In this paper, we investigate whether PRMs can be used mid-generation to provide early signals that enable the rejection of suboptimal candidates before full generation of step is complete. We introduce the hypothesis that PRMs are also Partial Reward Models, meaning that the scores they assign to partially completed reasoning step are predictive of final output quality. This allows for principled early rejection based on intermediate token-level signals. We support this hypothesis both theoretically, by proving that the risk of discarding optimal beams decreases exponentially with generation length and empirically, by demonstrating a strong correlation between partial and final rewards across multiple reward models. On math reasoning benchmarks, our method achieves up to 1.4x-9x reduction in inference FLOPs without degrading final performance. These results suggest that early rejection is a powerful mechanism for improving the compute-efficiency of reasoning in LLMs.

---

## How to run the codes

### Running Vanilla Beam Search

```bash
python main_beam_search.py \
    --num_problems=500 \
    --num_beams=32 \
    --beam_width=4 \
    --max_steps=20 \
    --search_batch=30 \
    --scoring_batch=5 \
    --monitor_gpu \
    --use_deepspeed \
    --dataset=math500 \
    --llm_model_name=Llama-3.2-3B-Instruct \
    --prm_model_name=math-shepherd

```

### Running Partial Beam Search

```bash
python main_partial_beam_search.py \
    --dataset math500 \
    --num_problems 500 \
    --num_beams 16 \
    --beam_width 4 \
    --max_steps 20 \
    --tau_tokens 64 \
    --llm_model_name Qwen2.5-3B-Instruct \
    --prm_model_name skywork-o1-1.5b \
    --monitor_gpu \
    --use_deepspeed
```

## Supported Datasets

1. **MATH-500** (`math500`): 500 challenging math problems across topics
2. **AIME 2024** (`aime`): American Invitational Mathematics Examination
3. **AMC 2023** (`amc23`): American Mathematics Competition
4. **SAT Math** (`sat_math`): AGIEval SAT Math problems

## Supported LLMs

1. **Qwen2.5-3B-Instruct** (`Qwen2.5-3B-Instruct`)
2. **Llama-3.2-3B-Instruct** (`Llama-3.2-3B-Instruct`)

## Supported PRMs 

1. **MathShepherd-Mistral-7B** (`math-shepherd`)
2. **Skywork-PRM-1.5B** (`skywork-o1-1.5b`)

## Output

Results are saved to a directory named by pattern:

```
{DATASET}_OUTPUT_{algorithm}_{llm_model}_{prm_model}_{num_beams}_{beam_width}_{max_steps}/
```

Each problem's solution is saved as a JSON file containing:
- Problem statement
- Metadata (subject, level, year, etc.)
- All generated solutions with scores
- Best solution


## Acknowledgment

Some parts of this code are based on [HF's Search-and-Learn](https://github.com/huggingface/search-and-learn).

## Citation 

If you use or build upon this work, please cite:

```
@inproceedings{cheshmi-etal-2025-accelerating,
    title = "Accelerating {LLM} Reasoning via Early Rejection with Partial Reward Modeling",
    author = "Cheshmi, Seyyed Saeid  and Khan, Azal Ahmad  and Wang, Xinran  and Liu, Zirui  and Anwar, Ali",
    editor = "Christodoulopoulos, Christos  and Chakraborty, Tanmoy  and Rose, Carolyn  and Peng, Violet",
    booktitle = "Findings of the Association for Computational Linguistics: EMNLP 2025",
    month = nov,
    year = "2025",
    address = "Suzhou, China",
    publisher = "Association for Computational Linguistics",
    url = "https://aclanthology.org/2025.findings-emnlp.551/",
    pages = "10433--10447",
    ISBN = "979-8-89176-335-7",
}
```
