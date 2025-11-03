import torch
import gc
import logging
import sys
import os
from typing import List, Dict, Any, Tuple
from datasets import Dataset





sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from core.text_utils import split_into_steps
from core.scoring import parallel_score_solutions

logger = logging.getLogger(__name__)

def parallel_generate_steps(llm, prompts: List[str], batch_size: int, step: int,
                           temperature: float = 0.8, max_new_tokens: int = 64) -> List[str]:
    """Generate steps with true parallel processing."""

    dataset = Dataset.from_dict({"text": prompts})


    generation_kwargs = {
        "max_new_tokens": max_new_tokens,
        "temperature": temperature,
        "num_return_sequences": 1,
        "stop_sequence": "ки\n",
        "return_full_text": False,
        "pad_token_id": llm.tokenizer.eos_token_id,
    }
    
    output_dataset = llm(
        dataset["text"],
        batch_size=batch_size,
        **generation_kwargs
    )
    
    outputs = []
    for output in output_dataset:
        if isinstance(output, dict):
            outputs.append(output.get('generated_text', '').strip())
        elif isinstance(output, list):
            outputs.append(output[0].get('generated_text', '').strip())
        else:
            outputs.append(str(output).strip())

    torch.cuda.empty_cache()
    gc.collect()
            
    return outputs

def parallel_complete_steps(llm, candidate_prompts: List[str], batch_size: int, tau_tokens: int, temperature: float = 0.8) -> Tuple[List[str], int]:
    """Complete a partially generated candidate step using dataset processing.
    
    Returns:
        Tuple containing:
        - List of completed outputs 
        - Number of outputs generated
    """
    dataset = Dataset.from_dict({"text": candidate_prompts})
    
    
    generation_kwargs = {
        "max_new_tokens": tau_tokens,
        "temperature": temperature,
        "num_return_sequences": 1,
        "stop_sequence": "ки\n",
        "return_full_text": False,
        "pad_token_id": llm.tokenizer.eos_token_id,
    }
    
    output_dataset = llm(
        dataset["text"],
        batch_size=batch_size,
        **generation_kwargs
    )
    
    outputs = []
    output_count = 0
    for output in output_dataset:
        if isinstance(output, dict):
            outputs.append(output.get('generated_text', '').strip())
            output_count += 1
        elif isinstance(output, list):
            outputs.append(output[0].get('generated_text', '').strip())
            output_count += 1
        else:
            outputs.append(str(output).strip())
            output_count += 1

    torch.cuda.empty_cache()
    gc.collect()
            
    return outputs, output_count

def is_solution_complete(text):
    """Check if the solution has reached a logical conclusion with an answer."""
    if "\\boxed{" in text:
        boxed_index = text.rfind("\\boxed{")
        
        after_boxed = text[boxed_index:]
        if "}" in after_boxed:
            closing_index = after_boxed.find("}") + boxed_index
            
            conclusion_phrases = ["Therefore", "The answer is", "The final answer is", "Thus", "So", "Hence"]
            for phrase in conclusion_phrases:
                if phrase in text[:boxed_index] and text.rfind(phrase) > text.rfind("##"):
                    return True
            
            remaining_text = text[closing_index:].strip()
            valid_endings = [".", "I hope it is correct", ""]
            for end in valid_endings:
                if remaining_text.startswith(end) or remaining_text == "":
                    return True
    
    return False

def partial_beam_search(problems: List[str], llm, prm_tokenizer, prm_model, 
                       num_beams: int = 4, beam_width: int = 2, 
                       max_steps: int = 4, b1: int = 8, b2: int = 4,
                       scoring_batch_size: int = 5,
                       tau_tokens: int = 64, temperature: float = 0.8,
                       prm_type: str = 'math-shepherd') -> List[List[Dict[str, Any]]]:
    """
    Implements Beam Search with Partial Scoring following the pseudocode:
    1. Initialize a set of N candidate beams for each problem
    2. For each beam, generate beam_width expansions using batch size b1
    3. Score each partial step using the PRM (only scoring step)
    4. Select the top N/M candidates based on scores
    5. Complete these selected candidates using batch size b2
    6. Use completed candidates for next iteration
    7. Repeat until EOS or max depth
    
    Args:
        problems: List of problems to solve
        llm: Language model for generation
        prm_tokenizer: Tokenizer for PRM
        prm_model: Policy Reward Model for scoring
        num_beams: Number of beams to maintain
        beam_width: Number of expansions per beam
        max_steps: Maximum number of steps to generate
        b1: Batch size for partial step generation
        b2: Batch size for step completion
        tau_tokens: Number of tokens for partial generation
        scoring_batch_size: Batch size for scoring
        temperature: Temperature for generation
        prm_type: Type of PRM model being used ('math-shepherd' or 'skywork-o1')
        
    Returns:
        List of beams for each problem
    """
    # if b1 < b2:
    #     raise ValueError("b1 must be greater than b2")
    
    logger.info('initializing partial beam search')
    from core.models import format_prompt
    initial_prompts = [format_prompt(problem) for problem in problems]
    num_problems = len(problems)
    

    all_beams = [
        [{"text": "", "score": 0.0, "steps": [], "complete": False, "completion_counts": []}] 
        for _ in range(num_problems)
    ]
    step = 0
    
    # Marker for separating steps
    step_marker = " ки\n"
    
    while step < max_steps:

        all_prompts = []
        problem_indices = []
        beam_indices = []
        
        for problem_idx, (problem, initial_prompt) in enumerate(zip(problems, initial_prompts)):
            current_beams = all_beams[problem_idx]
            
            for beam_idx, beam in enumerate(current_beams):
                if beam["complete"]:
                    continue
                    

                for _ in range(beam_width):
                    all_prompts.append(initial_prompt + beam["text"])
                    problem_indices.append(problem_idx)
                    beam_indices.append(beam_idx)
        
        if not all_prompts:
            # All beams across all problems are complete
            break
            
        # PHASE 1: Generate partial steps using batch size b1
        logger.info(f'generating partial steps...')
        partial_steps = parallel_generate_steps(
            llm,
            all_prompts,
            batch_size=b1,
            step=step,
            temperature=temperature,
            max_new_tokens=tau_tokens  
        )
        

        all_partial_candidates = [[] for _ in range(num_problems)]
        all_partial_texts = []
        problem_text_map = []  
        
        for i, partial_step in enumerate(partial_steps):
            if partial_step:
                problem_idx = problem_indices[i]
                beam_idx = beam_indices[i]
                current_beam = all_beams[problem_idx][beam_idx]
                
       
                full_text = current_beam["text"] + partial_step + step_marker
                

                is_complete = is_solution_complete(full_text)
                

                candidate = {
                    "text": full_text,
                    "partial_text": partial_step,
                    "steps": current_beam["steps"].copy(), 
                    "parent_beam": beam_idx,
                    "score": 0.0,
                    "complete": is_complete,
                    "completion_counts": current_beam["completion_counts"].copy()  
                }
     
                if is_complete:
                    candidate["steps"].append(partial_step)
                
                all_partial_candidates[problem_idx].append(candidate)
                all_partial_texts.append(full_text)
                problem_text_map.append(problem_idx)
        
        if all(not candidates for candidates in all_partial_candidates):
            break
            
        # PHASE 2: Score all partial steps together
        problem_solution_pairs = [(problems[problem_text_map[i]], text) for i, text in enumerate(all_partial_texts)]
        logger.info(f'scoring partial steps...')
        if problem_solution_pairs:
            partial_scores = parallel_score_solutions(
                problem_solution_pairs,
                batch_size=scoring_batch_size,
                prm_tokenizer=prm_tokenizer,
                prm_model=prm_model,
                prm_type=prm_type
            )
            
            for i, (text, score) in enumerate(zip(all_partial_texts, partial_scores)):
                problem_idx = problem_text_map[i]
                
                
                for candidate in all_partial_candidates[problem_idx]:
                    if candidate["text"] == text:
                        candidate["score"] = float(score[-1].item())
            
            # PHASE 3: Select top candidates for each problem independently
            all_top_candidates = []
            for problem_idx, candidates in enumerate(all_partial_candidates):
                if candidates:
                    top_candidates = sorted(candidates, key=lambda x: x["score"], reverse=True)
                    top_candidates = top_candidates[:num_beams//beam_width]
                    all_top_candidates.append(top_candidates)
                else:
                    all_top_candidates.append([])
                
            # PHASE 4: Complete these selected candidates for each problem
            all_completed_candidates = [[] for _ in range(num_problems)]
            
            # First, handle already complete candidates
            for problem_idx, top_candidates in enumerate(all_top_candidates):
                complete_candidates = [c for c in top_candidates if c["complete"]]
                for c in complete_candidates:
                    
                    c["clean_text"] = c["text"].replace(step_marker, "\n")
                    c["full_text"] = c["clean_text"]
                    all_completed_candidates[problem_idx].append(c)
            
            
            all_completion_prompts = []
            all_incomplete_indices = [] 
            
            for problem_idx, top_candidates in enumerate(all_top_candidates):
                incomplete_candidates = [c for c in top_candidates if not c["complete"]]
                for candidate_idx, candidate in enumerate(incomplete_candidates):
                    all_completion_prompts.append(initial_prompts[problem_idx] + candidate["text"])
                    all_incomplete_indices.append((problem_idx, candidate_idx))
            
           
            if all_completion_prompts:
                logger.info(f'completing partial steps...')
                completions, completion_count = parallel_complete_steps(
                    llm,
                    all_completion_prompts,
                    batch_size=b2,
                    tau_tokens=256,  
                    temperature=temperature
                )
                
               
                for i, completion in enumerate(completions):
                    if completion:
                        problem_idx, candidate_idx = all_incomplete_indices[i]
                        
                        incomplete_candidates = [c for c in all_top_candidates[problem_idx] if not c["complete"]]
                        if candidate_idx < len(incomplete_candidates):
                            candidate = incomplete_candidates[candidate_idx].copy()
                            
                            completed_text = candidate["text"] + completion + step_marker
                            
                            is_complete = is_solution_complete(completed_text)
                            
                            full_step = candidate["partial_text"]
                            if is_complete or len(completion) > 5:  
                                full_step += completion
                            
                            candidate["steps"].append(full_step)
                            candidate["text"] = completed_text
                            candidate["clean_text"] = completed_text.replace(step_marker, "\n")
                            candidate["full_text"] = candidate["clean_text"]
                            candidate["complete"] = is_complete
                            
                            candidate["completion_counts"].append(completion_count)
                            
                            all_completed_candidates[problem_idx].append(candidate)
            
            # PHASE 5: Update all_beams with the completed candidates for each problem
            for problem_idx, completed_candidates in enumerate(all_completed_candidates):
                if completed_candidates:
                    all_beams[problem_idx] = completed_candidates

            step += 1
            

            torch.cuda.empty_cache()
            

            if all(all(beam["complete"] for beam in beams) for beams in all_beams):
                break
        else:
       
            break
    

    for problem_beams in all_beams:
        for beam in problem_beams:
            if "full_text" not in beam:
                if "clean_text" in beam:
                    beam["full_text"] = beam["clean_text"]
                else:
                    beam["full_text"] = beam["text"].replace(step_marker, "\n")
            
            beam["steps"] = split_into_steps(beam["full_text"])
            
            if "completion_counts" not in beam:
                beam["completion_counts"] = []
    
    return [sorted(beams, key=lambda x: x["score"], reverse=True) for beams in all_beams]

