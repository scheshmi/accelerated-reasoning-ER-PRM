import torch
import gc
import logging
import sys
import os
from typing import List, Dict, Any
from datasets import Dataset

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from core.text_utils import split_into_steps
from core.scoring import parallel_score_solutions

logger = logging.getLogger(__name__)

def parallel_generate_steps(llm, prompts: List[str], step: int, search_batch_size: int, temperature: float = 0.8) -> List[str]:
    """Generate steps with true parallel processing."""
    dataset = Dataset.from_dict({"text": prompts})
    
    generation_kwargs = {
        "max_new_tokens": 256,
        "temperature": temperature,
        "num_return_sequences": 1,
        "stop_sequence": "ки\n",
        "return_full_text": False,
        "pad_token_id": llm.tokenizer.eos_token_id,
    }
    
    outputs = []
    
    output_dataset = llm(
        dataset["text"],
        batch_size=search_batch_size,  # Set batch_size here
        **generation_kwargs
    )
    
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

def parallel_beam_search(problems: List[str], llm, prm_tokenizer, prm_model, 
                        scoring_batch_size: int, search_batch_size: int,
                        num_beams: int = 4, beam_width: int = 2, 
                        max_steps: int = 4, prm_type: str = 'math-shepherd') -> List[List[Dict[str, Any]]]:
    """Memory-optimized parallel beam search for multiple problems simultaneously."""
    logger.info(f"Initializing beams ....")
    from core.models import format_prompt
    initial_prompts = [format_prompt(problem) for problem in problems]
    num_problems = len(problems)
    
    all_beams = [
        [{"text": "", "score": 0, "steps": []} for _ in range(num_beams)]
        for _ in range(num_problems)
    ]
    
    try:
        for step in range(max_steps):
            all_prompts = []
            beam_indices = []
            problem_indices = []
            
            for problem_idx, (problem, initial_prompt) in enumerate(zip(problems, initial_prompts)):
                current_beams = all_beams[problem_idx]
                
                for beam_idx, beam in enumerate(current_beams):
                    current_text = beam["text"]
                    for _ in range(beam_width):
                        all_prompts.append(initial_prompt + current_text)
                        beam_indices.append(beam_idx)
                        problem_indices.append(problem_idx)
            
            logger.info(f"Generating steps ....")
            new_steps = parallel_generate_steps(llm, all_prompts, step, search_batch_size)
            
            candidates = [[] for _ in range(num_problems)]
            final_answer_marker = "Therefore, the final answer"
            
            for i, new_step in enumerate(new_steps):
                if new_step:  
                    problem_idx = problem_indices[i]
                    beam_idx = beam_indices[i]
                    current_beam = all_beams[problem_idx][beam_idx]
                    current_full_text = current_beam["text"]
                    current_steps = current_beam["steps"]

                   
                    new_text_segment = new_step + " ки\n"
                    
             
                    new_individual_steps = split_into_steps(new_step)
                    
      
                    combined_full_text = current_full_text + new_text_segment
                    combined_steps = current_steps + new_individual_steps
                    
       
                    if final_answer_marker in combined_full_text:
                        final_answer_index = combined_full_text.find(final_answer_marker)
                        end_of_line_index = combined_full_text.find('\n', final_answer_index)
                        if end_of_line_index == -1:
                            end_of_line_index = len(combined_full_text)
                        else:
                            end_of_line_index += 1

                        truncated_full_text = combined_full_text[:end_of_line_index]
                        final_steps = split_into_steps(truncated_full_text.replace(" ки\n", "\n"))

                        candidates[problem_idx].append({
                            "text": truncated_full_text,
                            "steps": final_steps
                        })
                    else:
                        candidates[problem_idx].append({
                            "text": combined_full_text,
                            "steps": combined_steps
                        })
            
            all_scored_candidates = []
            
            flat_candidates = []
            flat_problem_solution_pairs = []
            problem_id_map = []
            
            for problem_idx, problem_candidates in enumerate(candidates):
                for candidate in problem_candidates:
                    flat_candidates.append(candidate["text"]) 
                    flat_problem_solution_pairs.append((problems[problem_idx], candidate["text"]))
                    problem_id_map.append(problem_idx)

            if flat_candidates:
                logger.info(f"Scoring candidates ....")

                batch_scores = parallel_score_solutions(
                    flat_problem_solution_pairs,
                    prm_tokenizer, 
                    prm_model, 
                    scoring_batch_size,
                    prm_type
                )
                
    
                for i, (candidate_text, score) in enumerate(zip(flat_candidates, batch_scores)):
                    problem_idx = problem_id_map[i]
                    

                    for candidate in candidates[problem_idx]:
                        if candidate["text"] == candidate_text:
                            candidate["score"] = float(score[-1].item())

                for problem_idx in range(num_problems):
                    problem_candidates = candidates[problem_idx]
                    if problem_candidates:
                        sorted_candidates = sorted(problem_candidates, key=lambda x: x["score"], reverse=True)
                        all_beams[problem_idx] = sorted_candidates[:num_beams//beam_width]
            
            torch.cuda.empty_cache()
            
            all_finished = True
            for problem_idx in range(num_problems):
                problem_finished = any(final_answer_marker in beam["text"] for beam in all_beams[problem_idx])
                all_finished = all_finished and problem_finished
            
            if all_finished:
                break
                
    except Exception as e:
        logger.error(f"Error in beam search: {str(e)}")
        
    return all_beams

