import torch
import gc
import logging
from typing import List, Tuple
from .skywork import parallel_score_solutions_skywork_o1

logger = logging.getLogger(__name__)

def parallel_score_solutions(problem_solution_pairs: List[Tuple[str, str]], prm_tokenizer, prm_model, batch_size: int, prm_type: str = 'math-shepherd') -> List[torch.Tensor]:
    """Score solutions with memory safety.
    
    Args:
        problem_solution_pairs: List of (problem, solution) pairs to score
        prm_tokenizer: Tokenizer for the PRM model
        prm_model: The PRM model
        batch_size: Batch size for scoring
        prm_type: Type of PRM model being used ('math-shepherd' or 'skywork-o1')
        
    Returns:
        List of scores for each solution
    """
    if prm_type == 'skywork-o1':
        return parallel_score_solutions_skywork_o1(problem_solution_pairs, prm_tokenizer, prm_model, batch_size)
    
    # Default to math-shepherd scoring
    good_token, bad_token = '+', '-'
    candidate_tokens = prm_tokenizer.encode(f"{good_token} {bad_token}")[1:]
    step_tag_id = prm_tokenizer.encode("ки")[-1]
    
    batch_scores = []
    
    for i in range(0, len(problem_solution_pairs), batch_size):
        batch_pairs = problem_solution_pairs[i:i + batch_size]
        input_texts = [f"{problem} {solution}"[:2048] for problem, solution in batch_pairs]
        
        inputs = prm_tokenizer(
            input_texts,
            padding='max_length',
            return_tensors="pt",
            max_length=2048
        ).to(prm_model.device)
        
        with torch.no_grad():
            logits = prm_model(**inputs).logits[:,:,candidate_tokens]
            scores = logits.softmax(dim=-1)[:,:,0]
            
            for seq_idx, seq in enumerate(inputs.input_ids):
                step_positions = (seq == step_tag_id).nonzero().squeeze(-1)
                if step_positions.numel() > 0:
                    seq_scores = scores[seq_idx][step_positions]
                    seq_scores = seq_scores.float()
                    batch_scores.append(seq_scores.cpu())
                else:
                    batch_scores.append(torch.tensor([0.5], dtype=torch.float32))  #
        
        del inputs
        del logits
        del scores
        
        torch.cuda.empty_cache()
        gc.collect()

    return batch_scores

