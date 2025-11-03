import os
import json
from pathlib import Path

def save_problem_result(problem_idx, beam, problems, problem_ids, problem_subjects, problem_levels, problem_answers, output_path):
    """Save the results for a single problem.
    
    Args:
        problem_idx: Index of the problem in the dataset
        beam: The best beam (solution) for this problem
        problems: List of all problems
        problem_ids: List of all problem IDs
        problem_subjects: List of all problem subjects
        problem_levels: List of all problem levels
        problem_answers: List of all problem answers
        output_path: Base path to save results
    
    Returns:
        str: Path to the saved output file
    """
    final_answer_marker = "Therefore, the final answer"
    complete_beams = [b for b in beam if final_answer_marker in b.get("text", "")]
    
    candidate_beams = complete_beams if complete_beams else beam
    
    processed_beams = []
    for b in candidate_beams:
        beam_text = b["text"]
        if beam_text.endswith(" ки\n"):
            beam_text = beam_text[:-4]  
        
        processed_beams.append({
            "steps": b["steps"],
            "num_steps": len(b["steps"]),
            "score": b["score"] if "score" in b else 0.0,
            "full_text": beam_text,
            "completion_counts": b.get("completion_counts", []),
            "total_completions": sum(b.get("completion_counts", []))
        })
    
    output_data = {
        "problem": problems[problem_idx],
        "subject": problem_subjects[problem_idx],
        "unique_id": problem_ids[problem_idx],
        "level": problem_levels[problem_idx],  
        "correct_answer": problem_answers[problem_idx],
        "solutions": processed_beams,
        "best_solution": processed_beams[0] if processed_beams else {}
    }
    
    subject_dir = output_path / problem_subjects[problem_idx].lower().replace(" ", "_")
    if not os.path.exists(subject_dir):
        os.makedirs(subject_dir)
    
    problem_filename = problem_ids[problem_idx].split('/')[-1]
    output_file = subject_dir / f"{problem_filename}"
    with open(output_file, 'w') as f:
        json.dump(output_data, f, indent=2)
    
    return output_file

def save_aime_problem_result(problem_idx, beams, problems, problem_ids, problem_subjects, problem_levels, problem_answers, problem_years, output_path):
    """Save the results for a single AIME problem.
    
    Args:
        problem_idx: Index of the problem in the dataset
        beams: The beam solutions for this problem
        problems: List of all problems
        problem_ids: List of all problem IDs
        problem_subjects: List of all problem subjects
        problem_levels: List of all problem levels
        problem_answers: List of all problem answers
        problem_years: List of all problem years
        output_path: Base path to save results
    
    Returns:
        str: Path to the saved output file
    """
    final_answer_marker = "Therefore, the final answer"
    complete_beams = [b for b in beams if final_answer_marker in b.get("full_text", "")]
    
    candidate_beams = complete_beams if complete_beams else beams
    
    processed_beams = []
    for b in candidate_beams:
        beam_text = b["text"]
        if beam_text.endswith(" ки\n"):
            beam_text = beam_text[:-4]  
        
        processed_beams.append({
            "steps": b["steps"],
            "num_steps": len(b["steps"]),
            "score": b["score"] if "score" in b else 0.0,
            "full_text": beam_text,
            "completion_counts": b.get("completion_counts", []),
            "total_completions": sum(b.get("completion_counts", []))
        })
    
    output_data = {
        "problem": problems[problem_idx],
        "subject": problem_subjects[problem_idx],
        "unique_id": problem_ids[problem_idx],
        "level": problem_levels[problem_idx],
        "year": problem_years[problem_idx],  
        "correct_answer": problem_answers[problem_idx],
        "solutions": processed_beams,
        "best_solution": processed_beams[0] if processed_beams else {}
    }
    
    year_dir = output_path / f"aime_{problem_years[problem_idx]}"
    if not os.path.exists(year_dir):
        os.makedirs(year_dir)
    
    output_file = year_dir / f"problem_{problem_ids[problem_idx]}.json"
    with open(output_file, 'w') as f:
        json.dump(output_data, f, indent=2)
    
    return output_file

def save_amc23_problem_result(problem_idx, beams, problems, problem_ids, problem_subjects, problem_answers, problem_urls, output_path):
    """Save the results for a single AMC 2023 problem.
    
    Args:
        problem_idx: Index of the problem in the dataset
        beams: The beam solutions for this problem
        problems: List of all problems
        problem_ids: List of all problem IDs
        problem_subjects: List of all problem subjects
        problem_answers: List of all problem answers
        problem_urls: List of problem URLs
        output_path: Base path to save results
    
    Returns:
        str: Path to the saved output file
    """
    final_answer_marker = "Therefore, the final answer"
    complete_beams = [b for b in beams if final_answer_marker in b.get("full_text", "")]
    
    candidate_beams = complete_beams if complete_beams else beams
    
    processed_beams = []
    for b in candidate_beams:
        beam_text = b["text"]
        
        if beam_text.endswith(" ки\n"):
            beam_text = beam_text[:-4]  
        
        processed_beams.append({
            "steps": b["steps"],
            "num_steps": len(b["steps"]),
            "score": b["score"] if "score" in b else 0.0,
            "full_text": beam_text,
            "completion_counts": b.get("completion_counts", []),
            "total_completions": sum(b.get("completion_counts", []))
        })
    
    output_data = {
        "problem": problems[problem_idx],
        "subject": problem_subjects[problem_idx],
        "unique_id": problem_ids[problem_idx],
        "url": problem_urls[problem_idx], 
        "correct_answer": problem_answers[problem_idx],
        "solutions": processed_beams,
        "best_solution": processed_beams[0] if processed_beams else {}
    }
    
    amc_dir = output_path / "amc_2023"
    if not os.path.exists(amc_dir):
        os.makedirs(amc_dir)
    
    output_file = amc_dir / f"problem_{problem_ids[problem_idx]}.json"
    with open(output_file, 'w') as f:
        json.dump(output_data, f, indent=2)
    
    return output_file

def save_agieval_sat_math_problem_result(problem_idx, beams, problems, problem_ids, problem_options, 
                                      problem_gold_indices, problem_answers, output_path):
    """Save the results for a single AGIEval SAT Math problem.
    
    Args:
        problem_idx: Index of the problem in the dataset
        beams: The beam solutions for this problem
        problems: List of all problems
        problem_ids: List of all problem IDs
        problem_options: List of all problem options
        problem_gold_indices: List of gold answer indices
        problem_answers: List of all problem answers
        output_path: Base path to save results
    
    Returns:
        str: Path to the saved output file
    """
    final_answer_marker = "Therefore, the final answer"
    complete_beams = [b for b in beams if final_answer_marker in b.get("full_text", "")]
    
    candidate_beams = complete_beams if complete_beams else beams
    
    processed_beams = []
    for b in candidate_beams:
        beam_text = b["text"]

        if beam_text.endswith(" ки\n"):
            beam_text = beam_text[:-4]  

        processed_beams.append({
            "steps": b["steps"],
            "num_steps": len(b["steps"]),
            "score": b["score"] if "score" in b else 0.0,
            "full_text": beam_text,
            "completion_counts": b.get("completion_counts", []),
            "total_completions": sum(b.get("completion_counts", []))
        })
    
    output_data = {
        "problem": problems[problem_idx],
        "unique_id": problem_ids[problem_idx],
        "options": problem_options[problem_idx],
        "gold_index": int(problem_gold_indices[problem_idx]),
        "correct_answer": problem_answers[problem_idx],
        "solutions": processed_beams,
        "best_solution": processed_beams[0] if processed_beams else {}
    }
    
    sat_math_dir = output_path / "sat_math"
    if not os.path.exists(sat_math_dir):
        os.makedirs(sat_math_dir)
    
    output_file = sat_math_dir / f"{problem_ids[problem_idx]}.json"
    with open(output_file, 'w') as f:
        json.dump(output_data, f, indent=2)
    
    return output_file

