import os
import json
import logging
import torch
import gc
from pathlib import Path
from datetime import datetime
from datasets import load_dataset
from deepspeed.profiling.flops_profiler import FlopsProfiler

logger = logging.getLogger(__name__)

def process_math500_dataset(
    search_algorithm,
    output_dir: str,
    search_batch_size: int,
    scoring_batch_size: int,
    num_problems: int,
    num_beams: int,
    beam_width: int,
    max_steps: int,
    profile: bool,
    profile_frequency: int,
    monitor_gpu: bool,
    monitor_interval: float,
    use_deepspeed: bool,
    prm_model_name: str,
    llm_model_name: str,
    llm, prm_tokenizer, prm_model, prm_type,
    get_batch_size_func,
    save_result_func,
    start_gpu_monitoring_func,
    stop_gpu_monitoring_func,
    get_system_info_func,
    b1=None, b2=None, tau_tokens=None, temperature=0.8
):
    """Process problems from the MATH-500 dataset.
    
    Args:
        search_algorithm: The search algorithm function to use
        output_dir: Directory to save results
        search_batch_size: Batch size for LLM search operations
        scoring_batch_size: Batch size for PRM scoring operations
        num_problems: Number of problems to process (None for all)
        num_beams: Number of beams for beam search
        beam_width: Beam width for beam search
        max_steps: Maximum number of steps for beam search
        profile: Whether to enable profiling for each batch
        profile_frequency: How often to profile (every N batches)
        monitor_gpu: Whether to enable continuous GPU monitoring
        monitor_interval: Interval in seconds between GPU measurements
        use_deepspeed: Whether to use DeepSpeed's Flops Profiler
        prm_model_name: PRM model to use
        llm_model_name: LLM model to use
        llm: The LLM model
        prm_tokenizer: PRM tokenizer
        prm_model: PRM model
        prm_type: Type of PRM
        get_batch_size_func: Function to get batch sizes
        save_result_func: Function to save results
        start_gpu_monitoring_func: Function to start GPU monitoring
        stop_gpu_monitoring_func: Function to stop GPU monitoring
        get_system_info_func: Function to get system info
        b1: Batch size for partial generation (optional)
        b2: Batch size for completion (optional)
        tau_tokens: Max tokens for partial generation (optional)
        temperature: Temperature for generation
    """
    ds_profiler = None
    if use_deepspeed:
        ds_profile_dir = os.path.join(output_dir, "deepspeed_flops_profile")
        os.makedirs(ds_profile_dir, exist_ok=True)
        logger.info("DeepSpeed FlopsProfiler will be initialized after model loading")

    if profile and not use_deepspeed:
        from torch.profiler import profile, ProfilerActivity, tensorboard_trace_handler
        profile_dir = os.path.join(output_dir, "profiling_logs")
        os.makedirs(profile_dir, exist_ok=True)
        total_flops = 0
        
    monitor_thread = None
    if monitor_gpu:
        monitor_dir = os.path.join(output_dir, "gpu_monitoring")
        monitor_thread = start_gpu_monitoring_func(monitor_dir, monitor_interval)

    logger.info("Initial System State:")
    for gpu in get_system_info_func():
        logger.info(f"GPU {gpu['id']} ({gpu['name']}):")
        logger.info(f"Total Memory: {gpu['memory_total']} MB")
        logger.info(f"Used Memory: {gpu['memory_used']} MB")
        logger.info(f"Free Memory: {gpu['memory_free']} MB")
        logger.info(f"Temperature: {gpu['temperature']}°C")

    ds_profiler_prm = None
    ds_profiler_llm = None
    if use_deepspeed:
        ds_profiler_prm = FlopsProfiler(prm_model)
        ds_profiler_llm = FlopsProfiler(llm.model)
        logger.info("DeepSpeed FlopsProfilers initialized for both PRM and LLM models")

    output_path = Path(output_dir)
    if not os.path.exists(output_path):
        os.makedirs(output_path)
    
    logger.info("Loading MATH-500 dataset from HuggingFace...")
    math_dataset = load_dataset("HuggingFaceH4/MATH-500")
    
    problems = []
    problem_ids = []
    problem_subjects = []
    problem_answers = []
    problem_levels = []
    
    dataset_items = math_dataset["test"]
    
    if num_problems is not None:
        dataset_items = dataset_items.select(range(min(num_problems, len(dataset_items))))
    
    for item in dataset_items:
        problems.append(item["problem"])
        problem_ids.append(item["unique_id"])
        problem_subjects.append(item["subject"])
        problem_answers.append(item["answer"])
        problem_levels.append(item["level"])
    batch_size_result = get_batch_size_func(llm_model_name if b1 is None else num_beams)
    if isinstance(batch_size_result, tuple):
        processing_batch_size, llm_batch_size = batch_size_result
    else:
        llm_batch_size = batch_size_result
        processing_batch_size = int(llm_batch_size / num_beams)
    
    logger.info(f"Processing batch size: {processing_batch_size}, llm_batch_size: {llm_batch_size}")
    total_batches = (len(problems) // processing_batch_size) + (1 if len(problems) % processing_batch_size > 0 else 0)
    
    try:
        for batch_idx in range(0, len(problems), processing_batch_size):
            try:
                batch_number = batch_idx // processing_batch_size + 1
                logger.info(f"Processing batch {batch_number}/{total_batches}")
                
                batch_problems = problems[batch_idx:batch_idx + processing_batch_size]
                
                should_profile = batch_number % profile_frequency == 0
                
                if should_profile and use_deepspeed and ds_profiler_prm is not None:
                    logger.info(f"Using DeepSpeed FlopsProfiler for batch {batch_number}")
                    
                    prm_profile_dir = os.path.join(ds_profile_dir, "prm_model")
                    llm_profile_dir = os.path.join(ds_profile_dir, "llm_model")
                    os.makedirs(prm_profile_dir, exist_ok=True)
                    os.makedirs(llm_profile_dir, exist_ok=True)
                    
                    ds_profiler_llm.start_profile()
                    ds_profiler_prm.start_profile()
                    
                    if b1 is not None:
                        batch_beams = search_algorithm(
                            batch_problems, llm, prm_tokenizer, prm_model,
                            num_beams=num_beams, beam_width=beam_width, max_steps=max_steps,
                            b1=llm_batch_size, b2=b2, scoring_batch_size=scoring_batch_size,
                            tau_tokens=tau_tokens, temperature=temperature, prm_type=prm_type
                        )
                    else:
                        batch_beams = search_algorithm(
                            batch_problems, llm, prm_tokenizer, prm_model,
                            scoring_batch_size, llm_batch_size,
                            num_beams=num_beams, beam_width=beam_width,
                            max_steps=max_steps, prm_type=prm_type
                        )
                    
                    ds_profiler_prm.stop_profile()
                    ds_profiler_llm.stop_profile()
                    
                    prm_flops = ds_profiler_prm.get_total_flops()
                    prm_macs = ds_profiler_prm.get_total_macs()
                    llm_flops = ds_profiler_llm.get_total_flops()
                    llm_macs = ds_profiler_llm.get_total_macs()
                    
                    prm_tflops = prm_flops / 1e12
                    llm_tflops = llm_flops / 1e12
                    total_tflops = (prm_flops + llm_flops) / 1e12
                    
                    logger.info(f"Batch {batch_number} Profiling Results:")
                    logger.info(f"PRM Model - Total FLOPS: {prm_flops:,} ({prm_tflops:.4f} TFLOPS)")
                    logger.info(f"PRM Model - Total MACs: {prm_macs:,}")
                    logger.info(f"LLM Model - Total FLOPS: {llm_flops:,} ({llm_tflops:.4f} TFLOPS)")
                    logger.info(f"LLM Model - Total MACs: {llm_macs:,}")
                    logger.info(f"Combined Models - Total FLOPS: {prm_flops+llm_flops:,} ({total_tflops:.4f} TFLOPS")
                    
                    prm_profile_path = os.path.join(prm_profile_dir, f"batch_{batch_number}_summary.txt")
                    prm_detailed_path = os.path.join(prm_profile_dir, f"batch_{batch_number}_detailed_profile.txt")
                    
                    try:
                        ds_profiler_prm.print_model_profile(output_file=prm_detailed_path)
                        logger.info(f"Saved detailed PRM model profile to {prm_detailed_path}")
                    except TypeError:
                        logger.warning("PRM print_model_profile doesn't support output_file in this version")
                    
                    with open(prm_profile_path, 'w') as f:
                        f.write(f"PRM Model Profile for Batch {batch_number}\n")
                        f.write(f"Total FLOPS: {prm_flops:,} ({prm_tflops:.4f} TFLOPS)\n")
                        f.write(f"Total MACs: {prm_macs:,}\n")
                    
                    llm_profile_path = os.path.join(llm_profile_dir, f"batch_{batch_number}_summary.txt")
                    llm_detailed_path = os.path.join(llm_profile_dir, f"batch_{batch_number}_detailed_profile.txt")
                    
                    try:
                        ds_profiler_llm.print_model_profile(output_file=llm_detailed_path)
                        logger.info(f"Saved detailed LLM model profile to {llm_detailed_path}")
                    except TypeError:
                        logger.warning("LLM print_model_profile doesn't support output_file in this version")
                    
                    with open(llm_profile_path, 'w') as f:
                        f.write(f"LLM Model Profile for Batch {batch_number}\n")
                        f.write(f"Total FLOPS: {llm_flops:,} ({llm_tflops:.4f} TFLOPS)\n")
                        f.write(f"Total MACs: {llm_macs:,}\n")
                    
                    combined_summary_path = os.path.join(ds_profile_dir, f"batch_{batch_number}_combined_summary.json")
                    with open(combined_summary_path, 'w') as f:
                        json.dump({
                            'prm_flops': prm_flops,
                            'prm_tflops': prm_tflops,
                            'prm_macs': prm_macs,
                            'llm_flops': llm_flops,
                            'llm_tflops': llm_tflops,
                            'llm_macs': llm_macs,
                            'total_flops': prm_flops + llm_flops,
                            'total_tflops': total_tflops,
                            'total_macs': prm_macs + llm_macs,
                            'batch_size': len(batch_problems),
                            'timestamp': datetime.now().strftime("%Y-%m-%d %H:%M:%S")
                        }, f, indent=2)
                    
                    ds_profiler_prm.end_profile()
                    ds_profiler_llm.end_profile()
                    
                elif should_profile and profile and not use_deepspeed:
                    logger.info(f"Profiling batch {batch_number} with PyTorch profiler")
                    batch_trace_handler = tensorboard_trace_handler(
                        os.path.join(profile_dir, f"batch_{batch_number}")
                    )
                    
                    with profile(
                        activities=[ProfilerActivity.CUDA, ProfilerActivity.CPU],
                        with_flops=True,
                        profile_memory=False,
                        with_stack=False,
                        record_shapes=False
                    ) as prof:
                        if b1 is not None:
                            batch_beams = search_algorithm(
                                batch_problems, llm, prm_tokenizer, prm_model,
                                num_beams=num_beams, beam_width=beam_width, max_steps=max_steps,
                                b1=llm_batch_size, b2=b2, scoring_batch_size=scoring_batch_size,
                                tau_tokens=tau_tokens, temperature=temperature, prm_type=prm_type
                            )
                        else:
                            batch_beams = search_algorithm(
                                batch_problems, llm, prm_tokenizer, prm_model,
                                scoring_batch_size, llm_batch_size,
                                num_beams=num_beams, beam_width=beam_width,
                                max_steps=max_steps, prm_type=prm_type
                            )
                    
                    batch_flops = 0
                    for evt in prof.key_averages():
                        if hasattr(evt, 'flops') and evt.flops > 0:
                            batch_flops += evt.flops
                    
                    total_flops += batch_flops
                    logger.info(f"Batch {batch_number} FLOPs: {batch_flops/1e12:.2f} TFLOPs")
                    
                    with open(os.path.join(profile_dir, f"batch_{batch_number}_flops.txt"), "w") as f:
                        f.write(f"{batch_flops/1e12:.2f}")
                    
                    gpu_info = get_system_info_func()
                    with open(os.path.join(profile_dir, f"batch_{batch_number}_gpu_info.json"), "w") as f:
                        json.dump(gpu_info, f, indent=2)
                    
                    torch.cuda.empty_cache()
                    gc.collect()
                else:
                    if b1 is not None:
                        batch_beams = search_algorithm(
                            batch_problems, llm, prm_tokenizer, prm_model,
                            num_beams=num_beams, beam_width=beam_width, max_steps=max_steps,
                            b1=llm_batch_size, b2=b2, scoring_batch_size=scoring_batch_size,
                            tau_tokens=tau_tokens, temperature=temperature, prm_type=prm_type
                        )
                    else:
                        batch_beams = search_algorithm(
                            batch_problems, llm, prm_tokenizer, prm_model,
                            scoring_batch_size, llm_batch_size,
                            num_beams=num_beams, beam_width=beam_width,
                            max_steps=max_steps, prm_type=prm_type
                        )
                
                for i, beams in enumerate(batch_beams):
                    idx = batch_idx + i
                    if idx < len(problems):
                        output_file = save_result_func(
                            idx, beams, problems, problem_ids, problem_subjects, 
                            problem_levels, problem_answers, output_path
                        )
                        logger.info(f"Processed problem {idx+1}/{len(problems)}: {problem_ids[idx]} -> {output_file}")
                
                torch.cuda.empty_cache()
                gc.collect()
                
            except Exception as e:
                logger.error(f"Error processing batch {batch_idx}: {str(e)}", exc_info=True)
    
    finally:
        if monitor_gpu and monitor_thread:
            stop_gpu_monitoring_func(monitor_thread)
        
        if use_deepspeed:
            if ds_profiler_prm is not None:
                ds_profiler_prm.end_profile()
            if ds_profiler_llm is not None:
                ds_profiler_llm.end_profile()
        
        if profile and not use_deepspeed:
            logger.info(f"Total FLOPs across all profiled batches: {total_flops/1e12:.2f} TFLOPs")
            with open(os.path.join(profile_dir, "total_flops.txt"), "w") as f:
                f.write(f"{total_flops/1e12:.2f}")
        
        logger.info(f"Processing complete.")


def process_aime_dataset(
    search_algorithm,
    output_dir: str,
    search_batch_size: int,
    scoring_batch_size: int,
    num_problems: int,
    num_beams: int,
    beam_width: int,
    max_steps: int,
    profile: bool,
    profile_frequency: int,
    monitor_gpu: bool,
    monitor_interval: float,
    use_deepspeed: bool,
    prm_model_name: str,
    llm_model_name: str,
    llm, prm_tokenizer, prm_model, prm_type,
    get_batch_size_func,
    save_result_func,
    start_gpu_monitoring_func,
    stop_gpu_monitoring_func,
    get_system_info_func,
    b1=None, b2=None, tau_tokens=None, temperature=0.8
):
    """Process problems from the AIME 2024 dataset."""
    ds_profiler_prm = None
    ds_profiler_llm = None
    if use_deepspeed:
        ds_profile_dir = os.path.join(output_dir, "deepspeed_flops_profile")
        os.makedirs(ds_profile_dir, exist_ok=True)
        ds_profiler_prm = FlopsProfiler(prm_model)
        ds_profiler_llm = FlopsProfiler(llm.model)
        logger.info("DeepSpeed FlopsProfilers initialized")

    monitor_thread = None
    if monitor_gpu:
        monitor_dir = os.path.join(output_dir, "gpu_monitoring")
        monitor_thread = start_gpu_monitoring_func(monitor_dir, monitor_interval)

    output_path = Path(output_dir)
    os.makedirs(output_path, exist_ok=True)
    
    logger.info("Loading AIME 2024 dataset from HuggingFace...")
    aime_dataset = load_dataset("HuggingFaceH4/aime_2024")
    
    problems = []
    problem_ids = []
    problem_subjects = []
    problem_answers = []
    problem_levels = []
    problem_years = []
    
    dataset_items = aime_dataset["train"]
    
    if num_problems is not None:
        dataset_items = dataset_items.select(range(min(num_problems, len(dataset_items))))
    
    for item in dataset_items:
        problems.append(item["problem"])
        problem_ids.append(str(item["id"]))
        problem_subjects.append("AIME")
        problem_answers.append(item["answer"])
        problem_levels.append("AIME")
        problem_years.append(item["year"])
    
    logger.info(f"Processing {len(problems)} problems from AIME 2024 dataset")
    batch_size_result = get_batch_size_func(llm_model_name if b1 is None else num_beams)
    if isinstance(batch_size_result, tuple):
        processing_batch_size, llm_batch_size = batch_size_result
    else:
        llm_batch_size = batch_size_result
        processing_batch_size = int(llm_batch_size / num_beams)
    
    logger.info(f"Processing batch size: {processing_batch_size}, llm_batch_size: {llm_batch_size}")
    total_batches = (len(problems) // processing_batch_size) + (1 if len(problems) % processing_batch_size > 0 else 0)
    
    try:
        for batch_idx in range(0, len(problems), processing_batch_size):
            try:
                batch_number = batch_idx // processing_batch_size + 1
                logger.info(f"Processing batch {batch_number}/{total_batches}")
                
                batch_problems = problems[batch_idx:batch_idx + processing_batch_size]
                
                should_profile = batch_number % profile_frequency == 0
                
                if should_profile and use_deepspeed and ds_profiler_prm is not None:
                    logger.info(f"Using DeepSpeed FlopsProfiler for batch {batch_number}")
                    
                    prm_profile_dir = os.path.join(ds_profile_dir, "prm_model")
                    llm_profile_dir = os.path.join(ds_profile_dir, "llm_model")
                    os.makedirs(prm_profile_dir, exist_ok=True)
                    os.makedirs(llm_profile_dir, exist_ok=True)
                    
                    ds_profiler_llm.start_profile()
                    ds_profiler_prm.start_profile()
                    if b1 is not None:
                        batch_beams = search_algorithm(
                            batch_problems, llm, prm_tokenizer, prm_model,
                            num_beams=num_beams, beam_width=beam_width, max_steps=max_steps,
                            b1=llm_batch_size, b2=b2, scoring_batch_size=scoring_batch_size,
                            tau_tokens=tau_tokens, temperature=temperature, prm_type=prm_type
                        )
                    else:
                        batch_beams = search_algorithm(
                            batch_problems, llm, prm_tokenizer, prm_model,
                            scoring_batch_size, llm_batch_size,
                            num_beams=num_beams, beam_width=beam_width,
                            max_steps=max_steps, prm_type=prm_type
                        )
                    
                    ds_profiler_prm.stop_profile()
                    ds_profiler_llm.stop_profile()
                    
                    prm_flops = ds_profiler_prm.get_total_flops()
                    prm_macs = ds_profiler_prm.get_total_macs()
                    llm_flops = ds_profiler_llm.get_total_flops()
                    llm_macs = ds_profiler_llm.get_total_macs()
                    
                    prm_tflops = prm_flops / 1e12
                    llm_tflops = llm_flops / 1e12
                    total_tflops = (prm_flops + llm_flops) / 1e12
                    
                    logger.info(f"Batch {batch_number} Profiling Results:")
                    logger.info(f"PRM Model - Total FLOPS: {prm_flops:,} ({prm_tflops:.4f} TFLOPS)")
                    logger.info(f"PRM Model - Total MACs: {prm_macs:,}")
                    logger.info(f"LLM Model - Total FLOPS: {llm_flops:,} ({llm_tflops:.4f} TFLOPS)")
                    logger.info(f"LLM Model - Total MACs: {llm_macs:,}")
                    logger.info(f"Combined Models - Total FLOPS: {prm_flops+llm_flops:,} ({total_tflops:.4f} TFLOPS)")
                    
                    prm_profile_path = os.path.join(prm_profile_dir, f"batch_{batch_number}_summary.txt")
                    with open(prm_profile_path, 'w') as f:
                        f.write(f"PRM Model Profile for Batch {batch_number}\n")
                        f.write(f"Total FLOPS: {prm_flops:,} ({prm_tflops:.4f} TFLOPS)\n")
                        f.write(f"Total MACs: {prm_macs:,}\n")
                    
                    llm_profile_path = os.path.join(llm_profile_dir, f"batch_{batch_number}_summary.txt")
                    with open(llm_profile_path, 'w') as f:
                        f.write(f"LLM Model Profile for Batch {batch_number}\n")
                        f.write(f"Total FLOPS: {llm_flops:,} ({llm_tflops:.4f} TFLOPS)\n")
                        f.write(f"Total MACs: {llm_macs:,}\n")
                    
                    combined_summary_path = os.path.join(ds_profile_dir, f"batch_{batch_number}_combined_summary.json")
                    with open(combined_summary_path, 'w') as f:
                        json.dump({
                            'prm_flops': prm_flops,
                            'prm_tflops': prm_tflops,
                            'prm_macs': prm_macs,
                            'llm_flops': llm_flops,
                            'llm_tflops': llm_tflops,
                            'llm_macs': llm_macs,
                            'total_flops': prm_flops + llm_flops,
                            'total_tflops': total_tflops,
                            'total_macs': prm_macs + llm_macs,
                            'batch_size': len(batch_problems),
                            'timestamp': datetime.now().strftime("%Y-%m-%d %H:%M:%S")
                        }, f, indent=2)
                    
                    ds_profiler_prm.end_profile()
                    ds_profiler_llm.end_profile()
                else:
                    if b1 is not None:
                        batch_beams = search_algorithm(
                            batch_problems, llm, prm_tokenizer, prm_model,
                            num_beams=num_beams, beam_width=beam_width, max_steps=max_steps,
                            b1=llm_batch_size, b2=b2, scoring_batch_size=scoring_batch_size,
                            tau_tokens=tau_tokens, temperature=temperature, prm_type=prm_type
                        )
                    else:
                        batch_beams = search_algorithm(
                            batch_problems, llm, prm_tokenizer, prm_model,
                            scoring_batch_size, llm_batch_size,
                            num_beams=num_beams, beam_width=beam_width,
                            max_steps=max_steps, prm_type=prm_type
                        )
                
                for i, beams in enumerate(batch_beams):
                    idx = batch_idx + i
                    if idx < len(problems):
                        output_file = save_result_func(
                            idx, beams, problems, problem_ids, problem_subjects, 
                            problem_levels, problem_answers, problem_years, output_path
                        )
                        logger.info(f"Processed problem {idx+1}/{len(problems)}: {problem_ids[idx]} -> {output_file}")
                
                torch.cuda.empty_cache()
                gc.collect()
                
            except Exception as e:
                logger.error(f"Error processing batch {batch_idx}: {str(e)}", exc_info=True)
    
    finally:
        if monitor_gpu and monitor_thread:
            stop_gpu_monitoring_func(monitor_thread)
        if use_deepspeed:
            if ds_profiler_prm is not None:
                ds_profiler_prm.end_profile()
            if ds_profiler_llm is not None:
                ds_profiler_llm.end_profile()
        logger.info(f"Processing complete.")


def process_amc23_dataset(
    search_algorithm,
    output_dir: str,
    search_batch_size: int,
    scoring_batch_size: int,
    num_problems: int,
    num_beams: int,
    beam_width: int,
    max_steps: int,
    profile: bool,
    profile_frequency: int,
    monitor_gpu: bool,
    monitor_interval: float,
    use_deepspeed: bool,
    prm_model_name: str,
    llm_model_name: str,
    llm, prm_tokenizer, prm_model, prm_type,
    get_batch_size_func,
    save_result_func,
    start_gpu_monitoring_func,
    stop_gpu_monitoring_func,
    get_system_info_func,
    b1=None, b2=None, tau_tokens=None, temperature=0.8
):
    """Process problems from the AMC 2023 dataset."""
    ds_profiler_prm = None
    ds_profiler_llm = None
    if use_deepspeed:
        ds_profile_dir = os.path.join(output_dir, "deepspeed_flops_profile")
        os.makedirs(ds_profile_dir, exist_ok=True)
        ds_profiler_prm = FlopsProfiler(prm_model)
        ds_profiler_llm = FlopsProfiler(llm.model)
        logger.info("DeepSpeed FlopsProfilers initialized")

    monitor_thread = None
    if monitor_gpu:
        monitor_dir = os.path.join(output_dir, "gpu_monitoring")
        monitor_thread = start_gpu_monitoring_func(monitor_dir, monitor_interval)

    output_path = Path(output_dir)
    os.makedirs(output_path, exist_ok=True)
    
    logger.info("Loading AMC 2023 dataset from HuggingFace...")
    amc_dataset = load_dataset("zwhe99/amc23")
    problems = []
    problem_ids = []
    problem_subjects = []
    problem_answers = []
    problem_urls = []
    
    dataset_items = amc_dataset["test"]
    
    if num_problems is not None:
        dataset_items = dataset_items.select(range(min(num_problems, len(dataset_items))))
    
    for item in dataset_items:
        problems.append(item["question"])
        problem_ids.append(str(item["id"]))
        problem_subjects.append("AMC")
        problem_answers.append(str(item["answer"]))
        problem_urls.append(item["url"])
    
    logger.info(f"Processing {len(problems)} problems from AMC 2023 dataset")
    
    batch_size_result = get_batch_size_func(llm_model_name if b1 is None else num_beams)
    if isinstance(batch_size_result, tuple):
        processing_batch_size, llm_batch_size = batch_size_result
    else:
        llm_batch_size = batch_size_result
        processing_batch_size = int(llm_batch_size / num_beams)
    
    logger.info(f"Processing batch size: {processing_batch_size}, llm_batch_size: {llm_batch_size}")
    total_batches = (len(problems) // processing_batch_size) + (1 if len(problems) % processing_batch_size > 0 else 0)
    
    try:
        for batch_idx in range(0, len(problems), processing_batch_size):
            try:
                batch_number = batch_idx // processing_batch_size + 1
                logger.info(f"Processing batch {batch_number}/{total_batches}")
                
                batch_problems = problems[batch_idx:batch_idx + processing_batch_size]
                
                should_profile = batch_number % profile_frequency == 0
                
                if should_profile and use_deepspeed and ds_profiler_prm is not None:
                    logger.info(f"Using DeepSpeed FlopsProfiler for batch {batch_number}")
                    
                    prm_profile_dir = os.path.join(ds_profile_dir, "prm_model")
                    llm_profile_dir = os.path.join(ds_profile_dir, "llm_model")
                    os.makedirs(prm_profile_dir, exist_ok=True)
                    os.makedirs(llm_profile_dir, exist_ok=True)
                    
                    ds_profiler_llm.start_profile()
                    ds_profiler_prm.start_profile()
                    if b1 is not None:
                        batch_beams = search_algorithm(
                            batch_problems, llm, prm_tokenizer, prm_model,
                            num_beams=num_beams, beam_width=beam_width, max_steps=max_steps,
                            b1=llm_batch_size, b2=b2, scoring_batch_size=scoring_batch_size,
                            tau_tokens=tau_tokens, temperature=temperature, prm_type=prm_type
                        )
                    else:
                        batch_beams = search_algorithm(
                            batch_problems, llm, prm_tokenizer, prm_model,
                            scoring_batch_size, llm_batch_size,
                            num_beams=num_beams, beam_width=beam_width,
                            max_steps=max_steps, prm_type=prm_type
                        )
                    
                    ds_profiler_prm.stop_profile()
                    ds_profiler_llm.stop_profile()
                    
                    prm_flops = ds_profiler_prm.get_total_flops()
                    prm_macs = ds_profiler_prm.get_total_macs()
                    llm_flops = ds_profiler_llm.get_total_flops()
                    llm_macs = ds_profiler_llm.get_total_macs()
                    
                    prm_tflops = prm_flops / 1e12
                    llm_tflops = llm_flops / 1e12
                    total_tflops = (prm_flops + llm_flops) / 1e12
                    
                    logger.info(f"Batch {batch_number} Profiling Results:")
                    logger.info(f"PRM Model - Total FLOPS: {prm_flops:,} ({prm_tflops:.4f} TFLOPS)")
                    logger.info(f"PRM Model - Total MACs: {prm_macs:,}")
                    logger.info(f"LLM Model - Total FLOPS: {llm_flops:,} ({llm_tflops:.4f} TFLOPS)")
                    logger.info(f"LLM Model - Total MACs: {llm_macs:,}")
                    logger.info(f"Combined Models - Total FLOPS: {prm_flops+llm_flops:,} ({total_tflops:.4f} TFLOPS)")
                    
                    prm_profile_path = os.path.join(prm_profile_dir, f"batch_{batch_number}_summary.txt")
                    with open(prm_profile_path, 'w') as f:
                        f.write(f"PRM Model Profile for Batch {batch_number}\n")
                        f.write(f"Total FLOPS: {prm_flops:,} ({prm_tflops:.4f} TFLOPS)\n")
                        f.write(f"Total MACs: {prm_macs:,}\n")
                    
                    llm_profile_path = os.path.join(llm_profile_dir, f"batch_{batch_number}_summary.txt")
                    with open(llm_profile_path, 'w') as f:
                        f.write(f"LLM Model Profile for Batch {batch_number}\n")
                        f.write(f"Total FLOPS: {llm_flops:,} ({llm_tflops:.4f} TFLOPS)\n")
                        f.write(f"Total MACs: {llm_macs:,}\n")
                    
                    combined_summary_path = os.path.join(ds_profile_dir, f"batch_{batch_number}_combined_summary.json")
                    with open(combined_summary_path, 'w') as f:
                        json.dump({
                            'prm_flops': prm_flops,
                            'prm_tflops': prm_tflops,
                            'prm_macs': prm_macs,
                            'llm_flops': llm_flops,
                            'llm_tflops': llm_tflops,
                            'llm_macs': llm_macs,
                            'total_flops': prm_flops + llm_flops,
                            'total_tflops': total_tflops,
                            'total_macs': prm_macs + llm_macs,
                            'batch_size': len(batch_problems),
                            'timestamp': datetime.now().strftime("%Y-%m-%d %H:%M:%S")
                        }, f, indent=2)
                    
                    ds_profiler_prm.end_profile()
                    ds_profiler_llm.end_profile()
                else:
                    if b1 is not None:
                        batch_beams = search_algorithm(
                            batch_problems, llm, prm_tokenizer, prm_model,
                            num_beams=num_beams, beam_width=beam_width, max_steps=max_steps,
                            b1=llm_batch_size, b2=b2, scoring_batch_size=scoring_batch_size,
                            tau_tokens=tau_tokens, temperature=temperature, prm_type=prm_type
                        )
                    else:
                        batch_beams = search_algorithm(
                            batch_problems, llm, prm_tokenizer, prm_model,
                            scoring_batch_size, llm_batch_size,
                            num_beams=num_beams, beam_width=beam_width,
                            max_steps=max_steps, prm_type=prm_type
                        )
                
                for i, beams in enumerate(batch_beams):
                    idx = batch_idx + i
                    if idx < len(problems):
                        output_file = save_result_func(
                            idx, beams, problems, problem_ids, problem_subjects, 
                            problem_answers, problem_urls, output_path
                        )
                        logger.info(f"Processed problem {idx+1}/{len(problems)}: {problem_ids[idx]} -> {output_file}")
                
                torch.cuda.empty_cache()
                gc.collect()
                
            except Exception as e:
                logger.error(f"Error processing batch {batch_idx}: {str(e)}", exc_info=True)
    
    finally:
        if monitor_gpu and monitor_thread:
            stop_gpu_monitoring_func(monitor_thread)
        if use_deepspeed:
            if ds_profiler_prm is not None:
                ds_profiler_prm.end_profile()
            if ds_profiler_llm is not None:
                ds_profiler_llm.end_profile()
        logger.info(f"Processing complete.")


def process_agieval_sat_math_dataset(
    search_algorithm,
    output_dir: str,
    search_batch_size: int,
    scoring_batch_size: int,
    num_problems: int,
    num_beams: int,
    beam_width: int,
    max_steps: int,
    profile: bool,
    profile_frequency: int,
    monitor_gpu: bool,
    monitor_interval: float,
    use_deepspeed: bool,
    prm_model_name: str,
    llm_model_name: str,
    llm, prm_tokenizer, prm_model, prm_type,
    get_batch_size_func,
    save_result_func,
    start_gpu_monitoring_func,
    stop_gpu_monitoring_func,
    get_system_info_func,
    b1=None, b2=None, tau_tokens=None, temperature=0.8
):
    """Process problems from the AGIEval SAT Math dataset."""
    ds_profiler_prm = None
    ds_profiler_llm = None
    if use_deepspeed:
        ds_profile_dir = os.path.join(output_dir, "deepspeed_flops_profile")
        os.makedirs(ds_profile_dir, exist_ok=True)
        ds_profiler_prm = FlopsProfiler(prm_model)
        ds_profiler_llm = FlopsProfiler(llm.model)
        logger.info("DeepSpeed FlopsProfilers initialized")

    monitor_thread = None
    if monitor_gpu:
        monitor_dir = os.path.join(output_dir, "gpu_monitoring")
        monitor_thread = start_gpu_monitoring_func(monitor_dir, monitor_interval)

    output_path = Path(output_dir)
    os.makedirs(output_path, exist_ok=True)
    
    logger.info("Loading AGIEval SAT Math dataset from HuggingFace...")
    sat_math_dataset = load_dataset("hails/agieval-sat-math")
    problems = []
    problem_ids = []
    problem_answers = []
    problem_options = []
    problem_gold_indices = []
    
    dataset_items = sat_math_dataset["test"]
    
    if num_problems is not None:
        dataset_items = dataset_items.select(range(min(num_problems, len(dataset_items))))
    
    for idx, item in enumerate(dataset_items):
        query = item["query"]
        
        answer_choices_pos = query.find("Answer Choices:")
        if answer_choices_pos == -1:
            answer_choices_pos = query.rfind("A:")
        
        if answer_choices_pos != -1:
            problem_text = query[:answer_choices_pos].strip()
        else:
            problem_text = query
            
        final_marker_pos = problem_text.rfind("A: Among")
        if final_marker_pos != -1:
            problem_text = problem_text[:final_marker_pos].strip()
        
        if problem_text.startswith("Q: "):
            problem_text = problem_text[3:].strip()
            
        problems.append(problem_text)
        problem_ids.append(f"sat_math_{idx}")
        
        choices = item["choices"]
        problem_options.append(choices)
        
        gold_idx = item["gold"][0]
        problem_gold_indices.append(gold_idx)
        
        if 0 <= gold_idx < len(choices):
            correct_answer = choices[gold_idx]
            if correct_answer.startswith("(") and ")" in correct_answer:
                option_prefix_end = correct_answer.find(")") + 1
                correct_answer = correct_answer[option_prefix_end:].strip()
            problem_answers.append(correct_answer)
        else:
            problem_answers.append("Unknown")
    
    logger.info(f"Processing {len(problems)} problems from AGIEval SAT Math dataset")
    
    batch_size_result = get_batch_size_func(llm_model_name if b1 is None else num_beams)
    if isinstance(batch_size_result, tuple):
        processing_batch_size, llm_batch_size = batch_size_result
    else:
        llm_batch_size = batch_size_result
        processing_batch_size = int(llm_batch_size / num_beams)
    
    logger.info(f"Processing batch size: {processing_batch_size}, llm_batch_size: {llm_batch_size}")
    total_batches = (len(problems) // processing_batch_size) + (1 if len(problems) % processing_batch_size > 0 else 0)
    
    try:
        for batch_idx in range(0, len(problems), processing_batch_size):
            try:
                batch_number = batch_idx // processing_batch_size + 1
                logger.info(f"Processing batch {batch_number}/{total_batches}")
                
                batch_problems = problems[batch_idx:batch_idx + processing_batch_size]
                
                should_profile = batch_number % profile_frequency == 0
                
                if should_profile and use_deepspeed and ds_profiler_prm is not None:
                    logger.info(f"Using DeepSpeed FlopsProfiler for batch {batch_number}")
                    
                    prm_profile_dir = os.path.join(ds_profile_dir, "prm_model")
                    llm_profile_dir = os.path.join(ds_profile_dir, "llm_model")
                    os.makedirs(prm_profile_dir, exist_ok=True)
                    os.makedirs(llm_profile_dir, exist_ok=True)
                    
                    ds_profiler_llm.start_profile()
                    ds_profiler_prm.start_profile()
                    if b1 is not None:
                        batch_beams = search_algorithm(
                            batch_problems, llm, prm_tokenizer, prm_model,
                            num_beams=num_beams, beam_width=beam_width, max_steps=max_steps,
                            b1=llm_batch_size, b2=b2, scoring_batch_size=scoring_batch_size,
                            tau_tokens=tau_tokens, temperature=temperature, prm_type=prm_type
                        )
                    else:
                        batch_beams = search_algorithm(
                            batch_problems, llm, prm_tokenizer, prm_model,
                            scoring_batch_size, llm_batch_size,
                            num_beams=num_beams, beam_width=beam_width,
                            max_steps=max_steps, prm_type=prm_type
                        )
                    
                    ds_profiler_prm.stop_profile()
                    ds_profiler_llm.stop_profile()
                    
                    prm_flops = ds_profiler_prm.get_total_flops()
                    prm_macs = ds_profiler_prm.get_total_macs()
                    llm_flops = ds_profiler_llm.get_total_flops()
                    llm_macs = ds_profiler_llm.get_total_macs()
                    
                    prm_tflops = prm_flops / 1e12
                    llm_tflops = llm_flops / 1e12
                    total_tflops = (prm_flops + llm_flops) / 1e12
                    
                    logger.info(f"Batch {batch_number} Profiling Results:")
                    logger.info(f"PRM Model - Total FLOPS: {prm_flops:,} ({prm_tflops:.4f} TFLOPS)")
                    logger.info(f"PRM Model - Total MACs: {prm_macs:,}")
                    logger.info(f"LLM Model - Total FLOPS: {llm_flops:,} ({llm_tflops:.4f} TFLOPS)")
                    logger.info(f"LLM Model - Total MACs: {llm_macs:,}")
                    logger.info(f"Combined Models - Total FLOPS: {prm_flops+llm_flops:,} ({total_tflops:.4f} TFLOPS)")
                    
                    prm_profile_path = os.path.join(prm_profile_dir, f"batch_{batch_number}_summary.txt")
                    with open(prm_profile_path, 'w') as f:
                        f.write(f"PRM Model Profile for Batch {batch_number}\n")
                        f.write(f"Total FLOPS: {prm_flops:,} ({prm_tflops:.4f} TFLOPS)\n")
                        f.write(f"Total MACs: {prm_macs:,}\n")
                    
                    llm_profile_path = os.path.join(llm_profile_dir, f"batch_{batch_number}_summary.txt")
                    with open(llm_profile_path, 'w') as f:
                        f.write(f"LLM Model Profile for Batch {batch_number}\n")
                        f.write(f"Total FLOPS: {llm_flops:,} ({llm_tflops:.4f} TFLOPS)\n")
                        f.write(f"Total MACs: {llm_macs:,}\n")
                    
                    combined_summary_path = os.path.join(ds_profile_dir, f"batch_{batch_number}_combined_summary.json")
                    with open(combined_summary_path, 'w') as f:
                        json.dump({
                            'prm_flops': prm_flops,
                            'prm_tflops': prm_tflops,
                            'prm_macs': prm_macs,
                            'llm_flops': llm_flops,
                            'llm_tflops': llm_tflops,
                            'llm_macs': llm_macs,
                            'total_flops': prm_flops + llm_flops,
                            'total_tflops': total_tflops,
                            'total_macs': prm_macs + llm_macs,
                            'batch_size': len(batch_problems),
                            'timestamp': datetime.now().strftime("%Y-%m-%d %H:%M:%S")
                        }, f, indent=2)
                    
                    ds_profiler_prm.end_profile()
                    ds_profiler_llm.end_profile()
                else:
                    if b1 is not None:
                        batch_beams = search_algorithm(
                            batch_problems, llm, prm_tokenizer, prm_model,
                            num_beams=num_beams, beam_width=beam_width, max_steps=max_steps,
                            b1=llm_batch_size, b2=b2, scoring_batch_size=scoring_batch_size,
                            tau_tokens=tau_tokens, temperature=temperature, prm_type=prm_type
                        )
                    else:
                        batch_beams = search_algorithm(
                            batch_problems, llm, prm_tokenizer, prm_model,
                            scoring_batch_size, llm_batch_size,
                            num_beams=num_beams, beam_width=beam_width,
                            max_steps=max_steps, prm_type=prm_type
                        )
                
                for i, beams in enumerate(batch_beams):
                    idx = batch_idx + i
                    if idx < len(problems):
                        output_file = save_result_func(
                            idx, beams, problems, problem_ids, problem_options, 
                            problem_gold_indices, problem_answers, output_path
                        )
                        logger.info(f"Processed problem {idx+1}/{len(problems)}: {problem_ids[idx]} -> {output_file}")
                
                torch.cuda.empty_cache()
                gc.collect()
                
            except Exception as e:
                logger.error(f"Error processing batch {batch_idx}: {str(e)}", exc_info=True)
    
    finally:
        if monitor_gpu and monitor_thread:
            stop_gpu_monitoring_func(monitor_thread)
        if use_deepspeed:
            if ds_profiler_prm is not None:
                ds_profiler_prm.end_profile()
            if ds_profiler_llm is not None:
                ds_profiler_llm.end_profile()
        logger.info(f"Processing complete.")

