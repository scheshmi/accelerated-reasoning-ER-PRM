import torch
import os
import logging
import argparse

# os.environ["NCCL_P2P_DISABLE"] = "1"
# os.environ["NCCL_IB_DISABLE"] = "1"

from utils.gpu_monitoring import start_gpu_monitoring, stop_gpu_monitoring, get_system_info
from utils.result_savers import save_problem_result, save_aime_problem_result, save_amc23_problem_result, save_agieval_sat_math_problem_result
from utils.batch_utils import get_batch_size_partial

from core.models import setup_models
from algorithms.partial_beam_search import partial_beam_search

from dataset_processors.processors import (
    process_math500_dataset,
    process_aime_dataset,
    process_amc23_dataset,
    process_agieval_sat_math_dataset
)

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run Partial Beam Search for math problem solving with optional profiling")
    parser.add_argument("--output_dir", type=str, default="MATH500_OUTPUT_partial_beam_search", help="Output directory")
    parser.add_argument("--llm_model_name", type=str, default='Llama-3.2-3B-Instruct', help="LLM model name")
    parser.add_argument("--prm_model_name", type=str, default='math-shepherd',
                      choices=['math-shepherd', 'skywork-o1-1.5b'], 
                      help="PRM model to use (math-shepherd or skywork-o1-1.5b)")
    parser.add_argument("--scoring_batch", type=int, default=5, help="Scoring batch size")
    parser.add_argument("--num_problems", type=int, default=10, help="Number of problems to process")
    parser.add_argument("--num_beams", type=int, default=16, help="Number of beams for beam search")
    parser.add_argument("--beam_width", type=int, default=4, help="Beam width for beam search")
    parser.add_argument("--max_steps", type=int, default=20, help="Maximum number of steps for beam search")
    parser.add_argument("--b1", type=int, default=5, help="Batch size for partial step generation")
    parser.add_argument("--b2", type=int, default=100, help="Batch size for step completion")
    parser.add_argument("--tau_tokens", type=int, default=64, help="Max new tokens for partial generation")
    parser.add_argument("--temperature", type=float, default=0.8, help="Temperature for generation")
    parser.add_argument("--profile", action="store_true", help="Enable profiling")
    parser.add_argument("--profile_freq", type=int, default=1, help="Profile frequency (every N batches)")
    parser.add_argument("--monitor_gpu", action="store_true", help="Enable continuous GPU monitoring")
    parser.add_argument("--monitor_interval", type=float, default=1.0, 
                        help="Interval between GPU measurements in seconds")
    parser.add_argument("--use_deepspeed", action="store_true", 
                        help="Use DeepSpeed's FlopsProfiler for FLOPS measurement")
    parser.add_argument("--dataset", type=str, default="math500", choices=["math500", "aime", "amc23", "sat_math"],
                      help="Dataset to use (math500, aime, amc23, or sat_math)")
    args = parser.parse_args()

    print("Configuration:")
    print(f"Output Directory: {args.output_dir}")
    print(f"Dataset: {args.dataset}")
    print(f"LLM Model: {args.llm_model_name}")
    print(f"PRM Model: {args.prm_model_name}")
    print(f"Scoring Batch Size: {args.scoring_batch}")
    print(f"Number of Problems: {args.num_problems}")
    print(f"Number of Beams: {args.num_beams}")
    print(f"Beam Width: {args.beam_width}")
    print(f"Max Steps: {args.max_steps}")
    print(f"Tau Tokens: {args.tau_tokens}")
    print(f"B1: {args.b1}")
    print(f"B2: {args.b2}")
    print(f"Temperature: {args.temperature}")
    print(f"Profiling: {'Enabled' if args.profile else 'Disabled'}")
    if args.profile:
        print(f"Profile Frequency: Every {args.profile_freq} batch(es)")
    print(f"GPU Monitoring: {'Enabled' if args.monitor_gpu else 'Disabled'}")
    if args.monitor_gpu:
        print(f"Monitor Interval: {args.monitor_interval} seconds")
    print(f"DeepSpeed FLOPS Profiler: {'Enabled' if args.use_deepspeed else 'Disabled'}")
    if args.use_deepspeed:
        print(f"Profile Frequency: Every {args.profile_freq} batch(es)")

    dataset_name = args.dataset.upper()
    args.output_dir = f'{dataset_name}_OUTPUT_partial_beam_search_{args.llm_model_name}_{args.prm_model_name}_{args.num_beams}_{args.beam_width}_{args.max_steps}_{args.tau_tokens}'

    llm, prm_tokenizer, prm_model, prm_type = setup_models(args.llm_model_name, args.prm_model_name)

    if args.dataset == "math500":
        process_math500_dataset(
            search_algorithm=partial_beam_search,
            output_dir=args.output_dir,
            search_batch_size=None,  
            scoring_batch_size=args.scoring_batch,
            num_problems=args.num_problems,
            num_beams=args.num_beams,
            beam_width=args.beam_width,
            max_steps=args.max_steps,
            profile=args.profile,
            profile_frequency=args.profile_freq,
            monitor_gpu=args.monitor_gpu,
            monitor_interval=args.monitor_interval,
            use_deepspeed=args.use_deepspeed,
            prm_model_name=args.prm_model_name,
            llm_model_name=args.llm_model_name,
            llm=llm,
            prm_tokenizer=prm_tokenizer,
            prm_model=prm_model,
            prm_type=prm_type,
            get_batch_size_func=get_batch_size_partial,
            save_result_func=save_problem_result,
            start_gpu_monitoring_func=start_gpu_monitoring,
            stop_gpu_monitoring_func=stop_gpu_monitoring,
            get_system_info_func=get_system_info,
            b1=args.b1,
            b2=args.b2,
            tau_tokens=args.tau_tokens,
            temperature=args.temperature
        )
    elif args.dataset == "aime":
        process_aime_dataset(
            search_algorithm=partial_beam_search,
            output_dir=args.output_dir,
            search_batch_size=None,
            scoring_batch_size=args.scoring_batch,
            num_problems=args.num_problems,
            num_beams=args.num_beams,
            beam_width=args.beam_width,
            max_steps=args.max_steps,
            profile=args.profile,
            profile_frequency=args.profile_freq,
            monitor_gpu=args.monitor_gpu,
            monitor_interval=args.monitor_interval,
            use_deepspeed=args.use_deepspeed,
            prm_model_name=args.prm_model_name,
            llm_model_name=args.llm_model_name,
            llm=llm,
            prm_tokenizer=prm_tokenizer,
            prm_model=prm_model,
            prm_type=prm_type,
            get_batch_size_func=get_batch_size_partial,
            save_result_func=save_aime_problem_result,
            start_gpu_monitoring_func=start_gpu_monitoring,
            stop_gpu_monitoring_func=stop_gpu_monitoring,
            get_system_info_func=get_system_info,
            b1=args.b1,
            b2=args.b2,
            tau_tokens=args.tau_tokens,
            temperature=args.temperature
        )
    elif args.dataset == "amc23":
        process_amc23_dataset(
            search_algorithm=partial_beam_search,
            output_dir=args.output_dir,
            search_batch_size=None,
            scoring_batch_size=args.scoring_batch,
            num_problems=args.num_problems,
            num_beams=args.num_beams,
            beam_width=args.beam_width,
            max_steps=args.max_steps,
            profile=args.profile,
            profile_frequency=args.profile_freq,
            monitor_gpu=args.monitor_gpu,
            monitor_interval=args.monitor_interval,
            use_deepspeed=args.use_deepspeed,
            prm_model_name=args.prm_model_name,
            llm_model_name=args.llm_model_name,
            llm=llm,
            prm_tokenizer=prm_tokenizer,
            prm_model=prm_model,
            prm_type=prm_type,
            get_batch_size_func=get_batch_size_partial,
            save_result_func=save_amc23_problem_result,
            start_gpu_monitoring_func=start_gpu_monitoring,
            stop_gpu_monitoring_func=stop_gpu_monitoring,
            get_system_info_func=get_system_info,
            b1=args.b1,
            b2=args.b2,
            tau_tokens=args.tau_tokens,
            temperature=args.temperature
        )
    elif args.dataset == "sat_math":
        process_agieval_sat_math_dataset(
            search_algorithm=partial_beam_search,
            output_dir=args.output_dir,
            search_batch_size=None,
            scoring_batch_size=args.scoring_batch,
            num_problems=args.num_problems,
            num_beams=args.num_beams,
            beam_width=args.beam_width,
            max_steps=args.max_steps,
            profile=args.profile,
            profile_frequency=args.profile_freq,
            monitor_gpu=args.monitor_gpu,
            monitor_interval=args.monitor_interval,
            use_deepspeed=args.use_deepspeed,
            prm_model_name=args.prm_model_name,
            llm_model_name=args.llm_model_name,
            llm=llm,
            prm_tokenizer=prm_tokenizer,
            prm_model=prm_model,
            prm_type=prm_type,
            get_batch_size_func=get_batch_size_partial,
            save_result_func=save_agieval_sat_math_problem_result,
            start_gpu_monitoring_func=start_gpu_monitoring,
            stop_gpu_monitoring_func=stop_gpu_monitoring,
            get_system_info_func=get_system_info,
            b1=args.b1,
            b2=args.b2,
            tau_tokens=args.tau_tokens,
            temperature=args.temperature
        )

