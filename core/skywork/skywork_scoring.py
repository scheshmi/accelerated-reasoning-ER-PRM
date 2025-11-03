import torch
import numpy as np
from typing import List, Tuple, Dict, Any
import gc
import logging
from transformers import AutoModelForCausalLM, AutoTokenizer
import sys
import os

# Add the path to the Skywork PRM module
# sys.path.append(os.path.join(os.path.dirname(__file__), '..', '..', 'search-and-learn', 'src', 'sal', 'models', 'skywork_o1_prm'))
from .prm_model import SkyworkPRMModel

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def setup_skywork_o1_prm_model():
    """Initialize the Skywork O1 1.5B PRM model."""
    try:
        # Import modules here to avoid import errors when not using this model


        model_name = "Skywork/Skywork-o1-Open-PRM-Qwen-2.5-1.5B"
        
        logger.info("Loading Skywork O1 1.5B PRM model...")
        # Load the base model first
        # prm_model = AutoModelForCausalLM.from_pretrained(
        #     model_name,
        #     device_map='auto',
        #     torch_dtype=torch.bfloat16,
        #     low_cpu_mem_usage=True,
        #     # attn_implementation="flash_attention_2"
        # ).eval()
        
        # Wrap it with the SkyworkPRMModel class
        prm_model = SkyworkPRMModel.from_pretrained(
            model_name,
            device_map="auto",
            torch_dtype=torch.bfloat16,
            low_cpu_mem_usage=True,
            # attn_implementation="flash_attention_2"
        ).eval()

        
        # Load the tokenizer
        prm_tokenizer = AutoTokenizer.from_pretrained(
            model_name,
            # use_fast=False
            trust_remote_code=True
        )
        prm_tokenizer.padding_side = 'left'
        
        if prm_tokenizer.pad_token is None:
            prm_tokenizer.pad_token = prm_tokenizer.eos_token
            prm_tokenizer.pad_token_id = prm_tokenizer.eos_token_id
        
        
        return prm_tokenizer, prm_model
    
    except Exception as e:
        logger.error(f"Error setting up Skywork O1 PRM model: {str(e)}")
        raise

def prepare_input(problem, solution, tokenizer, step_token):
    """Prepare input for the Skywork O1 PRM model.
    
    Args:
        problem: The math problem text
        solution: The generated solution text
        tokenizer: The tokenizer for the PRM model
        step_token: The token used to mark steps
        
    Returns:
        input_ids: The token IDs for the model input
        steps: The list of steps in the solution
        reward_flags: Flags indicating which tokens to compute rewards for
    """
    tokenizer.padding_side = 'left'
    prompt_ids = tokenizer.encode(tokenizer.bos_token + problem + "\n")
    response_ids = []
    steps = []
    reward_flags = [0] * len(prompt_ids)
    step_token_id = tokenizer.encode(step_token)[-1]
    
    for idx, step in enumerate(solution.split(step_token)):
        if step != "":
            step_ids = tokenizer.encode(step)
        else:
            step_ids = []
        step_ids += [step_token_id]
        step = step + step_token
        flag = [0] * len(step_ids)
        flag[-1] = 1
        response_ids.extend(step_ids)
        reward_flags.extend(flag)
        steps.append(step)
    
    input_ids = prompt_ids + response_ids
    return input_ids, steps, reward_flags

def prepare_batch_input_for_model(input_ids, reward_flags, pad_token_id):
    """Prepare batched inputs for the Skywork O1 PRM model.
    
    Args:
        input_ids: List of token ID sequences
        reward_flags: List of reward flag sequences
        pad_token_id: Token ID used for padding
        
    Returns:
        padded_input_ids: Padded tensor of input IDs
        padded_attention_mask: Attention mask for padded input
        padded_reward_flags: Padded tensor of reward flags
    """
    padded_input_ids = torch.nn.utils.rnn.pad_sequence(
        [torch.LongTensor(ids) for ids in input_ids],
        batch_first=True,
        padding_value=pad_token_id,
    )
    padded_attention_mask = torch.nn.utils.rnn.pad_sequence(
        [torch.LongTensor([1] * len(ids)) for ids in input_ids],
        batch_first=True,
        padding_value=0,
    )
    padded_reward_flags = torch.nn.utils.rnn.pad_sequence(
        [torch.LongTensor(reward_flag) for reward_flag in reward_flags],
        batch_first=True,
        padding_value=0,
    )
    return padded_input_ids, padded_attention_mask, padded_reward_flags

def derive_step_rewards(rewards, reward_flags):
    """Extract step rewards from model outputs.
    
    Args:
        rewards: Tensor of rewards from the model
        reward_flags: Flags indicating which positions to extract rewards from
        
    Returns:
        batch_step_rewards: List of reward sequences for each sample in the batch
    """
    batch_size = rewards.shape[0]
    batch_step_rewards = []
    
    for i in range(batch_size):
        rewards_indices = torch.nonzero(reward_flags[i] == 1).view(-1)
        step_rewards = []
        
        if rewards_indices.numel() > 0:
            step_rewards = [
                rewards[i][rewards_indices[j]].item() for j in range(len(rewards_indices))
            ]
        
        batch_step_rewards.append(step_rewards)
    
    return batch_step_rewards

def parallel_score_solutions_skywork_o1(problem_solution_pairs: List[Tuple[str, str]], prm_tokenizer, prm_model, batch_size: int) -> List[torch.Tensor]:
    """Score solutions using the Skywork O1 PRM model.
    
    Args:
        problem_solution_pairs: List of (problem, solution) pairs to score
        prm_tokenizer: Tokenizer for the PRM model
        prm_model: The PRM model (SkyworkPRMModel instance)
        batch_size: Batch size for scoring
        
    Returns:
        List of scores for each solution
    """
    # Ensure padding_side is set to 'left' for Flash Attention version of Qwen2
    prm_tokenizer.padding_side = 'left'
    
    batch_scores = []
    step_token = "ки\n"  # Use the same step token as in the beam search
    pad_token_id = prm_tokenizer.pad_token_id
    
    # Get the actual device from the model's parameters
    device = next(prm_model.parameters()).device
    
    for i in range(0, len(problem_solution_pairs), batch_size):
        batch_pairs = problem_solution_pairs[i:i + batch_size]
        
        # Process input for Skywork O1 PRM model format
        batch_input_ids = []
        batch_reward_flags = []
        
        for problem, solution in batch_pairs:
            # Format the input for Skywork O1
            input_ids, _, reward_flags = prepare_input(problem, solution, prm_tokenizer, step_token)
            batch_input_ids.append(input_ids)
            batch_reward_flags.append(reward_flags)
        
        # Prepare batch inputs
        padded_input_ids, padded_attention_mask, padded_reward_flags = prepare_batch_input_for_model(
            batch_input_ids, batch_reward_flags, pad_token_id
        )
        
        # Move tensors to device
        padded_input_ids = padded_input_ids.to(device)
        padded_attention_mask = padded_attention_mask.to(device)
        padded_reward_flags = padded_reward_flags.to(device)
        
        with torch.no_grad():
            # Forward pass through the model
            _, _, rewards = prm_model(
                input_ids=padded_input_ids,
                attention_mask=padded_attention_mask,
                return_probs=True  # Get probabilities directly
            )
            
            # Extract step rewards
            step_rewards = derive_step_rewards(rewards, padded_reward_flags)
            
            # Convert step rewards to tensors and add to batch scores
            for step_reward in step_rewards:
                if len(step_reward) > 0:
                    batch_scores.append(torch.tensor(step_reward, dtype=torch.float32))
                else:
                    # Default score if no steps were found
                    batch_scores.append(torch.tensor([0.5], dtype=torch.float32))
        
        # Clean up memory
        del padded_input_ids
        del padded_attention_mask
        del padded_reward_flags
        del rewards
        
        torch.cuda.empty_cache()
        gc.collect()
    
    return batch_scores 