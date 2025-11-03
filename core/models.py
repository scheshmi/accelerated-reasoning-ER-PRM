
import torch
import logging
from transformers import AutoTokenizer, AutoModelForCausalLM, pipeline
from .skywork import setup_skywork_o1_prm_model

logger = logging.getLogger(__name__)

def format_prompt(problem: str) -> str:
    """Format the problem with step tags for generation."""
    return f"""Solve the following math problem efficiently and clearly:

- For simple problems (2 steps or fewer):
Provide a concise solution with minimal explanation.

- For complex problems (3 steps or more):
Use this step-by-step format:

## Step 1: [Concise description]
[Brief explanation and calculations]

## Step 2: [Concise description]
[Brief explanation and calculations]

...

Regardless of the approach, always conclude with:

Therefore, the final answer is: $\\boxed{{answer}}$. I hope it is correct.

Where [answer] is just the final number or expression that solves the problem.

{problem}
"""

def setup_models(llm_model_name='Llama-3.2-3B-Instruct', prm_model_name='math-shepherd'):
    """Initialize the models with strict memory management.
    
    Args:
        llm_model_name: LLM model to use ('Llama-3.2-3B-Instruct' or 'Qwen2.5-3B-Instruct')
        prm_model_name: Which PRM model to use. Options: 'math-shepherd', 'skywork-o1-1.5b'
        
    Returns:
        llm: The LLM pipeline
        prm_tokenizer: The tokenizer for the PRM model
        prm_model: The PRM model
        prm_type: Type of PRM model being used
    """
    logger.info(f"Setting up models with memory optimization... Using PRM model: {prm_model_name}")
    

    # os.environ['CUDA_LAUNCH_BLOCKING'] = '1'
    torch.backends.cuda.max_split_size_mb = 1024
    torch.backends.cuda.matmul.allow_tf32 = True
    torch.backends.cuda.enable_flash_sdp(True)
    
    try:
        if prm_model_name == 'skywork-o1-1.5b':
            prm_tokenizer, prm_model = setup_skywork_o1_prm_model()
            prm_type = 'skywork-o1'
        else:  
            prm_model = AutoModelForCausalLM.from_pretrained(
                'peiyi9979/math-shepherd-mistral-7b-prm',
                device_map='auto',
                torch_dtype=torch.bfloat16,
                low_cpu_mem_usage=True,
                attn_implementation="flash_attention_2"
            ).eval()

            prm_tokenizer = AutoTokenizer.from_pretrained(
                'peiyi9979/math-shepherd-mistral-7b-prm',
                legacy=True, use_fast=False
            )
            if prm_tokenizer.pad_token is None:
                prm_tokenizer.pad_token = prm_tokenizer.eos_token
                prm_tokenizer.pad_token_id = prm_tokenizer.eos_token_id
            
            prm_type = 'math-shepherd'


        if llm_model_name == 'Llama-3.2-3B-Instruct':
            llm_tokenizer = AutoTokenizer.from_pretrained("meta-llama/Llama-3.2-3B-Instruct")
            if llm_tokenizer.pad_token is None:
                llm_tokenizer.pad_token = llm_tokenizer.eos_token
                llm_tokenizer.pad_token_id = llm_tokenizer.eos_token_id
            
            llm_tokenizer.padding_side = 'left'
                
            llm_model = AutoModelForCausalLM.from_pretrained(
                "meta-llama/Llama-3.2-3B-Instruct",
                torch_dtype=torch.bfloat16,
                device_map='auto',
                low_cpu_mem_usage=True,
                attn_implementation="flash_attention_2"
            ).eval()
        
        if llm_model_name == 'Qwen2.5-3B-Instruct':
            llm_tokenizer = AutoTokenizer.from_pretrained("Qwen/Qwen2.5-3B-Instruct")
            if llm_tokenizer.pad_token is None:
                llm_tokenizer.pad_token = llm_tokenizer.eos_token
                llm_tokenizer.pad_token_id = llm_tokenizer.eos_token_id 
                
            llm_tokenizer.padding_side = 'left'
                
            llm_model = AutoModelForCausalLM.from_pretrained(
                "Qwen/Qwen2.5-3B-Instruct",
                torch_dtype=torch.bfloat16,
                device_map='auto',
                low_cpu_mem_usage=True,
                attn_implementation="flash_attention_2"
            ).eval()
            
        llm = pipeline(
            "text-generation",
            model=llm_model,
            tokenizer=llm_tokenizer,
            device_map='auto',
        )
        
        if llm.tokenizer.pad_token_id is None:
            llm.tokenizer.pad_token = llm.tokenizer.eos_token
            llm.tokenizer.pad_token_id = llm.tokenizer.eos_token_id

        logger.info("Models loaded successfully")
        return llm, prm_tokenizer, prm_model, prm_type
    except Exception as e:
        logger.error(f"Error setting up models: {str(e)}")
        raise

