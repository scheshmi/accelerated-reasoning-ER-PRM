

def get_batch_size_beam(llm_model_name: str):
    """Get batch size for beam search algorithm.
    
    Args:
        llm_model_name: Name of the LLM model
        
    Returns:
        int: Batch size for the LLM
    """
    if llm_model_name == "Qwen2.5-3B-Instruct" or llm_model_name == "Llama-3.2-3B-Instruct":
        return 64
    else:
        return 16

def get_batch_size_partial(num_beams):
    """Get batch sizes for partial beam search algorithm.
    
    Args:
        num_beams: Number of beams
        
    Returns:
        tuple: (processing_batch_size, llm_batch_size)
    """
    if num_beams == 64:
        return 3, 50
    elif num_beams == 32:
        return 4, 50
    else:
        return 75//num_beams, 75

