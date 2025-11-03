from typing import List

def split_into_steps(text: str) -> List[str]:
    """Split generated text into individual steps."""
    steps = []
    current_step = []
    lines = text.split('\n')
    
    for line in lines:
        if line.strip().startswith('##') and current_step:
            steps.append('\n'.join(current_step).strip())
            current_step = []
        if line.strip():  
            current_step.append(line)
            
    if current_step:
        steps.append('\n'.join(current_step).strip())
        
    if not steps and text.strip():
        steps = [text.strip()]
        
    return steps

