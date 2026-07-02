import json
import os

def load_export(input_path):
    """
    Load an exported chat transcript and parse it into turns.
    Each turn should be a dictionary with 'role' (e.g. 'user', 'assistant') and 'text'.
    """
    if not os.path.exists(input_path):
        raise FileNotFoundError(f"File not found: {input_path}")
        
    _, ext = os.path.splitext(input_path.lower())
    
    if ext == '.json':
        with open(input_path, 'r', encoding='utf-8') as f:
            data = json.load(f)
            # Adapt this logic based on actual export formats (e.g. ChatGPT, Claude)
            # For now, assume it's a simple list of dicts: [{'role': 'user', 'text': '...'}]
            return data
    elif ext in ['.md', '.txt']:
        # Basic parsing for markdown/text dumps
        turns = []
        with open(input_path, 'r', encoding='utf-8') as f:
            content = f.read()
            
        # Check if it has explicit role markers
        if 'User:' in content or 'Human:' in content:
            lines = content.split('\n')
            current_role = None
            current_text = []
            
            for line in lines:
                if line.startswith('User:') or line.startswith('Human:'):
                    if current_role:
                        turns.append({"role": current_role, "text": '\n'.join(current_text).strip()})
                    current_role = 'user'
                    current_text = [line.split(':', 1)[1].strip()]
                elif line.startswith('Assistant:') or line.startswith('AI:'):
                    if current_role:
                        turns.append({"role": current_role, "text": '\n'.join(current_text).strip()})
                    current_role = 'assistant'
                    current_text = [line.split(':', 1)[1].strip()]
                elif current_role:
                    current_text.append(line)
            
            if current_role and current_text:
                turns.append({"role": current_role, "text": '\n'.join(current_text).strip()})
        else:
            # Unstructured raw text dump (like a UI copy-paste)
            # Split by double-newlines to get semantic paragraphs
            blocks = content.split('\n\n')
            for block in blocks:
                block = block.strip()
                if block and block != 'Show more':
                    turns.append({"role": "unknown", "text": block})
                    
        return turns
    else:
        raise ValueError(f"Unsupported file extension: {ext}")
