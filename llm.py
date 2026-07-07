import os
from dotenv import load_dotenv
from openai import OpenAI

# Load variables from .env file
load_dotenv()

def get_llm_client():
    api_key = os.getenv("OPENROUTER_API_KEY")
    if not api_key or api_key == "your_openrouter_api_key_here":
        raise ValueError("Please set your OPENROUTER_API_KEY in the .env file")
        
    # OpenRouter provides an OpenAI-compatible API
    client = OpenAI(
        base_url="https://openrouter.ai/api/v1",
        api_key=api_key,
    )
    return client

def generate_text(prompt, system_prompt="You are a helpful assistant."):
    client = get_llm_client()
    model = os.getenv("LLM_MODEL", "meta-llama/llama-3-8b-instruct:free")
    
    response = client.chat.completions.create(
        model=model,
        messages=[
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": prompt}
        ],
        # OpenRouter optional headers for rankings (good practice)
        extra_headers={
            "HTTP-Referer": "https://github.com/skanenje/Ai-author", 
            "X-Title": "AI Book Builder"
        }
    )
    
    return response.choices[0].message.content

if __name__ == "__main__":
    # Test the connection when you run this file directly
    print("Testing OpenRouter connection...")
    try:
        reply = generate_text("Briefly tell me what the Kalam cosmological argument is.")
        print(f"\nResponse from {os.getenv('LLM_MODEL')}:\n{reply}")
    except Exception as e:
        print(f"Error: {e}")
