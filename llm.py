import os
from dotenv import load_dotenv
from google import genai

# Load variables from .env file
load_dotenv()

def get_llm_client():
    api_key = os.getenv("GEMINI_API_KEY")
    if not api_key or api_key == "your_gemini_api_key_here":
        raise ValueError("Please set your GEMINI_API_KEY in the .env file")
        
    client = genai.Client(api_key=api_key)
    return client

def generate_text(prompt: str, system_prompt: str = "You are a helpful assistant.") -> str:
    """
    Generate a text response from the LLM based on the given prompt.
    """
    client = get_llm_client()
    model = os.getenv("LLM_MODEL", "gemini-2.5-flash")
    
    response = client.models.generate_content(
        model=model,
        contents=prompt,
        config=genai.types.GenerateContentConfig(
            system_instruction=system_prompt,
        )
    )
    
    return response.text

if __name__ == "__main__":
    # Test the connection when you run this file directly
    print("Testing OpenRouter connection...")
    try:
        reply = generate_text("Briefly tell me what the Kalam cosmological argument is.")
        print(f"\nResponse from {os.getenv('LLM_MODEL')}:\n{reply}")
    except Exception as e:
        print(f"Error: {e}")
