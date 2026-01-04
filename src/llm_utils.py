import os
from openai import OpenAI
from dotenv import load_dotenv

# Load environment variables from .env file
load_dotenv()

def get_fraud_explanation(structured_explanation, probability, is_fraud):
    """
    Sends the SHAP structured explanation to OpenAI's GPT model 
    to generate an analyst-friendly natural language summary.
    """
    api_key = os.getenv("OPENAI_API_KEY")
    if not api_key:
        return "Error: OPENAI_API_KEY not found in environment."

    client = OpenAI(api_key=api_key)

    status = "FLAGGED AS FRAUD" if is_fraud else "NOT FLAGGED (SAFE)"
    
    prompt = f"""
    You are a professional fraud risk analyst. 
    A machine learning model has analyzed a transaction and determined it is: {status}.
    Model Probability Score: {probability:.4f}

    TECHNICAL FACTORS IMPACTING THE SCORE:
    {structured_explanation}

    TASK:
    Generate a concise, professional report (2-3 sentences) explaining the model's decision.
    - If it is FLAGGED, explain what triggered the alarm.
    - If it is NOT FLAGGED, explain why it was considered safe despite any potentially unusual features.
    
    Do not mention technical terms like "SHAP" or "values". Talk like an analyst to a team.
    """

    try:
        response = client.chat.completions.create(
            model="gpt-4o-mini",
            messages=[
                {"role": "system", "content": "You are a helpful and professional fraud analysis assistant."},
                {"role": "user", "content": prompt}
            ],
            max_tokens=150,
            temperature=0.7
        )
        return response.choices[0].message.content.strip()
    except Exception as e:
        return f"Error connecting to OpenAI: {str(e)}"

if __name__ == "__main__":
    # Test the function with dummy technical factors
    test_explanation = """
    - V14 (value: -4.6838) increased fraud risk.
    - V10 (value: -2.6494) increased fraud risk.
    - V4 (value: 3.5846) increased fraud risk.
    - V12 (value: -2.5875) increased fraud risk.
    - V8 (value: 0.8784) decreased fraud risk.
    """
    print("Generating AI Explanation...")
    print(get_fraud_explanation(test_explanation))
