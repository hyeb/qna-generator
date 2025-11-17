import os
from google import genai
from google.genai import types


def generator_questions(context_text: str, client: genai.Client):
    """
    context_text를 받아 질문 생성
    """

    if not context_text:
        print("입력 Context가 비어있습니다")
        return None
    
    prompt = f""" 
    
    """

    model_name = 'geminia-2.5-flash'
    client = genai.Client(api_key=GEMINI_API_KEY)
    
    response = client.models.generate_content(
        model=model_name,
        contents=prompt,
    )


    return response.text