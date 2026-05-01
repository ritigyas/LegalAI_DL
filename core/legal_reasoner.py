import google.generativeai as genai
import os


genai.configure(api_key="AIzaSyDpl5yizB8_lZVGzmkDF2dt5vgU0IFkXM0")
model = genai.GenerativeModel("gemini-flash-latest")



def generate_output(query, context, cases, domain):
    prompt = f"""
    You are a strict legal AI system.

    Analyze the input and RETURN ONLY in this format:

    ------------------------

    Legal Issue:
    (clearly define the issue)

    Jurisdiction:
    (India or relevant court)

    Key Facts:
    (bullet points)

    Primary Precedents:
    (mention real or relevant cases)

    Live Court Updates:
    (if none, say not available)

    Legal Analysis (IRAC):
    Issue:
    Rule:
    Application:
    Conclusion:

    ------------------------

    USER QUERY: {query}
    DOMAIN: {domain}

    CONTEXT:
    {context}

    CASES:
    {cases}
    """

    response = model.generate_content(
        prompt,
        generation_config={"max_output_tokens": 99000}  # 🔥 FIXED
    )

    return response.text