from typing import List
from src.schema import RetrievalResult
import os
from groq import Groq

class Generator:
    def __init__(self, model_name: str = "llama-3.3-70b-versatile"):
        self.model_name = model_name
        self.client = Groq(api_key=os.environ.get("GROQ_API_KEY"))

    def format_prompt(self, query: str, context_results: List[RetrievalResult]) -> str:
        """
        Combines the user query and retrieved chunks into a structured prompt.
        """
        # Extract the text from our results
        context_text = "\n\n".join([
            f"--- Context Fragment {i+1} (Source: {res.chunk.metadata['source']}) ---\n{res.chunk.content}"
            for i, res in enumerate(context_results)
        ])

        prompt = f"""
        You are a highly knowledgeable assistant. Use the provided context fragments 
        to answer the user's question accurately. 
        
        Rules:
        1. If the answer is not in the context, say: "I'm sorry, I don't have enough information in my local knowledge base to answer that."
        2. Always cite the source file name in your answer.
        3. Keep the answer concise and professional.

        [CONTEXT]
        {context_text}

        [USER QUESTION]
        {query}

        [YOUR ANSWER]
        """
        return prompt

    def generate_answer_stream(self, prompt: str):
        """
        Sends the prompt to the LLM and streams the response to the console.
        """
        print("\n--- Assistant Response ---")
        
        try:
            stream = self.client.chat.completions.create(
                model=self.model_name,
                messages=[{"role": "user", "content": prompt}],
                stream=True,
            )

            for chunk in stream:
                content = chunk.choices[0].delta.content
                if content is not None:
                    print(content, end='', flush=True) # The 'flush' is key for real-time display
        except Exception as e:
            print(f"Error communicating with Groq API. Please verify your GROQ_API_KEY environment variable. Error details: {e}")
        
        print("\n" + "-"*25)