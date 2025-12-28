from typing import List, Dict
import ollama

class ChatHistory:
    def __init__(self, max_history: int = 5):
        self.history: List[Dict[str, str]] = []
        self.max_history = max_history

    def add_user_message(self, message: str):
        self._add_message("user", message)

    def add_ai_message(self, message: str):
        self._add_message("assistant", message)

    def _add_message(self, role: str, content: str):
        self.history.append({"role": role, "content": content})
        if len(self.history) > self.max_history * 2: # Keep pairs roughly
             self.history = self.history[-(self.max_history * 2):]

    def get_history(self) -> List[Dict[str, str]]:
        return self.history

    def condense_question(self, new_question: str) -> str:
        """
        Uses LLM to rephrase the new question based on history.
        """
        if not self.history:
            return new_question
        
        # Format history for prompt
        history_text = "\n".join([f"{msg['role'].capitalize()}: {msg['content']}" for msg in self.history])
        
        prompt = f"""Given the following conversation and a follow-up question, rephrase the follow-up question to be a standalone question.
        
        Chat History:
        {history_text}
        
        Follow Up Input: {new_question}
        
        Standalone Question:"""
        
        response = ollama.generate(model='qwen2.5:14b', prompt=prompt)
        return response['response'].strip()
