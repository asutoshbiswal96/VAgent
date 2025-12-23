import os
from typing import Optional


class GeminiClient:
    """Wrapper to call Gemini (free) if `GEMINI_API_KEY` is set; otherwise fallback to a simple canned reply generator.

    To enable Gemini: set `GEMINI_API_KEY` in environment or `.env` and install `google-generativeai`.
    """

    def __init__(self, api_key: Optional[str] = None):
        self.api_key = api_key or os.environ.get('GEMINI_API_KEY')
        self.enabled = bool(self.api_key)
        if self.enabled:
            try:
                import google.generativeai as genai
                genai.configure(api_key=self.api_key)
                self._genai = genai
            except Exception as e:
                self.enabled = False
                self._genai = None
        else:
            self._genai = None

    def generate(self, prompt: str, max_output_tokens: int = 256) -> str:
        if self.enabled and self._genai:
            try:
                response = self._genai.text.synthesize(prompt=prompt, model='gemini-lite')
                return response.text
            except Exception as e:
                return f"[Gemini call failed: {e}]"
        # fallback simple generation
        return self._simple_fallback(prompt)

    def _simple_fallback(self, prompt: str) -> str:
        # Very simple heuristic: look for keywords and reply politely.
        if 'remind' in prompt.lower() or 'due date' in prompt.lower():
            return "Hello, this is LIC reminder: your premium is due soon. Please arrange payment or contact us for help."
        return "Hello, I'm LIC virtual assistant. How can I help you today?"
