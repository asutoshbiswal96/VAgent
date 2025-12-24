import os
import base64
import io
from typing import Optional
from dotenv import load_dotenv
import os
load_dotenv(override=True)

class GeminiClient:
    """Wrapper to call Gemini (free) if `GEMINI_API_KEY` is set; otherwise fallback to a simple canned reply generator.

    To enable Gemini: set `GEMINI_API_KEY` in environment or `.env` and install `google-generativeai`.
    """

    def __init__(self, api_key: Optional[str] = None):
        self.api_key = api_key or os.environ.get('GEMINI_API_KEY')
        print(f'Gemini API Key found: {bool(self.api_key)}')
        self.enabled = bool(self.api_key)
        print(f'GeminiClient initialized. Enabled: {self.enabled}')
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
                # Use text synthesis for text responses when available
                if hasattr(self._genai, 'text') and hasattr(self._genai.text, 'synthesize'):
                    response = self._genai.text.synthesize(prompt=prompt, model='gemini-2.5-flash-native-audio-preview')
                    return getattr(response, 'text', str(response))
                # otherwise try a generic generate path
                if hasattr(self._genai, 'generate'):
                    response = self._genai.generate(prompt=prompt)
                    return getattr(response, 'text', str(response))
            except Exception as e:
                return f"[Gemini call failed: {e}]"
        # fallback simple generation
        return self._simple_fallback(prompt)

    def transcribe_audio(self, audio_bytes: bytes, model: str = 'gemini-2.5-flash-native-audio-preview') -> str:
        """Transcribe audio bytes to text using Gemini audio transcription if available.

        Returns a transcription string or an informative error message.
        """
        if not (self.enabled and self._genai):
            return "[transcription unavailable: Gemini not configured]"
        try:
            audio_api = getattr(self._genai, 'audio', None)
            # Common SDKs expose an audio.transcribe or audio.speech.transcribe
            if audio_api and hasattr(audio_api, 'transcribe'):
                resp = audio_api.transcribe(model=model, audio=audio_bytes)
                return getattr(resp, 'text', str(resp))
            # fallback: some SDKs may expose speech or speech_to_text
            if audio_api and hasattr(audio_api, 'speech') and hasattr(audio_api.speech, 'transcribe'):
                resp = audio_api.speech.transcribe(model=model, audio=audio_bytes)
                return getattr(resp, 'text', str(resp))
            return "[transcription not supported by installed genai SDK]"
        except Exception as e:
            return f"[transcription failed: {e}]"

    def synthesize_audio(self, text: str, model: str = 'gemini-2.5-flash-native-audio-preview', audio_format: str = 'wav') -> bytes:
        """Synthesize audio bytes from text using Gemini audio synthesis when available.

        Returns raw audio bytes (or UTF-8 text bytes as a fallback).
        """
        if not (self.enabled and self._genai):
            return text.encode('utf-8')
        try:
            audio_api = getattr(self._genai, 'audio', None)
            if audio_api and hasattr(audio_api, 'synthesize'):
                # SDKs differ in signatures; attempt a few common ones
                resp = audio_api.synthesize(model=model, input=text, format=audio_format)
                # Try several common response shapes
                if hasattr(resp, 'audio') and resp.audio:
                    return resp.audio
                if hasattr(resp, 'audio_bytes') and resp.audio_bytes:
                    return resp.audio_bytes
                # base64 payloads
                for attr in ('audioContent', 'audio_content', 'audio_base64', 'audio'):
                    val = getattr(resp, attr, None)
                    if isinstance(val, str):
                        try:
                            return base64.b64decode(val)
                        except Exception:
                            pass
                return str(resp).encode('utf-8')
            # try speech.synthesize
            if audio_api and hasattr(audio_api, 'speech') and hasattr(audio_api.speech, 'synthesize'):
                resp = audio_api.speech.synthesize(model=model, input=text, format=audio_format)
                if hasattr(resp, 'audio') and resp.audio:
                    return resp.audio
                if isinstance(resp, bytes):
                    return resp
                return str(resp).encode('utf-8')
            return text.encode('utf-8')
        except Exception as e:
            return f"[synthesis failed: {e}]".encode('utf-8')

    def generate_audio_conversation(self, input_audio: bytes) -> bytes:
        """End-to-end audio-to-audio: transcribe input audio, generate text reply, and synthesize audio reply.

        Returns audio bytes (or UTF-8 encoded text bytes on fallback).
        """
        # First attempt: try direct audio->audio generation if the SDK supports it
        if self.enabled and self._genai:
            audio_api = getattr(self._genai, 'audio', None)
            # helper to extract audio bytes from various response shapes
            def _extract_audio(resp):
                if resp is None:
                    return None
                if isinstance(resp, (bytes, bytearray)):
                    return resp
                if hasattr(resp, 'audio') and resp.audio:
                    return resp.audio
                if hasattr(resp, 'audio_bytes') and resp.audio_bytes:
                    return resp.audio_bytes
                for attr in ('audioContent', 'audio_content', 'audio_base64', 'audio'):
                    val = getattr(resp, attr, None)
                    if isinstance(val, (bytes, bytearray)):
                        return val
                    if isinstance(val, str):
                        try:
                            return base64.b64decode(val)
                        except Exception:
                            pass
                # sometimes the response contains nested fields
                try:
                    d = dict(resp.__dict__)
                except Exception:
                    d = {}
                for v in d.values():
                    if isinstance(v, (bytes, bytearray)):
                        return v
                    if isinstance(v, str):
                        try:
                            return base64.b64decode(v)
                        except Exception:
                            pass
                return None

            # Try audio_api.generate / audio_api.chat / audio_api.create variants
            try:
                if audio_api:
                    for fn in ('generate', 'create', 'chat'):
                        if hasattr(audio_api, fn):
                            try:
                                # try a few common argument names
                                for kw in ({'model': 'gemini-2.5-flash-native-audio-preview', 'audio': input_audio},
                                           {'model': 'gemini-2.5-flash-native-audio-preview', 'input': input_audio},
                                           {'model': 'gemini-2.5-flash-native-audio-preview', 'content': input_audio}):
                                    try:
                                        resp = getattr(audio_api, fn)(**kw)
                                        out = _extract_audio(resp)
                                        if out:
                                            return out
                                    except TypeError:
                                        continue
                            except Exception:
                                continue

                # Try top-level generate with audio input (multimodal signatures vary)
                if hasattr(self._genai, 'generate'):
                    try:
                        # try a few plausible signatures
                        for kw in ({'model': 'gemini-2.5-flash-native-audio-preview', 'audio': input_audio},
                                   {'model': 'gemini-2.5-flash-native-audio-preview', 'input_audio': input_audio},
                                   {'model': 'gemini-2.5-flash-native-audio-preview', 'input': input_audio},
                                   {'model': 'gemini-2.5-flash-native-audio-preview', 'multimodal_input': {'audio': input_audio}}):
                            try:
                                resp = self._genai.generate(**kw)
                                out = _extract_audio(resp)
                                if out:
                                    return out
                            except TypeError:
                                continue
                    except Exception:
                        pass
            except Exception:
                pass

        # Fallback: transcribe -> generate text -> synthesize audio
        transcription = self.transcribe_audio(input_audio)
        text_reply = self.generate(transcription)
        audio_out = self.synthesize_audio(text_reply)
        return audio_out

    def _simple_fallback(self, prompt: str) -> str:
        # Very simple heuristic: look for keywords and reply politely.
        if 'remind' in prompt.lower() or 'due date' in prompt.lower():
            return "Hello, this is LIC reminder: your premium is due soon. Please arrange payment or contact us for help."
        return "Hello, I'm LIC virtual assistant. How can I help you today?"
