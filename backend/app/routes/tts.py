"""
POST /api/tts

Uses openai/gpt-4o-audio-preview via OpenRouter.
Parses the SSE stream directly with httpx to reliably extract PCM16 audio chunks,
then wraps them in a WAV container for browser playback.
"""

import base64
import json
import logging
import re
import struct

import httpx
from fastapi import APIRouter, HTTPException
from fastapi.responses import Response
from pydantic import BaseModel

from app.config import get_settings
from app.services.rag_pipeline import _get_llm
from langchain_core.prompts import ChatPromptTemplate

logger = logging.getLogger(__name__)
router = APIRouter()

ALLOWED_VOICES = {"alloy", "echo", "fable", "onyx", "nova", "shimmer"}

_SPOKEN_REWRITE_PROMPT = ChatPromptTemplate.from_messages([
    (
        "system",
        (
            "You are a skilled audio narrator for a learning app. "
            "Rewrite the following text as a natural, engaging spoken explanation — "
            "like a knowledgeable teacher talking warmly to a curious student. "
            "Remove ALL markdown (headers, bullets, bold, asterisks, code blocks). "
            "Use smooth transitions. Be clear and engaging. "
            "Return ONLY the spoken script, nothing else."
        ),
    ),
    ("human", "{text}"),
])


def _strip_markdown(text: str) -> str:
    text = re.sub(r"#{1,6}\s+", "", text)
    text = re.sub(r"\*{1,3}(.+?)\*{1,3}", r"\1", text)
    text = re.sub(r"`{1,3}[^`]*`{1,3}", "", text)
    text = re.sub(r"\[([^\]]+)\]\([^)]+\)", r"\1", text)
    text = re.sub(r"^\s*[-*+]\s+", "", text, flags=re.MULTILINE)
    text = re.sub(r"\n{2,}", ". ", text)
    text = re.sub(r"\n", " ", text)
    return text.strip()


def _pcm16_to_wav(pcm_data: bytes, sample_rate: int = 24000, channels: int = 1) -> bytes:
    data_size   = len(pcm_data)
    byte_rate   = sample_rate * channels * 2
    block_align = channels * 2
    header = struct.pack(
        "<4sI4s4sIHHIIHH4sI",
        b"RIFF", 36 + data_size,
        b"WAVE",
        b"fmt ", 16,
        1, channels, sample_rate,
        byte_rate, block_align, 16,
        b"data", data_size,
    )
    return header + pcm_data


async def _stream_pcm16(url: str, headers: dict, body: dict) -> list[bytes]:
    """Extracted for testability — streams OpenRouter SSE and returns PCM16 chunks."""
    pcm_chunks: list[bytes] = []
    async with httpx.AsyncClient(timeout=60) as client:
        async with client.stream("POST", url, headers=headers, json=body) as resp:
            if resp.status_code != 200:
                raw = await resp.aread()
                raise HTTPException(status_code=502, detail=raw.decode())
            async for line in resp.aiter_lines():
                if not line.startswith("data: "):
                    continue
                payload_str = line[6:]
                if payload_str.strip() == "[DONE]":
                    break
                try:
                    data = json.loads(payload_str)
                    choices = data.get("choices", [])
                    if not choices:
                        continue
                    audio = choices[0].get("delta", {}).get("audio") or {}
                    chunk_b64 = audio.get("data")
                    if chunk_b64:
                        pcm_chunks.append(base64.b64decode(chunk_b64))
                except Exception:
                    continue
    return pcm_chunks


class TTSRequest(BaseModel):
    text: str
    voice: str = "nova"


@router.post("/tts", summary="AI speech via gpt-4o-audio-preview on OpenRouter")
async def text_to_speech(payload: TTSRequest) -> Response:
    settings = get_settings()
    voice = payload.voice if payload.voice in ALLOWED_VOICES else "nova"

    # Stage 1: LLM rewrite
    llm = _get_llm()
    try:
        chain = _SPOKEN_REWRITE_PROMPT | llm
        result = chain.invoke({"text": payload.text})
        spoken_text = result.content.strip()
    except Exception as exc:
        logger.warning("Audio rewrite failed (%s) — using stripped text.", exc)
        spoken_text = _strip_markdown(payload.text)

    # Stage 2: Stream audio via raw httpx SSE
    try:
        pcm_chunks = await _stream_pcm16(
            url=f"{settings.openrouter_base_url}/chat/completions",
            headers={
                "Authorization": f"Bearer {settings.openrouter_api_key}",
                "Content-Type": "application/json",
                "HTTP-Referer": "https://my-learning-ai-assistant.local",
                "X-Title": "My Learning AI Assistant",
            },
            body={
                "model": "openai/gpt-4o-audio-preview",
                "modalities": ["text", "audio"],
                "audio": {"voice": voice, "format": "pcm16"},
                "messages": [{"role": "user", "content": spoken_text}],
                "stream": True,
            },
        )
    except HTTPException:
        raise
    except Exception as exc:
        logger.error("TTS stream error: %s", exc)
        raise HTTPException(status_code=502, detail=f"TTS error: {exc}")

    if not pcm_chunks:
        raise HTTPException(status_code=502, detail="No audio data received from model.")

    wav_bytes = _pcm16_to_wav(b"".join(pcm_chunks))
    return Response(
        content=wav_bytes,
        media_type="audio/wav",
        headers={"Content-Disposition": "attachment; filename=speech.wav"},
    )