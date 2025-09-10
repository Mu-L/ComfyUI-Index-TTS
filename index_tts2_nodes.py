import os
import numpy as np
import torch
from typing import Any, Dict, Tuple

from .indextts2 import IndexTTS2Loader, IndexTTS2Engine


class IndexTTS2Node:
    """
    ComfyUI node: Index TTS 2 (basic)
    - English display name
    - Qwen soft-instruction optional (exposed as a boolean input)
    - Reuses AudioCleanupNode externally (not here)
    """

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "text": ("STRING", {"multiline": True, "default": "Hello, this is IndexTTS2."}),
                "reference_audio": ("AUDIO",),
                "mode": (["Auto", "Duration", "Tokens"], {"default": "Auto"}),
            },
            "optional": {
                # Optional external inputs (ONLY two):
                # 1) Use emotion reference audio -> connect emo_ref_audio, weight controlled by emotion_weight
                # 2) Use emotion vector -> connect emo_vector (JSON from companion node)
                "emo_ref_audio": ("AUDIO",),
                "emo_vector": ("STRING", {"multiline": False, "default": ""}),
                # Extra params (ONLY two):
                # - emotion_weight works only when emo_ref_audio connected
                # - emotion_description triggers text-emotion mode when non-empty
                "emotion_weight": ("FLOAT", {"default": 0.8, "min": 0.0, "max": 1.6, "step": 0.05}),
                "emotion_description": ("STRING", {"multiline": True, "default": ""}),
                "duration_sec": ("FLOAT", {"default": 0.0, "min": 0.0, "max": 600.0, "step": 0.1}),
                "token_count": ("INT", {"default": 0, "min": 0, "max": 8192, "step": 1}),
                "speed": ("FLOAT", {"default": 1.0, "min": 0.25, "max": 3.0, "step": 0.05}),
                # Advanced generation parameters (always visible, like HF demo)
                "do_sample": ("BOOL", {"default": False}),
                "temperature": ("FLOAT", {"default": 0.8, "min": 0.1, "max": 2.0, "step": 0.05}),
                "top_p": ("FLOAT", {"default": 0.9, "min": 0.0, "max": 1.0, "step": 0.01}),
                "top_k": ("INT", {"default": 30, "min": 0, "max": 100, "step": 1}),
                "num_beams": ("INT", {"default": 3, "min": 1, "max": 10, "step": 1}),
                "repetition_penalty": ("FLOAT", {"default": 10.0, "min": 1.0, "max": 10.0, "step": 0.1}),
                "length_penalty": ("FLOAT", {"default": 0.0, "min": -2.0, "max": 2.0, "step": 0.1}),
                "max_mel_tokens": ("INT", {"default": 1500, "min": 50, "max": 1815, "step": 5}),
                "max_tokens_per_sentence": ("INT", {"default": 120, "min": 0, "max": 600, "step": 5}),
                "seed": ("INT", {"default": 0, "min": 0, "max": 2**32 - 1}),
                "return_subtitles": ("BOOL", {"default": True}),
            },
        }

    RETURN_TYPES = ("AUDIO", "INT", "STRING")
    RETURN_NAMES = ("audio", "seed", "subtitle")
    FUNCTION = "generate"
    CATEGORY = "audio"

    def __init__(self):
        self.loader = IndexTTS2Loader()
        self.engine = IndexTTS2Engine(self.loader)

    @staticmethod
    def _process_audio_input(audio: Any) -> Tuple[np.ndarray, int]:
        # Accept ComfyUI audio dict or (wave, sr)
        if isinstance(audio, dict) and "waveform" in audio and "sample_rate" in audio:
            wave = audio["waveform"]
            sr = int(audio["sample_rate"])
            if isinstance(wave, torch.Tensor):
                # Expect [B, C, T] or [T]
                if wave.dim() == 3:
                    wave = wave[0, 0].detach().cpu().numpy()
                elif wave.dim() == 1:
                    wave = wave.detach().cpu().numpy()
                else:
                    wave = wave.flatten().detach().cpu().numpy()
            elif isinstance(wave, np.ndarray):
                if wave.ndim == 3:
                    wave = wave[0, 0]
                elif wave.ndim == 2:
                    wave = wave[0]
            return wave.astype(np.float32), sr
        elif isinstance(audio, tuple) and len(audio) == 2:
            wave, sr = audio
            if isinstance(wave, torch.Tensor):
                wave = wave.detach().cpu().numpy()
            return wave.astype(np.float32), int(sr)
        else:
            raise ValueError("reference_audio must be AUDIO type or (wave, sr)")

    def generate(
        self,
        text: str,
        reference_audio: Any,
        mode: str = "Auto",
        # Optional emotion inputs
        emo_ref_audio: Any = None,
        emo_vector: str = "",
        # Extra params (ONLY two)
        emotion_weight: float = 0.8,
        emotion_description: str = "",
        duration_sec: float = 0.0,
        token_count: int = 0,
        speed: float = 1.0,
        do_sample: bool = False,
        temperature: float = 0.8,
        top_p: float = 0.9,
        top_k: int = 30,
        num_beams: int = 3,
        repetition_penalty: float = 10.0,
        length_penalty: float = 0.0,
        max_mel_tokens: int = 1500,
        max_tokens_per_sentence: int = 120,
        seed: int = 0,
        return_subtitles: bool = True,
    ):
        ref = self._process_audio_input(reference_audio)
        emo_ref = self._process_audio_input(emo_ref_audio) if emo_ref_audio is not None else None

        # Determine emotion mode priority: emo_vector > emotion_description > emo_ref_audio > base
        emo_vec = None
        if isinstance(emo_vector, str) and emo_vector.strip():
            try:
                import json as _json
                v = _json.loads(emo_vector)
                if isinstance(v, (list, tuple)):
                    emo_vec = [float(x) for x in v]
            except Exception:
                emo_vec = None

        emo_text_val = None
        if emo_vec is None and isinstance(emotion_description, str) and emotion_description.strip():
            emo_text_val = emotion_description.strip()
            emo_ref = None  # text 模式优先，不再使用 emo_ref

        # weight works only if emo_ref exists
        _emo_weight = float(emotion_weight)

        sr, wave, sub = self.engine.generate(
            text=text,
            reference_audio=ref,
            mode=mode,
            duration_sec=(duration_sec if duration_sec > 0 else None),
            token_count=(token_count if token_count > 0 else None),
            speed=speed,
            # Advanced gen params
            do_sample=bool(do_sample),
            temperature=temperature,
            top_p=top_p,
            top_k=top_k,
            num_beams=num_beams,
            repetition_penalty=repetition_penalty,
            length_penalty=length_penalty,
            max_mel_tokens=max_mel_tokens,
            max_tokens_per_sentence=max_tokens_per_sentence,
            # Emotion control
            emotion_control_method=None,
            emo_text=emo_text_val,
            emo_ref_audio=emo_ref,
            emo_vector=emo_vec,
            emo_weight=_emo_weight,
            seed=seed,
            return_subtitles=return_subtitles,
        )

        # Pack back to ComfyUI audio dict
        wave_t = torch.tensor(wave, dtype=torch.float32).unsqueeze(0).unsqueeze(0)
        audio = {"waveform": wave_t, "sample_rate": int(sr)}
        subtitle = sub or ""
        return (audio, seed, subtitle)
