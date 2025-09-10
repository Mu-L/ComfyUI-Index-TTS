import json
import numpy as np
from typing import List, Tuple


class IndexTTSEmotionVectorNode:
    """
    ComfyUI node: Index TTS Emotion Vector
    - Outputs an emotion vector as JSON string for Index TTS 2's `emo_vector` input.
    - Matches HF demo with 8 sliders: Happy, Angry, Sad, Fear, Hate, Love, Surprise, Neutral
    - Optional random sampling (seeded) when you want quick stochastic presets.
    """

    EMO_ORDER = [
        "Happy", "Angry", "Sad", "Fear", "Hate", "Love", "Surprise", "Neutral"
    ]

    @classmethod
    def INPUT_TYPES(cls):
        sliders = {
            "Happy": ("FLOAT", {"default": 0.0, "min": 0.0, "max": 1.4, "step": 0.01}),
            "Angry": ("FLOAT", {"default": 0.0, "min": 0.0, "max": 1.4, "step": 0.01}),
            "Sad": ("FLOAT", {"default": 0.0, "min": 0.0, "max": 1.4, "step": 0.01}),
            "Fear": ("FLOAT", {"default": 0.0, "min": 0.0, "max": 1.4, "step": 0.01}),
            "Hate": ("FLOAT", {"default": 0.0, "min": 0.0, "max": 1.4, "step": 0.01}),
            "Love": ("FLOAT", {"default": 0.0, "min": 0.0, "max": 1.4, "step": 0.01}),
            "Surprise": ("FLOAT", {"default": 0.0, "min": 0.0, "max": 1.4, "step": 0.01}),
            "Neutral": ("FLOAT", {"default": 0.0, "min": 0.0, "max": 1.4, "step": 0.01}),
        }
        return {
            "required": {
                **sliders,
            },
            "optional": {
                "random_sampling": ("BOOL", {"default": False}),
                "seed": ("INT", {"default": 0, "min": 0, "max": 2**32 - 1}),
                "normalize": ("BOOL", {"default": True}),
                "top_k_random": ("INT", {"default": 2, "min": 1, "max": 8, "step": 1}),
            },
        }

    RETURN_TYPES = ("STRING",)
    RETURN_NAMES = ("emo_vector",)
    FUNCTION = "build_vector"
    CATEGORY = "audio"

    def _sample_random_vector(self, seed: int, top_k: int) -> List[float]:
        rng = np.random.default_rng(int(seed))
        # Dirichlet over 8 dims, then zero out all but top_k for sparse expressive control
        vec = rng.dirichlet(np.ones(8)).astype(np.float32)
        idx = np.argsort(vec)[::-1]
        mask = np.zeros_like(vec)
        mask[idx[: max(1, int(top_k))]] = 1.0
        vec = (vec * mask)
        s = float(vec.sum())
        if s > 0:
            vec = vec / s
        return vec.tolist()

    def build_vector(
        self,
        Happy: float,
        Angry: float,
        Sad: float,
        Fear: float,
        Hate: float,
        Love: float,
        Surprise: float,
        Neutral: float,
        random_sampling: bool = False,
        seed: int = 0,
        normalize: bool = True,
        top_k_random: int = 2,
    ) -> Tuple[str]:
        if random_sampling:
            vec = self._sample_random_vector(seed, top_k_random)
        else:
            vec = [Happy, Angry, Sad, Fear, Hate, Love, Surprise, Neutral]
            if normalize:
                s = float(sum(max(0.0, float(x)) for x in vec))
                if s > 0:
                    vec = [float(max(0.0, float(x)))/s for x in vec]
                else:
                    # fallback to neutral
                    vec = [0.0]*7 + [1.0]
        return (json.dumps(vec, ensure_ascii=False),)
