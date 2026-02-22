"""
Voice Cloning Script — powered by Chatterbox TTS (Resemble AI, 2025)
----------------------------------------------------------------------
Clones a voice from a reference WAV file and outputs a new WAV of that
voice speaking the given text prompt. No training required.

Requirements:
    pip install chatterbox-tts torch torchaudio

Usage:
    python voice_clone.py --sample voice_sample.wav --text "Hello!" --output output.wav

All parameters:
    --exaggeration  0.25–2.0  Emotion/expressiveness. Higher = more dramatic. (default: 0.5)
    --cfg-weight    0.0–1.0   Voice reference adherence. Lower = relaxed pacing. (default: 0.5)
    --temperature   0.05–5.0  Output variation/randomness. Lower = consistent. (default: 0.8)
    --seed          integer   Fixed seed for reproducible output. Omit for random.
    --turbo                   Use Chatterbox Turbo model (faster, supports [laugh]/[cough] tags)

Turbo model paralinguistic tags (use in --text):
    [laugh]   [chuckle]   [cough]   [sigh]   [gasp]

Tips for best results:
    - Reference audio should be 5–20 seconds of clean, clear speech
    - Avoid background music or noise in the sample
    - GPU (CUDA/MPS) will be used automatically if available; CPU also works
    - For consistent results across runs, set --seed to any fixed integer
"""

import argparse
import torch
import torchaudio as ta
from pathlib import Path


def get_device() -> str:
    if torch.cuda.is_available():
        return "cuda"
    if torch.backends.mps.is_available():
        return "mps"
    return "cpu"


def clone_voice(
    sample_path: str,
    text: str,
    output_path: str,
    exaggeration: float = 0.5,
    cfg_weight: float = 0.5,
    temperature: float = 0.3,
    seed: int | None = None,
    turbo: bool = False,
) -> None:
    """
    Clone a voice from a reference audio file and synthesize the given text.

    Args:
        sample_path:   Path to the reference WAV file.
        text:          The text you want the cloned voice to say.
                       When using Turbo, you can embed tags like [laugh], [cough], [sigh].
        output_path:   Where to save the output WAV file.
        exaggeration:  Emotion/expressiveness intensity (0.25–2.0). Default 0.5.
                       0.25 = flat/neutral, 1.0 = natural, 2.0 = very dramatic.
        cfg_weight:    How closely to match the reference voice (0.0–1.0). Default 0.5.
                       Lower values relax pacing; try 0.3 if speaker talks fast.
        temperature:   Output randomness/variation (0.05–5.0). Default 0.8.
                       Lower = more consistent takes, higher = more varied.
        seed:          Optional integer seed for reproducible output. None = random.
        turbo:         Use Chatterbox Turbo model (faster, lower VRAM, supports
                       paralinguistic tags like [laugh], [cough], [chuckle]).
    """
    sample = Path(sample_path)
    if not sample.exists():
        raise FileNotFoundError(f"Sample file not found: {sample_path}")

    device = get_device()
    print(f"Using device: {device}")

    if seed is not None:
        torch.manual_seed(seed)
        print(f"Seed set to: {seed}")

    if turbo:
        from chatterbox.tts_turbo import ChatterboxTurboTTS
        print("Loading Chatterbox Turbo model (downloads on first run)...")
        model = ChatterboxTurboTTS.from_pretrained(device=device)
    else:
        from chatterbox.tts import ChatterboxTTS
        print("Loading Chatterbox TTS model (downloads on first run, ~1–2 GB)...")
        model = ChatterboxTTS.from_pretrained(device=device)

    print(f'Synthesizing: "{text}"')
    wav = model.generate(
        text=text,
        audio_prompt_path=str(sample),
        exaggeration=exaggeration,
        cfg_weight=cfg_weight,
        temperature=temperature,
    )

    ta.save(output_path, wav, model.sr)
    print(f"Done! Output saved to: {output_path}")


def main():
    parser = argparse.ArgumentParser(
        description="Clone a voice from a WAV sample using Chatterbox TTS.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument(
        "--sample", required=True,
        help="Path to the reference voice WAV file (5–20 seconds of clean audio recommended)."
    )
    parser.add_argument(
        "--text", required=True,
        help="The text you want the cloned voice to say. "
             "With --turbo, you can use tags like [laugh], [cough], [chuckle]."
    )
    parser.add_argument(
        "--output", default="output.wav",
        help="Output WAV file path (default: output.wav)."
    )
    parser.add_argument(
        "--exaggeration", type=float, default=0.5,
        help="Emotion/expressiveness intensity, 0.25–2.0 (default: 0.5). "
             "0.25=flat, 1.0=natural, 2.0=very dramatic."
    )
    parser.add_argument(
        "--cfg-weight", type=float, default=0.5,
        dest="cfg_weight",
        help="Voice reference adherence, 0.0–1.0 (default: 0.5). "
             "Lower = more relaxed pacing. Try 0.3 if speaker talks fast."
    )
    parser.add_argument(
        "--temperature", type=float, default=0.8,
        help="Output variation/randomness, 0.05–5.0 (default: 0.8). "
             "Lower = consistent takes, higher = more varied results."
    )
    parser.add_argument(
        "--seed", type=int, default=None,
        help="Fixed integer seed for reproducible output (default: random). "
             "Set this to get identical results across multiple runs."
    )
    parser.add_argument(
        "--turbo", action="store_true",
        help="Use Chatterbox Turbo model: faster, lower VRAM, and supports "
             "paralinguistic tags in text like [laugh], [cough], [chuckle], [sigh], [gasp]."
    )
    args = parser.parse_args()

    clone_voice(
        sample_path=args.sample,
        text=args.text,
        output_path=args.output,
        exaggeration=args.exaggeration,
        cfg_weight=args.cfg_weight,
        temperature=args.temperature,
        seed=args.seed,
        turbo=args.turbo,
    )


if __name__ == "__main__":
    main()