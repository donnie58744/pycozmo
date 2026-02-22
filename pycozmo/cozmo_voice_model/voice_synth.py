"""
Cozmo Voice Synthesizer — Pure DSP Pipeline
---------------------------------------------
Generates speech via espeak-ng then processes it with DSP effects to
sound like Cozmo's voice. No AI models used.

Effects applied:
    1. Pitch shift up       — higher, robotic tone
    2. Ring modulation      — metallic buzz characteristic
    3. Bitcrush             — lo-fi, tiny speaker quality

Requirements:
    brew install espeak-ng          # macOS
    sudo apt install espeak-ng      # Linux
    pip install numpy scipy soundfile

Usage:
    python cozmo_tts.py --text "Hello, I am Cozmo!" --output cozmo_out.wav

Tuning tips:
    --pitch-shift     Semitones to shift up. 4–8 is a good Cozmo range.
    --ring-freq       Ring mod carrier frequency in Hz. 80–150 for metallic buzz.
    --ring-mix        Blend of ring mod 0.0–1.0. 0.4 is subtle, 0.9 is heavy.
    --bit-depth       Bitcrush depth. 8 = lo-fi robot, 16 = clean.
    --speed           espeak-ng words per minute. Higher = faster Cozmo chatter.
    --pitch           espeak-ng base pitch 0–99. Higher = more robotic.
"""

import argparse
import subprocess
import tempfile
import numpy as np
import soundfile as sf
from pathlib import Path
from scipy.signal import resample_poly
from math import gcd


# ── DSP Effects ────────────────────────────────────────────────────────────────

def pitch_shift(audio: np.ndarray, sr: int, semitones: float) -> np.ndarray:
    """
    Shift pitch by resampling — no AI, pure sample rate trick.
    Shifts up by speeding up the audio then resampling back to original length.
    """
    factor = 2 ** (semitones / 12.0)
    # Resample to a shorter length (speeds up = pitch up), then stretch back
    orig_len = len(audio)
    new_len = int(orig_len / factor)
    shifted = resample_poly(audio, new_len, orig_len)
    # Resample back to original length to preserve duration
    return resample_poly(shifted, orig_len, new_len)


def ring_modulate(audio: np.ndarray, sr: int, carrier_freq: float, mix: float) -> np.ndarray:
    """
    Ring modulation: multiply signal by a sine carrier wave.
    Creates metallic, buzzy overtones characteristic of robot voices.
    """
    t = np.arange(len(audio)) / sr
    carrier = np.sin(2 * np.pi * carrier_freq * t).astype(np.float32)
    modulated = audio * carrier
    return audio * (1.0 - mix) + modulated * mix


def bitcrush(audio: np.ndarray, bit_depth: int) -> np.ndarray:
    """
    Reduce bit depth to create lo-fi digital distortion.
    Lower bit_depth = crunchier, more robotic sound.
    """
    levels = 2 ** bit_depth
    crushed = np.round(audio * levels) / levels
    return crushed.astype(np.float32)


def normalize(audio: np.ndarray, target_peak: float = 0.9) -> np.ndarray:
    """Normalize audio to a target peak amplitude."""
    peak = np.max(np.abs(audio))
    if peak > 0:
        audio = audio * (target_peak / peak)
    return audio


# ── Pipeline ───────────────────────────────────────────────────────────────────

def generate_espeak(text: str, output_path: str, speed: int, pitch: int, voice: str) -> None:
    """Run espeak-ng to generate base TTS audio."""
    cmd = [
        "espeak-ng",
        "-v", voice,
        "-s", str(speed),
        "-p", str(pitch),
        "-w", output_path,
        text,
    ]
    result = subprocess.run(cmd, capture_output=True, text=True)
    if result.returncode != 0:
        raise RuntimeError(f"espeak-ng failed:\n{result.stderr}")
    print(f"  espeak-ng: generated base speech")


def apply_cozmo_dsp(
    input_path: str,
    output_path: str,
    pitch_semitones: float = 6.0,
    ring_freq: float = 120.0,
    ring_mix: float = 0.5,
    bit_depth: int = 10,
    keep_intermediate: bool = False,
) -> None:
    """Load audio, apply DSP chain, save output."""
    audio, sr = sf.read(input_path, dtype="float32")

    # Mono mixdown if stereo
    if audio.ndim > 1:
        audio = audio.mean(axis=1)

    print(f"  Audio: {len(audio)/sr:.2f}s @ {sr}Hz")

    # 1. Pitch shift
    print(f"  [1/3] Pitch shifting +{pitch_semitones} semitones...")
    audio = pitch_shift(audio, sr, pitch_semitones)

    # 2. Ring modulation
    print(f"  [2/3] Ring modulating @ {ring_freq}Hz (mix={ring_mix})...")
    audio = ring_modulate(audio, sr, ring_freq, ring_mix)

    # 3. Bitcrush
    print(f"  [3/3] Bitcrushing to {bit_depth}-bit...")
    audio = bitcrush(audio, bit_depth)

    # Normalize before saving
    audio = normalize(audio)

    sf.write(output_path, audio, sr)
    print(f"  Saved: {output_path}")


def run_pipeline(
    text: str,
    output_path: str,
    espeak_speed: int = 160,
    espeak_pitch: int = 70,
    espeak_voice: str = "en",
    pitch_semitones: float = 6.0,
    ring_freq: float = 120.0,
    ring_mix: float = 0.5,
    bit_depth: int = 10,
    keep_intermediate: bool = False,
) -> None:
    print("\n── Step 1: espeak-ng ──────────────────────────")
    with tempfile.NamedTemporaryFile(suffix=".wav", delete=not keep_intermediate) as tmp:
        espeak_wav = tmp.name

    generate_espeak(text, espeak_wav, speed=espeak_speed, pitch=espeak_pitch, voice=espeak_voice)

    if keep_intermediate:
        espeak_out = output_path.replace(".wav", "_espeak_raw.wav")
        import shutil
        shutil.copy(espeak_wav, espeak_out)
        print(f"  Raw espeak saved to: {espeak_out}")

    print("\n── Step 2: DSP Effects ────────────────────────")
    apply_cozmo_dsp(
        input_path=espeak_wav,
        output_path=output_path,
        pitch_semitones=pitch_semitones,
        ring_freq=ring_freq,
        ring_mix=ring_mix,
        bit_depth=bit_depth,
    )

    print(f"\n✓ Done! Cozmo voice saved to: {output_path}")


# ── CLI ────────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(
        description="Synthesize Cozmo's voice from text using espeak-ng + DSP.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument("--text", required=True, help="Text for Cozmo to say.")
    parser.add_argument("--output", default="cozmo_out.wav", help="Output WAV path (default: cozmo_out.wav).")

    esp = parser.add_argument_group("espeak-ng options")
    esp.add_argument("--speed", type=int, default=160,
                     help="Words per minute (default: 160). Higher = faster Cozmo chatter.")
    esp.add_argument("--pitch", type=int, default=70,
                     help="Base pitch 0–99 (default: 70). Higher = more robotic.")
    esp.add_argument("--voice", type=str, default="en",
                     help="espeak voice (default: 'en'). Try 'en+m3' or 'en+f3'.")

    dsp = parser.add_argument_group("DSP effect options")
    dsp.add_argument("--pitch-shift", type=float, default=6.0, dest="pitch_semitones",
                     help="Semitones to pitch shift up (default: 6.0). Range 2–10.")
    dsp.add_argument("--ring-freq", type=float, default=120.0,
                     help="Ring mod carrier frequency Hz (default: 120). Range 80–200.")
    dsp.add_argument("--ring-mix", type=float, default=0.5,
                     help="Ring mod blend 0.0–1.0 (default: 0.5). Higher = buzzier.")
    dsp.add_argument("--bit-depth", type=int, default=10,
                     help="Bitcrush depth (default: 10). Lower = crunchier. Range 6–14.")
    dsp.add_argument("--keep-intermediate", action="store_true",
                     help="Also save the raw espeak WAV before DSP for comparison.")

    args = parser.parse_args()

    run_pipeline(
        text=args.text,
        output_path=args.output,
        espeak_speed=args.speed,
        espeak_pitch=args.pitch,
        espeak_voice=args.voice,
        pitch_semitones=args.pitch_semitones,
        ring_freq=args.ring_freq,
        ring_mix=args.ring_mix,
        bit_depth=args.bit_depth,
        keep_intermediate=args.keep_intermediate,
    )


if __name__ == "__main__":
    main()