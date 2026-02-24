import sox

def apply_cozmo_effects(input_path: str, output_path: str) -> None:
    tfm = sox.Transformer()

    tfm.pitch(6)
    tfm.speed(1.1)
    tfm.gain(2)
    tfm.contrast(30)

    tfm.norm()

    tfm.build(input_path, output_path)
    print(f"[SoX] Cozmo voice saved → {output_path}")