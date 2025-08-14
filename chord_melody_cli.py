#!/usr/bin/env python3
"""Command line utility for chord and melody detection with optional stem separation."""

from __future__ import annotations

import argparse
import json
import os
import tempfile

import numpy as np
import librosa
import soundfile as sf

from chord_player_v43 import detect_chords, StemSeparator


def main() -> None:
    p = argparse.ArgumentParser(description="Chord and melody detection")
    p.add_argument("audio", help="Input audio file")
    p.add_argument("--json", type=str, help="Write results to JSON file")
    p.add_argument("--stems", choices=["hpss", "two", "four"], help="Separate stems")
    p.add_argument("--stems-out", type=str, help="Directory to write stem WAVs")
    args = p.parse_args()

    y, sr = librosa.load(args.audio, sr=None, mono=False)
    y_for_analysis = librosa.to_mono(y)
    stems = {}

    if args.stems:
        sep = StemSeparator(fs=sr)
        if args.stems == "hpss":
            stems, srr = sep.hpss(y, sr)
            stems = {k: (np.stack([v, v], axis=0) if v.ndim == 1 else v) for k, v in stems.items()}
        else:
            mode = "two_stems" if args.stems == "two" else "four_stems"
            stems, srr, _ = sep.demucs(y, sr, mode=mode)
        if args.stems_out:
            os.makedirs(args.stems_out, exist_ok=True)
            for name, buf in stems.items():
                sf.write(os.path.join(args.stems_out, f"{name}.wav"), buf.T, srr)
        if "instrumental" in stems:
            y_for_analysis = librosa.to_mono(stems["instrumental"])
            sr = srr

    if args.stems and "instrumental" in stems:
        tmp = tempfile.NamedTemporaryFile(suffix=".wav", delete=False)
        sf.write(tmp.name, y_for_analysis, sr)
        tmp.close()
        chords, key_sig, stats = detect_chords(tmp.name)
        os.unlink(tmp.name)
    else:
        chords, key_sig, stats = detect_chords(args.audio)

    print(f"Detected key: {key_sig}")
    print("\nChords:")
    for t, name in chords:
        print(f"{t:6.2f}s: {name}")

    melody = stats.get("melody", [])
    print("\nMelody:")
    if melody:
        for t, note in melody:
            print(f"{t:6.2f}s: {note}")
    else:
        print("  (no melody detected)")

    if args.json:
        with open(args.json, "w", encoding="utf-8") as f:
            json.dump({"key": key_sig, "chords": chords, "stats": stats}, f, indent=2)


if __name__ == "__main__":
    main()

