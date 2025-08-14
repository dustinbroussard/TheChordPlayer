#!/usr/bin/env python3
"""Command line utility for chord and melody detection.

Usage::

    python chord_melody_cli.py <audio-file>

The script prints detected chord progression and a simple melody line with
timestamps. It relies on the analysis routines implemented in
``chord_player_v43``.
"""

from __future__ import annotations

import sys

from chord_player_v43 import detect_chords


def main() -> None:
    if len(sys.argv) < 2:
        print("Usage: python chord_melody_cli.py <audio-file>")
        sys.exit(1)

    path = sys.argv[1]

    chords, key_sig, stats = detect_chords(path)

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


if __name__ == "__main__":
    main()

