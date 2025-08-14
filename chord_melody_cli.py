#!/usr/bin/env python3
from __future__ import annotations

import sys, json, argparse, os
from pathlib import Path

from chord_player_v43 import detect_chords  # uses the full analyzer (chords+melody)  # noqa

def positive_float(x: str) -> float:
    v = float(x)
    if v <= 0:
        raise argparse.ArgumentTypeError("must be > 0")
    return v

def main() -> None:
    p = argparse.ArgumentParser(
        description="Chord + melody analyzer (CLI) — matches the v4.3 GUI engine."
    )
    p.add_argument("audio", help="Input audio file (wav/mp3/flac/ogg/m4a/...)")
    # Analysis controls (mirror GUI settings)
    p.add_argument("--smoothing", choices=["adaptive", "gaussian", "median"], default="adaptive",
                   help="Chroma smoothing strategy (default: adaptive)")
    p.add_argument("--template-threshold", type=float, default=0.25,
                   help="Similarity threshold for labeling chords (default: 0.25)")
    p.add_argument("--min-chord-duration", type=positive_float, default=0.30,
                   help="Minimum duration seconds per chord segment (default: 0.30)")
    p.add_argument("--fast", action="store_true",
                   help="Enable performance mode (reduced chord types)")
    p.add_argument("--no-key", action="store_true",
                   help="Disable key-aware analysis")
    p.add_argument("--single-feature", action="store_true",
                   help="Use only CQT (skip multi-feature blend)")
    p.add_argument("--melody-threshold", type=float, default=0.80,
                   help="Confidence for melody note emission (PYIN voiced_prob) default: 0.80")

    # Outputs
    p.add_argument("--json", metavar="PATH", help="Write full results as JSON")
    p.add_argument("--txt", metavar="PATH", help="Write chords as a text file")
    p.add_argument("--midi", metavar="PATH", help="Write a guide MIDI (root notes)")
    p.add_argument("--quiet", "-q", action="store_true", help="Suppress console printing")

    args = p.parse_args()

    in_path = args.audio
    if not os.path.isfile(in_path):
        print(f"Error: file not found: {in_path}", file=sys.stderr)
        sys.exit(2)

    # Run analysis (maps directly to detect_chords kwargs in v4.3)
    chords, key_sig, stats = detect_chords(
        in_path,
        use_multi=not args.single_feature,
        smoothing=args.smoothing,
        min_chord_duration=args.min_chord_duration,
        template_threshold=args.template_threshold,
        key_aware=not args.no_key,
        performance_mode=args.fast,
    )

    # Optionally tighten melody threshold post-hoc
    if "melody" in stats and stats["melody"]:
        # stats["melody"] is [(time, note)], already thresholded in detect_melody
        # Keeping arg here in case you later expose melody confidence stream.
        pass

    # Console output
    if not args.quiet:
        print(f"Detected key: {key_sig}")
        print("\nChords:")
        for t, name in chords:
            print(f"{t:6.2f}s: {name}")
        print("\nMelody:")
        mel = stats.get("melody", [])
        if mel:
            for t, note in mel:
                print(f"{t:6.2f}s: {note}")
        else:
            print("  (no melody detected)")

    # Exports
    if args.json:
        payload = {
            "key_signature": key_sig,
            "tempo": stats.get("estimated_tempo", 120.0),
            "audio_duration": stats.get("audio_duration", 0.0),
            "analysis_stats": stats,
            "chord_progression": [{"time": t, "chord": n} for t, n in chords],
            "melody": [{"time": t, "note": n} for (t, n) in stats.get("melody", [])],
        }
        Path(args.json).parent.mkdir(parents=True, exist_ok=True)
        with open(args.json, "w", encoding="utf-8") as f:
            json.dump(payload, f, indent=2)

    if args.txt:
        lines = [f"{t:6.2f}s\t{n}" for t, n in chords]
        Path(args.txt).parent.mkdir(parents=True, exist_ok=True)
        with open(args.txt, "w", encoding="utf-8") as f:
            f.write("\n".join(lines) + "\n")

    if args.midi:
        try:
            import mido
            from mido import MidiFile, MidiTrack, Message, MetaMessage
        except Exception as e:
            print(f"Warning: MIDI export skipped (mido not available): {e}", file=sys.stderr)
        else:
            mid = MidiFile()
            track = MidiTrack()
            mid.tracks.append(track)

            tempo_bpm = float(stats.get("estimated_tempo", 120.0))
            micro_per_beat = int(60000000 / max(1.0, tempo_bpm))
            track.append(MetaMessage("set_tempo", tempo=micro_per_beat))

            ticks_per_beat = 480
            mid.ticks_per_beat = ticks_per_beat

            # Helper: map note name to pitch class
            NOTE_NAMES = ['C','C#','D','D#','E','F','F#','G','G#','A','A#','B']

            prev_time = 0.0
            for i, (t_sec, chord) in enumerate(chords):
                # Duration to next event
                if i < len(chords) - 1:
                    dur_sec = max(0.05, chords[i+1][0] - t_sec)
                else:
                    dur_sec = max(1.0, stats.get("audio_duration", 0.0) - t_sec)

                # convert wall-time to ticks
                delta_ticks = int((t_sec - prev_time) * ticks_per_beat * tempo_bpm / 60.0)
                dur_ticks = int(dur_sec * ticks_per_beat * tempo_bpm / 60.0)

                # Extract root (guide note only, same as GUI export behavior)
                root = chord.split('/')[0] if chord else "N.C."
                note_on_emitted = False

                if root != "N.C." and root:
                    # crude parse of root letter with accidental
                    if len(root) >= 2 and root[1] in ['#', 'b']:
                        root_name = root[:2]
                    else:
                        root_name = root[0]
                    # Flats: map to enharmonic sharps for NOTE_NAMES
                    root_name = root_name.replace('Db', 'C#').replace('Eb', 'D#') \
                                         .replace('Gb', 'F#').replace('Ab', 'G#') \
                                         .replace('Bb', 'A#')

                    if root_name in NOTE_NAMES:
                        pc = NOTE_NAMES.index(root_name)
                        midi_pitch = 60 + pc  # C4-based reference
                        track.append(Message("note_on", note=midi_pitch, velocity=64, time=delta_ticks))
                        track.append(Message("note_off", note=midi_pitch, velocity=64, time=dur_ticks))
                        note_on_emitted = True

                if not note_on_emitted:
                    # advance time with a rest
                    track.append(Message("note_on", note=0, velocity=0, time=delta_ticks))
                    track.append(Message("note_off", note=0, velocity=0, time=dur_ticks))

                prev_time = t_sec

            Path(args.midi).parent.mkdir(parents=True, exist_ok=True)
            mid.save(args.midi)

    # Exit with 0 if we found *anything* useful
    sys.exit(0 if chords or stats.get("melody") else 3)

if __name__ == "__main__":
    main()
