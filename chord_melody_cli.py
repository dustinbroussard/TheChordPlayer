#!/usr/bin/env python3
"""
Chord+Melody CLI
----------------
Turn chord-analysis JSON (like the one Dustin shared) into a playable MIDI file,
optionally merging a provided melody, or auto-generating a scale-aware melody.

Usage:
  python chord_melody_cli.py --input example_input.json --output demo_output.mid \
      --mode dorian --melody auto --swing 0.0

Requirements:
  pip install mido python-rtmidi  (rtmidi only if you want realtime; not needed for file export)

JSON Format (minimal fields used):
{
  "key_signature": "E Minor",
  "tempo": 103.359375,
  "audio_duration": 248.76,
  "chord_progression": [
    {"time": 0.95, "chord": "E6sus4"},
    {"time": 3.29, "chord": "Esus2"},
    ...
  ]
}

Melody input (optional): CSV with header time,pitch where time=seconds, pitch=MIDI note number.
If omitted or "--melody auto", a melodic line is generated using the chosen mode.

Outputs:
  - MIDI file with 2 tracks: Chords and Melody
  - Basic console summary
"""

import argparse
import json
import math
from dataclasses import dataclass
from typing import List, Optional, Tuple, Dict

import mido

# -----------------------------
# Music helpers
# -----------------------------

NOTE_TO_SEMITONE = {
    'C': 0, 'C#': 1, 'Db': 1,
    'D': 2, 'D#': 3, 'Eb': 3,
    'E': 4, 'E#': 5, 'Fb': 4,
    'F': 5, 'F#': 6, 'Gb': 6,
    'G': 7, 'G#': 8, 'Ab': 8,
    'A': 9, 'A#': 10, 'Bb': 10,
    'B': 11, 'Cb': 11, 'B#': 0,
}

def clamp(x, lo, hi):
    return max(lo, min(hi, x))

def midi_of(note_name: str, octave: int) -> int:
    """Return MIDI note number for note name + octave, with C4=60"""
    semitone = NOTE_TO_SEMITONE[note_name]
    return 12 * (octave + 1) + semitone

def parse_root(chord_name: str) -> str:
    # Root is first letter + optional #/b
    if len(chord_name) >= 2 and chord_name[1] in ['#', 'b']:
        return chord_name[:2]
    return chord_name[0]

def intervals_for_chord(chord: str) -> List[int]:
    """
    Very small pragmatic chord spellings by type.
    Intervals in semitones from root, chosen to voice simply in guitar/piano-friendly stacks.
    """
    c = chord.lower()
    # default: ambiguous → sus2/4-ish triad
    intervals = [0, 7, 14]  # root+5+octave by default (power-ish)
    # Add flavors
    if 'sus2' in c:
        intervals = [0, 2, 7]
    if 'sus4' in c:
        intervals = [0, 5, 7]
    if 'add9' in c or '9' in c:
        intervals = [0, 7, 14, 16]  # add9 on top
    if '6' in c:
        # add6 close to top
        if 9 not in intervals:
            intervals = sorted(set(intervals + [9]))
    if 'maj7' in c or 'M9' in c.lower():
        # Maj7 color
        if 11 not in intervals:
            intervals = sorted(set(intervals + [11]))
    if 'm13' in c or 'm11' in c:
        # minor quality
        base = [0, 3, 7]
        if '11' in c:
            base += [17]
        if '13' in c:
            base += [21]
        return sorted(set(base))
    if 'maj' in c and 'maj7' not in c:
        # plain major
        return [0, 4, 7]
    if 'dim' in c:
        return [0, 3, 6]
    if 'aug' in c:
        return [0, 4, 8]
    if 'm' in c and 'maj' not in c:
        # minor triad
        return [0, 3, 7]
    if '11' in c and 'm' not in c:
        intervals = sorted(set(intervals + [17]))
    if '13' in c and 'm' not in c:
        intervals = sorted(set(intervals + [21]))
    if '7#9' in c:
        intervals = sorted(set(intervals + [10, 15]))  # dom7 + #9
    return sorted(set(intervals))

def scale_for_mode(key_root: str, mode: str) -> List[int]:
    """
    Return scale degrees (semitones from tonic) for a given mode.
    """
    modes = {
        'aeolian': [0, 2, 3, 5, 7, 8, 10],    # natural minor
        'dorian':  [0, 2, 3, 5, 7, 9, 10],
        'pent_min':[0, 3, 5, 7, 10],          # minor pentatonic
    }
    return modes.get(mode.lower(), modes['dorian'])

def tonic_of_key(key_sig: str) -> str:
    # crude parse: first token is tonic note
    token = key_sig.strip().split()[0]
    return token

# -----------------------------
# Data structures
# -----------------------------

@dataclass
class ChordEvent:
    time: float  # seconds
    name: str

@dataclass
class SongData:
    key_signature: str
    tempo: float
    audio_duration: float
    chords: List[ChordEvent]

# -----------------------------
# Parsing & quantization
# -----------------------------

def load_json(path: str) -> SongData:
    with open(path, 'r', encoding='utf-8') as f:
        data = json.load(f)
    chords = [ChordEvent(time=float(c['time']), name=str(c['chord'])) for c in data['chord_progression']]
    tempo = float(data.get('tempo') or data.get('analysis_stats', {}).get('estimated_tempo', 100.0))
    return SongData(
        key_signature=data.get('key_signature', 'E Minor'),
        tempo=tempo,
        audio_duration=float(data.get('audio_duration', chords[-1].time if chords else 60.0)),
        chords=chords
    )

def seconds_to_beats(seconds: float, bpm: float) -> float:
    return seconds * bpm / 60.0

def beats_to_ticks(beats: float, ppq: int) -> int:
    return int(round(beats * ppq))

# -----------------------------
# Melody handling
# -----------------------------

def load_melody_csv(path: str) -> List[Tuple[float, int]]:
    # Expected header: time,pitch
    out = []
    with open(path, 'r', encoding='utf-8') as f:
        for i, line in enumerate(f):
            line = line.strip()
            if i == 0 and 'time' in line and 'pitch' in line:
                continue
            if not line:
                continue
            t_str, p_str = line.split(',')[:2]
            out.append((float(t_str), int(p_str)))
    return out

def auto_melody(song: SongData, mode: str, density: float = 0.75) -> List[Tuple[float, int, float]]:
    """
    Generate (time_sec, midi, dur_sec).
    - Strong beats get chord tones
    - Passing notes from scale
    - Range E3..E5
    """
    rng = random.Random(42)
    bpm = song.tempo
    ppq_beats = []

    # Precompute scale in semitones from tonic
    tonic = tonic_of_key(song.key_signature)
    tonic_semi = NOTE_TO_SEMITONE[tonic]
    scale = scale_for_mode(song.key_signature, mode)

    # Helper to pick nearest scale tone around target
    def nearest_scale_midi(target_midi: int) -> int:
        # Map target to tonic-relative semitone class and snap to scale
        rel = (target_midi - (12 + tonic_semi)) % 12
        # choose closest allowed degree
        best = None
        best_d = 999
        for deg in scale:
            d = min((rel - deg) % 12, (deg - rel) % 12)
            if d < best_d:
                best_d = d
                best = deg
        # reconstruct absolute midi near target
        base = (target_midi // 12) * 12 + (12 + tonic_semi) + best
        # Nudge by octave to be closest
        candidates = [base - 12, base, base + 12]
        return min(candidates, key=lambda x: abs(x - target_midi))

    # Build events at 1/8 notes by default
    step_beats = 0.5  # eighths
    events = []
    # Build a quick lookup for chord by time
    chords = song.chords[:]
    chords.sort(key=lambda c: c.time)
    # Append sentinel
    chords.append(ChordEvent(time=song.audio_duration, name="End"))
    # Iterate over chord windows
    for i in range(len(chords) - 1):
        c0, c1 = chords[i], chords[i+1]
        # chord info
        root_name = parse_root(c0.name)
        root_semi = NOTE_TO_SEMITONE.get(root_name, NOTE_TO_SEMITONE[tonic])
        intervals = intervals_for_chord(c0.name)
        chord_tones = [(12 + root_semi + iv) % 12 for iv in intervals]

        # Time window
        t = c0.time
        while t < c1.time - 1e-6:
            # Decide note or rest
            if rng.random() < density:
                # Strong beat? place chord tone, else scale tone
                beat = seconds_to_beats(t, bpm)
                strong = abs(beat - round(beat)) < 1e-6  # near integer beat
                # Target register around E4=64..E5=76
                base_target = 64 + int(12 * rng.random())
                if strong and chord_tones:
                    # choose nearest chord tone to target
                    target = base_target
                    # lift chord tone to closest
                    ct_abs = []
                    for ct in chord_tones:
                        # map ct to abs near target
                        base = (target // 12) * 12 + ct + 12
                        for k in (-12, 0, 12, 24):
                            ct_abs.append(base + k)
                    pitch = min(ct_abs, key=lambda x: abs(x - base_target))
                else:
                    # scale tone
                    pitch = nearest_scale_midi(base_target)
                # clamp to E3..E5 (52..76)
                pitch = clamp(pitch, 52, 76)
                dur = min( (c1.time - t), (60.0 / bpm) * step_beats )
                events.append((t, int(pitch), dur))
            t += (60.0 / bpm) * step_beats
    return events

# -----------------------------
# MIDI building
# -----------------------------

def build_midi(song: SongData, mode: str, melody: Optional[List[Tuple[float,int,float]]] = None,
               ppq: int = 480, swing: float = 0.0, chord_velocity: int = 64, melody_velocity: int = 80) -> mido.MidiFile:
    mid = mido.MidiFile(type=1)
    mid.ticks_per_beat = ppq

    # Meta track
    meta = mido.MidiTrack()
    meta.append(mido.MetaMessage('set_tempo', tempo=mido.bpm2tempo(song.tempo), time=0))
    meta.append(mido.MetaMessage('time_signature', numerator=4, denominator=4, time=0))
    mid.tracks.append(meta)

    # Chord track
    chord_tr = mido.MidiTrack()
    mid.tracks.append(chord_tr)

    # Melody track
    mel_tr = mido.MidiTrack()
    mid.tracks.append(mel_tr)

    # Build chord notes as blocks between chord change times
    chords = song.chords[:]
    chords.sort(key=lambda c: c.time)
    chords.append(ChordEvent(time=song.audio_duration, name="End"))

    def sec_to_tick(sec: float) -> int:
        beats = seconds_to_beats(sec, song.tempo)
        return beats_to_ticks(beats, mid.ticks_per_beat)

    cur_tick = 0
    for i in range(len(chords)-1):
        c0, c1 = chords[i], chords[i+1]
        if c0.name == "End":
            break
        root_name = parse_root(c0.name)
        root_semi = NOTE_TO_SEMITONE.get(root_name, NOTE_TO_SEMITONE[tonic_of_key(song.key_signature)])
        intervals = intervals_for_chord(c0.name)

        # Voice chord around C3..C4 region
        base_oct = 3
        chord_notes = sorted(set([midi_of('C', base_oct) - ((12 + NOTE_TO_SEMITONE['C']) - (12 + root_semi)) + iv for iv in intervals]))
        chord_notes = [n for n in chord_notes if 43 <= n <= 64]  # clamp to G2..E4

        start_tick = sec_to_tick(c0.time)
        end_tick   = sec_to_tick(c1.time)
        delta = start_tick - cur_tick
        if delta < 0: delta = 0

        # Note on for all chord tones
        for j, n in enumerate(chord_notes):
            chord_tr.append(mido.Message('note_on', note=int(n), velocity=chord_velocity, time=delta if j==0 else 0))
            delta = 0
        # Note off when chord changes
        dur_ticks = max(end_tick - start_tick, 0)
        for j, n in enumerate(chord_notes):
            chord_tr.append(mido.Message('note_off', note=int(n), velocity=0, time=dur_ticks if j==0 else 0))
            dur_ticks = 0
        cur_tick = end_tick

    # Melody events
    if melody is None:
        melody = []

    # Apply swing to 8ths if requested (swing>0 shifts off-beat later; 0..0.5)
    def swingify(t0: float) -> float:
        if swing <= 1e-6:
            return t0
        beat = seconds_to_beats(t0, song.tempo)
        frac = beat - math.floor(beat)
        # if in the second half of the beat (~off-beat), delay slightly
        if 0.5 - 1e-6 <= frac <= 1.0 + 1e-6:
            delay = (60.0 / song.tempo) * (swing * 0.5)  # up to 1/16 note delay
            return t0 + delay
        return t0

    mel_tick_cursor = 0
    for (t, p, d) in melody:
        t_sw = swingify(t)
        start_tick = sec_to_tick(t_sw)
        end_tick = sec_to_tick(t_sw + d)
        dt = max(start_tick - mel_tick_cursor, 0)
        mel_tr.append(mido.Message('note_on', note=int(p), velocity=melody_velocity, time=dt))
        mel_tr.append(mido.Message('note_off', note=int(p), velocity=0, time=max(end_tick - start_tick, 0)))
        mel_tick_cursor = end_tick

    return mid

# -----------------------------
# Main
# -----------------------------

def main():
    ap = argparse.ArgumentParser(description="Combine chord JSON with melody into a MIDI file.")
    ap.add_argument('--input', required=True, help='Path to chord-analysis JSON')
    ap.add_argument('--output', required=True, help='Output MIDI file path')
    ap.add_argument('--mode', choices=['dorian','aeolian','pent_min'], default='dorian', help='Scale for auto-melody')
    ap.add_argument('--melody', default='auto', help='"auto" or path to CSV time,pitch (MIDI)')
    ap.add_argument('--swing', type=float, default=0.0, help='0..0.5 swing amount (affects off-beat 8ths)')
    ap.add_argument('--no_melody', action='store_true', help='Export chords only')
    ap.add_argument('--density', type=float, default=0.75, help='Auto-melody density 0..1')
    args = ap.parse_args()

    song = load_json(args.input)

    if args.no_melody:
        mel = []
    else:
        if args.melody.lower() == 'auto':
            mel = auto_melody(song, args.mode, density=args.density)
        else:
            # CSV time,pitch → create short default durations (1/8)
            pairs = load_melody_csv(args.melody)
            step = 60.0 / song.tempo * 0.5
            mel = [(t, p, step) for (t, p) in pairs]

    mid = build_midi(song, args.mode, melody=mel, swing=args.swing)
    mid.save(args.output)

    print(f"Wrote MIDI: {args.output}")
    print(f"Tempo: {song.tempo:.2f} BPM | Duration: {song.audio_duration:.2f}s | Chords: {len(song.chords)}")
    print(f"Melody notes: {len(mel)}")

if __name__ == '__main__':
    main()
