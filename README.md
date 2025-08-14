# TheChordPlayer

Utilities for analysing audio recordings and extracting musical information.

## Command line usage

Run the combined chord and melody detector on an audio file:

```
python chord_melody_cli.py <audio-file>
```

The tool prints the detected key, chord progression, and a simple melody line
with timestamps.

## GUI player

`chord_player_v43.py` now includes a realtime audio player with a 5‑band
equaliser and waveform/spectrum visualiser. Optional dependencies:

```
pip install sounddevice pyqtgraph
```

If realtime audio is unavailable the player falls back to the previous
`pygame` based playback.

