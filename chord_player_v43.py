#!/usr/bin/env python3
"""
Chord Player Pro v4.3 — Enhanced and Fixed Version
- Fixed encoding issues and UI problems
- Re-enabled safe audio playback with error handling
- Improved performance and stability
- Better error handling and user feedback

Dependencies (Python 3.10+ recommended):
  pip install numpy librosa soundfile scipy scikit-learn mido PyQt6 pygame

Run:
  python chord_player_v43.py
"""

from __future__ import annotations
import sys, os, time, json, hashlib, shutil, threading, tempfile, subprocess
from typing import Callable, Dict, List, Sequence, Tuple, Optional

import numpy as np
import librosa
import soundfile as sf
from scipy.ndimage import median_filter, gaussian_filter1d
from scipy.signal import iirpeak
from sklearn.preprocessing import normalize

# Qt imports (optional)
try:  # pragma: no cover - optional GUI dependency
    from PyQt6.QtWidgets import (
        QApplication, QWidget, QVBoxLayout, QPushButton, QLabel, QFileDialog,
        QMessageBox, QHBoxLayout, QCheckBox, QSpinBox, QProgressBar, QTabWidget,
        QTextEdit, QGroupBox, QGridLayout, QComboBox, QDoubleSpinBox, QSplitter,
        QListWidget, QFrame, QScrollArea, QSlider, QFormLayout
    )
    from PyQt6.QtCore import QTimer, Qt, QThread, pyqtSignal, pyqtSlot, QObject, QSettings
    from PyQt6.QtGui import QFont, QPixmap, QPalette, QColor
    QT_AVAILABLE = True
except Exception:  # pragma: no cover - headless environments
    QT_AVAILABLE = False

    # Minimal stubs so the analysis code can be imported without Qt
    class QObject:  # type: ignore
        pass

    class QWidget:  # type: ignore
        pass

    class QApplication:  # type: ignore
        pass

    class QVBoxLayout:  # type: ignore
        def __init__(self, *args, **kwargs):
            pass

    class QPushButton:  # type: ignore
        def __init__(self, *args, **kwargs):
            pass

    class QLabel:  # type: ignore
        def __init__(self, *args, **kwargs):
            pass

    class QFileDialog:  # type: ignore
        pass

    class QMessageBox:  # type: ignore
        @staticmethod
        def information(*args, **kwargs):
            pass

    class QHBoxLayout(QVBoxLayout):
        pass

    class QCheckBox:  # type: ignore
        def __init__(self, *args, **kwargs):
            pass

    class QSpinBox:  # type: ignore
        def __init__(self, *args, **kwargs):
            pass

    class QProgressBar:  # type: ignore
        def __init__(self, *args, **kwargs):
            pass

    class QTabWidget:  # type: ignore
        pass

    class QTextEdit:  # type: ignore
        pass

    class QGroupBox:  # type: ignore
        def __init__(self, *args, **kwargs):
            pass

    class QGridLayout(QVBoxLayout):
        pass

    class QComboBox:  # type: ignore
        def __init__(self, *args, **kwargs):
            pass

    class QDoubleSpinBox(QSpinBox):
        pass

    class QSplitter:  # type: ignore
        def __init__(self, *args, **kwargs):
            pass

    class QListWidget:  # type: ignore
        pass

    class QFrame:  # type: ignore
        pass

    class QScrollArea:  # type: ignore
        pass

    class QSlider:  # type: ignore
        pass

    class QFormLayout(QVBoxLayout):  # type: ignore
        def addRow(self, *args, **kwargs):
            pass
        def rowCount(self):
            return 0
        def removeRow(self, *args, **kwargs):
            pass

    class QTimer:  # type: ignore
        def __init__(self, *args, **kwargs):
            pass

    class QThread:  # type: ignore
        pass

    class pyqtSignal:  # type: ignore
        def __init__(self, *args, **kwargs):
            pass

        def emit(self, *args, **kwargs):
            pass

    class pyqtSlot:  # type: ignore
        def __init__(self, *args, **kwargs):
            pass

        def __call__(self, func):
            return func

    class QSettings:  # type: ignore
        def __init__(self, *args, **kwargs):
            pass

    class Qt:  # type: ignore
        class Orientation:
            Horizontal = 0

    class QFont:  # type: ignore
        pass

    class QPixmap:  # type: ignore
        pass

    class QPalette:  # type: ignore
        pass

    class QColor:  # type: ignore
        pass

if QT_AVAILABLE:
    from PyQt6 import QtCore
    import pyqtgraph as pg

# Audio playback
try:
    import pygame
    PYGAME_AVAILABLE = True
except ImportError:
    PYGAME_AVAILABLE = False
    print("Warning: pygame not available. Audio playback disabled.")

# ---------------- Config ----------------
CHORDS_CACHE_DIR = "chord_cache_v43"
CHORDS_CACHE_EXTENSION = ".json"
SR = 22050
FFT_SIZE = 16384
HOP_LENGTH = 512
NOTE_NAMES = ['C','C#','D','D#','E','F','F#','G','G#','A','A#','B']

CHORD_DEFS: Dict[str, List[int]] = {
    '': [0,4,7], 'm': [0,3,7], 'dim': [0,3,6], 'aug': [0,4,8],
    'sus2': [0,2,7], 'sus4': [0,5,7],
    '6': [0,4,7,9], 'm6': [0,3,7,9], '6/9': [0,4,7,9,14%12],
    '7': [0,4,7,10], 'M7': [0,4,7,11], 'maj7': [0,4,7,11], 'm7': [0,3,7,10],
    'm7b5': [0,3,6,10], 'dim7': [0,3,6,9], 'mM7': [0,3,7,11],
    '9': [0,4,7,10,14%12], 'M9': [0,4,7,11,14%12], 'm9': [0,3,7,10,14%12],
    'add9': [0,4,7,14%12], '11': [0,4,7,10,14%12,17%12], '13': [0,4,7,10,14%12,21%12],
    '7#5': [0,4,8,10], '7b5': [0,4,6,10], '7#9': [0,4,7,10,15%12], '7b9': [0,4,7,10,13%12],
    '7#11': [0,4,7,10,18%12], 'add11': [0,4,7,17%12], 'madd9': [0,3,7,14%12],
    '6sus4': [0,5,7,9], 'm6/9': [0,3,7,9,14%12], 'maj7#11': [0,4,7,11,18%12],
    'm11': [0,3,7,10,14%12,17%12], 'm13': [0,3,7,10,14%12,21%12],
}

CHORD_PRIORITY = {
    '': 1.0, 'm': 1.0, '7': 0.9, 'm7': 0.9, 'M7': 0.8, 'maj7': 0.8, 
    'dim': 0.7, 'sus4': 0.6, 'sus2': 0.6, '6': 0.5, 'add9': 0.4
}

CHORD_WEIGHTS = {
    'root': 1.0, 'third': 0.9, 'fifth': 0.7, 'seventh': 0.6, 
    'ninth': 0.4, 'eleventh': 0.3, 'thirteenth': 0.2
}

# --------------- Utilities ---------------
class PerformanceMonitor:
    def __init__(self):
        self.timings: Dict[str, float] = {}
        
    def start(self, name: str):
        self.timings[name] = time.time()
        
    def end(self, name: str) -> float:
        if name in self.timings:
            d = time.time() - self.timings[name]
            print(f"[PERF] {name}: {d:.3f}s")
            return d
        return 0.0

perf = PerformanceMonitor()

def get_file_hash(path: str) -> str:
    h = hashlib.sha256()
    try:
        with open(path, 'rb') as f:
            for chunk in iter(lambda: f.read(8192), b""):
                h.update(chunk)
        return h.hexdigest()
    except Exception as e:
        print(f"[WARN] Could not hash file {path}: {e}")
        return str(hash(path))

def format_time(seconds: float) -> str:
    seconds = max(0.0, float(seconds))
    m = int(seconds // 60)
    s = int(round(seconds - m*60))
    return f"{m:02d}:{s:02d}"

# ----------------- STEMS ----------------------------------------------------
class StemSeparator:
    """Two engines: HPSS (fast) and Demucs (quality)."""

    def __init__(self, fs=44100):
        self.fs = fs

    def hpss(self, y: np.ndarray, sr: int):
        """Returns dict with 'instrumental' and 'drums'."""
        y_mono = librosa.to_mono(y) if y.ndim > 1 else y
        H, P = librosa.effects.hpss(y_mono)
        stems = {
            "instrumental": H.astype(np.float32),
            "drums": P.astype(np.float32),
        }
        return stems, sr

    def demucs(self, y: np.ndarray, sr: int, mode: str = "two_stems"):
        """Run Demucs separation."""
        tmpdir = tempfile.mkdtemp(prefix="stems_")
        in_wav = os.path.join(tmpdir, "input.wav")
        if y.ndim == 1:
            y = np.stack([y, y], axis=0)
        sf.write(in_wav, y.T.astype(np.float32), sr)
        outdir = os.path.join(tmpdir, "out")
        os.makedirs(outdir, exist_ok=True)
        cmd = [
            "python", "-m", "demucs.separate", "-d", "cpu", "-n", "htdemucs", "--out", outdir, in_wav
        ]
        if mode == "two_stems":
            cmd.insert(5, "--two-stems=vocals")
        try:
            subprocess.run(cmd, check=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
        except Exception as e:
            shutil.rmtree(tmpdir, ignore_errors=True)
            raise RuntimeError(f"Demucs failed: {e}")
        model_dir = next(p for p in os.listdir(outdir))
        song_dir = os.path.join(outdir, model_dir, "input")
        stems = {}
        for fname in os.listdir(song_dir):
            if not fname.endswith(".wav"):
                continue
            name = os.path.splitext(fname)[0]
            data, srr = sf.read(os.path.join(song_dir, fname), dtype="float32", always_2d=True)
            stems[name] = data.T
        key_map = {"vocals": "vocals", "no_vocals": "instrumental", "drums": "drums", "bass": "bass", "other": "other"}
        normalized = {key_map.get(k, k): v for k, v in stems.items()}
        return normalized, srr, tmpdir

# ---- EQ ENGINE -------------------------------------------------------------
class BiquadPeaking:
    def __init__(self, fs, f0, gain_db=0.0, Q=1.0):
        self.fs = fs
        self.f0 = f0
        self.Q = Q
        self.set_gain(gain_db)
        self.z1 = 0.0
        self.z2 = 0.0

    def set_gain(self, gain_db):
        A = 10 ** (gain_db / 40.0)
        w0 = 2 * np.pi * self.f0 / self.fs
        alpha = np.sin(w0) / (2 * self.Q)

        b0 = 1 + alpha * A
        b1 = -2 * np.cos(w0)
        b2 = 1 - alpha * A
        a0 = 1 + alpha / A
        a1 = -2 * np.cos(w0)
        a2 = 1 - alpha / A

        self.b0 = b0 / a0
        self.b1 = b1 / a0
        self.b2 = b2 / a0
        self.a1 = a1 / a0
        self.a2 = a2 / a0

    def process(self, x):
        y = np.empty_like(x)
        z1, z2 = self.z1, self.z2
        b0, b1, b2, a1, a2 = self.b0, self.b1, self.b2, self.a1, self.a2
        for i, xn in enumerate(x):
            yn = b0 * xn + z1
            z1 = b1 * xn - a1 * yn + z2
            z2 = b2 * xn - a2 * yn
            y[i] = yn
        self.z1, self.z2 = z1, z2
        return y

class FiveBandEQ:
    def __init__(self, fs):
        self.fs = fs
        self.bands = [
            ("60", BiquadPeaking(fs, 60.0, 0.0, Q=1.0)),
            ("230", BiquadPeaking(fs, 230.0, 0.0, Q=1.0)),
            ("910", BiquadPeaking(fs, 910.0, 0.0, Q=1.0)),
            ("4.1k", BiquadPeaking(fs, 4100.0, 0.0, Q=1.0)),
            ("14k", BiquadPeaking(fs, 14000.0, 0.0, Q=0.9)),
        ]

    def set_gain(self, idx, gain_db):
        self.bands[idx][1].set_gain(gain_db)

    def process(self, x):
        y = x
        for _, biq in self.bands:
            y = biq.process(y)
        return np.clip(y, -1.0, 1.0)

# ---- REALTIME AUDIO STREAM -------------------------------------------------
try:
    import sounddevice as sd
    HAVE_SD = True
except Exception:  # pragma: no cover - optional dependency
    HAVE_SD = False

class RTPlayer:
    def __init__(self, fs=44100, blocksize=1024, channels=1, eq=None, on_visual=None):
        self.fs = fs
        self.blocksize = blocksize
        self.channels = channels
        self.eq = eq
        self.on_visual = on_visual
        self.buffer = np.zeros(0, dtype=np.float32)
        self.lock = threading.Lock()
        self.stream = None
        self.gain = 1.0

    def load(self, mono_float32_pcm):
        with self.lock:
            self.buffer = mono_float32_pcm.astype(np.float32)

    def start(self):
        if not HAVE_SD:
            raise RuntimeError("sounddevice not available")
        if self.stream:
            self.stop()
        self.pos = 0

        def cb(outdata, frames, time, status):
            if status:
                pass
            with self.lock:
                end = min(self.pos + frames, len(self.buffer))
                chunk = self.buffer[self.pos:end]
                self.pos = end
            if len(chunk) < frames:
                pad = np.zeros(frames - len(chunk), dtype=np.float32)
                chunk = np.concatenate([chunk, pad])

            if self.eq:
                chunk = self.eq.process(chunk)

            if self.on_visual:
                self.on_visual(chunk.copy())

            chunk *= self.gain
            out = np.tile(chunk.reshape(-1, 1), (1, self.channels))
            outdata[:] = out

        self.stream = sd.OutputStream(
            samplerate=self.fs,
            channels=self.channels,
            blocksize=self.blocksize,
            dtype="float32",
            callback=cb,
        )
        self.stream.start()

    def stop(self):
        if self.stream:
            self.stream.stop()
            self.stream.close()
            self.stream = None

if QT_AVAILABLE:
    # ---- VISUALIZER --------------------------------------------------------
    class VisualizerWidget(QWidget):
        def __init__(self, fs, parent=None):
            super().__init__(parent)
            self.fs = fs
            layout = QVBoxLayout(self)
            self.wave = pg.PlotWidget()
            self.wave.setYRange(-1.05, 1.05)
            self.wave_curve = self.wave.plot(pen=pg.mkPen(width=1))
            self.spec = pg.PlotWidget()
            self.spec.setLogMode(x=False, y=True)
            self.spec_curve = self.spec.plot(stepMode=True, fillLevel=-120)
            layout.addWidget(self.wave)
            layout.addWidget(self.spec)
            self._buf = np.zeros(2048, dtype=np.float32)

        @pyqtSlot(object)
        def push_audio(self, chunk):
            L = len(chunk)
            if L >= len(self._buf):
                self._buf[:] = chunk[-len(self._buf) :]
            else:
                self._buf = np.roll(self._buf, -L)
                self._buf[-L:] = chunk
            self.wave_curve.setData(self._buf)
            w = np.hanning(len(chunk))
            Y = np.fft.rfft(chunk * w)
            mag_db = 20 * np.log10(np.maximum(1e-7, np.abs(Y)))
            freqs = np.fft.rfftfreq(len(chunk), d=1.0 / self.fs)
            bins = 60
            idx = np.linspace(1, len(freqs) - 1, bins).astype(int)
            self.spec_curve.setData(freqs[idx], mag_db[idx])

    # ---- EQ CONTROL PANEL --------------------------------------------------
    class EQPanel(QWidget):
        gainChanged = pyqtSignal(int, float)

        def __init__(self, labels=("60", "230", "910", "4.1k", "14k"), parent=None):
            super().__init__(parent)
            layout = QHBoxLayout(self)
            self.sliders = []
            for i, lab in enumerate(labels):
                vbox = QVBoxLayout()
                lbl = QLabel(lab)
                s = QSlider(Qt.Orientation.Vertical)
                s.setRange(-12 * 2, 12 * 2)
                s.setValue(0)
                s.valueChanged.connect(lambda val, idx=i: self._emit(idx, val / 2.0))
                vbox.addWidget(lbl, alignment=Qt.AlignmentFlag.AlignHCenter)
                vbox.addWidget(s)
                layout.addLayout(vbox)
                self.sliders.append(s)

        def _emit(self, idx, gain_db):
            self.gainChanged.emit(idx, gain_db)

    class StemsPanel(QWidget):
        requestSeparate = pyqtSignal(str)
        exportStem = pyqtSignal(str)

        def __init__(self, parent=None):
            super().__init__(parent)
            layout = QVBoxLayout(self)
            row = QHBoxLayout()
            self.btn_hpss = QPushButton("HPSS (fast)")
            self.btn_two = QPushButton("Demucs: Vocals/Inst")
            self.btn_four = QPushButton("Demucs: 4-Stem")
            for b in (self.btn_hpss, self.btn_two, self.btn_four):
                row.addWidget(b)
            layout.addLayout(row)
            self.btn_hpss.clicked.connect(lambda: self.requestSeparate.emit("hpss"))
            self.btn_two.clicked.connect(lambda: self.requestSeparate.emit("two"))
            self.btn_four.clicked.connect(lambda: self.requestSeparate.emit("four"))
            self.mixer = QFormLayout()
            self._sliders = {}
            box = QGroupBox("Stem Mixer (dB)")
            box.setLayout(self.mixer)
            layout.addWidget(box)

        def set_stems(self, stems_keys):
            while self.mixer.rowCount():
                self.mixer.removeRow(0)
            self._sliders.clear()
            for k in stems_keys:
                s = QSlider(Qt.Orientation.Horizontal)
                s.setRange(-24 * 2, 12 * 2)
                s.setValue(0)
                self.mixer.addRow(QLabel(k), s)
                self._sliders[k] = s

        def gains_db(self):
            return {k: s.value() / 2.0 for k, s in self._sliders.items()}

# --------------- Audio Playback ---------------
class AudioPlayer:
    def __init__(self):
        self.is_playing = False
        self.is_paused = False
        self.current_position = 0.0
        self.audio_data = None
        self.sample_rate = SR
        self.pygame_initialized = False
        
    def initialize_pygame(self):
        if not PYGAME_AVAILABLE:
            return False
        try:
            pygame.mixer.pre_init(frequency=self.sample_rate, size=-16, channels=1, buffer=1024)
            pygame.mixer.init()
            self.pygame_initialized = True
            return True
        except Exception as e:
            print(f"[WARN] Pygame initialization failed: {e}")
            return False
    
    def load_audio(self, file_path: str) -> bool:
        try:
            y, sr = librosa.load(file_path, sr=self.sample_rate, mono=True)
            self.audio_data = y
            self.sample_rate = sr
            return True
        except Exception as e:
            print(f"[ERROR] Failed to load audio: {e}")
            return False
    
    def play(self):
        if not self.pygame_initialized and not self.initialize_pygame():
            return False
        
        if self.audio_data is None:
            return False
        
        try:
            # Convert to pygame format
            audio_int16 = (self.audio_data * 32767).astype(np.int16)
            sound = pygame.sndarray.make_sound(audio_int16)
            pygame.mixer.Sound.play(sound)
            self.is_playing = True
            self.is_paused = False
            return True
        except Exception as e:
            print(f"[ERROR] Playback failed: {e}")
            return False
    
    def stop(self):
        if self.pygame_initialized:
            try:
                pygame.mixer.stop()
            except Exception:
                pass
        self.is_playing = False
        self.is_paused = False
        self.current_position = 0.0
    
    def pause(self):
        if self.pygame_initialized and self.is_playing:
            try:
                pygame.mixer.pause()
                self.is_paused = True
            except Exception:
                pass
    
    def unpause(self):
        if self.pygame_initialized and self.is_paused:
            try:
                pygame.mixer.unpause()
                self.is_paused = False
            except Exception:
                pass

# --------------- Analysis Core ---------------
def create_chord_templates() -> Dict[str, np.ndarray]:
    os.makedirs(CHORDS_CACHE_DIR, exist_ok=True)
    cache_file = os.path.join(CHORDS_CACHE_DIR, "templates_v43.json")
    
    if os.path.exists(cache_file):
        try:
            with open(cache_file, 'r', encoding='utf-8') as f:
                data = json.load(f)
            return {k: np.array(v, dtype=np.float32) for k, v in data.items()}
        except Exception as e:
            print(f"[CACHE] Template load failed: {e}")
    
    print("[COMPUTE] Building chord templates...")
    perf.start("template_creation")
    
    items = sorted(CHORD_DEFS.items(), key=lambda kv: CHORD_PRIORITY.get(kv[0], 0.3), reverse=True)
    templates: Dict[str, np.ndarray] = {}
    
    for root_idx, note in enumerate(NOTE_NAMES):
        for suffix, intervals in items:
            vec = np.zeros(12, dtype=np.float32)
            for i, interval in enumerate(intervals):
                pitch_class = (root_idx + interval) % 12
                interval_mod = interval % 12
                
                # Assign weights based on interval type
                if interval_mod == 0:
                    w = CHORD_WEIGHTS['root']
                elif interval_mod in (3,4):
                    w = CHORD_WEIGHTS['third']
                elif interval_mod in (6,7,8):
                    w = CHORD_WEIGHTS['fifth']
                elif interval_mod in (10,11):
                    w = CHORD_WEIGHTS['seventh']
                elif interval_mod in (2, 14%12):
                    w = CHORD_WEIGHTS['ninth']
                elif interval_mod in (5, 17%12):
                    w = CHORD_WEIGHTS['eleventh']
                else:
                    w = CHORD_WEIGHTS['thirteenth']
                
                vec[pitch_class] = max(vec[pitch_class], w * (1.0 - 0.08*i))
            
            if np.sum(vec) > 0:
                vec = vec / np.linalg.norm(vec)
            templates[note+suffix] = vec
    
    try:
        with open(cache_file, 'w', encoding='utf-8') as f:
            json.dump({k: v.tolist() for k,v in templates.items()}, f)
    except Exception as e:
        print(f"[CACHE] Save failed: {e}")
    
    perf.end("template_creation")
    return templates

def extract_features(y: np.ndarray, sr: int = SR) -> Dict[str, np.ndarray | float]:
    perf.start("feature_extraction")
    feats: Dict[str, np.ndarray | float] = {}
    
    try:
        tempo, _ = librosa.beat.beat_track(y=y, sr=sr, hop_length=HOP_LENGTH)
        feats['tempo'] = float(tempo)
    except Exception as e:
        print(f"[WARN] Tempo extraction failed: {e}")
        feats['tempo'] = 120.0
    
    try:
        feats['chroma_cqt'] = librosa.feature.chroma_cqt(
            y=y, sr=sr, hop_length=HOP_LENGTH, 
            fmin=librosa.note_to_hz('C1'), n_chroma=12, n_octaves=6
        )
        feats['chroma_stft'] = librosa.feature.chroma_stft(
            y=y, sr=sr, hop_length=HOP_LENGTH, n_fft=FFT_SIZE
        )
        
        y_harm, _ = librosa.effects.hpss(y, margin=(1.0, 5.0))
        feats['chroma_harm'] = librosa.feature.chroma_cqt(y=y_harm, sr=sr, hop_length=HOP_LENGTH)
        
        feats['spectral_centroid'] = librosa.feature.spectral_centroid(y=y, sr=sr, hop_length=HOP_LENGTH)
        feats['rms'] = librosa.feature.rms(y=y, hop_length=HOP_LENGTH)
        
    except Exception as e:
        print(f"[ERROR] Feature extraction failed: {e}")
        raise
    
    perf.end("feature_extraction")
    return feats

def smooth_chroma(chroma: np.ndarray, features: Dict[str, np.ndarray|float], method: str='adaptive') -> np.ndarray:
    perf.start("smoothing")
    
    if method == 'adaptive' and 'spectral_centroid' in features:
        centroid = np.asarray(features['spectral_centroid'])[0]
        cdiff = np.abs(np.diff(centroid))
        tempo = float(features.get('tempo', 120.0))
        sigma = 2.0 * (120.0 / max(tempo, 60.0)) * (0.5 + 1.0/(1.0 + 1000.0*np.mean(cdiff)))
        
        out = np.zeros_like(chroma)
        for i in range(chroma.shape[0]):
            out[i] = gaussian_filter1d(chroma[i], sigma=sigma)
    elif method == 'median':
        out = np.vstack([median_filter(chroma[i], size=5) for i in range(chroma.shape[0])])
    else:
        out = gaussian_filter1d(chroma, sigma=1.5, axis=1)
    
    perf.end("smoothing")
    return out

def detect_key(chroma: np.ndarray) -> Tuple[str, float]:
    perf.start("key_detection")
    
    avg = np.mean(chroma, axis=1)
    major = np.array([6.35, 2.23, 3.48, 2.33, 4.38, 4.09, 2.52, 5.19, 2.39, 3.66, 2.29, 2.88])
    minor = np.array([6.33, 2.68, 3.52, 5.38, 2.60, 3.53, 2.54, 4.75, 3.98, 2.69, 3.34, 3.17])
    
    scores: List[Tuple[str, float]] = []
    
    for shift in range(12):
        try:
            mc = np.corrcoef(avg, np.roll(major, shift))[0,1]
            nc = np.corrcoef(avg, np.roll(minor, shift))[0,1]
            if not np.isnan(mc):
                scores.append((f"{NOTE_NAMES[shift]} Major", mc))
            if not np.isnan(nc):
                scores.append((f"{NOTE_NAMES[shift]} Minor", nc))
        except Exception:
            continue
    
    scores.sort(key=lambda x: x[1], reverse=True)
    perf.end("key_detection")
    
    if not scores:
        return ("Unknown", 0.0)
    
    best = scores[0]
    second = scores[1] if len(scores) > 1 else (scores[0][0], scores[0][1]-1)
    conf = min(1.0, max(0.0, float(best[1] - second[1])))
    
    return best[0], conf

def post_process(chords: List[Tuple[float,str]], min_dur: float=0.3) -> Tuple[List[Tuple[float,str]], float]:
    if not chords:
        return ([], 0.0)
    
    perf.start("post_processing")
    
    merged: List[Tuple[float,str]] = []
    cur = chords[0][1]
    start = chords[0][0]
    last_time = start
    
    for t, name in chords[1:]:
        if name == cur and t - last_time < 0.2:
            last_time = t
        else:
            if last_time - start >= min_dur:
                merged.append((start, cur))
            cur = name
            start = t
            last_time = t
    
    if last_time - start >= min_dur:
        merged.append((start, cur))
    
    unique_ratio = len(set(n for _,n in merged)) / max(1, len(merged))

    perf.end("post_processing")
    return merged, unique_ratio

def detect_melody(y: np.ndarray, sr: int = SR, *, threshold: float = 0.8) -> List[Tuple[float, str]]:
    """Detect a simple melody line using fundamental frequency estimation.

    Returns a list of (time, note_name) tuples whenever the detected note
    changes. Frames with low confidence are ignored. This is intentionally
    lightweight and meant for quick preview rather than transcription-level
    accuracy.
    """
    perf.start("melody_detection")
    melody: List[Tuple[float, str]] = []

    try:
        f0, voiced_flag, voiced_prob = librosa.pyin(
            y,
            fmin=librosa.note_to_hz('C2'),
            fmax=librosa.note_to_hz('C7'),
            sr=sr,
            hop_length=HOP_LENGTH,
        )
        times = librosa.times_like(f0, sr=sr, hop_length=HOP_LENGTH)

        last_note: Optional[str] = None
        last_time: float = 0.0

        for t, f, vf, vp in zip(times, f0, voiced_flag, voiced_prob):
            if not vf or vp < threshold or f is None or np.isnan(f):
                note = None
            else:
                try:
                    note = librosa.hz_to_note(float(f))
                except Exception:
                    note = None

            if note != last_note:
                if last_note is not None:
                    melody.append((last_time, last_note))
                last_note = note
                last_time = float(t)

        if last_note is not None:
            melody.append((last_time, last_note))

    except Exception as e:
        print(f"[ERROR] Melody detection failed: {e}")

    perf.end("melody_detection")
    return [m for m in melody if m[1] is not None]

def detect_chords(path: str, *, progress: Callable[[int],None]|None=None,
                  status: Callable[[str],None]|None=None,
                  use_multi: bool=True, smoothing: str='adaptive',
                  min_chord_duration: float=0.3, template_threshold: float=0.25,
                  key_aware: bool=True, performance_mode: bool=False,
                  max_chord_types: int=80) -> Tuple[List[Tuple[float,str]], str, Dict]:
    
    os.makedirs(CHORDS_CACHE_DIR, exist_ok=True)
    
    try:
        fh = get_file_hash(path)
        cache_key = hash(str((use_multi,smoothing,min_chord_duration,template_threshold,key_aware,performance_mode,max_chord_types)))
        cache_file = os.path.join(CHORDS_CACHE_DIR, f"{fh}_v43_{cache_key}{CHORDS_CACHE_EXTENSION}")
        
        if os.path.exists(cache_file):
            try:
                with open(cache_file,'r', encoding='utf-8') as f:
                    data = json.load(f)
                stats = data.get('stats', {})
                if 'melody' not in stats:
                    # Melody was not cached; compute quickly now
                    y_tmp, sr_tmp = librosa.load(path, sr=SR, mono=True)
                    stats['melody'] = detect_melody(y_tmp, sr_tmp)
                    data['stats'] = stats
                    try:
                        with open(cache_file,'w', encoding='utf-8') as wf:
                            json.dump(data, wf)
                    except Exception:
                        pass
                if status:
                    status("Loaded from cache.")
                return data['chords'], data.get('key_signature','Unknown'), stats
            except Exception:
                pass
    except Exception as e:
        print(f"[WARN] Cache check failed: {e}")

    st = { 
        'cache_hit': False, 'audio_duration': 0.0, 'total_frames': 0,
        'chord_changes': 0, 'key_confidence': 0.0, 'performance_mode': performance_mode 
    }

    if status: status("Loading audio...")
    if progress: progress(5)
    
    perf.start("audio_loading")
    y, sr = librosa.load(path, sr=SR, mono=True)
    st['audio_duration'] = len(y)/sr
    perf.end("audio_loading")

    if status: status("Extracting features...")
    if progress: progress(20)
    feats = extract_features(y, sr)

    if status: status("Preparing chroma...")
    if progress: progress(35)
    
    if use_multi and not performance_mode:
        chroma = (0.4*np.asarray(feats['chroma_cqt']) + 
                 0.3*np.asarray(feats['chroma_stft']) + 
                 0.3*np.asarray(feats['chroma_harm']))
    else:
        chroma = np.asarray(feats['chroma_cqt'])

    chroma = smooth_chroma(chroma, feats, method=smoothing)
    chroma = normalize(chroma, axis=0, norm='l2')
    st['total_frames'] = chroma.shape[1]

    if status: status("Detecting key...")
    if progress: progress(50)
    key_sig, key_conf = detect_key(chroma) if key_aware else ("Unknown", 0.0)
    st['key_confidence'] = float(key_conf)

    if status: status("Detecting melody...")
    if progress: progress(55)
    st['melody'] = detect_melody(y, sr)

    if status: status("Building templates...")
    if progress: progress(60)
    templates = create_chord_templates()

    if performance_mode:
        keep = ['', 'm', '7', 'm7', 'M7', 'maj7', 'dim', 'sus4', 'sus2']
        templates = {k:v for k,v in templates.items() if any(k.endswith(s) for s in keep)}

    names = list(templates.keys())
    T = np.stack([templates[n] for n in names], axis=1)  # [12, N]

    if status: status("Matching chords...")
    if progress: progress(70)
    perf.start("chord_matching")
    
    times = librosa.frames_to_time(np.arange(chroma.shape[1]), sr=SR, hop_length=HOP_LENGTH)
    sims = T.T @ chroma  # [N, F]
    best_idx = np.argmax(sims, axis=0)
    best_val = np.max(sims, axis=0)
    
    chords: List[Tuple[float,str]] = []
    for idx, val, t in zip(best_idx, best_val, times):
        chord_name = names[int(idx)] if float(val) >= template_threshold else "N.C."
        chords.append((float(t), chord_name))
    
    perf.end("chord_matching")

    if status: status("Post-processing...")
    if progress: progress(85)
    chords, complexity = post_process(chords, min_dur=min_chord_duration)
    st['chord_changes'] = len(chords)
    st['harmonic_complexity'] = float(complexity)
    st['estimated_tempo'] = float(feats.get('tempo', 120.0))

    if progress: progress(95)
    
    # Save to cache
    try:
        with open(cache_file,'w', encoding='utf-8') as f:
            json.dump({'chords': chords, 'key_signature': key_sig, 'stats': st}, f)
    except Exception as e:
        print(f"[CACHE] Write failed: {e}")
    
    if progress: progress(100)
    if status: status("Analysis complete.")
    
    return chords, key_sig, st

# --------------- Worker ---------------
class ChordAnalysisWorker(QObject):
    progress = pyqtSignal(int)
    finished = pyqtSignal(list, str, dict)  # chords, key, stats
    error = pyqtSignal(str)
    status = pyqtSignal(str)

    def __init__(self, file_path: str, settings: Dict):
        super().__init__()
        self.file_path = file_path
        self.settings = settings

    def run(self):
        try:
            self.status.emit("Starting analysis...")
            chords, key_sig, stats = detect_chords(
                self.file_path,
                progress=self.progress.emit,
                status=self.status.emit,
                use_multi=self.settings['use_multi_features'],
                smoothing=self.settings['smoothing_method'],
                min_chord_duration=self.settings['min_chord_duration'],
                template_threshold=self.settings['template_threshold'],
                key_aware=self.settings['key_aware'],
                performance_mode=self.settings['performance_mode'],
                max_chord_types=self.settings['max_chord_types'],
            )
            self.finished.emit(chords, key_sig, stats)
        except Exception as e:
            import traceback
            error_msg = f"Analysis failed: {str(e)}\n\n{traceback.format_exc()}"
            self.error.emit(error_msg)

# --------------- UI ---------------
class App(QWidget):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("Enhanced Chord Player Pro v4.3")
        self.resize(1300, 850)
        self.settings = QSettings("MusicTech", "ChordPlayerProV43")

        self.chords: List[Tuple[float,str]] = []
        self.key_signature: str = "Unknown"
        self.stats: Dict = {}
        self.audio_duration: float = 0.0
        self.current_file_path: Optional[str] = None
        self.analysis_thread: Optional[QThread] = None
        self.worker: Optional[ChordAnalysisWorker] = None

        # Realtime playback
        self.audio_mono_float32: Optional[np.ndarray] = None
        self.fs: int = 44100
        self.player: Optional[RTPlayer] = None
        self.eq: Optional[FiveBandEQ] = None
        self.vis: Optional[VisualizerWidget] = None
        self.eq_panel: Optional[EQPanel] = None
        self.stems_panel: Optional[StemsPanel] = None
        self.visualizer_container: Optional[QWidget] = None
        self.rt_playing: bool = False
        self.use_rt_player: bool = HAVE_SD and QT_AVAILABLE

        self.separator = StemSeparator()
        self.current_stems: Optional[Dict[str, np.ndarray]] = None
        self.current_stems_sr: Optional[int] = None
        self.demucs_tmpdir: Optional[str] = None

        # Audio player fallback
        self.audio_player = AudioPlayer()
        self.playback_timer = QTimer(self)
        self.playback_timer.timeout.connect(self.update_playback_position)
        self.current_playback_time = 0.0

        self.init_ui()
        self.apply_theme()
        self.load_settings()

    def init_ui(self):
        main = QVBoxLayout(self)
        self.splitter = QSplitter(Qt.Orientation.Horizontal)
        main.addWidget(self.splitter)

        left = self.build_left()
        right = self.build_right()
        self.splitter.addWidget(left)
        self.splitter.addWidget(right)
        self.splitter.setSizes([750, 550])
        
        self.create_status_bar(main)
        
        self.timer = QTimer(self)
        self.timer.setInterval(100)
        self.timer.timeout.connect(self.update_display)

    def build_left(self) -> QWidget:
        w = QWidget()
        v = QVBoxLayout(w)
        
        # File controls
        g = QGroupBox("🎵 Audio File & Playback Controls")
        h = QHBoxLayout(g)
        
        self.btn_load = QPushButton("📁 Load Audio File")
        self.btn_load.clicked.connect(self.load_audio)
        h.addWidget(self.btn_load)
        
        self.btn_play = QPushButton("▶️ Play")
        self.btn_play.setEnabled(False)
        self.btn_play.clicked.connect(self.play_audio)
        h.addWidget(self.btn_play)
        
        self.btn_pause = QPushButton("⏸️ Pause")
        self.btn_pause.setEnabled(False)
        self.btn_pause.clicked.connect(self.toggle_pause)
        h.addWidget(self.btn_pause)
        
        self.btn_stop = QPushButton("⏹️ Stop")
        self.btn_stop.setEnabled(False)
        self.btn_stop.clicked.connect(self.stop_playback)
        h.addWidget(self.btn_stop)
        
        v.addWidget(g)
        
        # Volume control
        vol_group = QGroupBox("🔊 Volume")
        vol_layout = QHBoxLayout(vol_group)
        self.volume_slider = QSlider(Qt.Orientation.Horizontal)
        self.volume_slider.setRange(0, 100)
        self.volume_slider.setValue(70)
        self.volume_slider.valueChanged.connect(self.update_volume)
        self.volume_label = QLabel("70%")
        vol_layout.addWidget(self.volume_slider)
        vol_layout.addWidget(self.volume_label)
        v.addWidget(vol_group)
        
        # Progress
        g2 = QGroupBox("📊 Analysis Progress")
        v2 = QVBoxLayout(g2)
        self.progress_bar = QProgressBar()
        self.progress_bar.setTextVisible(True)
        v2.addWidget(self.progress_bar)
        
        hh = QHBoxLayout()
        self.status_label = QLabel("Ready")
        self.perf_label = QLabel("")
        self.perf_label.setStyleSheet("color:#666;font-size:10px;")
        hh.addWidget(self.status_label)
        hh.addStretch()
        hh.addWidget(self.perf_label)
        v2.addLayout(hh)
        v.addWidget(g2)
        
        # Current analysis
        g3 = QGroupBox("🎼 Current Analysis")
        v3 = QVBoxLayout(g3)
        self.chord_label = QLabel("Load an audio file to begin")
        self.chord_label.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self.chord_label.setMinimumHeight(120)
        v3.addWidget(self.chord_label)
        
        row = QHBoxLayout()
        self.key_label = QLabel("Key: Unknown")
        self.key_label.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self.time_label = QLabel("00:00 / 00:00")
        self.time_label.setAlignment(Qt.AlignmentFlag.AlignCenter)
        row.addWidget(self.key_label)
        row.addWidget(self.time_label)
        v3.addLayout(row)
        
        row2 = QHBoxLayout()
        self.conf_label = QLabel("Confidence: -")
        self.conf_label.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self.tempo_label = QLabel("♩ = - BPM")
        self.tempo_label.setAlignment(Qt.AlignmentFlag.AlignCenter)
        row2.addWidget(self.conf_label)
        row2.addWidget(self.tempo_label)
        v3.addLayout(row2)
        v.addWidget(g3)
        
        # Progression list
        g4 = QGroupBox("🎶 Chord Progression")
        v4 = QVBoxLayout(g4)
        
        controls = QHBoxLayout()
        self.show_times_cb = QCheckBox("Show Times")
        self.show_times_cb.setChecked(True)
        self.show_times_cb.stateChanged.connect(self.update_chord_list_display)
        controls.addWidget(self.show_times_cb)
        
        self.max_chords_spin = QSpinBox()
        self.max_chords_spin.setRange(5, 50)
        self.max_chords_spin.setValue(12)
        self.max_chords_spin.setPrefix("Show last ")
        self.max_chords_spin.setSuffix(" chords")
        self.max_chords_spin.valueChanged.connect(self.update_chord_list_display)
        controls.addWidget(self.max_chords_spin)
        controls.addStretch()
        v4.addLayout(controls)
        
        self.chord_list = QListWidget()
        self.chord_list.setMaximumHeight(200)
        self.chord_list.setAlternatingRowColors(True)
        v4.addWidget(self.chord_list)
        v.addWidget(g4)
        
        return w

    def build_right(self) -> QWidget:
        w = QWidget()
        v = QVBoxLayout(w)
        tabs = QTabWidget()
        v.addWidget(tabs)
        
        # Settings tab
        st = QWidget()
        tabs.addTab(st, "⚙️ Settings")
        self.setup_settings_tab(st)
        
        # Results tab
        rt = QWidget()
        tabs.addTab(rt, "📈 Analysis")
        self.setup_results_tab(rt)
        
        # Performance tab
        pt = QWidget()
        tabs.addTab(pt, "🚀 Performance")
        self.setup_perf_tab(pt)
        
        return w

    def create_status_bar(self, layout: QVBoxLayout):
        frame = QFrame()
        h = QHBoxLayout(frame)
        h.setContentsMargins(5,2,5,2)
        
        self.file_info = QLabel("No file loaded")
        self.file_info.setStyleSheet("color:#666;")
        self.cache_status = QLabel("Cache: Ready")
        self.cache_status.setStyleSheet("color:#666;")
        
        h.addWidget(self.file_info)
        h.addStretch()
        h.addWidget(self.cache_status)
        layout.addWidget(frame)

    def setup_settings_tab(self, parent: QWidget):
        scroll = QScrollArea()
        scroll.setWidgetResizable(True)
        container = QWidget()
        layout = QVBoxLayout(container)
        
        # Analysis Quality
        q = QGroupBox("🎯 Analysis Quality")
        grid = QGridLayout(q)
        
        grid.addWidget(QLabel("Performance Mode:"), 0, 0)
        self.performance_cb = QCheckBox("Enable fast analysis")
        grid.addWidget(self.performance_cb, 0, 1)
        
        grid.addWidget(QLabel("Multi-Feature Analysis:"), 1, 0)
        self.multi_feature_cb = QCheckBox("Use multiple algorithms")
        self.multi_feature_cb.setChecked(True)
        grid.addWidget(self.multi_feature_cb, 1, 1)
        
        layout.addWidget(q)
        
        # Signal Processing
        s = QGroupBox("🎛️ Signal Processing")
        g2 = QGridLayout(s)
        
        g2.addWidget(QLabel("Smoothing Method:"), 0, 0)
        self.smoothing_combo = QComboBox()
        self.smoothing_combo.addItems(['adaptive','gaussian','median'])
        g2.addWidget(self.smoothing_combo, 0, 1)
        
        g2.addWidget(QLabel("Detection Threshold:"), 1, 0)
        self.threshold_spin = QDoubleSpinBox()
        self.threshold_spin.setRange(0.1,0.9)
        self.threshold_spin.setSingleStep(0.05)
        self.threshold_spin.setValue(0.25)
        g2.addWidget(self.threshold_spin, 1, 1)
        
        layout.addWidget(s)
        
        # Musical Analysis
        m = QGroupBox("🎼 Musical Analysis")
        g3 = QGridLayout(m)
        
        g3.addWidget(QLabel("Min Chord Duration:"), 0, 0)
        self.min_duration_spin = QDoubleSpinBox()
        self.min_duration_spin.setRange(0.1,5.0)
        self.min_duration_spin.setSingleStep(0.1)
        self.min_duration_spin.setValue(0.30)
        self.min_duration_spin.setSuffix(" sec")
        g3.addWidget(self.min_duration_spin, 0, 1)
        
        g3.addWidget(QLabel("Key-Aware Analysis:"), 1, 0)
        self.key_aware_cb = QCheckBox("Use key context")
        self.key_aware_cb.setChecked(True)
        g3.addWidget(self.key_aware_cb, 1, 1)
        
        g3.addWidget(QLabel("Max Chord Types:"), 2, 0)
        self.max_chord_types_spin = QSpinBox()
        self.max_chord_types_spin.setRange(20,200)
        self.max_chord_types_spin.setValue(80)
        g3.addWidget(self.max_chord_types_spin, 2, 1)
        
        layout.addWidget(m)
        
        # Export & Presets
        e = QGroupBox("💾 Export & Presets")
        v = QVBoxLayout(e)
        
        # Presets
        preset_row = QHBoxLayout()
        b_fast = QPushButton("⚡ Fast")
        b_bal = QPushButton("⚖️ Balanced")
        b_acc = QPushButton("🎯 Accurate")
        
        b_fast.clicked.connect(lambda: self.apply_preset('fast'))
        b_bal.clicked.connect(lambda: self.apply_preset('balanced'))
        b_acc.clicked.connect(lambda: self.apply_preset('accurate'))
        
        preset_row.addWidget(b_fast)
        preset_row.addWidget(b_bal)
        preset_row.addWidget(b_acc)
        v.addLayout(preset_row)
        
        # Export buttons
        export_row = QHBoxLayout()
        self.btn_export_txt = QPushButton("📄 Export Text")
        self.btn_export_txt.setEnabled(False)
        self.btn_export_txt.clicked.connect(lambda: self.export_results('txt'))
        
        self.btn_export_json = QPushButton("📊 Export JSON")
        self.btn_export_json.setEnabled(False)
        self.btn_export_json.clicked.connect(lambda: self.export_results('json'))
        
        self.btn_export_midi = QPushButton("🎹 Export MIDI")
        self.btn_export_midi.setEnabled(False)
        self.btn_export_midi.clicked.connect(lambda: self.export_results('midi'))
        
        export_row.addWidget(self.btn_export_txt)
        export_row.addWidget(self.btn_export_json)
        export_row.addWidget(self.btn_export_midi)
        v.addLayout(export_row)
        
        layout.addWidget(e)
        layout.addStretch()
        
        scroll.setWidget(container)
        parent.setLayout(QVBoxLayout())
        parent.layout().addWidget(scroll)

    def setup_results_tab(self, parent: QWidget):
        v = QVBoxLayout(parent)
        
        # Analysis Summary
        g = QGroupBox("📊 Analysis Summary")
        grid = QGridLayout(g)
        
        self.total_chords_label = QLabel("Total Chords: 0")
        grid.addWidget(self.total_chords_label, 0, 0)
        
        self.avg_duration_label = QLabel("Avg Duration: 0.0s")
        grid.addWidget(self.avg_duration_label, 0, 1)
        
        self.key_confidence_label = QLabel("Key Confidence: 0%")
        grid.addWidget(self.key_confidence_label, 1, 0)
        
        self.analysis_time_label = QLabel("Analysis Time: -")
        grid.addWidget(self.analysis_time_label, 1, 1)
        
        self.cache_hit_label = QLabel("Cache Status: -")
        grid.addWidget(self.cache_hit_label, 2, 0)
        
        self.complexity_label = QLabel("Harmonic Complexity: -")
        grid.addWidget(self.complexity_label, 2, 1)
        
        v.addWidget(g)
        
        # Results tabs
        tabs = QTabWidget()
        
        # Progression tab
        prog = QWidget()
        prog_v = QVBoxLayout(prog)
        self.results_text = QTextEdit()
        self.results_text.setReadOnly(True)
        self.results_text.setFont(QFont("Consolas", 10))
        prog_v.addWidget(self.results_text)
        tabs.addTab(prog, "Progression")
        
        # Statistics tab
        stats = QWidget()
        stats_v = QVBoxLayout(stats)
        self.stats_text = QTextEdit()
        self.stats_text.setReadOnly(True)
        self.stats_text.setFont(QFont("Consolas", 10))
        stats_v.addWidget(self.stats_text)
        tabs.addTab(stats, "Statistics")
        
        v.addWidget(tabs)

    def setup_perf_tab(self, parent: QWidget):
        v = QVBoxLayout(parent)
        
        # Performance metrics
        g = QGroupBox("⚡ Performance & Cache")
        grid = QGridLayout(g)
        
        self.perf_total_label = QLabel("Total Analysis: -")
        grid.addWidget(self.perf_total_label, 0, 0)
        
        self.perf_loading_label = QLabel("Audio Loading: -")
        grid.addWidget(self.perf_loading_label, 0, 1)
        
        self.perf_features_label = QLabel("Feature Extraction: -")
        grid.addWidget(self.perf_features_label, 1, 0)
        
        self.perf_matching_label = QLabel("Chord Matching: -")
        grid.addWidget(self.perf_matching_label, 1, 1)
        
        v.addWidget(g)
        
        # Cache controls
        cache_row = QHBoxLayout()
        self.btn_clear_cache = QPushButton("🗑️ Clear Cache")
        self.btn_clear_cache.clicked.connect(self.clear_cache)
        
        self.btn_cache_info = QPushButton("ℹ️ Cache Info")
        self.btn_cache_info.clicked.connect(self.show_cache_info)
        
        cache_row.addWidget(self.btn_clear_cache)
        cache_row.addWidget(self.btn_cache_info)
        cache_row.addStretch()
        v.addLayout(cache_row)
        
        self.cache_info_text = QTextEdit()
        self.cache_info_text.setReadOnly(True)
        self.cache_info_text.setMaximumHeight(150)
        v.addWidget(self.cache_info_text)
        
        v.addStretch()

    def apply_theme(self):
        self.setStyleSheet("""
            QWidget { 
                font-family: 'Segoe UI', Arial, sans-serif; 
                font-size: 11px; 
                background: #FAFAFA; 
            }
            QGroupBox { 
                font-weight: 600; 
                border: 2px solid #E0E0E0; 
                border-radius: 12px; 
                margin-top: 12px; 
                padding-top: 12px; 
                background: #FFF; 
            }
            QGroupBox::title { 
                left: 16px; 
                padding: 0 12px; 
                color: #1976D2; 
                font-size: 12px; 
                font-weight: 700; 
            }
            QPushButton { 
                padding: 10px 18px; 
                border-radius: 8px; 
                background: #2196F3; 
                color: #fff; 
                border: none; 
                font-weight: 600; 
            }
            QPushButton:hover {
                background: #1976D2;
            }
            QPushButton:disabled { 
                background: #E0E0E0; 
                color: #9E9E9E; 
            }
            QProgressBar { 
                border: 2px solid #E0E0E0; 
                border-radius: 8px; 
                background: #F5F5F5; 
                min-height: 22px; 
                font-weight: 600; 
            }
            QProgressBar::chunk { 
                background: #4CAF50; 
                border-radius: 6px; 
            }
            QListWidget { 
                border: 2px solid #E0E0E0; 
                border-radius: 8px; 
                background: #FFF; 
                alternate-background-color: #F8F9FA; 
                font-family: 'Consolas','Monaco',monospace; 
            }
            QSlider::groove:horizontal {
                border: 1px solid #999999;
                height: 8px;
                background: qlineargradient(x1:0, y1:0, x2:0, y2:1, stop:0 #B1B1B1, stop:1 #c4c4c4);
                margin: 2px 0;
                border-radius: 4px;
            }
            QSlider::handle:horizontal {
                background: qlineargradient(x1:0, y1:0, x2:1, y2:1, stop:0 #b4b4b4, stop:1 #8f8f8f);
                border: 1px solid #5c5c5c;
                width: 18px;
                margin: -2px 0;
                border-radius: 9px;
            }
        """)
        
        # Special styling for key UI elements
        self.chord_label.setStyleSheet("""
            QLabel { 
                border: 4px solid #2196F3; 
                border-radius: 20px; 
                padding: 26px; 
                margin: 8px;
                background: qlineargradient(x1:0,y1:0,x2:0,y2:1, stop:0 #E3F2FD, stop:1 #90CAF9);
                color: #0D47A1; 
                font-size: 32px; 
                font-weight: 700; 
                letter-spacing: 2px; 
            }
        """)
        
        self.key_label.setStyleSheet("""
            QLabel { 
                border: 2px solid #FF8F00; 
                border-radius: 10px; 
                padding: 10px; 
                background: #FFECB3; 
                font-weight: 700; 
            }
        """)
        
        self.time_label.setStyleSheet("""
            QLabel { 
                border: 2px solid #7B1FA2; 
                border-radius: 10px; 
                padding: 10px; 
                background: #E1BEE7; 
                font-weight: 700; 
                font-family: 'Consolas'; 
            }
        """)
        
        self.conf_label.setStyleSheet("""
            QLabel { 
                border: 1px solid #4CAF50; 
                border-radius: 8px; 
                padding: 8px; 
                background: #C8E6C9; 
                font-weight: 600; 
            }
        """)
        
        self.tempo_label.setStyleSheet("""
            QLabel { 
                border: 1px solid #FF9800; 
                border-radius: 8px; 
                padding: 8px; 
                background: #FFE0B2; 
                font-weight: 600; 
            }
        """)

    def load_settings(self):
        self.performance_cb.setChecked(self.settings.value("performance_mode", False, type=bool))
        self.multi_feature_cb.setChecked(self.settings.value("multi_features", True, type=bool))
        self.smoothing_combo.setCurrentText(self.settings.value("smoothing_method", "adaptive"))
        self.min_duration_spin.setValue(self.settings.value("min_duration", 0.30, type=float))
        self.threshold_spin.setValue(self.settings.value("threshold", 0.25, type=float))
        self.key_aware_cb.setChecked(self.settings.value("key_aware", True, type=bool))
        self.max_chord_types_spin.setValue(self.settings.value("max_chord_types", 80, type=int))

    def save_settings(self):
        self.settings.setValue("performance_mode", self.performance_cb.isChecked())
        self.settings.setValue("multi_features", self.multi_feature_cb.isChecked())
        self.settings.setValue("smoothing_method", self.smoothing_combo.currentText())
        self.settings.setValue("min_duration", self.min_duration_spin.value())
        self.settings.setValue("threshold", self.threshold_spin.value())
        self.settings.setValue("key_aware", self.key_aware_cb.isChecked())
        self.settings.setValue("max_chord_types", self.max_chord_types_spin.value())

    def apply_preset(self, name: str):
        n = (name or '').lower()
        if n == 'fast':
            self.performance_cb.setChecked(True)
            self.multi_feature_cb.setChecked(False)
            self.smoothing_combo.setCurrentText('gaussian')
            self.threshold_spin.setValue(0.30)
            self.min_duration_spin.setValue(0.50)
            self.max_chord_types_spin.setValue(30)
        elif n == 'balanced':
            self.performance_cb.setChecked(False)
            self.multi_feature_cb.setChecked(True)
            self.smoothing_combo.setCurrentText('adaptive')
            self.threshold_spin.setValue(0.25)
            self.min_duration_spin.setValue(0.30)
            self.max_chord_types_spin.setValue(80)
        elif n == 'accurate':
            self.performance_cb.setChecked(False)
            self.multi_feature_cb.setChecked(True)
            self.smoothing_combo.setCurrentText('adaptive')
            self.threshold_spin.setValue(0.20)
            self.min_duration_spin.setValue(0.20)
            self.max_chord_types_spin.setValue(150)
        else:
            QMessageBox.information(self, "Presets", f"Unknown preset: {name}")
            return
        
        self.save_settings()
        QMessageBox.information(
            self, "Preset Applied", 
            f"'{n.title()}' preset has been applied. Settings will be used for the next analysis."
        )

    # Audio Control Methods
    def update_volume(self):
        volume = self.volume_slider.value()
        self.volume_label.setText(f"{volume}%")
        if self.use_rt_player and self.player:
            self.player.gain = volume / 100.0
        elif PYGAME_AVAILABLE and self.audio_player.pygame_initialized:
            try:
                pygame.mixer.music.set_volume(volume / 100.0)
            except Exception:
                pass

    def play_audio(self):
        if not self.current_file_path:
            QMessageBox.information(self, "Playback", "No audio file loaded.")
            return
        if self.use_rt_player and self.player:
            try:
                if self.current_stems:
                    mix, srr = self._mixdown_from_stems()
                    if mix is not None:
                        mono = np.mean(mix, axis=0)
                        self.player.fs = srr
                        self.player.load(mono)
                else:
                    self.player.load(self.audio_mono_float32)
                self.player.start()
                self.btn_play.setText("⏸️ Playing")
                self.btn_pause.setEnabled(False)
                self.btn_stop.setEnabled(True)
                self.playback_timer.start(100)

                self.current_playback_time = 0.0
                self.rt_playing = True
                return
            except Exception as e:
                QMessageBox.warning(self, "Playback Error", f"Realtime audio failed:\n{e}")
        if not PYGAME_AVAILABLE:
            QMessageBox.warning(
                self, "Playback Error",
                "Audio playback requires pygame. Install with:\npip install pygame",
            )
            return
        if self.audio_player.is_playing and self.audio_player.is_paused:
            self.audio_player.unpause()
            self.btn_play.setText("▶️ Play")
            self.btn_pause.setText("⏸️ Pause")
        else:
            if self.audio_player.load_audio(self.current_file_path):
                if self.audio_player.play():
                    self.btn_play.setText("⏸️ Playing")
                    self.btn_pause.setEnabled(True)
                    self.btn_stop.setEnabled(True)
                    self.playback_timer.start(100)
                    self.current_playback_time = 0.0
                else:
                    QMessageBox.warning(self, "Playback Error", "Failed to start audio playback.")
            else:
                QMessageBox.warning(self, "Playback Error", "Failed to load audio file for playback.")

    def toggle_pause(self):
        if self.use_rt_player:
            return
        if self.audio_player.is_playing:
            if self.audio_player.is_paused:
                self.audio_player.unpause()
                self.btn_pause.setText("⏸️ Pause")
            else:
                self.audio_player.pause()
                self.btn_pause.setText("▶️ Resume")

    def stop_playback(self):
        if self.use_rt_player and self.player:
            self.player.stop()
            self.rt_playing = False
        else:
            self.audio_player.stop()
        self.playback_timer.stop()
        self.current_playback_time = 0.0
        self.btn_play.setText("▶️ Play")
        self.btn_pause.setText("⏸️ Pause")
        self.btn_pause.setEnabled(False)
        self.btn_stop.setEnabled(False)

    def update_playback_position(self):
        if self.use_rt_player:
            if self.rt_playing:
                self.current_playback_time += 0.1
                if self.current_playback_time >= self.audio_duration:
                    self.stop_playback()
        elif self.audio_player.is_playing and not self.audio_player.is_paused:
            self.current_playback_time += 0.1
            if self.current_playback_time >= self.audio_duration:
                self.stop_playback()

    def _mixdown_from_stems(self):
        gains = self.stems_panel.gains_db() if self.stems_panel else {}
        if not self.current_stems:
            return None, None
        L = max(v.shape[1] for v in self.current_stems.values())
        mix = np.zeros((2, L), dtype=np.float32)
        for name, buf in self.current_stems.items():
            g = 10 ** (gains.get(name, 0.0) / 20.0)
            if buf.shape[1] < L:
                pad = np.zeros((buf.shape[0], L - buf.shape[1]), dtype=np.float32)
                b = np.concatenate([buf, pad], axis=1)
            else:
                b = buf[:, :L]
            mix += g * b
        mx = np.max(np.abs(mix)) or 1.0
        if mx > 0.98:
            mix = 0.98 * mix / mx
        return mix, self.current_stems_sr

    def _on_separate(self, kind: str):
        if self.audio_mono_float32 is None:
            return
        y = self.audio_mono_float32
        sr = self.fs
        if kind == "hpss":
            stems, srr = self.separator.hpss(y, sr)
            fixed = {k: (np.stack([v, v], axis=0) if v.ndim == 1 else v) for k, v in stems.items()}
            self.current_stems = fixed
            self.current_stems_sr = srr
            self.demucs_tmpdir = None
        else:
            mode = "two_stems" if kind == "two" else "four_stems"
            stems, srr, tdir = self.separator.demucs(y, sr, mode=mode)
            self.current_stems = stems
            self.current_stems_sr = srr
            self.demucs_tmpdir = tdir
        if self.stems_panel:
            self.stems_panel.set_stems(sorted(self.current_stems.keys()))
        if "instrumental" in self.current_stems:
            try:
                tmp = tempfile.NamedTemporaryFile(suffix=".wav", delete=False)
                sf.write(tmp.name, self.current_stems["instrumental"].T, self.current_stems_sr)
                tmp.close()
                chords, key_sig, stats = detect_chords(tmp.name)
                os.unlink(tmp.name)
                self.chords = chords
                self.key_signature = key_sig
                self.stats = stats
                self.update_chord_list_display()
                self.key_label.setText(f"Key: {key_sig}")
                key_conf = int(round(100 * stats.get('key_confidence', 0.0)))
                self.conf_label.setText(f"Confidence: {key_conf}%")
                tempo = int(round(stats.get('estimated_tempo', 120)))
                self.tempo_label.setText(f"♩ = {tempo} BPM")
            except Exception as e:
                print(f"[STEMS] analysis failed: {e}")

    def export_current_stems(self, out_dir: str):
        if not self.current_stems or not self.current_stems_sr:
            return
        os.makedirs(out_dir, exist_ok=True)
        for name, buf in self.current_stems.items():
            sf.write(os.path.join(out_dir, f"{name}.wav"), buf.T, self.current_stems_sr)

    # Analysis Methods
    def load_audio(self):
        file_path, _ = QFileDialog.getOpenFileName(
            self, "Select Audio File", 
            self.settings.value("last_dir", ""),
            "Audio Files (*.wav *.mp3 *.flac *.aac *.m4a *.ogg *.opus *.wma);;All Files (*)"
        )
        
        if not file_path:
            return
        
        self.settings.setValue("last_dir", os.path.dirname(file_path))
        
        try:
            raw, sr_raw = librosa.load(file_path, sr=None, mono=True)
            self.audio_duration = len(raw) / sr_raw
            self.current_file_path = file_path
        except Exception as e:
            QMessageBox.critical(self, "Load Error", f"Could not read audio file.\n\n{e}")
            return

        peak = np.max(np.abs(raw)) or 1.0
        self.audio_mono_float32 = (0.8 * raw / peak).astype(np.float32)
        self.fs = sr_raw

        if self.use_rt_player:
            self.eq = FiveBandEQ(self.fs)
            self.vis = VisualizerWidget(self.fs)
            self.eq_panel = EQPanel()
            self.stems_panel = StemsPanel()
            self.stems_panel.requestSeparate.connect(self._on_separate)

            def on_visual(chunk):
                QtCore.QMetaObject.invokeMethod(
                    self.vis,
                    "push_audio",
                    QtCore.Qt.ConnectionType.QueuedConnection,
                    QtCore.Q_ARG(object, chunk),
                )

            self.player = RTPlayer(fs=self.fs, eq=self.eq, on_visual=on_visual)
            self.eq_panel.gainChanged.connect(lambda idx, g: self.eq.set_gain(idx, g))

            if self.visualizer_container is None:
                vlayout = QVBoxLayout()
                vlayout.addWidget(self.vis)
                vlayout.addWidget(self.eq_panel)
                vlayout.addWidget(self.stems_panel)
                self.visualizer_container = QWidget()
                self.visualizer_container.setLayout(vlayout)
                self.splitter.addWidget(self.visualizer_container)
            else:
                layout = self.visualizer_container.layout()
                while layout.count():
                    item = layout.takeAt(0)
                    w = item.widget()
                    if w is not None:
                        w.setParent(None)
                layout.addWidget(self.vis)
                layout.addWidget(self.eq_panel)
                layout.addWidget(self.stems_panel)

        # Update UI
        filename = os.path.basename(file_path)
        filesize = os.path.getsize(file_path) / 1048576  # MB
        self.file_info.setText(f"File: {filename} ({filesize:.1f} MB)")
        
        # Enable playback controls
        self.btn_play.setEnabled(True)
        
        # Disable other controls during analysis
        self.btn_load.setEnabled(False)
        self.progress_bar.setValue(0)
        self.chord_label.setText("🔄 Analyzing audio...")
        self.status_label.setText("Initializing analysis...")
        
        # Clear previous results
        self.results_text.clear()
        self.stats_text.clear()
        self.chord_list.clear()

        # Gather settings
        settings = {
            'use_multi_features': self.multi_feature_cb.isChecked(),
            'smoothing_method': self.smoothing_combo.currentText(),
            'min_chord_duration': self.min_duration_spin.value(),
            'template_threshold': self.threshold_spin.value(),
            'key_aware': self.key_aware_cb.isChecked(),
            'performance_mode': self.performance_cb.isChecked(),
            'max_chord_types': self.max_chord_types_spin.value(),
        }
        
        self.save_settings()
        
        mode_text = "Performance" if settings['performance_mode'] else "Quality"
        smoothing_text = settings['smoothing_method'].title()
        self.perf_label.setText(f"{mode_text} | {smoothing_text} smoothing")

        # Start analysis thread
        self.analysis_thread = QThread()
        self.worker = ChordAnalysisWorker(file_path, settings)
        self.worker.moveToThread(self.analysis_thread)
        
        # Connect signals
        self.analysis_thread.started.connect(self.worker.run)
        self.worker.progress.connect(self.update_progress)
        self.worker.status.connect(self.status_label.setText)
        self.worker.finished.connect(self.on_analysis_finished)
        self.worker.error.connect(self.on_analysis_error)
        
        self.analysis_thread.start()

    def update_progress(self, value: int):
        self.progress_bar.setValue(int(value))

    def on_analysis_finished(self, chords: List[Tuple[float,str]], key_sig: str, stats: Dict):
        # Clean up thread
        if self.analysis_thread:
            self.analysis_thread.quit()
            self.analysis_thread.wait(2000)
        self.analysis_thread = None
        self.worker = None

        # Store results
        self.chords = chords
        self.key_signature = key_sig
        self.stats = stats
        
        # Update UI with results
        self.key_label.setText(f"Key: {key_sig}")
        
        key_conf = int(round(100 * stats.get('key_confidence', 0.0)))
        self.conf_label.setText(f"Confidence: {key_conf}%")
        
        tempo = int(round(stats.get('estimated_tempo', 120)))
        self.tempo_label.setText(f"♩ = {tempo} BPM")
        
        self.time_label.setText(f"00:00 / {format_time(self.audio_duration)}")
        
        # Show current chord
        current_chord = self.chords[0][1] if self.chords else "No chords detected"
        self.chord_label.setText(current_chord)

        # Enable export buttons
        has_results = len(self.chords) > 0
        self.btn_export_txt.setEnabled(has_results)
        self.btn_export_json.setEnabled(has_results)
        self.btn_export_midi.setEnabled(has_results)

        # Update results displays
        self.update_chord_list_display()
        self.results_text.setPlainText(self.progression_text())
        self.stats_text.setPlainText(json.dumps(self.stats, indent=2))
        
        # Update summary statistics
        self.total_chords_label.setText(f"Total Chords: {len(self.chords)}")
        
        avg_dur = (self.audio_duration / max(1, len(self.chords))) if self.audio_duration else 0.0
        self.avg_duration_label.setText(f"Avg Duration: {avg_dur:.2f}s")
        
        self.key_confidence_label.setText(f"Key Confidence: {key_conf}%")
        self.analysis_time_label.setText("Analysis Time: (see console PERF logs)")
        
        cache_status = "Hit" if self.stats.get('cache_hit', False) else "Miss"
        self.cache_hit_label.setText(f"Cache Status: {cache_status}")
        
        complexity = self.stats.get('harmonic_complexity', '-')
        self.complexity_label.setText(f"Harmonic Complexity: {complexity}")

        # Re-enable controls
        self.btn_load.setEnabled(True)
        self.status_label.setText("Analysis complete.")
        
        # Start update timer
        self.timer.start()

    def on_analysis_error(self, error_msg: str):
        # Clean up thread
        if self.analysis_thread:
            self.analysis_thread.quit()
            self.analysis_thread.wait(2000)
        self.analysis_thread = None
        self.worker = None
        
        self.btn_load.setEnabled(True)
        QMessageBox.critical(self, "Analysis Error", error_msg)
        self.status_label.setText("Analysis failed.")

    def update_chord_list_display(self):
        self.chord_list.clear()
        if not self.chords:
            return
        
        max_display = self.max_chords_spin.value()
        recent_chords = self.chords[-max_display:]
        show_times = self.show_times_cb.isChecked()
        
        for t, name in recent_chords:
            if show_times:
                item_text = f"{format_time(t)}  |  {name}"
            else:
                item_text = name
            self.chord_list.addItem(item_text)

    def progression_text(self) -> str:
        if not self.chords:
            return "No chord progression detected."
        
        lines = []
        for t, name in self.chords:
            lines.append(f"{format_time(t)}\t{name}")
        
        return "\n".join(lines)

    def update_display(self):
        # Update current chord display based on playback position
        if self.chords and self.audio_player.is_playing:
            current_time = self.current_playback_time
            current_chord = "N.C."
            
            for t, name in self.chords:
                if t <= current_time:
                    current_chord = name
                else:
                    break
            
            self.chord_label.setText(current_chord)
        
        # Update time display
        if self.audio_player.is_playing:
            current_time_str = format_time(self.current_playback_time)
            total_time_str = format_time(self.audio_duration)
            self.time_label.setText(f"{current_time_str} / {total_time_str}")
        else:
            total_time_str = format_time(self.audio_duration)
            self.time_label.setText(f"00:00 / {total_time_str}")

    # Export Methods
    def export_results(self, format_type: str):
        if not self.chords:
            QMessageBox.information(self, "Export", "No results to export.")
            return
        
        if format_type == 'txt':
            file_path, _ = QFileDialog.getSaveFileName(
                self, "Save Text File", "chords.txt", "Text Files (*.txt)"
            )
            if file_path:
                try:
                    with open(file_path, 'w', encoding='utf-8') as f:
                        f.write(self.progression_text())
                    QMessageBox.information(self, "Export", f"Text file saved to: {file_path}")
                except Exception as e:
                    QMessageBox.critical(self, "Export Error", f"Failed to save text file:\n{e}")
        
        elif format_type == 'json':
            file_path, _ = QFileDialog.getSaveFileName(
                self, "Save JSON File", "chords.json", "JSON Files (*.json)"
            )
            if file_path:
                try:
                    export_data = {
                        'key_signature': self.key_signature,
                        'tempo': self.stats.get('estimated_tempo', 120),
                        'audio_duration': self.audio_duration,
                        'analysis_stats': self.stats,
                        'chord_progression': [
                            {'time': t, 'chord': name} for t, name in self.chords
                        ]
                    }
                    with open(file_path, 'w', encoding='utf-8') as f:
                        json.dump(export_data, f, indent=2)
                    QMessageBox.information(self, "Export", f"JSON file saved to: {file_path}")
                except Exception as e:
                    QMessageBox.critical(self, "Export Error", f"Failed to save JSON file:\n{e}")
        
        elif format_type == 'midi':
            try:
                import mido
                from mido import MidiFile, MidiTrack, Message
            except ImportError:
                QMessageBox.warning(
                    self, "MIDI Export", 
                    "MIDI export requires the 'mido' library.\nInstall with: pip install mido"
                )
                return
            
            file_path, _ = QFileDialog.getSaveFileName(
                self, "Save MIDI File", "chords.mid", "MIDI Files (*.mid)"
            )
            if file_path:
                try:
                    mid = MidiFile()
                    track = MidiTrack()
                    mid.tracks.append(track)
                    
                    # Set tempo
                    tempo = int(self.stats.get('estimated_tempo', 120))
                    microseconds_per_beat = int(60000000 / tempo)
                    track.append(mido.MetaMessage('set_tempo', tempo=microseconds_per_beat))
                    
                    # Convert chords to MIDI
                    ticks_per_beat = 480
                    previous_time = 0.0
                    
                    for i, (time_sec, chord_name) in enumerate(self.chords):
                        # Calculate duration
                        if i < len(self.chords) - 1:
                            duration_sec = self.chords[i + 1][0] - time_sec
                        else:
                            duration_sec = max(1.0, self.audio_duration - time_sec)
                        
                        # Convert to MIDI ticks
                        delta_time = int((time_sec - previous_time) * ticks_per_beat * tempo / 60)
                        duration_ticks = int(duration_sec * ticks_per_beat * tempo / 60)
                        
                        if chord_name != "N.C." and chord_name:
                            # Extract root note
                            root_note = chord_name.split('/')[0]  # Handle inversions
                            if len(root_note) > 0:
                                try:
                                    if '#' in root_note:
                                        note_name = root_note[:2]
                                    elif 'b' in root_note:
                                        note_name = root_note[:2]
                                    else:
                                        note_name = root_note[0]
                                    
                                    if note_name in NOTE_NAMES:
                                        root_pc = NOTE_NAMES.index(note_name)
                                        midi_note = 60 + root_pc  # Middle C octave
                                        
                                        # Note on
                                        track.append(Message(
                                            'note_on', 
                                            note=midi_note, 
                                            velocity=64, 
                                            time=delta_time
                                        ))
                                        
                                        # Note off
                                        track.append(Message(
                                            'note_off', 
                                            note=midi_note, 
                                            velocity=64, 
                                            time=duration_ticks
                                        ))
                                        
                                        delta_time = 0  # Reset delta time after first message
                                except Exception:
                                    continue
                        
                        previous_time = time_sec
                    
                    mid.save(file_path)
                    QMessageBox.information(self, "Export", f"MIDI file saved to: {file_path}")
                    
                except Exception as e:
                    QMessageBox.critical(self, "Export Error", f"Failed to save MIDI file:\n{e}")

    # Cache Management
    def clear_cache(self):
        try:
            if os.path.isdir(CHORDS_CACHE_DIR):
                shutil.rmtree(CHORDS_CACHE_DIR)
            os.makedirs(CHORDS_CACHE_DIR, exist_ok=True)
            self.cache_info_text.setPlainText("Cache cleared successfully.")
            self.cache_status.setText("Cache: Cleared")
        except Exception as e:
            QMessageBox.warning(self, "Cache", f"Failed to clear cache:\n{e}")

    def show_cache_info(self):
        if not os.path.isdir(CHORDS_CACHE_DIR):
            self.cache_info_text.setPlainText("No cache directory found.")
            return
        
        try:
            files = [f for f in os.listdir(CHORDS_CACHE_DIR) if f.endswith('.json')]
            total_size = 0
            
            for filename in files:
                filepath = os.path.join(CHORDS_CACHE_DIR, filename)
                total_size += os.path.getsize(filepath)
            
            info_lines = [
                f"Cache Directory: {CHORDS_CACHE_DIR}",
                f"Total Files: {len(files)}",
                f"Total Size: {total_size / 1048576:.2f} MB",
                f"",
                "Recent Files:"
            ]
            
            # Show most recent files
            file_info = []
            for filename in files:
                filepath = os.path.join(CHORDS_CACHE_DIR, filename)
                mtime = os.path.getmtime(filepath)
                size = os.path.getsize(filepath)
                file_info.append((mtime, filename, size))
            
            file_info.sort(reverse=True)  # Most recent first
            
            for mtime, filename, size in file_info[:10]:  # Show top 10
                mod_time = time.strftime('%Y-%m-%d %H:%M', time.localtime(mtime))
                info_lines.append(f"  {filename[:30]}... ({size/1024:.1f} KB, {mod_time})")
            
            self.cache_info_text.setPlainText("\n".join(info_lines))
            
        except Exception as e:
            self.cache_info_text.setPlainText(f"Error reading cache info:\n{e}")

    def closeEvent(self, event):
        """Handle application close event"""
        # Stop any ongoing analysis
        if self.analysis_thread and self.analysis_thread.isRunning():
            if self.worker:
                self.worker.deleteLater()
            self.analysis_thread.quit()
            self.analysis_thread.wait(3000)
        
        # Stop audio playback
        self.stop_playback()
        
        # Clean up pygame
        if PYGAME_AVAILABLE and self.audio_player.pygame_initialized:
            try:
                pygame.mixer.quit()
            except Exception:
                pass
        
        # Save settings
        self.save_settings()
        
        event.accept()

# --------------- Main Application ---------------
def main():
    try:
        app = QApplication(sys.argv)
        app.setApplicationName("Chord Player Pro")
        app.setApplicationVersion("4.3")
        app.setOrganizationName("MusicTech")
        
        # Set application icon if available
        try:
            # You can add an icon file here
            # app.setWindowIcon(QIcon("icon.png"))
            pass
        except Exception:
            pass
        
        # Create and show main window
        main_window = App()
        main_window.show()
        
        # Check for required dependencies
        missing_deps = []
        try:
            import librosa
        except ImportError:
            missing_deps.append("librosa")
        
        try:
            import sklearn
        except ImportError:
            missing_deps.append("scikit-learn")
        
        if not PYGAME_AVAILABLE:
            missing_deps.append("pygame (for audio playback)")
        
        if missing_deps:
            QMessageBox.warning(
                main_window,
                "Missing Dependencies",
                f"Some features may not work properly. Missing:\n" + 
                "\n".join(f"• {dep}" for dep in missing_deps) +
                "\n\nInstall with: pip install " + " ".join(dep.split()[0] for dep in missing_deps)
            )
        
        sys.exit(app.exec())
        
    except Exception as e:
        print(f"Fatal error: {e}")
        import traceback
        traceback.print_exc()
        
        # Try to show error in GUI if possible
        try:
            app = QApplication(sys.argv)
            QMessageBox.critical(
                None, "Fatal Error", 
                f"Failed to start Chord Player Pro:\n\n{e}\n\nCheck console for details."
            )
        except Exception:
            pass
        
        sys.exit(1)

if __name__ == '__main__':
        main()
