import sys
import time
import os
import hashlib
import json
import threading
from concurrent.futures import ThreadPoolExecutor
import numpy as np
import librosa
import sounddevice as sd
from scipy import signal
from scipy.ndimage import median_filter, gaussian_filter1d
from sklearn.preprocessing import normalize
from sklearn.cluster import KMeans
from PyQt6.QtWidgets import (
    QApplication, QWidget, QVBoxLayout, QPushButton, QLabel, QFileDialog, 
    QMessageBox, QHBoxLayout, QSlider, QCheckBox, QSpinBox, QProgressBar,
    QTabWidget, QTextEdit, QGroupBox, QGridLayout, QComboBox, QDoubleSpinBox,
    QSplitter, QListWidget, QFrame
)
from PyQt6.QtCore import QTimer, Qt, QThread, pyqtSignal, QObject, QSettings
from PyQt6.QtGui import QFont, QPalette, QColor, QPixmap, QPainter

# --- Enhanced Configuration ---
CHORDS_CACHE_DIR = "chord_cache_v4"
CHORDS_CACHE_EXTENSION = ".json"
SR = 22050
FFT_SIZE = 16384
HOP_LENGTH = 512
NOTE_NAMES = ['C','C#','D','D#','E','F','F#','G','G#','A','A#','B']

# Enhanced chord definitions with more jazz and contemporary chords
CHORD_DEFS = {
    # Basic triads
    '': [0,4,7],           # Major
    'm': [0,3,7],          # Minor
    'dim': [0,3,6],        # Diminished
    'aug': [0,4,8],        # Augmented
    
    # Suspended chords
    'sus2': [0,2,7],       # Suspended 2nd
    'sus4': [0,5,7],       # Suspended 4th
    
    # 6th chords
    '6': [0,4,7,9],        # Major 6th
    'm6': [0,3,7,9],       # Minor 6th
    '6/9': [0,4,7,9,14%12], # Major 6th add 9
    
    # 7th chords
    '7': [0,4,7,10],       # Dominant 7th
    'M7': [0,4,7,11],      # Major 7th
    'm7': [0,3,7,10],      # Minor 7th
    'm7b5': [0,3,6,10],    # Half-diminished
    'dim7': [0,3,6,9],     # Diminished 7th
    'mM7': [0,3,7,11],     # Minor-major 7th
    
    # Extended chords
    '9': [0,4,7,10,14%12], # Dominant 9th
    'M9': [0,4,7,11,14%12],# Major 9th
    'm9': [0,3,7,10,14%12],# Minor 9th
    'add9': [0,4,7,14%12], # Add 9th
    '11': [0,4,7,10,14%12,17%12], # Dominant 11th
    '13': [0,4,7,10,14%12,21%12], # Dominant 13th
    
    # Altered dominants
    '7#5': [0,4,8,10],     # Augmented 7th
    '7b5': [0,4,6,10],     # Dominant 7th flat 5
    '7#9': [0,4,7,10,15%12], # Dominant 7th sharp 9
    '7b9': [0,4,7,10,13%12], # Dominant 7th flat 9
    '7#11': [0,4,7,10,18%12], # Dominant 7th sharp 11
    
    # Contemporary chords
    'add11': [0,4,7,17%12], # Add 11th
    'madd9': [0,3,7,14%12], # Minor add 9th
    '6sus4': [0,5,7,9],     # 6th suspended 4th
    'm6/9': [0,3,7,9,14%12], # Minor 6th add 9
    
    # Polychords (simplified representations)
    'maj7#11': [0,4,7,11,18%12], # Major 7th sharp 11
    'm11': [0,3,7,10,14%12,17%12], # Minor 11th
    'm13': [0,3,7,10,14%12,21%12], # Minor 13th
}

# Enhanced chord quality weights
CHORD_WEIGHTS = {
    'root': 1.0,
    'third': 0.9,
    'fifth': 0.7,
    'seventh': 0.6,
    'ninth': 0.4,
    'eleventh': 0.3,
    'thirteenth': 0.2,
    'bass': 0.8  # For slash chords
}

# --- Enhanced Analysis Worker ---
class ChordAnalysisWorker(QObject):
    progress = pyqtSignal(int)
    finished = pyqtSignal(list, str)  # chords, key_signature
    error = pyqtSignal(str)
    status_update = pyqtSignal(str)  # New signal for status updates
    
    def __init__(self, file_path, settings):
        super().__init__()
        self.file_path = file_path
        self.settings = settings
        
    def run(self):
        try:
            self.status_update.emit("Loading audio file...")
            chords, key_sig = detect_chords_enhanced(
                self.file_path, 
                progress_callback=self.progress.emit,
                status_callback=self.status_update.emit,
                **self.settings
            )
            self.finished.emit(chords, key_sig)
        except Exception as e:
            self.error.emit(str(e))

# --- Enhanced Detection Functions ---
def create_enhanced_chord_templates():
    """Create enhanced chord templates with better weighting"""
    templates = {}
    
    for root_idx, note in enumerate(NOTE_NAMES):
        for suffix, intervals in CHORD_DEFS.items():
            name = note + suffix
            vec = np.zeros(12)
            
            # Apply sophisticated harmonic weighting
            for i, interval in enumerate(intervals):
                pitch_class = (root_idx + interval) % 12
                
                # Determine harmonic importance
                interval_mod = interval % 12
                if interval_mod == 0:  # Root
                    weight = CHORD_WEIGHTS['root']
                elif interval_mod in [3, 4]:  # Third
                    weight = CHORD_WEIGHTS['third']
                elif interval_mod in [6, 7, 8]:  # Fifth
                    weight = CHORD_WEIGHTS['fifth']
                elif interval_mod in [10, 11]:  # Seventh
                    weight = CHORD_WEIGHTS['seventh']
                elif interval_mod in [2, 14%12]:  # Ninth
                    weight = CHORD_WEIGHTS['ninth']
                elif interval_mod in [5, 17%12]:  # Eleventh
                    weight = CHORD_WEIGHTS['eleventh']
                else:
                    weight = CHORD_WEIGHTS['thirteenth']
                
                # Position weighting (earlier intervals more important)
                position_weight = 1.0 - (i * 0.08)
                
                # Frequency-based weighting (favor mid-range frequencies)
                freq_weight = 1.0 - abs(pitch_class - 7) * 0.05
                
                vec[pitch_class] = weight * position_weight * freq_weight
                
            # Normalize template with L2 norm
            if np.sum(vec) > 0:
                vec = vec / np.linalg.norm(vec)
                
            templates[name] = vec
    
    return templates

def extract_enhanced_features(y, sr=SR):
    """Extract enhanced audio features for chord detection"""
    features = {}
    
    # Multi-resolution chroma with CQT
    features['chroma_cqt'] = librosa.feature.chroma_cqt(
        y=y, sr=sr, hop_length=HOP_LENGTH, fmin=librosa.note_to_hz('C1')
    )
    
    # STFT-based chroma for comparison
    features['chroma_stft'] = librosa.feature.chroma_stft(
        y=y, sr=sr, hop_length=HOP_LENGTH
    )
    
    # Harmonic chroma (separated from percussive)
    y_harmonic, _ = librosa.effects.hpss(y, margin=(1.0, 5.0))
    features['chroma_harmonic'] = librosa.feature.chroma_cqt(
        y=y_harmonic, sr=sr, hop_length=HOP_LENGTH
    )
    
    # Spectral centroid for brightness analysis
    features['spectral_centroid'] = librosa.feature.spectral_centroid(
        y=y, sr=sr, hop_length=HOP_LENGTH
    )
    
    # RMS energy for dynamics
    features['rms'] = librosa.feature.rms(y=y, hop_length=HOP_LENGTH)
    
    # Zero crossing rate for texture
    features['zcr'] = librosa.feature.zero_crossing_rate(y=y, hop_length=HOP_LENGTH)
    
    return features

def apply_intelligent_smoothing(chroma_sequence, features=None, method='adaptive'):
    """Apply intelligent smoothing based on musical context"""
    if method == 'adaptive' and features is not None:
        # Use spectral flux to determine stability regions
        if 'spectral_centroid' in features:
            centroid = features['spectral_centroid'][0]
            centroid_diff = np.abs(np.diff(centroid))
            
            # Adaptive sigma based on spectral stability
            sigma_base = 2.0
            sigma_adaptive = np.zeros_like(centroid_diff)
            
            for i in range(len(centroid_diff)):
                stability = 1.0 / (1.0 + centroid_diff[i] * 1000)
                sigma_adaptive[i] = sigma_base * (0.5 + stability)
            
            # Apply variable smoothing
            smoothed = np.zeros_like(chroma_sequence)
            for i in range(12):
                for t in range(chroma_sequence.shape[1] - 1):
                    sigma = sigma_adaptive[min(t, len(sigma_adaptive) - 1)]
                    window_size = int(sigma * 2)
                    
                    start = max(0, t - window_size)
                    end = min(chroma_sequence.shape[1], t + window_size + 1)
                    
                    weights = np.exp(-0.5 * ((np.arange(start, end) - t) / sigma) ** 2)
                    values = chroma_sequence[i, start:end]
                    
                    smoothed[i, t] = np.sum(values * weights) / np.sum(weights)
        else:
            # Fallback to standard gaussian
            smoothed = gaussian_filter1d(chroma_sequence, sigma=2.0, axis=1)
    else:
        # Standard median filtering
        smoothed = np.zeros_like(chroma_sequence)
        for i in range(12):
            smoothed[i, :] = median_filter(chroma_sequence[i, :], size=5)
    
    return smoothed

def detect_key_enhanced(chroma_features, method='krumhansl'):
    """Enhanced key detection with multiple methods"""
    avg_chroma = np.mean(chroma_features, axis=1)
    
    if method == 'krumhansl':
        # Krumhansl-Schmuckler profiles
        major_profile = np.array([6.35, 2.23, 3.48, 2.33, 4.38, 4.09, 2.52, 5.19, 2.39, 3.66, 2.29, 2.88])
        minor_profile = np.array([6.33, 2.68, 3.52, 5.38, 2.60, 3.53, 2.54, 4.75, 3.98, 2.69, 3.34, 3.17])
    elif method == 'temperley':
        # Temperley profiles (more modern)
        major_profile = np.array([5.0, 2.0, 3.5, 2.0, 4.5, 4.0, 2.0, 4.5, 2.0, 3.5, 1.5, 4.0])
        minor_profile = np.array([5.0, 2.0, 3.5, 4.5, 2.0, 4.0, 2.0, 4.5, 3.5, 2.0, 1.5, 4.0])
    
    key_correlations = []
    
    for shift in range(12):
        # Major key correlation
        major_corr = np.corrcoef(avg_chroma, np.roll(major_profile, shift))[0, 1]
        if not np.isnan(major_corr):
            key_correlations.append((NOTE_NAMES[shift] + ' Major', major_corr))
        
        # Minor key correlation
        minor_corr = np.corrcoef(avg_chroma, np.roll(minor_profile, shift))[0, 1]
        if not np.isnan(minor_corr):
            key_correlations.append((NOTE_NAMES[shift] + ' Minor', minor_corr))
    
    if key_correlations:
        best_key = max(key_correlations, key=lambda x: x[1])
        return best_key[0], best_key[1]
    else:
        return "Unknown", 0.0

def detect_chords_enhanced(path, progress_callback=None, status_callback=None, **settings):
    """Enhanced chord detection with improved accuracy"""
    
    # Default settings
    default_settings = {
        'use_harmonic_separation': True,
        'smoothing_method': 'adaptive',
        'min_chord_duration': 0.3,
        'template_threshold': 0.25,
        'key_aware': True,
        'use_multi_features': True
    }
    
    # Merge with provided settings
    for key, value in default_settings.items():
        if key not in settings:
            settings[key] = value
    
    # Cache management
    os.makedirs(CHORDS_CACHE_DIR, exist_ok=True)
    fh = get_file_hash(path)
    cache_key = f"{fh}_enhanced_{hash(str(sorted(settings.items())))}"
    cache_file = os.path.join(CHORDS_CACHE_DIR, cache_key + CHORDS_CACHE_EXTENSION)
    
    if os.path.exists(cache_file):
        try:
            with open(cache_file, 'r') as f:
                data = json.load(f)
                return data['chords'], data.get('key_signature', 'Unknown')
        except:
            pass
    
    if progress_callback:
        progress_callback(5)
    if status_callback:
        status_callback("Loading audio...")
    
    # Load audio
    y, sr = librosa.load(path, sr=SR)
    
    if progress_callback:
        progress_callback(15)
    if status_callback:
        status_callback("Extracting features...")
    
    # Extract enhanced features
    features = extract_enhanced_features(y, sr)
    
    if progress_callback:
        progress_callback(35)
    if status_callback:
        status_callback("Analyzing harmony...")
    
    # Choose primary chroma feature
    if settings['use_multi_features']:
        # Combine multiple chroma features with weighting
        chroma_combined = (
            0.4 * features['chroma_cqt'] +
            0.3 * features['chroma_harmonic'] +
            0.3 * features['chroma_stft']
        )
    else:
        chroma_combined = features['chroma_cqt']
    
    # Apply intelligent smoothing
    smoothed_chroma = apply_intelligent_smoothing(
        chroma_combined, features, method=settings['smoothing_method']
    )
    
    if progress_callback:
        progress_callback(55)
    if status_callback:
        status_callback("Detecting key signature...")
    
    # Key detection
    key_signature = "Unknown"
    if settings['key_aware']:
        key_signature, confidence = detect_key_enhanced(smoothed_chroma)
    
    if progress_callback:
        progress_callback(65)
    if status_callback:
        status_callback("Building chord templates...")
    
    # Normalize chroma
    chroma_norm = normalize(smoothed_chroma, axis=0, norm='l2')
    
    # Create enhanced templates
    templates = create_enhanced_chord_templates()
    
    if progress_callback:
        progress_callback(75)
    if status_callback:
        status_callback("Matching chords...")
    
    # Add periodic progress updates
    frame_count = chroma_norm.shape[1]
    update_interval = max(10, frame_count // 100)  # Update at least every 1%
    
    # Enhanced chord detection
    chords = []
    times = librosa.frames_to_time(np.arange(chroma_norm.shape[1]), sr=sr, hop_length=HOP_LENGTH)
    
    for frame_idx, (t, chroma_frame) in enumerate(zip(times, chroma_norm.T)):
        if frame_idx % update_interval == 0 and progress_callback:
            new_progress = min(99, 75 + int(20 * (frame_idx / frame_count)))
            progress_callback(new_progress)
            
        # First check for simple triads (major, minor)
        similarities = {}
        simple_templates = {
            k: v for k, v in templates.items() 
            if k in ['', 'm']  # Only major and minor triads first
        }
        
        # First pass with just major/minor chords
        for chord_name, template in simple_templates.items():
            cos_sim = np.dot(template, chroma_frame) / (np.linalg.norm(template) * np.linalg.norm(chroma_frame) + 1e-8)
            similarities[note + chord_name] = cos_sim  # Add root note
            
        # If no clear match (cosine similarity < 0.8), check dominant 7ths
        if not similarities or max(similarities.values()) < 0.8:
            dominant_templates = {k: v for k, v in templates.items() if k in ['7', 'm7']}
            for chord_name, template in dominant_templates.items():
                cos_sim = np.dot(template, chroma_frame) / (np.linalg.norm(template) * np.linalg.norm(chroma_frame) + 1e-8)
                similarities[note + chord_name] = cos_sim
                
        # Only check complex chords if no good matches (similarity < 0.75)
        if not similarities or max(similarities.values()) < 0.75:
            other_templates = {
                k: v for k, v in templates.items() 
                if k not in ['', 'm', '7', 'm7']
            }
            for chord_name, template in other_templates.items():
                cos_sim = np.dot(template, chroma_frame) / (np.linalg.norm(template) * np.linalg.norm(chroma_frame) + 1e-8)
                similarities[note + chord_name] = cos_sim
        
        # Find best match
        best_match = max(similarities.items(), key=lambda x: x[1])
        
        if best_match[1] >= settings['template_threshold']:
            chord_name = best_match[0]
        else:
            chord_name = "N.C."
        
        chords.append((t, chord_name))
    
    if progress_callback:
        progress_callback(90)
    if status_callback:
        status_callback("Post-processing...")
    
    # Enhanced post-processing
    chords = post_process_enhanced(chords, settings['min_chord_duration'], key_signature)
    
    # Cache results
    try:
        with open(cache_file, 'w') as f:
            json.dump({
                'chords': chords,
                'key_signature': key_signature,
                'analysis_settings': settings
            }, f)
    except:
        pass
    
    if progress_callback:
        progress_callback(100)
    if status_callback:
        status_callback("Analysis complete!")
    
    return chords, key_signature

def post_process_enhanced(chords, min_duration=0.3, key_signature=None):
    """Enhanced post-processing with musical intelligence"""
    if not chords:
        return []
    
    # Merge neighboring identical chords
    merged_chords = []
    current_chord = chords[0][1]
    start_time = chords[0][0]
    end_time = start_time
    
    for t, chord in chords[1:]:
        if chord == current_chord and t - end_time < 0.2:  # Merge if gap is small
            end_time = t
        else:
            if end_time - start_time >= min_duration:  # Only keep sufficiently long chords
                merged_chords.append((start_time, current_chord))
            current_chord = chord
            start_time = t
        end_time = t
    
    # Add last chord if valid
    if end_time - start_time >= min_duration:
        merged_chords.append((start_time, current_chord))
        
    # Further filtering based on key context
    filtered_chords = []
    for i, (t, chord) in enumerate(merged_chords):
        keep = True
        
        # Skip very short chords unless they're important or between different chords
        if i > 0 and i < len(merged_chords)-1:
            prev_chord = merged_chords[i-1][1]
            next_chord = merged_chords[i+1][1]
            
            # Remove repeated chords in sequence (A -> A -> B becomes A -> B)
            if chord == prev_chord and chord == next_chord:
                keep = False
                
            # Remove single-occurrence transition chords
            elif chord != prev_chord and chord != next_chord:
                duration = merged_chords[i+1][0] - t
                if duration < min_duration and not is_harmonically_important(chord, merged_chords, i):
                    keep = False
        
        if keep:
            filtered_chords.append((t, chord))
    
    return filtered_chords

def is_harmonically_important(chord, sequence, index):
    """Determine if chord is harmonically important"""
    # Dominant chords are usually important
    if any(x in chord for x in ['7', 'V', 'dom']):
        return True
    
    # Check for common progressions
    if index > 0 and index < len(sequence) - 1:
        prev_chord = sequence[index-1][1]
        next_chord = sequence[index+1][1]
        
        # V-I resolution
        if '7' in chord and any(x in next_chord for x in ['', 'm']):
            return True
    
    return False

def is_transition_chord(chord, sequence, index):
    """Check if chord is a valid transition chord"""
    if index == 0 or index >= len(sequence) - 1:
        return False
    
    prev_chord = sequence[index-1][1]
    next_chord = sequence[index+1][1]
    
    # Passing chords are often brief but important
    if chord != prev_chord and chord != next_chord:
        return True
    
    return False

def get_file_hash(path):
    """Generate SHA-256 hash of file for caching"""
    hasher = hashlib.sha256()
    with open(path, 'rb') as f:
        for chunk in iter(lambda: f.read(8192), b""):
            hasher.update(chunk)
    return hasher.hexdigest()

# --- Enhanced UI ---
class EnhancedChordPlayer(QWidget):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("Enhanced Chord Player Pro v4.0")
        self.resize(1000, 700)
        
        # Initialize variables
        self.chords = []
        self.key_signature = "Unknown"
        self.start_time = None
        self.audio_duration = 0
        self.analysis_thread = None
        self.worker = None
        self.settings = QSettings("MusicTech", "ChordPlayer")
        
        self.init_ui()
        self.apply_enhanced_theme()
        self.load_settings()
        
    def init_ui(self):
        """Initialize enhanced user interface"""
        main_layout = QVBoxLayout(self)
        
        # Create splitter for better layout
        splitter = QSplitter(Qt.Orientation.Horizontal)
        main_layout.addWidget(splitter)
        
        # Left panel - Controls and display
        left_widget = QWidget()
        left_layout = QVBoxLayout(left_widget)
        
        # File controls
        file_group = QGroupBox("Audio File")
        file_layout = QHBoxLayout(file_group)
        
        self.btn_load = QPushButton("Load Audio File")
        self.btn_load.clicked.connect(self.load_audio)
        file_layout.addWidget(self.btn_load)
        
        self.btn_stop = QPushButton("Stop Playback")
        self.btn_stop.clicked.connect(self.stop_playback)
        self.btn_stop.setEnabled(False)
        file_layout.addWidget(self.btn_stop)
        
        left_layout.addWidget(file_group)
        
        # Progress and status
        progress_group = QGroupBox("Analysis Progress")
        progress_layout = QVBoxLayout(progress_group)
        
        self.progress_bar = QProgressBar()
        progress_layout.addWidget(self.progress_bar)
        
        self.status_label = QLabel("Ready")
        progress_layout.addWidget(self.status_label)
        
        left_layout.addWidget(progress_group)
        
        # Current chord display (enhanced)
        chord_group = QGroupBox("Current Chord")
        chord_layout = QVBoxLayout(chord_group)
        
        self.chord_label = QLabel("Load an audio file to begin")
        self.chord_label.setAlignment(Qt.AlignmentFlag.AlignCenter)
        chord_layout.addWidget(self.chord_label)
        
        self.key_label = QLabel("Key: Unknown")
        self.key_label.setAlignment(Qt.AlignmentFlag.AlignCenter)
        chord_layout.addWidget(self.key_label)
        
        self.time_label = QLabel("00:00 / 00:00")
        self.time_label.setAlignment(Qt.AlignmentFlag.AlignCenter)
        chord_layout.addWidget(self.time_label)
        
        left_layout.addWidget(chord_group)
        
        # Chord progression display
        progression_group = QGroupBox("Recent Chords")
        progression_layout = QVBoxLayout(progression_group)
        
        self.chord_list = QListWidget()
        self.chord_list.setMaximumHeight(150)
        progression_layout.addWidget(self.chord_list)
        
        left_layout.addWidget(progression_group)
        
        splitter.addWidget(left_widget)
        
        # Right panel - Settings and results
        right_widget = QWidget()
        tabs = QTabWidget()
        right_widget_layout = QVBoxLayout(right_widget)
        right_widget_layout.addWidget(tabs)
        
        # Settings tab
        settings_tab = QWidget()
        tabs.addTab(settings_tab, "Settings")
        self.setup_enhanced_settings_tab(settings_tab)
        
        # Results tab
        results_tab = QWidget()
        tabs.addTab(results_tab, "Analysis Results")
        self.setup_results_tab(results_tab)
        
        splitter.addWidget(right_widget)
        
        # Set splitter proportions
        splitter.setSizes([400, 600])
        
        # Timer for updates
        self.timer = QTimer(self)
        self.timer.setInterval(50)
        self.timer.timeout.connect(self.update_display)
        
    def setup_enhanced_settings_tab(self, parent):
        """Setup enhanced settings interface"""
        layout = QVBoxLayout(parent)
        
        # Analysis settings
        analysis_group = QGroupBox("Analysis Settings")
        analysis_layout = QGridLayout(analysis_group)
        
        # Harmonic separation
        analysis_layout.addWidget(QLabel("Use Harmonic Separation:"), 0, 0)
        self.harmonic_cb = QCheckBox()
        self.harmonic_cb.setChecked(True)
        analysis_layout.addWidget(self.harmonic_cb, 0, 1)
        
        # Multi-feature analysis
        analysis_layout.addWidget(QLabel("Multi-Feature Analysis:"), 1, 0)
        self.multi_feature_cb = QCheckBox()
        self.multi_feature_cb.setChecked(True)
        analysis_layout.addWidget(self.multi_feature_cb, 1, 1)
        
        # Smoothing method
        analysis_layout.addWidget(QLabel("Smoothing Method:"), 2, 0)
        self.smoothing_combo = QComboBox()
        self.smoothing_combo.addItems(['adaptive', 'median', 'gaussian'])
        analysis_layout.addWidget(self.smoothing_combo, 2, 1)
        
        # Minimum chord duration
        analysis_layout.addWidget(QLabel("Min Chord Duration:"), 3, 0)
        self.min_duration_spin = QDoubleSpinBox()
        self.min_duration_spin.setRange(0.1, 5.0)
        self.min_duration_spin.setValue(0.3)
        self.min_duration_spin.setSingleStep(0.1)
        self.min_duration_spin.setSuffix(" sec")
        analysis_layout.addWidget(self.min_duration_spin, 3, 1)
        
        # Detection threshold
        analysis_layout.addWidget(QLabel("Detection Threshold:"), 4, 0)
        self.threshold_spin = QDoubleSpinBox()
        self.threshold_spin.setRange(0.1, 0.9)
        self.threshold_spin.setValue(0.25)
        self.threshold_spin.setSingleStep(0.05)
        analysis_layout.addWidget(self.threshold_spin, 4, 1)
        
        # Key-aware analysis
        analysis_layout.addWidget(QLabel("Key-Aware Analysis:"), 5, 0)
        self.key_aware_cb = QCheckBox()
        self.key_aware_cb.setChecked(True)
        analysis_layout.addWidget(self.key_aware_cb, 5, 1)
        
        layout.addWidget(analysis_group)
        
        # Export settings
        export_group = QGroupBox("Export Options")
        export_layout = QVBoxLayout(export_group)
        
        export_buttons = QHBoxLayout()
        
        self.btn_export_txt = QPushButton("Export as Text")
        self.btn_export_txt.clicked.connect(lambda: self.export_results('txt'))
        self.btn_export_txt.setEnabled(False)
        export_buttons.addWidget(self.btn_export_txt)
        
        self.btn_export_json = QPushButton("Export as JSON")
        self.btn_export_json.clicked.connect(lambda: self.export_results('json'))
        self.btn_export_json.setEnabled(False)
        export_buttons.addWidget(self.btn_export_json)
        
        export_layout.addLayout(export_buttons)
        layout.addWidget(export_group)
        layout.addStretch()
        
    def setup_results_tab(self, parent):
        """Setup results display"""
        layout = QVBoxLayout(parent)
        
        # Analysis summary
        summary_group = QGroupBox("Analysis Summary")
        summary_layout = QGridLayout(summary_group)
        
        self.total_chords_label = QLabel("Total Chords: 0")
        summary_layout.addWidget(self.total_chords_label, 0, 0)
        
        self.avg_duration_label = QLabel("Avg Duration: 0.0s")
        summary_layout.addWidget(self.avg_duration_label, 0, 1)
        
        self.key_confidence_label = QLabel("Key Confidence: 0%")
        summary_layout.addWidget(self.key_confidence_label, 1, 0)
        
        layout.addWidget(summary_group)
        
        # Detailed results
        self.results_text = QTextEdit()
        self.results_text.setReadOnly(True)
        self.results_text.setFont(QFont("Consolas", 10))
        layout.addWidget(self.results_text)
        
    def apply_enhanced_theme(self):
        """Apply enhanced visual theme"""
        # Set fonts
        title_font = QFont("Arial", 28, QFont.Weight.Bold)
        self.chord_label.setFont(title_font)
        
        subtitle_font = QFont("Arial", 14)
        self.key_label.setFont(subtitle_font)
        self.time_label.setFont(subtitle_font)
        
        # Enhanced chord display styling
        self.chord_label.setStyleSheet("""
            QLabel {
                border: 4px solid #2196F3;
                border-radius: 20px;
                padding: 40px;
                margin: 10px;
                background: qlineargradient(x1:0, y1:0, x2:0, y2:1,
                    stop:0 #E3F2FD, stop:0.5 #BBDEFB, stop:1 #90CAF9);
                color: #0D47A1;
                font-weight: bold;
            }
        """)
        
        self.key_label.setStyleSheet("""
            QLabel {
                background: qlineargradient(x1:0, y1:0, x2:1, y2:0,
                    stop:0 #FFF8E1, stop:1 #FFECB3);
                border: 2px solid #FF8F00;
                border-radius: 12px;
                padding: 12px;
                color: #E65100;
                font-weight: bold;
            }
        """)
        
        self.time_label.setStyleSheet("""
            QLabel {
                background: qlineargradient(x1:0, y1:0, x2:1, y2:0,
                    stop:0 #F3E5F5, stop:1 #E1BEE7);
                border: 2px solid #7B1FA2;
                border-radius: 12px;
                padding: 12px;
                color: #4A148C;
                font-weight: bold;
            }
        """)
        
        # Group box styling
        self.setStyleSheet("""
            QGroupBox {
                font-weight: bold;
                border: 2px solid #CCCCCC;
                border-radius: 8px;
                margin-top: 1ex;
                padding-top: 10px;
            }
            QGroupBox::title {
                subcontrol-origin: margin;
                left: 10px;
                padding: 0 10px 0 10px;
                color: #333333;
            }
            QPushButton {
                padding: 10px 20px;
                border-radius: 6px;
                background-color: #2196F3;
                color: white;
                border: none;
                font-weight: bold;
                font-size: 12px;
            }
            QPushButton:hover {
                background-color: #1976D2;
            }
            QPushButton:pressed {
                background-color: #0D47A1;
            }
            QPushButton:disabled {
                background-color: #BDBDBD;
                color: #757575;
            }
            QProgressBar {
                border: 2px solid #CCCCCC;
                border-radius: 5px;
                text-align: center;
                background-color: #F5F5F5;
                font-weight: bold;
            }
            QProgressBar::chunk {
                background: qlineargradient(x1:0, y1:0, x2:1, y2:0,
                    stop:0 #4CAF50, stop:1 #81C784);
                border-radius: 3px;
            }
        """)
        
    def load_settings(self):
        """Load user settings"""
        self.harmonic_cb.setChecked(self.settings.value("harmonic_separation", True, type=bool))
        self.multi_feature_cb.setChecked(self.settings.value("multi_features", True, type=bool))
        self.smoothing_combo.setCurrentText(self.settings.value("smoothing_method", "adaptive"))
        self.min_duration_spin.setValue(self.settings.value("min_duration", 0.3, type=float))
        self.threshold_spin.setValue(self.settings.value("threshold", 0.25, type=float))
        self.key_aware_cb.setChecked(self.settings.value("key_aware", True, type=bool))
        
    def save_settings(self):
        """Save user settings"""
        self.settings.setValue("harmonic_separation", self.harmonic_cb.isChecked())
        self.settings.setValue("multi_features", self.multi_feature_cb.isChecked())
        self.settings.setValue("smoothing_method", self.smoothing_combo.currentText())
        self.settings.setValue("min_duration", self.min_duration_spin.value())
        self.settings.setValue("threshold", self.threshold_spin.value())
        self.settings.setValue("key_aware", self.key_aware_cb.isChecked())
        
    def load_audio(self):
        """Load and analyze audio file with enhanced features"""
        file_path, _ = QFileDialog.getOpenFileName(
            self, "Select Audio File", "", 
            "Audio Files (*.wav *.mp3 *.flac *.aac *.m4a *.ogg *.opus *.wma)"
        )
        if not file_path:
            return
            
        # Save current settings
        self.save_settings()
            
        # Disable controls during analysis
        self.btn_load.setEnabled(False)
        self.btn_stop.setEnabled(False)
        self.chord_label.setText("Analyzing audio...")
        self.status_label.setText("Initializing analysis...")
        self.progress_bar.setValue(0)
        
        # Clear previous results
        self.chord_list.clear()
        self.results_text.clear()
        
        # Get current settings
        settings = {
            'use_harmonic_separation': self.harmonic_cb.isChecked(),
            'use_multi_features': self.multi_feature_cb.isChecked(),
            'smoothing_method': self.smoothing_combo.currentText(),
            'min_chord_duration': self.min_duration_spin.value(),
            'template_threshold': self.threshold_spin.value(),
            'key_aware': self.key_aware_cb.isChecked()
        }
        
        # Start analysis in separate thread
        self.analysis_thread = QThread()
        self.worker = ChordAnalysisWorker(file_path, settings)
        self.worker.moveToThread(self.analysis_thread)
        
        # Connect signals
        self.analysis_thread.started.connect(self.worker.run)
        self.worker.progress.connect(self.progress_bar.setValue)
        self.worker.status_update.connect(self.status_label.setText)
        self.worker.finished.connect(self.on_analysis_finished)
        self.worker.error.connect(self.on_analysis_error)
        
        self.analysis_thread.start()
        
        # Load audio for playback
        try:
            y, sr = librosa.load(file_path, sr=SR)
            self.audio_data = y
            self.audio_sr = sr
            self.audio_duration = len(y) / sr
            self.audio_file_path = file_path
        except Exception as e:
            QMessageBox.critical(self, "Error", f"Audio loading failed:\n{str(e)}")
            self.btn_load.setEnabled(True)
            
    def on_analysis_finished(self, chords, key_signature):
        """Handle completed analysis with enhanced features"""
        self.chords = chords
        self.key_signature = key_signature
        
        # Update key display
        self.key_label.setText(f"Key: {key_signature}")
        
        # Display analysis results
        self.display_enhanced_results()
        
        # Enable export buttons
        self.btn_export_txt.setEnabled(True)
        self.btn_export_json.setEnabled(True)
        
        # Start playback if audio loaded successfully
        if hasattr(self, 'audio_data'):
            try:
                sd.stop()
                sd.play(self.audio_data, samplerate=self.audio_sr)
                
                self.start_time = time.time()
                self.timer.start()
                
                self.status_label.setText("Playback started")
                self.btn_stop.setEnabled(True)
            except Exception as e:
                QMessageBox.warning(self, "Playback Error", 
                                  f"Could not start playback:\n{str(e)}")
        
        # Re-enable load button
        self.btn_load.setEnabled(True)
        
        # Clean up thread
        self.analysis_thread.quit()
        self.analysis_thread.wait()
        
    def on_analysis_error(self, error_msg):
        """Handle analysis error with better user feedback"""
        QMessageBox.critical(self, "Analysis Error", 
                           f"Chord detection failed:\n{error_msg}\n\n"
                           f"Please try:\n"
                           f"• Using a different audio file\n"
                           f"• Adjusting analysis settings\n"
                           f"• Checking file format compatibility")
        
        self.chord_label.setText("Analysis failed - Try another file")
        self.status_label.setText("Ready")
        self.btn_load.setEnabled(True)
        
        # Clean up thread
        if self.analysis_thread:
            self.analysis_thread.quit()
            self.analysis_thread.wait()
            
    def display_enhanced_results(self):
        """Display enhanced analysis results"""
        if not self.chords:
            return
            
        # Update summary statistics
        total_chords = len(self.chords)
        self.total_chords_label.setText(f"Total Chords: {total_chords}")
        
        # Calculate average chord duration
        if total_chords > 1:
            total_duration = self.chords[-1][0] - self.chords[0][0]
            avg_duration = total_duration / total_chords
            self.avg_duration_label.setText(f"Avg Duration: {avg_duration:.1f}s")
        
        # Detailed results text
        results_text = f"Enhanced Chord Analysis Results\n"
        results_text += f"{'=' * 50}\n\n"
        results_text += f"File: {getattr(self, 'audio_file_path', 'Unknown')}\n"
        results_text += f"Key Signature: {self.key_signature}\n"
        results_text += f"Total Segments: {total_chords}\n"
        results_text += f"Analysis Duration: {self.audio_duration:.1f} seconds\n\n"
        results_text += f"Chord Progression:\n"
        results_text += f"{'-' * 50}\n"
        results_text += f"{'Time':>8} {'Chord':>12} {'Duration':>10}\n"
        results_text += f"{'-' * 50}\n"
        
        for i, (time_stamp, chord) in enumerate(self.chords[:100]):  # Show first 100
            duration = ""
            if i < len(self.chords) - 1:
                dur_val = self.chords[i+1][0] - time_stamp
                duration = f"{dur_val:>8.1f}s"
            else:
                duration = f"{'---':>8}"
            
            results_text += f"{time_stamp:>8.1f}s {chord:>12} {duration}\n"
            
        if len(self.chords) > 100:
            results_text += f"\n... and {len(self.chords) - 100} more segments"
            
        # Add chord statistics
        chord_counts = {}
        for _, chord in self.chords:
            chord_counts[chord] = chord_counts.get(chord, 0) + 1
            
        most_common = sorted(chord_counts.items(), key=lambda x: x[1], reverse=True)[:10]
        
        results_text += f"\n\nMost Common Chords:\n"
        results_text += f"{'-' * 30}\n"
        for chord, count in most_common:
            percentage = (count / total_chords) * 100
            results_text += f"{chord:>12}: {count:>3} ({percentage:>5.1f}%)\n"
            
        self.results_text.setText(results_text)
        
    def stop_playback(self):
        """Stop audio playback with cleanup"""
        sd.stop()
        self.timer.stop()
        self.start_time = None
        self.btn_stop.setEnabled(False)
        self.chord_label.setText("Playback stopped")
        self.status_label.setText("Stopped")
        self.progress_bar.setValue(0)
        self.chord_list.clear()
        
    def update_display(self):
        """Update display during playback with enhanced features"""
        if self.start_time is None or not self.chords:
            return
            
        elapsed = time.time() - self.start_time
        
        # Update progress bar
        if self.audio_duration > 0:
            progress = min(100, int((elapsed / self.audio_duration) * 100))
            self.progress_bar.setValue(progress)
        
        # Update time display
        elapsed_str = f"{int(elapsed//60):02d}:{int(elapsed%60):02d}"
        total_str = f"{int(self.audio_duration//60):02d}:{int(self.audio_duration%60):02d}"
        self.time_label.setText(f"{elapsed_str} / {total_str}")
        
        # Find current chord
        current_chord = "♪"
        chord_color_class = "default"
        
        for i, (t, chord) in enumerate(self.chords):
            if t <= elapsed:
                if chord != "N.C.":
                    current_chord = chord
                    chord_color_class = self.get_enhanced_chord_color_class(chord)
                else:
                    current_chord = "♪"
                    chord_color_class = "default"
            else:
                break
        
        # Update chord display
        self.update_enhanced_chord_display(current_chord, chord_color_class)
        
        # Update recent chords list
        self.update_chord_list(elapsed)
        
        # Stop when audio ends
        if elapsed >= self.audio_duration:
            self.stop_playback()
            
    def get_enhanced_chord_color_class(self, chord):
        """Enhanced chord color classification"""
        chord_lower = chord.lower()
        
        if chord == "N.C." or chord == "♪":
            return "default"
        elif 'dim7' in chord_lower:
            return "diminished7"
        elif 'dim' in chord_lower:
            return "diminished"
        elif 'aug' in chord_lower:
            return "augmented"
        elif 'm7b5' in chord_lower:
            return "half_diminished"
        elif any(x in chord_lower for x in ['m7', 'm9', 'm11', 'm13']):
            return "minor_seventh"
        elif 'm' in chord_lower and not any(x in chord_lower for x in ['maj', 'major']):
            return "minor"
        elif any(x in chord_lower for x in ['7#', '7b', 'alt']):
            return "altered"
        elif any(x in chord_lower for x in ['7', '9', '11', '13']) and 'maj' not in chord_lower:
            return "dominant"
        elif any(x in chord_lower for x in ['maj7', 'm7', '9']):
            return "major_seventh"
        elif 'sus' in chord_lower:
            return "suspended"
        elif '6' in chord_lower:
            return "sixth"
        else:
            return "major"
            
    def update_enhanced_chord_display(self, chord, color_class):
        """Update chord display with enhanced styling"""
        styles = {
            "default": {
                "border": "#9E9E9E",
                "bg_start": "#F5F5F5",
                "bg_end": "#E0E0E0",
                "color": "#424242"
            },
            "major": {
                "border": "#4CAF50",
                "bg_start": "#E8F5E8",
                "bg_end": "#C8E6C9",
                "color": "#2E7D32"
            },
            "minor": {
                "border": "#2196F3",
                "bg_start": "#E3F2FD",
                "bg_end": "#BBDEFB",
                "color": "#1976D2"
            },
            "dominant": {
                "border": "#FF5722",
                "bg_start": "#FBE9E7",
                "bg_end": "#FFCCBC",
                "color": "#D84315"
            },
            "major_seventh": {
                "border": "#4CAF50",
                "bg_start": "#E8F5E8",
                "bg_end": "#A5D6A7",
                "color": "#1B5E20"
            },
            "minor_seventh": {
                "border": "#2196F3",
                "bg_start": "#E3F2FD",
                "bg_end": "#90CAF9",
                "color": "#0D47A1"
            },
            "diminished": {
                "border": "#9C27B0",
                "bg_start": "#F3E5F5",
                "bg_end": "#E1BEE7",
                "color": "#7B1FA2"
            },
            "diminished7": {
                "border": "#673AB7",
                "bg_start": "#EDE7F6",
                "bg_end": "#D1C4E9",
                "color": "#512DA8"
            },
            "half_diminished": {
                "border": "#795548",
                "bg_start": "#EFEBE9",
                "bg_end": "#D7CCC8",
                "color": "#5D4037"
            },
            "augmented": {
                "border": "#FF9800",
                "bg_start": "#FFF3E0",
                "bg_end": "#FFE0B2",
                "color": "#F57C00"
            },
            "suspended": {
                "border": "#607D8B",
                "bg_start": "#ECEFF1",
                "bg_end": "#CFD8DC",
                "color": "#455A64"
            },
            "altered": {
                "border": "#E91E63",
                "bg_start": "#FCE4EC",
                "bg_end": "#F8BBD9",
                "color": "#C2185B"
            },
            "sixth": {
                "border": "#00BCD4",
                "bg_start": "#E0F7FA",
                "bg_end": "#B2EBF2",
                "color": "#00838F"
            }
        }
        
        style_info = styles.get(color_class, styles["default"])
        
        full_style = f"""
            QLabel {{
                border: 4px solid {style_info['border']};
                border-radius: 20px;
                padding: 40px;
                margin: 10px;
                background: qlineargradient(x1:0, y1:0, x2:0, y2:1,
                    stop:0 {style_info['bg_start']}, stop:1 {style_info['bg_end']});
                color: {style_info['color']};
                font-size: 28px;
                font-weight: bold;
                text-align: center;
            }}
        """
        
        self.chord_label.setStyleSheet(full_style)
        self.chord_label.setText(chord)
        
    def update_chord_list(self, elapsed_time):
        """Update the recent chords list"""
        # Find chords within the last 10 seconds
        recent_chords = []
        for time_stamp, chord in self.chords:
            if time_stamp <= elapsed_time and elapsed_time - time_stamp <= 10:
                recent_chords.append((time_stamp, chord))
        
        # Update list widget
        self.chord_list.clear()
        for time_stamp, chord in recent_chords[-8:]:  # Show last 8 chords
            time_str = f"{int(time_stamp//60):02d}:{int(time_stamp%60):02d}"
            item_text = f"{time_str} - {chord}"
            self.chord_list.addItem(item_text)
        
        # Auto-scroll to bottom
        self.chord_list.scrollToBottom()
        
    def export_results(self, format_type):
        """Export analysis results in specified format"""
        if not self.chords:
            QMessageBox.warning(self, "No Data", "No chord analysis data to export.")
            return
            
        # Get save location
        if format_type == 'txt':
            file_path, _ = QFileDialog.getSaveFileName(
                self, "Export as Text", "chord_analysis.txt", "Text Files (*.txt)"
            )
        elif format_type == 'json':
            file_path, _ = QFileDialog.getSaveFileName(
                self, "Export as JSON", "chord_analysis.json", "JSON Files (*.json)"
            )
        else:
            return
            
        if not file_path:
            return
            
        try:
            if format_type == 'txt':
                self.export_as_text(file_path)
            elif format_type == 'json':
                self.export_as_json(file_path)
                
            QMessageBox.information(self, "Export Complete", 
                                  f"Analysis exported successfully to:\n{file_path}")
        except Exception as e:
            QMessageBox.critical(self, "Export Error", 
                               f"Failed to export analysis:\n{str(e)}")
            
    def export_as_text(self, file_path):
        """Export results as formatted text"""
        with open(file_path, 'w', encoding='utf-8') as f:
            f.write("Enhanced Chord Analysis Results\n")
            f.write("=" * 50 + "\n\n")
            f.write(f"Source File: {getattr(self, 'audio_file_path', 'Unknown')}\n")
            f.write(f"Key Signature: {self.key_signature}\n")
            f.write(f"Total Chord Segments: {len(self.chords)}\n")
            f.write(f"Analysis Duration: {self.audio_duration:.1f} seconds\n\n")
            
            f.write("Chord Progression:\n")
            f.write("-" * 50 + "\n")
            f.write(f"{'Time':>8} {'Chord':>15} {'Duration':>12}\n")
            f.write("-" * 50 + "\n")
            
            for i, (time_stamp, chord) in enumerate(self.chords):
                if i < len(self.chords) - 1:
                    duration = self.chords[i+1][0] - time_stamp
                    duration_str = f"{duration:8.1f}s"
                else:
                    duration_str = "---"
                    
                f.write(f"{time_stamp:8.1f}s {chord:>15} {duration_str:>12}\n")
                
    def export_as_json(self, file_path):
        """Export results as JSON"""
        export_data = {
            "metadata": {
                "source_file": getattr(self, 'audio_file_path', 'Unknown'),
                "key_signature": self.key_signature,
                "total_segments": len(self.chords),
                "duration": self.audio_duration,
                "export_timestamp": time.time(),
                "analysis_version": "4.0"
            },
            "chord_progression": [
                {
                    "time": time_stamp,
                    "chord": chord,
                    "duration": (self.chords[i+1][0] - time_stamp 
                               if i < len(self.chords) - 1 else None)
                }
                for i, (time_stamp, chord) in enumerate(self.chords)
            ]
        }
        
        with open(file_path, 'w', encoding='utf-8') as f:
            json.dump(export_data, f, indent=2, ensure_ascii=False)
            
    def closeEvent(self, event):
        """Handle application close"""
        # Stop playback
        sd.stop()
        
        # Save settings
        self.save_settings()
        
        # Clean up threads
        if self.analysis_thread and self.analysis_thread.isRunning():
            self.analysis_thread.quit()
            self.analysis_thread.wait(3000)  # Wait up to 3 seconds
            
        event.accept()

# --- Main Application ---
def main():
    """Enhanced main entry point"""
    import sys
    
    # Set up high DPI support
    if hasattr(Qt, 'AA_EnableHighDpiScaling'):
        QApplication.setAttribute(Qt.AA_EnableHighDpiScaling, True)
        QApplication.setAttribute(Qt.AA_UseHighDpiPixmaps, True)
    elif hasattr(Qt.ApplicationAttribute, 'AA_EnableHighDpiScaling'):
        QApplication.setAttribute(Qt.ApplicationAttribute.AA_EnableHighDpiScaling, True)
        QApplication.setAttribute(Qt.ApplicationAttribute.AA_UseHighDpiPixmaps, True)
    
    if hasattr(Qt, 'HighDpiScaleFactorRoundingPolicy'):
        QApplication.setHighDpiScaleFactorRoundingPolicy(
            Qt.HighDpiScaleFactorRoundingPolicy.PassThrough
        )
    
    app = QApplication(sys.argv)
    app.setApplicationName("Enhanced Chord Player Pro")
    app.setApplicationVersion("4.0")
    app.setOrganizationName("MusicTech Solutions")
    
    # Apply global application style
    app.setStyleSheet("""
        QWidget {
            font-family: 'Segoe UI', 'Arial', sans-serif;
            font-size: 11px;
        }
        QMainWindow {
            background-color: #FAFAFA;
        }
        QTabWidget::pane {
            border: 1px solid #CCCCCC;
            background-color: #FFFFFF;
            border-radius: 4px;
        }
        QTabBar::tab {
            background-color: #E0E0E0;
            padding: 10px 16px;
            margin-right: 2px;
            border-top-left-radius: 6px;
            border-top-right-radius: 6px;
            font-weight: bold;
        }
        QTabBar::tab:selected {
            background-color: #2196F3;
            color: white;
        }
        QTabBar::tab:hover:!selected {
            background-color: #F0F0F0;
        }
    """)
    
    # Create and show main window
    player = EnhancedChordPlayer()
    player.show()
    
    return app.exec()

if __name__ == "__main__":
    sys.exit(main())
        
