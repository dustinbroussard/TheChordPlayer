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
import mido
from mido import MidiFile, MidiTrack, Message
from scipy import signal
from scipy.ndimage import median_filter, gaussian_filter1d
from sklearn.preprocessing import normalize
from sklearn.cluster import KMeans
from PyQt6.QtWidgets import (
    QApplication, QWidget, QVBoxLayout, QPushButton, QLabel, QFileDialog, 
    QMessageBox, QHBoxLayout, QSlider, QCheckBox, QSpinBox, QProgressBar,
    QTabWidget, QTextEdit, QGroupBox, QGridLayout, QComboBox, QDoubleSpinBox,
    QSplitter, QListWidget, QFrame, QScrollArea, QToolTip
)
from PyQt6.QtCore import QTimer, Qt, QThread, pyqtSignal, QObject, QSettings
from PyQt6.QtGui import QFont, QPalette, QColor, QPixmap, QPainter, QAction

# --- Enhanced Configuration with Optimizations ---
CHORDS_CACHE_DIR = "chord_cache_v41"
CHORDS_CACHE_EXTENSION = ".json"
SR = 22050
FFT_SIZE = 16384
HOP_LENGTH = 512
NOTE_NAMES = ['C','C#','D','D#','E','F','F#','G','G#','A','A#','B']

# Enhanced chord definitions with better organization
CHORD_DEFS = {
    # Basic triads (high priority)
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
    
    # 7th chords (very common)
    '7': [0,4,7,10],       # Dominant 7th
    'M7': [0,4,7,11],      # Major 7th
    'maj7': [0,4,7,11],    # Alternative notation
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
    
    # Jazz extensions
    'maj7#11': [0,4,7,11,18%12], # Major 7th sharp 11
    'm11': [0,3,7,10,14%12,17%12], # Minor 11th
    'm13': [0,3,7,10,14%12,21%12], # Minor 13th
}

# Chord priority weights for detection order
CHORD_PRIORITY = {
    '': 1.0,      # Major - highest priority
    'm': 1.0,     # Minor - highest priority
    '7': 0.9,     # Dominant 7th
    'm7': 0.9,    # Minor 7th
    'M7': 0.8,    # Major 7th
    'maj7': 0.8,  # Major 7th alt
    'dim': 0.7,   # Diminished
    'sus4': 0.6,  # Suspended 4th
    'sus2': 0.6,  # Suspended 2nd
    '6': 0.5,     # Sixth chords
    'add9': 0.4,  # Add 9th
    # All others get default 0.3
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
    'bass': 0.8
}

# --- Performance Optimizations ---
class PerformanceMonitor:
    """Monitor and optimize performance"""
    def __init__(self):
        self.timings = {}
        
    def start_timer(self, name):
        self.timings[name] = time.time()
        
    def end_timer(self, name):
        if name in self.timings:
            duration = time.time() - self.timings[name]
            print(f"[PERF] {name}: {duration:.3f}s")
            return duration
        return 0

# Global performance monitor
perf_monitor = PerformanceMonitor()

# --- Enhanced Analysis Worker ---
class ChordAnalysisWorker(QObject):
    progress = pyqtSignal(int)
    finished = pyqtSignal(list, str, dict)  # chords, key_signature, analysis_stats
    error = pyqtSignal(str)
    status_update = pyqtSignal(str)
    
    def __init__(self, file_path, settings):
        super().__init__()
        self.file_path = file_path
        self.settings = settings
        
    def run(self):
        try:
            self.status_update.emit("Initializing enhanced analysis...")
            perf_monitor.start_timer("total_analysis")
            
            chords, key_sig, stats = detect_chords_enhanced(
                self.file_path, 
                progress_callback=self.progress.emit,
                status_callback=self.status_update.emit,
                **self.settings
            )
            
            total_time = perf_monitor.end_timer("total_analysis")
            stats['total_analysis_time'] = total_time
            
            self.finished.emit(chords, key_sig, stats)
        except Exception as e:
            self.error.emit(str(e))

# --- Optimized Detection Functions ---
def create_enhanced_chord_templates():
    """Create optimized chord templates with better caching"""
    cache_file = os.path.join(CHORDS_CACHE_DIR, "templates_v41.json")
    
    # Try to load from cache
    if os.path.exists(cache_file):
        try:
            with open(cache_file, 'r') as f:
                cached_data = json.load(f)
                templates = {k: np.array(v) for k, v in cached_data.items()}
                print(f"[CACHE] Loaded {len(templates)} chord templates from cache")
                return templates
        except:
            pass
    
    print("[COMPUTE] Computing chord templates...")
    perf_monitor.start_timer("template_creation")
    
    templates = {}
    
    # Create templates with priority-based ordering
    chord_items = list(CHORD_DEFS.items())
    chord_items.sort(key=lambda x: CHORD_PRIORITY.get(x[0], 0.3), reverse=True)
    
    for root_idx, note in enumerate(NOTE_NAMES):
        for suffix, intervals in chord_items:
            name = note + suffix
            vec = np.zeros(12, dtype=np.float32)  # Use float32 for memory efficiency
            
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
    
    # Cache templates for future use
    try:
        os.makedirs(CHORDS_CACHE_DIR, exist_ok=True)
        cache_data = {k: v.tolist() for k, v in templates.items()}
        with open(cache_file, 'w') as f:
            json.dump(cache_data, f)
        print(f"[CACHE] Saved {len(templates)} templates to cache")
    except Exception as e:
        print(f"[WARNING] Could not cache templates: {e}")
    
    perf_monitor.end_timer("template_creation")
    return templates

def extract_enhanced_features(y, sr=SR):
    """Extract optimized audio features for chord detection"""
    perf_monitor.start_timer("feature_extraction")
    
    features = {}
    
    # Add tempo estimation
    try:
        tempo, _ = librosa.beat.beat_track(y=y, sr=sr, hop_length=HOP_LENGTH)
        features['tempo'] = tempo
    except Exception as e:
        print(f"[WARNING] Tempo detection failed: {e}")
        features['tempo'] = 120.0  # Fallback BPM
    
    # Multi-resolution chroma with CQT (primary feature)
    features['chroma_cqt'] = librosa.feature.chroma_cqt(
        y=y, sr=sr, hop_length=HOP_LENGTH, fmin=librosa.note_to_hz('C1'),
        n_chroma=12, n_octaves=6
    )
    
    # STFT-based chroma for comparison (faster)
    features['chroma_stft'] = librosa.feature.chroma_stft(
        y=y, sr=sr, hop_length=HOP_LENGTH, n_fft=FFT_SIZE
    )
    
    # Harmonic-percussive separation (computationally expensive, make optional)
    try:
        y_harmonic, _ = librosa.effects.hpss(y, margin=(1.0, 5.0))
        features['chroma_harmonic'] = librosa.feature.chroma_cqt(
            y=y_harmonic, sr=sr, hop_length=HOP_LENGTH
        )
    except:
        # Fallback if HPSS fails
        features['chroma_harmonic'] = features['chroma_cqt']
    
    # Additional features for context
    features['spectral_centroid'] = librosa.feature.spectral_centroid(
        y=y, sr=sr, hop_length=HOP_LENGTH
    )
    
    features['rms'] = librosa.feature.rms(y=y, hop_length=HOP_LENGTH)
    
    perf_monitor.end_timer("feature_extraction")
    return features

def apply_intelligent_smoothing(chroma_sequence, features=None, method='adaptive'):
    """Optimized intelligent smoothing"""
    perf_monitor.start_timer("smoothing")
    
    if method == 'adaptive' and features is not None:
        # Use spectral flux to determine stability regions
        if 'spectral_centroid' in features:
            centroid = features['spectral_centroid'][0]
            centroid_diff = np.abs(np.diff(centroid))
            
            # Enhanced with tempo: Faster tempo -> less smoothing
            tempo = features.get('tempo', 120)
            sigma_base = 2.0 * (120 / max(tempo, 60))  # Scale sigma inversely
            stability = 1.0 / (1.0 + centroid_diff * 1000)
            sigma_adaptive = sigma_base * (0.5 + stability)
            
            # Apply gaussian smoothing with varying sigma
            smoothed = np.zeros_like(chroma_sequence)
            for i in range(12):
                # Use scipy's gaussian filter for efficiency
                smoothed[i, :] = gaussian_filter1d(
                    chroma_sequence[i, :], 
                    sigma=np.mean(sigma_adaptive)  # Use average sigma for simplicity
                )
        else:
            # Fallback to standard gaussian
            smoothed = gaussian_filter1d(chroma_sequence, sigma=2.0, axis=1)
    elif method == 'median':
        # Fast median filtering
        smoothed = np.zeros_like(chroma_sequence)
        for i in range(12):
            smoothed[i, :] = median_filter(chroma_sequence[i, :], size=5)
    else:
        # Simple gaussian smoothing (fastest)
        smoothed = gaussian_filter1d(chroma_sequence, sigma=1.5, axis=1)
    
    perf_monitor.end_timer("smoothing")
    return smoothed

def detect_key_enhanced(chroma_features, method='krumhansl'):
    """Enhanced key detection with confidence scoring"""
    perf_monitor.start_timer("key_detection")
    
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
    
    perf_monitor.end_timer("key_detection")
    
    if key_correlations:
        # Sort by correlation strength
        key_correlations.sort(key=lambda x: x[1], reverse=True)
        best_key = key_correlations[0]
        
        # Calculate confidence as difference between best and second-best
        confidence = best_key[1]
        if len(key_correlations) > 1:
            confidence = min(1.0, (best_key[1] - key_correlations[1][1]) * 2)
        
        return best_key[0], confidence
    else:
        return "Unknown", 0.0

def detect_chords_enhanced(path, progress_callback=None, status_callback=None, **settings):
    """Optimized chord detection with comprehensive analysis"""
    
    # Default settings with performance options
    default_settings = {
        'use_harmonic_separation': True,
        'smoothing_method': 'adaptive',
        'min_chord_duration': 0.3,
        'template_threshold': 0.25,
        'key_aware': True,
        'use_multi_features': True,
        'performance_mode': False,  # New setting for faster analysis
        'max_chord_types': 50       # Limit chord types for performance
    }
    
    # Merge with provided settings
    for key, value in default_settings.items():
        if key not in settings:
            settings[key] = value
    
    # Cache management
    os.makedirs(CHORDS_CACHE_DIR, exist_ok=True)
    fh = get_file_hash(path)
    cache_key = f"{fh}_enhanced_v41_{hash(str(sorted(settings.items())))}"
    cache_file = os.path.join(CHORDS_CACHE_DIR, cache_key + CHORDS_CACHE_EXTENSION)
    
    if os.path.exists(cache_file):
        try:
            with open(cache_file, 'r') as f:
                data = json.load(f)
                if status_callback:
                    status_callback("Loaded from cache!")
                return data['chords'], data.get('key_signature', 'Unknown'), data.get('stats', {})
        except:
            pass
    
    # Initialize analysis statistics
    analysis_stats = {
        'cache_hit': False,
        'audio_duration': 0,
        'total_frames': 0,
        'chord_changes': 0,
        'key_confidence': 0,
        'performance_mode': settings['performance_mode']
    }
    
    if progress_callback:
        progress_callback(5)
    if status_callback:
        status_callback("Loading audio file...")
    
    # Load audio
    perf_monitor.start_timer("audio_loading")
    y, sr = librosa.load(path, sr=SR)
    analysis_stats['audio_duration'] = len(y) / sr
    perf_monitor.end_timer("audio_loading")
    
    if progress_callback:
        progress_callback(15)
    if status_callback:
        status_callback("Extracting enhanced features...")
    
    # Extract enhanced features
    features = extract_enhanced_features(y, sr)
    
    if progress_callback:
        progress_callback(35)
    if status_callback:
        status_callback("Analyzing harmonic content...")
    
    # Choose primary chroma feature based on settings
    if settings['use_multi_features'] and not settings['performance_mode']:
        # Combine multiple chroma features with weighting
        chroma_combined = (
            0.4 * features['chroma_cqt'] +
            0.3 * features['chroma_harmonic'] +
            0.3 * features['chroma_stft']
        )
    else:
        # Use single feature for performance
        chroma_combined = features['chroma_cqt']
    
    # Apply intelligent smoothing
    smoothed_chroma = apply_intelligent_smoothing(
        chroma_combined, features, method=settings['smoothing_method']
    )
    
    analysis_stats['total_frames'] = smoothed_chroma.shape[1]
    
    if progress_callback:
        progress_callback(55)
    if status_callback:
        status_callback("Detecting key signature...")
    
    # Key detection
    key_signature = "Unknown"
    key_confidence = 0.0
    if settings['key_aware']:
        key_signature, key_confidence = detect_key_enhanced(smoothed_chroma)
        analysis_stats['key_confidence'] = key_confidence
    
    if progress_callback:
        progress_callback(65)
    if status_callback:
        status_callback("Building optimized chord templates...")
    
    # Normalize chroma
    chroma_norm = normalize(smoothed_chroma, axis=0, norm='l2')
    
    # Create enhanced templates
    templates = create_enhanced_chord_templates()
    
    # Limit templates for performance mode
    if settings['performance_mode']:
        # Keep only the most common chord types
        priority_chords = ['', 'm', '7', 'm7', 'M7', 'maj7', 'dim', 'sus4', 'sus2']
        filtered_templates = {}
        for note in NOTE_NAMES:
            for chord_type in priority_chords:
                chord_name = note + chord_type
                if chord_name in templates:
                    filtered_templates[chord_name] = templates[chord_name]
        templates = filtered_templates
        print(f"[PERF] Using {len(templates)} optimized templates")
    
    if progress_callback:
        progress_callback(75)
    if status_callback:
        status_callback("Performing chord matching...")
    
    # Enhanced chord detection with optimized matching
    perf_monitor.start_timer("chord_matching")
    chords = []
    times = librosa.frames_to_time(np.arange(chroma_norm.shape[1]), sr=sr, hop_length=HOP_LENGTH)
    
    # Pre-compute template matrix for vectorized operations
    template_names = list(templates.keys())
    template_matrix = np.array([templates[name] for name in template_names]).T
    
    frame_count = chroma_norm.shape[1]
    update_interval = max(10, frame_count // 50)  # More frequent updates
    
    # Vectorized chord detection for better performance
    batch_size = 100 if settings['performance_mode'] else 50
    
    for batch_start in range(0, frame_count, batch_size):
        batch_end = min(batch_start + batch_size, frame_count)
        batch_chroma = chroma_norm[:, batch_start:batch_end]
        
        # Vectorized similarity computation
        similarities = np.dot(template_matrix.T, batch_chroma)
        
        # Find best matches for each frame in batch
        best_indices = np.argmax(similarities, axis=0)
        best_scores = np.max(similarities, axis=0)
        
        for i, (frame_idx, best_idx, score) in enumerate(zip(
            range(batch_start, batch_end), best_indices, best_scores
        )):
            if frame_idx % update_interval == 0 and progress_callback:
                new_progress = min(95, 75 + int(20 * (frame_idx / frame_count)))
                progress_callback(new_progress)
            
            if score >= settings['template_threshold']:
                chord_name = template_names[best_idx]
            else:
                chord_name = "N.C."
            
            chords.append((times[frame_idx], chord_name))
    
    perf_monitor.end_timer("chord_matching")
    
    if progress_callback:
        progress_callback(90)
    if status_callback:
        status_callback("Post-processing results...")
    
    # Enhanced post-processing
    chords, complexity = post_process_enhanced(chords, settings['min_chord_duration'], key_signature)
    analysis_stats['chord_changes'] = len(chords)
    analysis_stats['harmonic_complexity'] = complexity
    analysis_stats['estimated_tempo'] = features.get('tempo', 120)
    
    # Cache results
    try:
        cache_data = {
            'chords': chords,
            'key_signature': key_signature,
            'stats': analysis_stats,
            'analysis_settings': settings
        }
        with open(cache_file, 'w') as f:
            json.dump(cache_data, f)
        print(f"[CACHE] Saved analysis results to cache")
    except Exception as e:
        print(f"[WARNING] Could not cache results: {e}")
    
    if progress_callback:
        progress_callback(100)
    if status_callback:
        status_callback("Analysis complete!")
    
    return chords, key_signature, analysis_stats

def post_process_enhanced(chords, min_duration=0.3, key_signature=None):
    """Enhanced post-processing with better musical intelligence"""
    if not chords:
        return []
    
    perf_monitor.start_timer("post_processing")
    
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
    
    # Advanced filtering with musical context
    filtered_chords = []
    for i, (t, chord) in enumerate(merged_chords):
        keep = True
        
        # Skip very short chords unless they're harmonically important
        if i > 0 and i < len(merged_chords)-1:
            prev_chord = merged_chords[i-1][1]
            next_chord = merged_chords[i+1][1]
            
            # Check for musical patterns
            if chord == prev_chord and chord == next_chord:
                keep = False  # Remove redundant repetitions
            elif chord != prev_chord and chord != next_chord:
                duration = merged_chords[i+1][0] - t if i < len(merged_chords)-1 else min_duration
                if duration < min_duration and not is_harmonically_important(chord, merged_chords, i):
                    keep = False
        
        if keep:
            filtered_chords.append((t, chord))
    
    perf_monitor.end_timer("post_processing")
    # Calculate harmonic complexity (unique chords ratio)
    complexity = len(set(c[1] for c in filtered_chords)) / max(len(filtered_chords), 1)
    return filtered_chords, complexity

def is_harmonically_important(chord, sequence, index):
    """Enhanced harmonic importance detection"""
    # Dominant chords are usually important
    if any(x in chord for x in ['7', 'V', 'dom']):
        return True
    
    # Diminished chords often serve as passing chords
    if 'dim' in chord:
        return True
    
    # Check for common progressions
    if index > 0 and index < len(sequence) - 1:
        prev_chord = sequence[index-1][1]
        next_chord = sequence[index+1][1]
        
        # V-I resolution patterns
        if '7' in chord and any(x in next_chord for x in ['', 'm']):
            return True
        
        # ii-V-I patterns
        if 'm7' in chord and '7' in next_chord:
            return True
    
    return False

def get_file_hash(path):
    """Generate SHA-256 hash of file for caching"""
    hasher = hashlib.sha256()
    with open(path, 'rb') as f:
        for chunk in iter(lambda: f.read(8192), b""):
            hasher.update(chunk)
    return hasher.hexdigest()

# --- Enhanced UI with Better UX ---
class EnhancedChordPlayer(QWidget):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("Enhanced Chord Player Pro v4.1")
        self.resize(1200, 800)
        
        # Initialize variables
        self.chords = []
        self.key_signature = "Unknown"
        self.analysis_stats = {}
        self.start_time = None
        self.audio_duration = 0
        self.analysis_thread = None
        self.worker = None
        self.settings = QSettings("MusicTech", "ChordPlayerPro")
        
        self.init_enhanced_ui()
        self.apply_modern_theme()
        self.load_settings()
        
    def init_enhanced_ui(self):
        """Initialize enhanced user interface with better organization"""
        main_layout = QVBoxLayout(self)
        
        # Create main splitter
        main_splitter = QSplitter(Qt.Orientation.Horizontal)
        main_layout.addWidget(main_splitter)
        
        # Left panel - Primary controls and display
        left_panel = self.create_left_panel()
        main_splitter.addWidget(left_panel)
        
        # Right panel - Advanced settings and analysis
        right_panel = self.create_right_panel()
        main_splitter.addWidget(right_panel)
        
        # Set splitter proportions (60% left, 40% right)
        main_splitter.setSizes([720, 480])
        
        # Status bar
        self.create_status_bar(main_layout)
        
        # Timer for updates
        self.timer = QTimer(self)
        self.timer.setInterval(50)
        self.timer.timeout.connect(self.update_display)
        
    def create_left_panel(self):
        """Create the main control and display panel"""
        left_widget = QWidget()
        left_layout = QVBoxLayout(left_widget)
        
        # File controls with enhanced styling
        file_group = QGroupBox("🎵 Audio File Controls")
        file_layout = QHBoxLayout(file_group)
        
        self.btn_load = QPushButton("📁 Load Audio File")
        self.btn_load.clicked.connect(self.load_audio)
        self.btn_load.setToolTip("Load an audio file for chord analysis\nSupported formats: WAV, MP3, FLAC, AAC, M4A, OGG")
        file_layout.addWidget(self.btn_load)
        
        self.btn_stop = QPushButton("⏹️ Stop Playback")
        self.btn_stop.clicked.connect(self.stop_playback)
        self.btn_stop.setEnabled(False)
        self.btn_stop.setToolTip("Stop audio playback and analysis")
        file_layout.addWidget(self.btn_stop)
        
        self.btn_pause = QPushButton("⏸️ Pause")
        self.btn_pause.clicked.connect(self.toggle_pause)
        self.btn_pause.setEnabled(False)
        self.btn_pause.setToolTip("Pause/Resume playback")
        file_layout.addWidget(self.btn_pause)
        
        left_layout.addWidget(file_group)
        
        # Progress and status with enhanced display
        progress_group = QGroupBox("📊 Analysis Progress")
        progress_layout = QVBoxLayout(progress_group)
        
        self.progress_bar = QProgressBar()
        self.progress_bar.setTextVisible(True)
        progress_layout.addWidget(self.progress_bar)
        
        # Status with performance indicators
        status_layout = QHBoxLayout()
        self.status_label = QLabel("Ready to analyze audio")
        self.performance_label = QLabel("")
        self.performance_label.setStyleSheet("color: #666; font-size: 10px;")
        status_layout.addWidget(self.status_label)
        status_layout.addStretch()
        status_layout.addWidget(self.performance_label)
        progress_layout.addLayout(status_layout)
        
        left_layout.addWidget(progress_group)
        
        # Enhanced current chord display
        chord_group = QGroupBox("🎼 Current Analysis")
        chord_layout = QVBoxLayout(chord_group)
        
        # Main chord display
        self.chord_label = QLabel("Load an audio file to begin")
        self.chord_label.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self.chord_label.setMinimumHeight(120)
        chord_layout.addWidget(self.chord_label)
        
        # Info row with key and time
        info_layout = QHBoxLayout()
        
        self.key_label = QLabel("Key: Unknown")
        self.key_label.setAlignment(Qt.AlignmentFlag.AlignCenter)
        info_layout.addWidget(self.key_label)
        
        self.time_label = QLabel("00:00 / 00:00")
        self.time_label.setAlignment(Qt.AlignmentFlag.AlignCenter)
        info_layout.addWidget(self.time_label)
        
        chord_layout.addLayout(info_layout)
        
        # Confidence and additional info
        detail_layout = QHBoxLayout()
        self.confidence_label = QLabel("Confidence: -")
        self.confidence_label.setAlignment(Qt.AlignmentFlag.AlignCenter)
        detail_layout.addWidget(self.confidence_label)
        
        self.tempo_label = QLabel("♩ = - BPM")
        self.tempo_label.setAlignment(Qt.AlignmentFlag.AlignCenter)
        detail_layout.addWidget(self.tempo_label)
        
        chord_layout.addLayout(detail_layout)
        left_layout.addWidget(chord_group)
        
        # Chord progression display with enhanced features
        progression_group = QGroupBox("🎶 Chord Progression")
        progression_layout = QVBoxLayout(progression_group)
        
        # Controls for progression view
        prog_controls = QHBoxLayout()
        self.show_times_cb = QCheckBox("Show Times")
        self.show_times_cb.setChecked(True)
        self.show_times_cb.stateChanged.connect(self.update_chord_list_display)
        prog_controls.addWidget(self.show_times_cb)
        
        self.max_chords_spin = QSpinBox()
        self.max_chords_spin.setRange(5, 50)
        self.max_chords_spin.setValue(12)
        self.max_chords_spin.setPrefix("Show last ")
        self.max_chords_spin.setSuffix(" chords")
        self.max_chords_spin.valueChanged.connect(self.update_chord_list_display)
        prog_controls.addWidget(self.max_chords_spin)
        
        prog_controls.addStretch()
        progression_layout.addLayout(prog_controls)
        
        self.chord_list = QListWidget()
        self.chord_list.setMaximumHeight(180)
        self.chord_list.setAlternatingRowColors(True)
        progression_layout.addWidget(self.chord_list)
        
        left_layout.addWidget(progression_group)
        
        return left_widget
        
    def create_right_panel(self):
        """Create the settings and analysis panel"""
        right_widget = QWidget()
        tabs = QTabWidget()
        right_widget_layout = QVBoxLayout(right_widget)
        right_widget_layout.addWidget(tabs)
        
        # Enhanced Settings tab
        settings_tab = QWidget()
        tabs.addTab(settings_tab, "⚙️ Settings")
        self.setup_enhanced_settings_tab(settings_tab)
        
        # Analysis Results tab
        results_tab = QWidget()
        tabs.addTab(results_tab, "📈 Analysis")
        self.setup_results_tab(results_tab)
        
        # Performance tab
        performance_tab = QWidget()
        tabs.addTab(performance_tab, "🚀 Performance")
        self.setup_performance_tab(performance_tab)
        
        return right_widget
        
    def create_status_bar(self, layout):
        """Create enhanced status bar"""
        status_frame = QFrame()
        status_frame.setFrameStyle(QFrame.Shape.StyledPanel)
        status_layout = QHBoxLayout(status_frame)
        status_layout.setContentsMargins(5, 2, 5, 2)
        
        self.file_info_label = QLabel("No file loaded")
        self.file_info_label.setStyleSheet("color: #666;")
        status_layout.addWidget(self.file_info_label)
        
        status_layout.addStretch()
        
        self.cache_status_label = QLabel("Cache: Ready")
        self.cache_status_label.setStyleSheet("color: #666;")
        status_layout.addWidget(self.cache_status_label)
        
        layout.addWidget(status_frame)
        
    def setup_enhanced_settings_tab(self, parent):
        """Setup enhanced settings interface with better organization"""
        scroll_area = QScrollArea()
        scroll_widget = QWidget()
        layout = QVBoxLayout(scroll_widget)
        
        # Analysis Quality Settings
        quality_group = QGroupBox("🎯 Analysis Quality")
        quality_layout = QGridLayout(quality_group)
        
        # Performance mode toggle
        quality_layout.addWidget(QLabel("Performance Mode:"), 0, 0)
        self.performance_cb = QCheckBox("Enable fast analysis")
        self.performance_cb.setToolTip("Faster analysis with reduced accuracy\nUses fewer chord types and simplified algorithms")
        quality_layout.addWidget(self.performance_cb, 0, 1)
        
        # Multi-feature analysis
        quality_layout.addWidget(QLabel("Multi-Feature Analysis:"), 1, 0)
        self.multi_feature_cb = QCheckBox("Use multiple algorithms")
        self.multi_feature_cb.setChecked(True)
        self.multi_feature_cb.setToolTip("Combines CQT, STFT, and harmonic features\nMore accurate but slower")
        quality_layout.addWidget(self.multi_feature_cb, 1, 1)
        
        # Harmonic separation
        quality_layout.addWidget(QLabel("Harmonic Separation:"), 2, 0)
        self.harmonic_cb = QCheckBox("Separate harmonic content")
        self.harmonic_cb.setChecked(True)
        self.harmonic_cb.setToolTip("Isolates harmonic content from percussive\nImproves chord detection accuracy")
        quality_layout.addWidget(self.harmonic_cb, 2, 1)
        
        layout.addWidget(quality_group)
        
        # Signal Processing Settings
        signal_group = QGroupBox("🎛️ Signal Processing")
        signal_layout = QGridLayout(signal_group)
        
        # Smoothing method
        signal_layout.addWidget(QLabel("Smoothing Method:"), 0, 0)
        self.smoothing_combo = QComboBox()
        self.smoothing_combo.addItems(['adaptive', 'gaussian', 'median'])
        self.smoothing_combo.setToolTip("adaptive: Context-aware smoothing\ngaussian: Standard smoothing\nmedian: Noise-resistant smoothing")
        signal_layout.addWidget(self.smoothing_combo, 0, 1)
        
        # Detection threshold
        signal_layout.addWidget(QLabel("Detection Threshold:"), 1, 0)
        self.threshold_spin = QDoubleSpinBox()
        self.threshold_spin.setRange(0.1, 0.9)
        self.threshold_spin.setValue(0.25)
        self.threshold_spin.setSingleStep(0.05)
        self.threshold_spin.setToolTip("Minimum confidence for chord detection\nLower = more chords detected\nHigher = only confident detections")
        signal_layout.addWidget(self.threshold_spin, 1, 1)
        
        layout.addWidget(signal_group)
        
        # Musical Analysis Settings
        musical_group = QGroupBox("🎼 Musical Analysis")
        musical_layout = QGridLayout(musical_group)
        
        # Minimum chord duration
        musical_layout.addWidget(QLabel("Min Chord Duration:"), 0, 0)
        self.min_duration_spin = QDoubleSpinBox()
        self.min_duration_spin.setRange(0.1, 5.0)
        self.min_duration_spin.setValue(0.3)
        self.min_duration_spin.setSingleStep(0.1)
        self.min_duration_spin.setSuffix(" sec")
        self.min_duration_spin.setToolTip("Minimum duration for a chord to be recognized\nFilters out very brief chord changes")
        musical_layout.addWidget(self.min_duration_spin, 0, 1)
        
        # Key-aware analysis
        musical_layout.addWidget(QLabel("Key-Aware Analysis:"), 1, 0)
        self.key_aware_cb = QCheckBox("Use key context")
        self.key_aware_cb.setChecked(True)
        self.key_aware_cb.setToolTip("Considers detected key for chord recognition\nImproves accuracy for tonal music")
        musical_layout.addWidget(self.key_aware_cb, 1, 1)
        
        # Maximum chord types
        musical_layout.addWidget(QLabel("Max Chord Types:"), 2, 0)
        self.max_chord_types_spin = QSpinBox()
        self.max_chord_types_spin.setRange(20, 200)
        self.max_chord_types_spin.setValue(100)
        self.max_chord_types_spin.setToolTip("Maximum number of chord types to consider\nLower values = faster analysis")
        musical_layout.addWidget(self.max_chord_types_spin, 2, 1)
        
        layout.addWidget(musical_group)
        
        # Export and Presets
        export_group = QGroupBox("💾 Export & Presets")
        export_layout = QVBoxLayout(export_group)
        
        # Preset buttons
        preset_layout = QHBoxLayout()
        
        self.btn_preset_fast = QPushButton("⚡ Fast")
        self.btn_preset_fast.clicked.connect(lambda: self.apply_preset('fast'))
        self.btn_preset_fast.setToolTip("Optimized for speed\nGood for real-time analysis")
        preset_layout.addWidget(self.btn_preset_fast)
        
        self.btn_preset_balanced = QPushButton("⚖️ Balanced")
        self.btn_preset_balanced.clicked.connect(lambda: self.apply_preset('balanced'))
        self.btn_preset_balanced.setToolTip("Balance of speed and accuracy\nRecommended for most users")
        preset_layout.addWidget(self.btn_preset_balanced)
        
        self.btn_preset_accurate = QPushButton("🎯 Accurate")
        self.btn_preset_accurate.clicked.connect(lambda: self.apply_preset('accurate'))
        self.btn_preset_accurate.setToolTip("Maximum accuracy\nBest for detailed analysis")
        preset_layout.addWidget(self.btn_preset_accurate)
        
        export_layout.addLayout(preset_layout)
        
        # Export buttons
        export_buttons = QHBoxLayout()
        
        self.btn_export_txt = QPushButton("📄 Export Text")
        self.btn_export_txt.clicked.connect(lambda: self.export_results('txt'))
        self.btn_export_txt.setEnabled(False)
        export_buttons.addWidget(self.btn_export_txt)
        
        self.btn_export_json = QPushButton("📊 Export JSON")
        self.btn_export_json.clicked.connect(lambda: self.export_results('json'))
        self.btn_export_json.setEnabled(False)
        export_buttons.addWidget(self.btn_export_json)
        
        self.btn_export_midi = QPushButton("🎹 Export MIDI")
        self.btn_export_midi.clicked.connect(lambda: self.export_results('midi'))
        self.btn_export_midi.setEnabled(False)
        self.btn_export_midi.setToolTip("Export chord progression as MIDI file")
        export_buttons.addWidget(self.btn_export_midi)
        
        export_layout.addLayout(export_buttons)
        layout.addWidget(export_group)
        
        layout.addStretch()
        scroll_area.setWidget(scroll_widget)
        
        main_layout = QVBoxLayout(parent)
        main_layout.addWidget(scroll_area)
        
    def setup_results_tab(self, parent):
        """Setup enhanced results display"""
        layout = QVBoxLayout(parent)
        
        # Analysis summary with enhanced metrics
        summary_group = QGroupBox("📊 Analysis Summary")
        summary_layout = QGridLayout(summary_group)
        
        self.total_chords_label = QLabel("Total Chords: 0")
        summary_layout.addWidget(self.total_chords_label, 0, 0)
        
        self.avg_duration_label = QLabel("Avg Duration: 0.0s")
        summary_layout.addWidget(self.avg_duration_label, 0, 1)
        
        self.key_confidence_label = QLabel("Key Confidence: 0%")
        summary_layout.addWidget(self.key_confidence_label, 1, 0)
        
        self.analysis_time_label = QLabel("Analysis Time: 0.0s")
        summary_layout.addWidget(self.analysis_time_label, 1, 1)
        
        self.cache_hit_label = QLabel("Cache Status: Miss")
        summary_layout.addWidget(self.cache_hit_label, 2, 0)
        
        self.complexity_label = QLabel("Harmonic Complexity: -")
        summary_layout.addWidget(self.complexity_label, 2, 1)
        
        layout.addWidget(summary_group)
        
        # Detailed results with tabs
        results_tabs = QTabWidget()
        
        # Chord progression tab
        prog_tab = QWidget()
        prog_layout = QVBoxLayout(prog_tab)
        
        self.results_text = QTextEdit()
        self.results_text.setReadOnly(True)
        self.results_text.setFont(QFont("Consolas", 10))
        prog_layout.addWidget(self.results_text)
        
        results_tabs.addTab(prog_tab, "Progression")
        
        # Statistics tab
        stats_tab = QWidget()
        stats_layout = QVBoxLayout(stats_tab)
        
        self.stats_text = QTextEdit()
        self.stats_text.setReadOnly(True)
        self.stats_text.setFont(QFont("Consolas", 10))
        stats_layout.addWidget(self.stats_text)
        
        results_tabs.addTab(stats_tab, "Statistics")
        
        layout.addWidget(results_tabs)
        
    def setup_performance_tab(self, parent):
        """Setup performance monitoring tab"""
        layout = QVBoxLayout(parent)
        
        # Performance metrics
        perf_group = QGroupBox("⚡ Performance Metrics")
        perf_layout = QGridLayout(perf_group)
        
        self.perf_total_label = QLabel("Total Analysis: -")
        perf_layout.addWidget(self.perf_total_label, 0, 0)
        
        self.perf_loading_label = QLabel("Audio Loading: -")
        perf_layout.addWidget(self.perf_loading_label, 0, 1)
        
        self.perf_features_label = QLabel("Feature Extraction: -")
        perf_layout.addWidget(self.perf_features_label, 1, 0)
        
        self.perf_matching_label = QLabel("Chord Matching: -")
        perf_layout.addWidget(self.perf_matching_label, 1, 1)
        
        layout.addWidget(perf_group)
        
        # Cache information
        cache_group = QGroupBox("💾 Cache Information")
        cache_layout = QVBoxLayout(cache_group)
        
        cache_info_layout = QHBoxLayout()
        self.btn_clear_cache = QPushButton("🗑️ Clear Cache")
        self.btn_clear_cache.clicked.connect(self.clear_cache)
        cache_info_layout.addWidget(self.btn_clear_cache)
        
        self.btn_cache_info = QPushButton("ℹ️ Cache Info")
        self.btn_cache_info.clicked.connect(self.show_cache_info)
        cache_info_layout.addWidget(self.btn_cache_info)
        
        cache_info_layout.addStretch()
        cache_layout.addLayout(cache_info_layout)
        
        self.cache_info_text = QTextEdit()
        self.cache_info_text.setReadOnly(True)
        self.cache_info_text.setMaximumHeight(150)
        cache_layout.addWidget(self.cache_info_text)
        
        layout.addWidget(cache_group)
        
        layout.addStretch()
        
    def apply_modern_theme(self):
        """Apply modern, professional theme"""
        # Enhanced color scheme
        self.setStyleSheet("""
            QWidget {
                font-family: 'Segoe UI', 'San Francisco', 'Helvetica Neue', Arial, sans-serif;
                font-size: 11px;
                background-color: #FAFAFA;
            }
            
            QGroupBox {
                font-weight: 600;
                border: 2px solid #E0E0E0;
                border-radius: 12px;
                margin-top: 12px;
                padding-top: 16px;
                background-color: #FFFFFF;
            }
            
            QGroupBox::title {
                subcontrol-origin: margin;
                left: 16px;
                padding: 0 12px 0 12px;
                color: #1976D2;
                font-size: 12px;
                font-weight: 700;
            }
            
            QPushButton {
                padding: 12px 24px;
                border-radius: 8px;
                background: qlineargradient(x1:0, y1:0, x2:0, y2:1,
                    stop:0 #2196F3, stop:1 #1976D2);
                color: white;
                border: none;
                font-weight: 600;
                font-size: 12px;
                min-height: 20px;
            }
            
            QPushButton:hover {
                background: qlineargradient(x1:0, y1:0, x2:0, y2:1,
                    stop:0 #1976D2, stop:1 #1565C0);
                transform: translateY(-1px);
            }
            
            QPushButton:pressed {
                background: qlineargradient(x1:0, y1:0, x2:0, y2:1,
                    stop:0 #1565C0, stop:1 #0D47A1);
            }
            
            QPushButton:disabled {
                background: #E0E0E0;
                color: #9E9E9E;
            }
            
            QProgressBar {
                border: 2px solid #E0E0E0;
                border-radius: 8px;
                text-align: center;
                background-color: #F5F5F5;
                font-weight: 600;
                font-size: 11px;
                min-height: 24px;
            }
            
            QProgressBar::chunk {
                background: qlineargradient(x1:0, y1:0, x2:1, y2:0,
                    stop:0 #4CAF50, stop:0.5 #66BB6A, stop:1 #81C784);
                border-radius: 6px;
            }
            
            QTabWidget::pane {
                border: 2px solid #E0E0E0;
                background-color: #FFFFFF;
                border-radius: 8px;
                margin-top: 4px;
            }
            
            QTabBar::tab {
                background: qlineargradient(x1:0, y1:0, x2:0, y2:1,
                    stop:0 #F5F5F5, stop:1 #E0E0E0);
                padding: 12px 20px;
                margin-right: 2px;
                border-top-left-radius: 8px;
                border-top-right-radius: 8px;
                font-weight: 600;
                color: #666666;
            }
            
            QTabBar::tab:selected {
                background: qlineargradient(x1:0, y1:0, x2:0, y2:1,
                    stop:0 #2196F3, stop:1 #1976D2);
                color: white;
            }
            
            QTabBar::tab:hover:!selected {
                background: qlineargradient(x1:0, y1:0, x2:0, y2:1,
                    stop:0 #EEEEEE, stop:1 #E0E0E0);
            }
            
            QListWidget {
                border: 2px solid #E0E0E0;
                border-radius: 8px;
                background-color: #FFFFFF;
                alternate-background-color: #F8F9FA;
                selection-background-color: #E3F2FD;
                font-family: 'Consolas', 'Monaco', monospace;
            }
            
            QListWidget::item {
                padding: 8px;
                border-bottom: 1px solid #F0F0F0;
            }
            
            QListWidget::item:selected {
                background-color: #2196F3;
                color: white;
            }
        """)
        
        # Enhanced chord display styling
        self.apply_chord_display_styling()
        
    def apply_preset(self, preset_name):
        """Apply predefined analysis presets (fast, balanced, accurate)."""
        preset_name = (preset_name or '').lower()
        if preset_name == 'fast':
            self.performance_cb.setChecked(True)
            self.multi_feature_cb.setChecked(False)
            self.harmonic_cb.setChecked(False)
            self.smoothing_combo.setCurrentText('gaussian')
            self.threshold_spin.setValue(0.30)
            self.min_duration_spin.setValue(0.50)
            self.max_chord_types_spin.setValue(30)
        elif preset_name == 'balanced':
            self.performance_cb.setChecked(False)
            self.multi_feature_cb.setChecked(True)
            self.harmonic_cb.setChecked(True)
            self.smoothing_combo.setCurrentText('adaptive')
            self.threshold_spin.setValue(0.25)
            self.min_duration_spin.setValue(0.30)
            self.max_chord_types_spin.setValue(80)
        elif preset_name == 'accurate':
            self.performance_cb.setChecked(False)
            self.multi_feature_cb.setChecked(True)
            self.harmonic_cb.setChecked(True)
            self.smoothing_combo.setCurrentText('adaptive')
            self.threshold_spin.setValue(0.20)
            self.min_duration_spin.setValue(0.20)
            self.max_chord_types_spin.setValue(150)
        else:
            QMessageBox.information(self, "Presets", f"Unknown preset: {preset_name}")
            return

        self.save_settings()
        QMessageBox.information(
            self, "Preset Applied",
            f"'{preset_name.title()}' preset has been applied.\n"
            f"Settings will be used for the next analysis."
        )

    def apply_chord_display_styling(self):
        """Apply enhanced styling to chord display elements"""
        # Main chord label styling
        chord_style = """
            QLabel {
                border: 4px solid #2196F3;
                border-radius: 20px;
                padding: 30px;
                margin: 8px;
                background: qlineargradient(x1:0, y1:0, x2:0, y2:1,
                    stop:0 #E3F2FD, stop:0.3 #BBDEFB, stop:0.7 #90CAF9, stop:1 #64B5F6);
                color: #0D47A1;
                font-size: 32px;
                font-weight: 700;
                text-align: center;
                letter-spacing: 2px;
            }
        """
        self.chord_label.setStyleSheet(chord_style)
        
        # Key signature styling
        key_style = """
            QLabel {
                background: qlineargradient(x1:0, y1:0, x2:1, y2:0,
                    stop:0 #FFF8E1, stop:0.5 #FFECB3, stop:1 #FFE082);
                border: 3px solid #FF8F00;
                border-radius: 15px;
                padding: 16px;
                color: #E65100;
                font-weight: 700;
                font-size: 14px;
            }
        """
        self.key_label.setStyleSheet(key_style)
        
        # Time display styling
        time_style = """
            QLabel {
                background: qlineargradient(x1:0, y1:0, x2:1, y2:0,
                    stop:0 #F3E5F5, stop:0.5 #E1BEE7, stop:1 #CE93D8);
                border: 3px solid #7B1FA2;
                border-radius: 15px;
                padding: 16px;
                color: #4A148C;
                font-weight: 700;
                font-size: 14px;
                font-family: 'Consolas', monospace;
            }
        """
        self.time_label.setStyleSheet(time_style)
        
        # Confidence styling
        confidence_style = """
            QLabel {
                background: qlineargradient(x1:0, y1:0, x2:1, y2:0,
                    stop:0 #E8F5E8, stop:1 #C8E6C9);
                border: 2px solid #4CAF50;
                border-radius: 12px;
                padding: 12px;
                color: #2E7D32;
                font-weight: 600;
                font-size: 12px;
            }
        """
        self.confidence_label.setStyleSheet(confidence_style)
        
        # Tempo styling
        tempo_style = """
            QLabel {
                background: qlineargradient(x1:0, y1:0, x2:1, y2:0,
                    stop:0 #FFF3E0, stop:1 #FFE0B2);
                border: 2px solid #FF9800;
                border-radius: 12px;
                padding: 12px;
                color: #F57C00;
                font-weight: 600;
                font-size: 12px;
            }
        """
        self.tempo_label.setStyleSheet(tempo_style)
        
    def apply_preset(self, preset_name):
        """Apply predefined analysis presets (fast, balanced, accurate)."""
        preset_name = (preset_name or '').lower()
        if preset_name == 'fast':
            self.performance_cb.setChecked(True)
            self.multi_feature_cb.setChecked(False)
            self.harmonic_cb.setChecked(False)
            self.smoothing_combo.setCurrentText('gaussian')
            self.threshold_spin.setValue(0.30)
            self.min_duration_spin.setValue(0.50)
            self.max_chord_types_spin.setValue(30)
        elif preset_name == 'balanced':
            self.performance_cb.setChecked(False)
            self.multi_feature_cb.setChecked(True)
            self.harmonic_cb.setChecked(True)
            self.smoothing_combo.setCurrentText('adaptive')
            self.threshold_spin.setValue(0.25)
            self.min_duration_spin.setValue(0.30)
            self.max_chord_types_spin.setValue(80)
        elif preset_name == 'accurate':
            self.performance_cb.setChecked(False)
            self.multi_feature_cb.setChecked(True)
            self.harmonic_cb.setChecked(True)
            self.smoothing_combo.setCurrentText('adaptive')
            self.threshold_spin.setValue(0.20)
            self.min_duration_spin.setValue(0.20)
            self.max_chord_types_spin.setValue(150)
        else:
            QMessageBox.information(self, "Presets", f"Unknown preset: {preset_name}")
            return

        self.save_settings()
        QMessageBox.information(
            self, "Preset Applied",
            f"'{preset_name.title()}' preset has been applied.\n"
            f"Settings will be used for the next analysis."
        )

    def load_settings(self):
        """Load user settings with new parameters"""
        self.performance_cb.setChecked(self.settings.value("performance_mode", False, type=bool))
        self.harmonic_cb.setChecked(self.settings.value("harmonic_separation", True, type=bool))
        self.multi_feature_cb.setChecked(self.settings.value("multi_features", True, type=bool))
        self.smoothing_combo.setCurrentText(self.settings.value("smoothing_method", "adaptive"))
        self.min_duration_spin.setValue(self.settings.value("min_duration", 0.3, type=float))
        self.threshold_spin.setValue(self.settings.value("threshold", 0.25, type=float))
        self.key_aware_cb.setChecked(self.settings.value("key_aware", True, type=bool))
        self.max_chord_types_spin.setValue(self.settings.value("max_chord_types", 100, type=int))
        
    def save_settings(self):
        """Save user settings with new parameters"""
        self.settings.setValue("performance_mode", self.performance_cb.isChecked())
        self.settings.setValue("harmonic_separation", self.harmonic_cb.isChecked())
        self.settings.setValue("multi_features", self.multi_feature_cb.isChecked())
        self.settings.setValue("smoothing_method", self.smoothing_combo.currentText())
        self.settings.setValue("min_duration", self.min_duration_spin.value())
        self.settings.setValue("threshold", self.threshold_spin.value())
        self.settings.setValue("key_aware", self.key_aware_cb.isChecked())
        self.settings.setValue("max_chord_types", self.max_chord_types_spin.value())
        
    def load_audio(self):
        """Enhanced audio loading with better error handling and feedback"""
        file_path, _ = QFileDialog.getOpenFileName(
            self, "Select Audio File", self.settings.value("last_directory", ""), 
            "Audio Files (*.wav *.mp3 *.flac *.aac *.m4a *.ogg *.opus *.wma);;All Files (*)"
        )
        if not file_path:
            return
            
        # Save directory for next time
        self.settings.setValue("last_directory", os.path.dirname(file_path))
        
        # Update file info
        file_name = os.path.basename(file_path)
        file_size = os.path.getsize(file_path) / (1024 * 1024)  # MB
        self.file_info_label.setText(f"File: {file_name} ({file_size:.1f} MB)")
        
        # Save current settings
        self.save_settings()
            
        # Disable controls during analysis
        self.btn_load.setEnabled(False)
        self.btn_stop.setEnabled(False)
        self.btn_pause.setEnabled(False)
        self.chord_label.setText("🔄 Analyzing audio...")
        self.status_label.setText("Initializing enhanced analysis...")
        self.progress_bar.setValue(0)
        
        # Clear previous results
        self.chord_list.clear()
        self.results_text.clear()
        self.stats_text.clear()
        
        # Get current settings
        settings = {
            'use_harmonic_separation': self.harmonic_cb.isChecked(),
            'use_multi_features': self.multi_feature_cb.isChecked(),
            'smoothing_method': self.smoothing_combo.currentText(),
            'min_chord_duration': self.min_duration_spin.value(),
            'template_threshold': self.threshold_spin.value(),
            'key_aware': self.key_aware_cb.isChecked(),
            'performance_mode': self.performance_cb.isChecked(),
            'max_chord_types': self.max_chord_types_spin.value()
        }
        
        # Update performance info
        mode_text = "Performance Mode" if settings['performance_mode'] else "Quality Mode"
        self.performance_label.setText(f"{mode_text} | {settings['smoothing_method'].title()} smoothing")
        
        # Start analysis in separate thread
        self.analysis_thread = QThread()
        self.worker = ChordAnalysisWorker(file_path, settings)
        self.worker.moveToThread(self.analysis_thread)
        
        # Connect signals
        self.analysis_thread.started.connect(self.worker.run)
        self.worker.progress.connect(self.update_progress)
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
            
            # Estimate tempo for display
            tempo, _ = librosa.beat.beat_track(y=y, sr=sr)
            self.estimated_tempo = tempo
            
        except Exception as e:
            QMessageBox.critical(self, "Audio Loading Error", 
                               f"Failed to load audio file:\n{str(e)}\n\n"
                               f"Please check:\n"
                               f"• File format is supported\n"
                               f"• File is not corrupted\n"
                               f"• Sufficient memory available")
            self.btn_load.setEnabled(True)
            
    def update_progress(self, value):
        """Enhanced progress update with time estimation"""
        self.progress_bar.setValue(value)
        if hasattr(self, 'analysis_start_time') and value > 0:
            elapsed = time.time() - self.analysis_start_time
            if value < 100:
                estimated_total = (elapsed / value) * 100
                remaining = estimated_total - elapsed
                self.progress_bar.setFormat(f"{value}% - ~{remaining:.0f}s remaining")
            else:
                self.progress_bar.setFormat(f"Complete in {elapsed:.1f}s")
        else:
            self.analysis_start_time = time.time()
            
    def on_analysis_finished(self, chords, key_signature, stats):
        """Enhanced analysis completion handler"""
        self.chords = chords
        self.key_signature = key_signature
        self.analysis_stats = stats
        
        # Update displays
        self.update_key_display(key_signature, stats.get('key_confidence', 0))
        self.display_enhanced_results()
        self.update_performance_display(stats)
        
        # Enable export buttons
        self.btn_export_txt.setEnabled(True)
        self.btn_export_json.setEnabled(True)
        self.btn_export_midi.setEnabled(True)
        
        # Update cache status
        cache_status = "Hit" if stats.get('cache_hit', False) else "Miss"
        self.cache_status_label.setText(f"Cache: {cache_status}")
        
        # Start playback if audio loaded successfully
        if hasattr(self, 'audio_data'):
            try:
                sd.stop()
                sd.play(self.audio_data, samplerate=self.audio_sr)
                
                self.start_time = time.time()
                self.is_paused = False
                self.timer.start()
                
                self.status_label.setText(f"✅ Analysis complete - Playback started ({len(chords)} chords detected)")
                self.btn_stop.setEnabled(True)
                self.btn_pause.setEnabled(True)
                
            except Exception as e:
                QMessageBox.warning(self, "Playback Error", 
                                  f"Analysis completed but playback failed:\n{str(e)}\n\n"
                                  f"You can still view and export the results.")
        
        # Re-enable load button
        self.btn_load.setEnabled(True)
        
        # Clean up thread
        self.analysis_thread.quit()
        self.analysis_thread.wait()
        
    def on_analysis_error(self, error_msg):
        """Enhanced error handling with better diagnostics"""
        error_details = f"Chord detection failed:\n{error_msg}\n\n"
        
        # Add diagnostic suggestions
        if "memory" in error_msg.lower():
            error_details += "💡 Suggestions:\n• Try enabling Performance Mode\n• Close other applications\n• Use a shorter audio file"
        elif "format" in error_msg.lower() or "codec" in error_msg.lower():
            error_details += "💡 Suggestions:\n• Convert to WAV or FLAC format\n• Check if file is corrupted\n• Try a different audio file"
        elif "permission" in error_msg.lower():
            error_details += "💡 Suggestions:\n• Check file permissions\n• Move file to a different location\n• Run as administrator"
        else:
            error_details += "💡 Suggestions:\n• Try Performance Mode for faster analysis\n• Reduce detection threshold\n• Check file format compatibility"
        
        QMessageBox.critical(self, "Analysis Error", error_details)
        
        self.chord_label.setText("❌ Analysis failed")
        self.status_label.setText("Ready - Try adjusting settings or different file")
        self.progress_bar.setValue(0)
        self.progress_bar.setFormat("")
        self.btn_load.setEnabled(True)
        
        # Clean up thread
        if self.analysis_thread:
            self.analysis_thread.quit()
            self.analysis_thread.wait()
            
    def update_key_display(self, key_signature, confidence):
        """Update key signature display with confidence"""
        confidence_pct = int(confidence * 100)
        self.key_label.setText(f"Key: {key_signature}")
        
        if confidence > 0:
            self.key_confidence_label.setText(f"Key Confidence: {confidence_pct}%")
            
            # Color-code confidence
            if confidence_pct >= 80:
                color = "#4CAF50"  # Green
            elif confidence_pct >= 60:
                color = "#FF9800"  # Orange
            else:
                color = "#F44336"  # Red
                
            self.key_confidence_label.setStyleSheet(f"""
                QLabel {{
                    background: qlineargradient(x1:0, y1:0, x2:1, y2:0,
                        stop:0 {color}30, stop:1 {color}20);
                    border: 2px solid {color};
                    border-radius: 12px;
                    padding: 8px;
                    color: {color};
                    font-weight: 600;
                }}
            """)
            
    def display_enhanced_results(self):
        """Display comprehensive analysis results"""
        if not self.chords:
            return
            
        # Update summary statistics
        total_chords = len(self.chords)
        self.total_chords_label.setText(f"Total Chords: {total_chords}")
        
        # Calculate metrics
        if total_chords > 1:
            durations = []
            for i in range(len(self.chords) - 1):
                duration = self.chords[i+1][0] - self.chords[i][0]
                durations.append(duration)
            
            avg_duration = np.mean(durations)
            self.avg_duration_label.setText(f"Avg Duration: {avg_duration:.1f}s")
            
            # Harmonic complexity (unique chords / total segments)
            unique_chords = len(set(chord for _, chord in self.chords if chord != "N.C."))
            complexity = (unique_chords / total_chords) * 100
            self.complexity_label.setText(f"Harmonic Complexity: {complexity:.1f}%")
        
        # Analysis time
        analysis_time = self.analysis_stats.get('total_analysis_time', 0)
        self.analysis_time_label.setText(f"Analysis Time: {analysis_time:.1f}s")
        
        # Cache status
        cache_hit = self.analysis_stats.get('cache_hit', False)
        cache_text = "Hit ✅" if cache_hit else "Miss ❌"
        self.cache_hit_label.setText(f"Cache Status: {cache_text}")
        
        # Detailed progression results
        self.update_progression_display()
        
        # Statistical analysis
        self.update_statistics_display()
        
    def update_progression_display(self):
        """Update detailed chord progression display"""
        results_text = f"🎼 Enhanced Chord Analysis Results\n"
        results_text += f"{'=' * 60}\n\n"
        results_text += f"📁 File: {os.path.basename(getattr(self, 'audio_file_path', 'Unknown'))}\n"
        results_text += f"🎵 Key Signature: {self.key_signature}\n"
        results_text += f"🎯 Total Segments: {len(self.chords)}\n"
        results_text += f"⏱️  Duration: {self.audio_duration:.1f} seconds\n"
        results_text += f"🚀 Performance Mode: {'✅' if self.analysis_stats.get('performance_mode', False) else '❌'}\n\n"
        
        results_text += f"🎶 Chord Progression:\n"
        results_text += f"{'-' * 60}\n"
        results_text += f"{'Time':>8} {'Chord':>15} {'Duration':>12} {'%':>8}\n"
        results_text += f"{'-' * 60}\n"
        
        total_duration = self.audio_duration
        for i, (time_stamp, chord) in enumerate(self.chords[:200]):  # Show first 200
            if i < len(self.chords) - 1:
                duration = self.chords[i+1][0] - time_stamp
                duration_str = f"{duration:8.1f}s"
                percentage = f"{(duration/total_duration)*100:5.1f}%"
            else:
                duration_str = "---"
                percentage = "---"
            
            # Add emoji for chord types
            chord_emoji = self.get_chord_emoji(chord)
            results_text += f"{time_stamp:8.1f}s {chord_emoji}{chord:>14} {duration_str:>12} {percentage:>8}\n"
            
        if len(self.chords) > 200:
            results_text += f"\n... and {len(self.chords) - 200} more segments\n"
            
        self.results_text.setText(results_text)
        
    def update_statistics_display(self):
        """Update statistical analysis display"""
        if not self.chords:
            return
            
        # Chord frequency analysis
        chord_counts = {}
        total_duration = 0
        chord_durations = {}
        
        for i, (time_stamp, chord) in enumerate(self.chords):
            chord_counts[chord] = chord_counts.get(chord, 0) + 1
            
            if i < len(self.chords) - 1:
                duration = self.chords[i+1][0] - time_stamp
                chord_durations[chord] = chord_durations.get(chord, 0) + duration
                total_duration += duration
        
        # Sort by frequency and duration
        most_common = sorted(chord_counts.items(), key=lambda x: x[1], reverse=True)[:15]
        longest_duration = sorted(chord_durations.items(), key=lambda x: x[1], reverse=True)[:10]
        
        stats_text = f"📊 Statistical Analysis\n"
        stats_text += f"{'=' * 50}\n\n"
        
        # Frequency analysis
        stats_text += f"🔢 Most Frequent Chords:\n"
        stats_text += f"{'-' * 40}\n"
        stats_text += f"{'Chord':>12} {'Count':>8} {'Freq %':>10}\n"
        stats_text += f"{'-' * 40}\n"
        for chord, count in most_common:
            percentage = (count / len(self.chords)) * 100
            emoji = self.get_chord_emoji(chord)
            stats_text += f"{emoji}{chord:>11} {count:>8} {percentage:>8.1f}%\n"
        
        # Duration analysis
        stats_text += f"\n⏱️  Longest Duration Chords:\n"
        stats_text += f"{'-' * 40}\n"
        stats_text += f"{'Chord':>12} {'Duration':>10} {'%':>8}\n"
        stats_text += f"{'-' * 40}\n"
        for chord, duration in longest_duration:
            percentage = (duration / total_duration) * 100 if total_duration > 0 else 0
            emoji = self.get_chord_emoji(chord)
            stats_text += f"{emoji}{chord:>11} {duration:>8.1f}s {percentage:>6.1f}%\n"
        
        # Key analysis
        if self.key_signature != "Unknown":
            stats_text += f"\n🎼 Key Analysis:\n"
            stats_text += f"{'-' * 30}\n"
            stats_text += f"Detected Key: {self.key_signature}\n"
            key_confidence = self.analysis_stats.get('key_confidence', 0) * 100
            stats_text += f"Confidence: {key_confidence:.1f}%\n"
            
            # Analyze chord-key relationships
            if 'Major' in self.key_signature:
                root = self.key_signature.replace(' Major', '')
                stats_text += f"Expected chords in {root} Major:\n"
                stats_text += f"  I: {root}, ii: {NOTE_NAMES[(NOTE_NAMES.index(root)+2)%12]}m, iii: {NOTE_NAMES[(NOTE_NAMES.index(root)+4)%12]}m\n"
                stats_text += f"  IV: {NOTE_NAMES[(NOTE_NAMES.index(root)+5)%12]}, V: {NOTE_NAMES[(NOTE_NAMES.index(root)+7)%12]}, vi: {NOTE_NAMES[(NOTE_NAMES.index(root)+9)%12]}m\n"
        
        # Performance metrics
        stats_text += f"\n🚀 Performance Metrics:\n"
        stats_text += f"{'-' * 30}\n"
        for key, value in self.analysis_stats.items():
            if 'time' in key and isinstance(value, (int, float)):
                stats_text += f"{key.replace('_', ' ').title()}: {value:.3f}s\n"
        
        self.stats_text.setText(stats_text)
        
    def get_chord_emoji(self, chord):
        """Get emoji representation for chord types"""
        if chord == "N.C.":
            return "🔇 "
        elif 'dim' in chord.lower():
            return "🔸 "
        elif 'aug' in chord.lower():
            return "🔹 "
        elif 'm' in chord and not any(x in chord for x in ['maj', 'M']):
            return "🎵 "
        elif any(x in chord for x in ['7', '9', '11', '13']):
            return "🎶 "
        elif 'sus' in chord.lower():
            return "🎭 "
        else:
            return "🎼 "
            
    def update_performance_display(self, stats):
        """Update performance metrics display"""
        self.perf_total_label.setText(f"Total Analysis: {stats.get('total_analysis_time', 0):.2f}s")
        
        # Individual timing components would need to be passed from the analysis
        # For now, show what we have
        cache_hit = stats.get('cache_hit', False)
        mode = "Performance" if stats.get('performance_mode', False) else "Quality"
        
        perf_text = f"Analysis completed in {mode} mode\n"
        perf_text += f"Cache status: {'Hit' if cache_hit else 'Miss'}\n"
        perf_text += f"Total frames processed: {stats.get('total_frames', 0)}\n"
        perf_text += f"Chord changes detected: {stats.get('chord_changes', 0)}\n"
        
        # Update cache info
        self.update_cache_info()
        
    def update_cache_info(self):
        """Update cache information display"""
        cache_dir = CHORDS_CACHE_DIR
        if os.path.exists(cache_dir):
            cache_files = [f for f in os.listdir(cache_dir) if f.endswith(CHORDS_CACHE_EXTENSION)]
            total_files = len(cache_files)
            
            # Calculate total cache size
            total_size = 0
            for file in cache_files:
                try:
                    file_path = os.path.join(cache_dir, file)
                    total_size += os.path.getsize(file_path)
                except:
                    pass
            
            cache_size_mb = total_size / (1024 * 1024)
            
            cache_info = f"Cache Directory: {cache_dir}\n"
            cache_info += f"Cached Analyses: {total_files}\n"
            cache_info += f"Total Size: {cache_size_mb:.1f} MB\n\n"
            cache_info += "Recent Cache Files:\n"
            
            # Show recent files
            recent_files = sorted(cache_files, key=lambda x: os.path.getmtime(os.path.join(cache_dir, x)), reverse=True)[:10]
            for file in recent_files:
                file_path = os.path.join(cache_dir, file)
                mod_time = time.ctime(os.path.getmtime(file_path))
                cache_info += f"• {file[:30]}... ({mod_time})\n"
            
            self.cache_info_text.setText(cache_info)
        else:
            self.cache_info_text.setText("No cache directory found.")
            
    def clear_cache(self):
        """Clear analysis cache"""
        reply = QMessageBox.question(self, "Clear Cache", 
                                   "Are you sure you want to clear the analysis cache?\n\n"
                                   "This will delete all cached chord analysis results, "
                                   "and future analyses will take longer until the cache is rebuilt.",
                                   QMessageBox.StandardButton.Yes | QMessageBox.StandardButton.No)
        
        if reply == QMessageBox.StandardButton.Yes:
            try:
                cache_dir = CHORDS_CACHE_DIR
                if os.path.exists(cache_dir):
                    import shutil
                    shutil.rmtree(cache_dir)
                    os.makedirs(cache_dir, exist_ok=True)
                    
                QMessageBox.information(self, "Cache Cleared", 
                                      "Analysis cache has been cleared successfully.")
                self.update_cache_info()
                self.cache_status_label.setText("Cache: Cleared")
                
            except Exception as e:
                QMessageBox.critical(self, "Error", f"Failed to clear cache:\n{str(e)}")
                
    def export_results(self, format_type):
        """Export analysis results in various formats"""
        if not self.chords:
            QMessageBox.warning(self, "Export Error", "No analysis results to export.")
            return
        
        file_path, _ = QFileDialog.getSaveFileName(
            self, 
            f"Export as {format_type.upper()}", 
            "", 
            f"{format_type.upper()} Files (*.{format_type})"
        )
        if not file_path:
            return
        
        try:
            if format_type == 'txt':
                with open(file_path, 'w') as f:
                    f.write(f"Key: {self.key_signature}\n")
                    f.write(f"Tempo: {self.analysis_stats.get('estimated_tempo', 0):.1f} BPM\n\n")
                    for t, chord in self.chords:
                        f.write(f"{t:.2f}s: {chord}\n")
                        
            elif format_type == 'json':
                data = {
                    'key_signature': self.key_signature,
                    'tempo': self.analysis_stats.get('estimated_tempo', 0),
                    'chords': self.chords,
                    'stats': self.analysis_stats
                }
                with open(file_path, 'w') as f:
                    json.dump(data, f, indent=4)
                    
            elif format_type == 'midi':
                mid = MidiFile()
                track = MidiTrack()
                mid.tracks.append(track)
                
                prev_time = 0
                ticks_per_beat = 480  # Standard resolution
                tempo = int(60000000 / self.analysis_stats.get('estimated_tempo', 120))  # Microsecs per beat
                track.append(Message('set_tempo', tempo=tempo, time=0))
                
                for t, chord in self.chords:
                    if chord == "N.C.": 
                        continue
                    
                    # Parse chord to MIDI notes (simplified: root + intervals)
                    root_note = NOTE_NAMES.index(chord[0]) + 60  # Middle C range
                    intervals = CHORD_DEFS.get(chord[1:], [0, 4, 7])  # Default major
                    notes = [root_note + i for i in intervals]
                    
                    delta_ticks = int((t - prev_time) * ticks_per_beat)
                    for note in notes:
                        track.append(Message('note_on', note=note, velocity=64, time=delta_ticks))
                    delta_ticks = 0  # Subsequent notes at same time
                    
                    # Hold for min duration (or until next)
                    hold_ticks = int(self.settings.value("min_duration", 0.3) * ticks_per_beat)
                    for note in notes:
                        track.append(Message('note_off', note=note, velocity=0, time=hold_ticks))
                    
                    prev_time = t
                
                mid.save(file_path)
            
            QMessageBox.information(self, "Export Success", f"Results exported to {file_path}")
        except Exception as e:
            QMessageBox.critical(self, "Export Error", f"Failed to export: {str(e)}")

    def show_cache_info(self):
        """Show detailed cache information"""
        self.update_cache_info()
        
    def toggle_pause(self):
        """Toggle playback pause/resume"""
        if not hasattr(self, 'is_paused'):
            return
            
        if self.is_paused:
            # Resume
            if hasattr(self, 'audio_data'):
                try:
                    # Calculate remaining audio
                    elapsed = time.time() - self.start_time if self.start_time else 0
                    remaining_samples = int((self.audio_duration - elapsed) * self.audio_sr)
                    if remaining_samples > 0:
                        start_sample = len(self.audio_data) - remaining_samples
                        sd.play(self.audio_data[start_sample:], samplerate=self.audio_sr)
                        
                    self.timer.start()
                    self.is_paused = False
                    self.btn_pause.setText("⏸️ Pause")
                    self.status_label.setText("Playback resumed")
                except Exception as e:
                    QMessageBox.warning(self, "Resume Error", f"Could not resume playback:\n{str(e)}")
        else:
            # Pause
            sd.stop()
            self.timer.stop()
            self.is_paused = True
            self.btn_pause.setText("▶️ Resume")
            self.status_label.setText("Playback paused")
            
    def stop_playback(self):
        """Enhanced stop playback with cleanup"""
        sd.stop()
        self.timer.stop()
        self.start_time = None
        self.is_paused = False
        
        self.btn_stop.setEnabled(False)
        self.btn_pause.setEnabled(False)
        self.btn_pause.setText("⏸️ Pause")
        
        self.chord_label.setText("⏹️ Playback stopped")
        self.status_label.setText("Stopped - Load another file or replay current")
        self.progress_bar.setValue(0)
        self.progress_bar.setFormat("")
        self.chord_list.clear()
        
        # Reset displays
        self.confidence_label.setText("Confidence: -")
        self.tempo_label.setText("♩ = - BPM")
        
    def update_display(self):
        """Enhanced display update during playback"""
        if self.start_time is None or not self.chords or self.is_paused:
            return
            
        elapsed = time.time() - self.start_time
        
        # Update progress bar
        if self.audio_duration > 0:
            progress = min(100, int((elapsed / self.audio_duration) * 100))
            self.progress_bar.setValue(progress)
            self.progress_bar.setFormat(f"{progress}% - Playing")
        
        # Update time display
        elapsed_str = f"{int(elapsed//60):02d}:{int(elapsed%60):02d}"
        total_str = f"{int(self.audio_duration//60):02d}:{int(self.audio_duration%60):02d}"
        self.time_label.setText(f"{elapsed_str} / {total_str}")
        
        # Find current chord with confidence estimation
        current_chord = "♪"
        chord_confidence = 0
        chord_color_class = "default"
        
        for i, (t, chord) in enumerate(self.chords):
            if t <= elapsed:
                if chord != "N.C.":
                    current_chord = chord
                    chord_color_class = self.get_enhanced_chord_color_class(chord)
                    
                    # Estimate confidence based on chord duration and stability
                    if i < len(self.chords) - 1:
                        chord_duration = self.chords[i+1][0] - t
                        chord_confidence = min(1.0, chord_duration / 2.0)  # Longer chords = higher confidence
                else:
                    current_chord = "♪"
                    chord_color_class = "default"
                    chord_confidence = 0
            else:
                break
        
        # Update chord display with enhanced styling
        self.update_enhanced_chord_display(current_chord, chord_color_class)
        
        # Update confidence display
        if chord_confidence > 0:
            self.confidence_label.setText(f"Confidence: {int(chord_confidence * 100)}%")
        else:
            self.confidence_label.setText("Confidence: -")
        
        # Update tempo display
        if hasattr(self, 'estimated_tempo'):
            self.tempo_label.setText(f"♩ ≈ {self.estimated_tempo:.0f} BPM")
        
        # Update recent chords list
        self.update_chord_list_display(elapsed)
        
        # Stop when audio ends
        if elapsed >= self.audio_duration:
            self.stop_playback()
            
    def get_enhanced_chord_color_class(self, chord):
        """Enhanced chord color classification with more categories"""
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
        elif any(x in chord_lower for x in ['13']):
            return "thirteenth"
        elif any(x in chord_lower for x in ['11']):
            return "eleventh"
        elif any(x in chord_lower for x in ['9']):
            return "ninth"
        elif any(x in chord_lower for x in ['7']) and 'maj' not in chord_lower:
            return "dominant"
        elif any(x in chord_lower for x in ['maj7', 'm7']):
            return "major_seventh"
        elif 'sus' in chord_lower:
            return "suspended"
        elif '6' in chord_lower:
            return "sixth"
        else:
            return "major"
            
    def update_enhanced_chord_display(self, chord, color_class):
        """Update chord display with comprehensive color scheme"""
        styles = {
            "default": {"border": "#9E9E9E", "bg_start": "#F5F5F5", "bg_end": "#E0E0E0", "color": "#424242"},
            "major": {"border": "#4CAF50", "bg_start": "#E8F5E8", "bg_end": "#C8E6C9", "color": "#2E7D32"},
            "minor": {"border": "#2196F3", "bg_start": "#E3F2FD", "bg_end": "#BBDEFB", "color": "#1976D2"},
            "dominant": {"border": "#FF5722", "bg_start": "#FBE9E7", "bg_end": "#FFCCBC", "color": "#D84315"},
            "major_seventh": {"border": "#4CAF50", "bg_start": "#E8F5E8", "bg_end": "#A5D6A7", "color": "#1B5E20"},
            "minor_seventh": {"border": "#2196F3", "bg_start": "#E3F2FD", "bg_end": "#90CAF9", "color": "#0D47A1"},
            "ninth": {"border": "#9C27B0", "bg_start": "#F3E5F5", "bg_end": "#E1BEE7", "color": "#7B1FA2"},
            "eleventh": {"border": "#673AB7", "bg_start": "#EDE7F6", "bg_end": "#D1C4E9", "color": "#512DA8"},
            "thirteenth": {"border": "#3F51B5", "bg_start": "#E8EAF6", "bg_end": "#C5CAE9", "color": "#303F9F"},
            "diminished": {"border": "#795548", "bg_start": "#EFEBE9", "bg_end": "#D7CCC8", "color": "#5D4037"},
            }
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
    QSplitter, QListWidget, QFrame, QScrollArea, QToolTip
)
from PyQt6.QtCore import QTimer, Qt, QThread, pyqtSignal, QObject, QSettings
from PyQt6.QtGui import QFont, QPalette, QColor, QPixmap, QPainter, QAction

# --- Enhanced Configuration with Optimizations ---
CHORDS_CACHE_DIR = "chord_cache_v41"
CHORDS_CACHE_EXTENSION = ".json"
SR = 22050
FFT_SIZE = 16384
HOP_LENGTH = 512
NOTE_NAMES = ['C','C#','D','D#','E','F','F#','G','G#','A','A#','B']

# Enhanced chord definitions with better organization
CHORD_DEFS = {
    # Basic triads (high priority)
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
    
    # 7th chords (very common)
    '7': [0,4,7,10],       # Dominant 7th
    'M7': [0,4,7,11],      # Major 7th
    'maj7': [0,4,7,11],    # Alternative notation
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
    
    # Jazz extensions
    'maj7#11': [0,4,7,11,18%12], # Major 7th sharp 11
    'm11': [0,3,7,10,14%12,17%12], # Minor 11th
    'm13': [0,3,7,10,14%12,21%12], # Minor 13th
}

# Chord priority weights for detection order
CHORD_PRIORITY = {
    '': 1.0,      # Major - highest priority
    'm': 1.0,     # Minor - highest priority
    '7': 0.9,     # Dominant 7th
    'm7': 0.9,    # Minor 7th
    'M7': 0.8,    # Major 7th
    'maj7': 0.8,  # Major 7th alt
    'dim': 0.7,   # Diminished
    'sus4': 0.6,  # Suspended 4th
    'sus2': 0.6,  # Suspended 2nd
    '6': 0.5,     # Sixth chords
    'add9': 0.4,  # Add 9th
    # All others get default 0.3
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
    'bass': 0.8
}

# --- Performance Optimizations ---
class PerformanceMonitor:
    """Monitor and optimize performance"""
    def __init__(self):
        self.timings = {}
        
    def start_timer(self, name):
        self.timings[name] = time.time()
        
    def end_timer(self, name):
        if name in self.timings:
            duration = time.time() - self.timings[name]
            print(f"[PERF] {name}: {duration:.3f}s")
            return duration
        return 0

# Global performance monitor
perf_monitor = PerformanceMonitor()

# --- Enhanced Analysis Worker ---
class ChordAnalysisWorker(QObject):
    progress = pyqtSignal(int)
    finished = pyqtSignal(list, str, dict)  # chords, key_signature, analysis_stats
    error = pyqtSignal(str)
    status_update = pyqtSignal(str)
    
    def __init__(self, file_path, settings):
        super().__init__()
        self.file_path = file_path
        self.settings = settings
        
    def run(self):
        try:
            self.status_update.emit("Initializing enhanced analysis...")
            perf_monitor.start_timer("total_analysis")
            
            chords, key_sig, stats = detect_chords_enhanced(
                self.file_path, 
                progress_callback=self.progress.emit,
                status_callback=self.status_update.emit,
                **self.settings
            )
            
            total_time = perf_monitor.end_timer("total_analysis")
            stats['total_analysis_time'] = total_time
            
            self.finished.emit(chords, key_sig, stats)
        except Exception as e:
            self.error.emit(str(e))

# --- Optimized Detection Functions ---
def create_enhanced_chord_templates():
    """Create optimized chord templates with better caching"""
    cache_file = os.path.join(CHORDS_CACHE_DIR, "templates_v41.json")
    
    # Try to load from cache
    if os.path.exists(cache_file):
        try:
            with open(cache_file, 'r') as f:
                cached_data = json.load(f)
                templates = {k: np.array(v) for k, v in cached_data.items()}
                print(f"[CACHE] Loaded {len(templates)} chord templates from cache")
                return templates
        except:
            pass
    
    print("[COMPUTE] Computing chord templates...")
    perf_monitor.start_timer("template_creation")
    
    templates = {}
    
    # Create templates with priority-based ordering
    chord_items = list(CHORD_DEFS.items())
    chord_items.sort(key=lambda x: CHORD_PRIORITY.get(x[0], 0.3), reverse=True)
    
    for root_idx, note in enumerate(NOTE_NAMES):
        for suffix, intervals in chord_items:
            name = note + suffix
            vec = np.zeros(12, dtype=np.float32)  # Use float32 for memory efficiency
            
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
    
    # Cache templates for future use
    try:
        os.makedirs(CHORDS_CACHE_DIR, exist_ok=True)
        cache_data = {k: v.tolist() for k, v in templates.items()}
        with open(cache_file, 'w') as f:
            json.dump(cache_data, f)
        print(f"[CACHE] Saved {len(templates)} templates to cache")
    except Exception as e:
        print(f"[WARNING] Could not cache templates: {e}")
    
    perf_monitor.end_timer("template_creation")
    return templates

def extract_enhanced_features(y, sr=SR):
    """Extract optimized audio features for chord detection"""
    perf_monitor.start_timer("feature_extraction")
    
    features = {}
    
    # Multi-resolution chroma with CQT (primary feature)
    features['chroma_cqt'] = librosa.feature.chroma_cqt(
        y=y, sr=sr, hop_length=HOP_LENGTH, fmin=librosa.note_to_hz('C1'),
        n_chroma=12, n_octaves=6
    )
    
    # STFT-based chroma for comparison (faster)
    features['chroma_stft'] = librosa.feature.chroma_stft(
        y=y, sr=sr, hop_length=HOP_LENGTH, n_fft=FFT_SIZE
    )
    
    # Harmonic-percussive separation (computationally expensive, make optional)
    try:
        y_harmonic, _ = librosa.effects.hpss(y, margin=(1.0, 5.0))
        features['chroma_harmonic'] = librosa.feature.chroma_cqt(
            y=y_harmonic, sr=sr, hop_length=HOP_LENGTH
        )
    except:
        # Fallback if HPSS fails
        features['chroma_harmonic'] = features['chroma_cqt']
    
    # Additional features for context
    features['spectral_centroid'] = librosa.feature.spectral_centroid(
        y=y, sr=sr, hop_length=HOP_LENGTH
    )
    
    features['rms'] = librosa.feature.rms(y=y, hop_length=HOP_LENGTH)
    
    perf_monitor.end_timer("feature_extraction")
    return features

def apply_intelligent_smoothing(chroma_sequence, features=None, method='adaptive'):
    """Optimized intelligent smoothing"""
    perf_monitor.start_timer("smoothing")
    
    if method == 'adaptive' and features is not None:
        # Use spectral flux to determine stability regions
        if 'spectral_centroid' in features:
            centroid = features['spectral_centroid'][0]
            centroid_diff = np.abs(np.diff(centroid))
            
            # Simplified adaptive sigma calculation
            sigma_base = 2.0
            stability = 1.0 / (1.0 + centroid_diff * 1000)
            sigma_adaptive = sigma_base * (0.5 + stability)
            
            # Apply gaussian smoothing with varying sigma
            smoothed = np.zeros_like(chroma_sequence)
            for i in range(12):
                # Use scipy's gaussian filter for efficiency
                smoothed[i, :] = gaussian_filter1d(
                    chroma_sequence[i, :], 
                    sigma=np.mean(sigma_adaptive)  # Use average sigma for simplicity
                )
        else:
            # Fallback to standard gaussian
            smoothed = gaussian_filter1d(chroma_sequence, sigma=2.0, axis=1)
    elif method == 'median':
        # Fast median filtering
        smoothed = np.zeros_like(chroma_sequence)
        for i in range(12):
            smoothed[i, :] = median_filter(chroma_sequence[i, :], size=5)
    else:
        # Simple gaussian smoothing (fastest)
        smoothed = gaussian_filter1d(chroma_sequence, sigma=1.5, axis=1)
    
    perf_monitor.end_timer("smoothing")
    return smoothed

def detect_key_enhanced(chroma_features, method='krumhansl'):
    """Enhanced key detection with confidence scoring"""
    perf_monitor.start_timer("key_detection")
    
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
    
    perf_monitor.end_timer("key_detection")
    
    if key_correlations:
        # Sort by correlation strength
        key_correlations.sort(key=lambda x: x[1], reverse=True)
        best_key = key_correlations[0]
        
        # Calculate confidence as difference between best and second-best
        confidence = best_key[1]
        if len(key_correlations) > 1:
            confidence = min(1.0, (best_key[1] - key_correlations[1][1]) * 2)
        
        return best_key[0], confidence
    else:
        return "Unknown", 0.0

def detect_chords_enhanced(path, progress_callback=None, status_callback=None, **settings):
    """Optimized chord detection with comprehensive analysis"""
    
    # Default settings with performance options
    default_settings = {
        'use_harmonic_separation': True,
        'smoothing_method': 'adaptive',
        'min_chord_duration': 0.3,
        'template_threshold': 0.25,
        'key_aware': True,
        'use_multi_features': True,
        'performance_mode': False,  # New setting for faster analysis
        'max_chord_types': 50       # Limit chord types for performance
    }
    
    # Merge with provided settings
    for key, value in default_settings.items():
        if key not in settings:
            settings[key] = value
    
    # Cache management
    os.makedirs(CHORDS_CACHE_DIR, exist_ok=True)
    fh = get_file_hash(path)
    cache_key = f"{fh}_enhanced_v41_{hash(str(sorted(settings.items())))}"
    cache_file = os.path.join(CHORDS_CACHE_DIR, cache_key + CHORDS_CACHE_EXTENSION)
    
    if os.path.exists(cache_file):
        try:
            with open(cache_file, 'r') as f:
                data = json.load(f)
                if status_callback:
                    status_callback("Loaded from cache!")
                return data['chords'], data.get('key_signature', 'Unknown'), data.get('stats', {})
        except:
            pass
    
    # Initialize analysis statistics
    analysis_stats = {
        'cache_hit': False,
        'audio_duration': 0,
        'total_frames': 0,
        'chord_changes': 0,
        'key_confidence': 0,
        'performance_mode': settings['performance_mode']
    }
    
    if progress_callback:
        progress_callback(5)
    if status_callback:
        status_callback("Loading audio file...")
    
    # Load audio
    perf_monitor.start_timer("audio_loading")
    y, sr = librosa.load(path, sr=SR)
    analysis_stats['audio_duration'] = len(y) / sr
    perf_monitor.end_timer("audio_loading")
    
    if progress_callback:
        progress_callback(15)
    if status_callback:
        status_callback("Extracting enhanced features...")
    
    # Extract enhanced features
    features = extract_enhanced_features(y, sr)
    
    if progress_callback:
        progress_callback(35)
    if status_callback:
        status_callback("Analyzing harmonic content...")
    
    # Choose primary chroma feature based on settings
    if settings['use_multi_features'] and not settings['performance_mode']:
        # Combine multiple chroma features with weighting
        chroma_combined = (
            0.4 * features['chroma_cqt'] +
            0.3 * features['chroma_harmonic'] +
            0.3 * features['chroma_stft']
        )
    else:
        # Use single feature for performance
        chroma_combined = features['chroma_cqt']
    
    # Apply intelligent smoothing
    smoothed_chroma = apply_intelligent_smoothing(
        chroma_combined, features, method=settings['smoothing_method']
    )
    
    analysis_stats['total_frames'] = smoothed_chroma.shape[1]
    
    if progress_callback:
        progress_callback(55)
    if status_callback:
        status_callback("Detecting key signature...")
    
    # Key detection
    key_signature = "Unknown"
    key_confidence = 0.0
    if settings['key_aware']:
        key_signature, key_confidence = detect_key_enhanced(smoothed_chroma)
        analysis_stats['key_confidence'] = key_confidence
    
    if progress_callback:
        progress_callback(65)
    if status_callback:
        status_callback("Building optimized chord templates...")
    
    # Normalize chroma
    chroma_norm = normalize(smoothed_chroma, axis=0, norm='l2')
    
    # Create enhanced templates
    templates = create_enhanced_chord_templates()
    
    # Limit templates for performance mode
    if settings['performance_mode']:
        # Keep only the most common chord types
        priority_chords = ['', 'm', '7', 'm7', 'M7', 'maj7', 'dim', 'sus4', 'sus2']
        filtered_templates = {}
        for note in NOTE_NAMES:
            for chord_type in priority_chords:
                chord_name = note + chord_type
                if chord_name in templates:
                    filtered_templates[chord_name] = templates[chord_name]
        templates = filtered_templates
        print(f"[PERF] Using {len(templates)} optimized templates")
    
    if progress_callback:
        progress_callback(75)
    if status_callback:
        status_callback("Performing chord matching...")
    
    # Enhanced chord detection with optimized matching
    perf_monitor.start_timer("chord_matching")
    chords = []
    times = librosa.frames_to_time(np.arange(chroma_norm.shape[1]), sr=sr, hop_length=HOP_LENGTH)
    
    # Pre-compute template matrix for vectorized operations
    template_names = list(templates.keys())
    template_matrix = np.array([templates[name] for name in template_names]).T
    
    frame_count = chroma_norm.shape[1]
    update_interval = max(10, frame_count // 50)  # More frequent updates
    
    # Vectorized chord detection for better performance
    batch_size = 100 if settings['performance_mode'] else 50
    
    for batch_start in range(0, frame_count, batch_size):
        batch_end = min(batch_start + batch_size, frame_count)
        batch_chroma = chroma_norm[:, batch_start:batch_end]
        
        # Vectorized similarity computation
        similarities = np.dot(template_matrix.T, batch_chroma)
        
        # Find best matches for each frame in batch
        best_indices = np.argmax(similarities, axis=0)
        best_scores = np.max(similarities, axis=0)
        
        for i, (frame_idx, best_idx, score) in enumerate(zip(
            range(batch_start, batch_end), best_indices, best_scores
        )):
            if frame_idx % update_interval == 0 and progress_callback:
                new_progress = min(95, 75 + int(20 * (frame_idx / frame_count)))
                progress_callback(new_progress)
            
            if score >= settings['template_threshold']:
                chord_name = template_names[best_idx]
            else:
                chord_name = "N.C."
            
            chords.append((times[frame_idx], chord_name))
    
    perf_monitor.end_timer("chord_matching")
    
    if progress_callback:
        progress_callback(90)
    if status_callback:
        status_callback("Post-processing results...")
    
    # Enhanced post-processing
    chords = post_process_enhanced(chords, settings['min_chord_duration'], key_signature)
    analysis_stats['chord_changes'] = len(chords)
    
    # Cache results
    try:
        cache_data = {
            'chords': chords,
            'key_signature': key_signature,
            'stats': analysis_stats,
            'analysis_settings': settings
        }
        with open(cache_file, 'w') as f:
            json.dump(cache_data, f)
        print(f"[CACHE] Saved analysis results to cache")
    except Exception as e:
        print(f"[WARNING] Could not cache results: {e}")
    
    if progress_callback:
        progress_callback(100)
    if status_callback:
        status_callback("Analysis complete!")
    
    return chords, key_signature, analysis_stats

def post_process_enhanced(chords, min_duration=0.3, key_signature=None):
    """Enhanced post-processing with better musical intelligence"""
    if not chords:
        return []
    
    perf_monitor.start_timer("post_processing")
    
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
    
    # Advanced filtering with musical context
    filtered_chords = []
    for i, (t, chord) in enumerate(merged_chords):
        keep = True
        
        # Skip very short chords unless they're harmonically important
        if i > 0 and i < len(merged_chords)-1:
            prev_chord = merged_chords[i-1][1]
            next_chord = merged_chords[i+1][1]
            
            # Check for musical patterns
            if chord == prev_chord and chord == next_chord:
                keep = False  # Remove redundant repetitions
            elif chord != prev_chord and chord != next_chord:
                duration = merged_chords[i+1][0] - t if i < len(merged_chords)-1 else min_duration
                if duration < min_duration and not is_harmonically_important(chord, merged_chords, i):
                    keep = False
        
        if keep:
            filtered_chords.append((t, chord))
    
    perf_monitor.end_timer("post_processing")
    return filtered_chords

def is_harmonically_important(chord, sequence, index):
    """Enhanced harmonic importance detection"""
    # Dominant chords are usually important
    if any(x in chord for x in ['7', 'V', 'dom']):
        return True
    
    # Diminished chords often serve as passing chords
    if 'dim' in chord:
        return True
    
    # Check for common progressions
    if index > 0 and index < len(sequence) - 1:
        prev_chord = sequence[index-1][1]
        next_chord = sequence[index+1][1]
        
        # V-I resolution patterns
        if '7' in chord and any(x in next_chord for x in ['', 'm']):
            return True
        
        # ii-V-I patterns
        if 'm7' in chord and '7' in next_chord:
            return True
    
    return False

def get_file_hash(path):
    """Generate SHA-256 hash of file for caching"""
    hasher = hashlib.sha256()
    with open(path, 'rb') as f:
        for chunk in iter(lambda: f.read(8192), b""):
            hasher.update(chunk)
    return hasher.hexdigest()

# --- Enhanced UI with Better UX ---
class EnhancedChordPlayer(QWidget):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("Enhanced Chord Player Pro v4.1")
        self.resize(1200, 800)
        
        # Initialize variables
        self.chords = []
        self.key_signature = "Unknown"
        self.analysis_stats = {}
        self.start_time = None
        self.audio_duration = 0
        self.analysis_thread = None
        self.worker = None
        self.settings = QSettings("MusicTech", "ChordPlayerPro")
        
        self.init_enhanced_ui()
        self.apply_modern_theme()
        self.load_settings()
        
    def init_enhanced_ui(self):
        """Initialize enhanced user interface with better organization"""
        main_layout = QVBoxLayout(self)
        
        # Create main splitter
        main_splitter = QSplitter(Qt.Orientation.Horizontal)
        main_layout.addWidget(main_splitter)
        
        # Left panel - Primary controls and display
        left_panel = self.create_left_panel()
        main_splitter.addWidget(left_panel)
        
        # Right panel - Advanced settings and analysis
        right_panel = self.create_right_panel()
        main_splitter.addWidget(right_panel)
        
        # Set splitter proportions (60% left, 40% right)
        main_splitter.setSizes([720, 480])
        
        # Status bar
        self.create_status_bar(main_layout)
        
        # Timer for updates
        self.timer = QTimer(self)
        self.timer.setInterval(50)
        self.timer.timeout.connect(self.update_display)
        
    def create_left_panel(self):
        """Create the main control and display panel"""
        left_widget = QWidget()
        left_layout = QVBoxLayout(left_widget)
        
        # File controls with enhanced styling
        file_group = QGroupBox("🎵 Audio File Controls")
        file_layout = QHBoxLayout(file_group)
        
        self.btn_load = QPushButton("📁 Load Audio File")
        self.btn_load.clicked.connect(self.load_audio)
        self.btn_load.setToolTip("Load an audio file for chord analysis\nSupported formats: WAV, MP3, FLAC, AAC, M4A, OGG")
        file_layout.addWidget(self.btn_load)
        
        self.btn_stop = QPushButton("⏹️ Stop Playback")
        self.btn_stop.clicked.connect(self.stop_playback)
        self.btn_stop.setEnabled(False)
        self.btn_stop.setToolTip("Stop audio playback and analysis")
        file_layout.addWidget(self.btn_stop)
        
        self.btn_pause = QPushButton("⏸️ Pause")
        self.btn_pause.clicked.connect(self.toggle_pause)
        self.btn_pause.setEnabled(False)
        self.btn_pause.setToolTip("Pause/Resume playback")
        file_layout.addWidget(self.btn_pause)
        
        left_layout.addWidget(file_group)
        
        # Progress and status with enhanced display
        progress_group = QGroupBox("📊 Analysis Progress")
        progress_layout = QVBoxLayout(progress_group)
        
        self.progress_bar = QProgressBar()
        self.progress_bar.setTextVisible(True)
        progress_layout.addWidget(self.progress_bar)
        
        # Status with performance indicators
        status_layout = QHBoxLayout()
        self.status_label = QLabel("Ready to analyze audio")
        self.performance_label = QLabel("")
        self.performance_label.setStyleSheet("color: #666; font-size: 10px;")
        status_layout.addWidget(self.status_label)
        status_layout.addStretch()
        status_layout.addWidget(self.performance_label)
        progress_layout.addLayout(status_layout)
        
        left_layout.addWidget(progress_group)
        
        # Enhanced current chord display
        chord_group = QGroupBox("🎼 Current Analysis")
        chord_layout = QVBoxLayout(chord_group)
        
        # Main chord display
        self.chord_label = QLabel("Load an audio file to begin")
        self.chord_label.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self.chord_label.setMinimumHeight(120)
        chord_layout.addWidget(self.chord_label)
        
        # Info row with key and time
        info_layout = QHBoxLayout()
        
        self.key_label = QLabel("Key: Unknown")
        self.key_label.setAlignment(Qt.AlignmentFlag.AlignCenter)
        info_layout.addWidget(self.key_label)
        
        self.time_label = QLabel("00:00 / 00:00")
        self.time_label.setAlignment(Qt.AlignmentFlag.AlignCenter)
        info_layout.addWidget(self.time_label)
        
        chord_layout.addLayout(info_layout)
        
        # Confidence and additional info
        detail_layout = QHBoxLayout()
        self.confidence_label = QLabel("Confidence: -")
        self.confidence_label.setAlignment(Qt.AlignmentFlag.AlignCenter)
        detail_layout.addWidget(self.confidence_label)
        
        self.tempo_label = QLabel("♩ = - BPM")
        self.tempo_label.setAlignment(Qt.AlignmentFlag.AlignCenter)
        detail_layout.addWidget(self.tempo_label)
        
        chord_layout.addLayout(detail_layout)
        left_layout.addWidget(chord_group)
        
        # Chord progression display with enhanced features
        progression_group = QGroupBox("🎶 Chord Progression")
        progression_layout = QVBoxLayout(progression_group)
        
        # Controls for progression view
        prog_controls = QHBoxLayout()
        self.show_times_cb = QCheckBox("Show Times")
        self.show_times_cb.setChecked(True)
        self.show_times_cb.stateChanged.connect(self.update_chord_list_display)
        prog_controls.addWidget(self.show_times_cb)
        
        self.max_chords_spin = QSpinBox()
        self.max_chords_spin.setRange(5, 50)
        self.max_chords_spin.setValue(12)
        self.max_chords_spin.setPrefix("Show last ")
        self.max_chords_spin.setSuffix(" chords")
        self.max_chords_spin.valueChanged.connect(self.update_chord_list_display)
        prog_controls.addWidget(self.max_chords_spin)
        
        prog_controls.addStretch()
        progression_layout.addLayout(prog_controls)
        
        self.chord_list = QListWidget()
        self.chord_list.setMaximumHeight(180)
        self.chord_list.setAlternatingRowColors(True)
        progression_layout.addWidget(self.chord_list)
        
        left_layout.addWidget(progression_group)
        
        return left_widget
        
    def create_right_panel(self):
        """Create the settings and analysis panel"""
        right_widget = QWidget()
        tabs = QTabWidget()
        right_widget_layout = QVBoxLayout(right_widget)
        right_widget_layout.addWidget(tabs)
        
        # Enhanced Settings tab
        settings_tab = QWidget()
        tabs.addTab(settings_tab, "⚙️ Settings")
        self.setup_enhanced_settings_tab(settings_tab)
        
        # Analysis Results tab
        results_tab = QWidget()
        tabs.addTab(results_tab, "📈 Analysis")
        self.setup_results_tab(results_tab)
        
        # Performance tab
        performance_tab = QWidget()
        tabs.addTab(performance_tab, "🚀 Performance")
        self.setup_performance_tab(performance_tab)
        
        return right_widget
        
    def create_status_bar(self, layout):
        """Create enhanced status bar"""
        status_frame = QFrame()
        status_frame.setFrameStyle(QFrame.Shape.StyledPanel)
        status_layout = QHBoxLayout(status_frame)
        status_layout.setContentsMargins(5, 2, 5, 2)
        
        self.file_info_label = QLabel("No file loaded")
        self.file_info_label.setStyleSheet("color: #666;")
        status_layout.addWidget(self.file_info_label)
        
        status_layout.addStretch()
        
        self.cache_status_label = QLabel("Cache: Ready")
        self.cache_status_label.setStyleSheet("color: #666;")
        status_layout.addWidget(self.cache_status_label)
        
        layout.addWidget(status_frame)
        
    def setup_enhanced_settings_tab(self, parent):
        """Setup enhanced settings interface with better organization"""
        scroll_area = QScrollArea()
        scroll_widget = QWidget()
        layout = QVBoxLayout(scroll_widget)
        
        # Analysis Quality Settings
        quality_group = QGroupBox("🎯 Analysis Quality")
        quality_layout = QGridLayout(quality_group)
        
        # Performance mode toggle
        quality_layout.addWidget(QLabel("Performance Mode:"), 0, 0)
        self.performance_cb = QCheckBox("Enable fast analysis")
        self.performance_cb.setToolTip("Faster analysis with reduced accuracy\nUses fewer chord types and simplified algorithms")
        quality_layout.addWidget(self.performance_cb, 0, 1)
        
        # Multi-feature analysis
        quality_layout.addWidget(QLabel("Multi-Feature Analysis:"), 1, 0)
        self.multi_feature_cb = QCheckBox("Use multiple algorithms")
        self.multi_feature_cb.setChecked(True)
        self.multi_feature_cb.setToolTip("Combines CQT, STFT, and harmonic features\nMore accurate but slower")
        quality_layout.addWidget(self.multi_feature_cb, 1, 1)
        
        # Harmonic separation
        quality_layout.addWidget(QLabel("Harmonic Separation:"), 2, 0)
        self.harmonic_cb = QCheckBox("Separate harmonic content")
        self.harmonic_cb.setChecked(True)
        self.harmonic_cb.setToolTip("Isolates harmonic content from percussive\nImproves chord detection accuracy")
        quality_layout.addWidget(self.harmonic_cb, 2, 1)
        
        layout.addWidget(quality_group)
        
        # Signal Processing Settings
        signal_group = QGroupBox("🎛️ Signal Processing")
        signal_layout = QGridLayout(signal_group)
        
        # Smoothing method
        signal_layout.addWidget(QLabel("Smoothing Method:"), 0, 0)
        self.smoothing_combo = QComboBox()
        self.smoothing_combo.addItems(['adaptive', 'gaussian', 'median'])
        self.smoothing_combo.setToolTip("adaptive: Context-aware smoothing\ngaussian: Standard smoothing\nmedian: Noise-resistant smoothing")
        signal_layout.addWidget(self.smoothing_combo, 0, 1)
        
        # Detection threshold
        signal_layout.addWidget(QLabel("Detection Threshold:"), 1, 0)
        self.threshold_spin = QDoubleSpinBox()
        self.threshold_spin.setRange(0.1, 0.9)
        self.threshold_spin.setValue(0.25)
        self.threshold_spin.setSingleStep(0.05)
        self.threshold_spin.setToolTip("Minimum confidence for chord detection\nLower = more chords detected\nHigher = only confident detections")
        signal_layout.addWidget(self.threshold_spin, 1, 1)
        
        layout.addWidget(signal_group)
        
        # Musical Analysis Settings
        musical_group = QGroupBox("🎼 Musical Analysis")
        musical_layout = QGridLayout(musical_group)
        
        # Minimum chord duration
        musical_layout.addWidget(QLabel("Min Chord Duration:"), 0, 0)
        self.min_duration_spin = QDoubleSpinBox()
        self.min_duration_spin.setRange(0.1, 5.0)
        self.min_duration_spin.setValue(0.3)
        self.min_duration_spin.setSingleStep(0.1)
        self.min_duration_spin.setSuffix(" sec")
        self.min_duration_spin.setToolTip("Minimum duration for a chord to be recognized\nFilters out very brief chord changes")
        musical_layout.addWidget(self.min_duration_spin, 0, 1)
        
        # Key-aware analysis
        musical_layout.addWidget(QLabel("Key-Aware Analysis:"), 1, 0)
        self.key_aware_cb = QCheckBox("Use key context")
        self.key_aware_cb.setChecked(True)
        self.key_aware_cb.setToolTip("Considers detected key for chord recognition\nImproves accuracy for tonal music")
        musical_layout.addWidget(self.key_aware_cb, 1, 1)
        
        # Maximum chord types
        musical_layout.addWidget(QLabel("Max Chord Types:"), 2, 0)
        self.max_chord_types_spin = QSpinBox()
        self.max_chord_types_spin.setRange(20, 200)
        self.max_chord_types_spin.setValue(100)
        self.max_chord_types_spin.setToolTip("Maximum number of chord types to consider\nLower values = faster analysis")
        musical_layout.addWidget(self.max_chord_types_spin, 2, 1)
        
        layout.addWidget(musical_group)
        
        # Export and Presets
        export_group = QGroupBox("💾 Export & Presets")
        export_layout = QVBoxLayout(export_group)
        
        # Preset buttons
        preset_layout = QHBoxLayout()
        
        self.btn_preset_fast = QPushButton("⚡ Fast")
        self.btn_preset_fast.clicked.connect(lambda: self.apply_preset('fast'))
        self.btn_preset_fast.setToolTip("Optimized for speed\nGood for real-time analysis")
        preset_layout.addWidget(self.btn_preset_fast)
        
        self.btn_preset_balanced = QPushButton("⚖️ Balanced")
        self.btn_preset_balanced.clicked.connect(lambda: self.apply_preset('balanced'))
        self.btn_preset_balanced.setToolTip("Balance of speed and accuracy\nRecommended for most users")
        preset_layout.addWidget(self.btn_preset_balanced)
        
        self.btn_preset_accurate = QPushButton("🎯 Accurate")
        self.btn_preset_accurate.clicked.connect(lambda: self.apply_preset('accurate'))
        self.btn_preset_accurate.setToolTip("Maximum accuracy\nBest for detailed analysis")
        preset_layout.addWidget(self.btn_preset_accurate)
        
        export_layout.addLayout(preset_layout)
        
        # Export buttons
        export_buttons = QHBoxLayout()
        
        self.btn_export_txt = QPushButton("📄 Export Text")
        self.btn_export_txt.clicked.connect(lambda: self.export_results('txt'))
        self.btn_export_txt.setEnabled(False)
        export_buttons.addWidget(self.btn_export_txt)
        
        self.btn_export_json = QPushButton("📊 Export JSON")
        self.btn_export_json.clicked.connect(lambda: self.export_results('json'))
        self.btn_export_json.setEnabled(False)
        export_buttons.addWidget(self.btn_export_json)
        
        self.btn_export_midi = QPushButton("🎹 Export MIDI")
        self.btn_export_midi.clicked.connect(lambda: self.export_results('midi'))
        self.btn_export_midi.setEnabled(False)
        self.btn_export_midi.setToolTip("Export chord progression as MIDI file")
        export_buttons.addWidget(self.btn_export_midi)
        
        export_layout.addLayout(export_buttons)
        layout.addWidget(export_group)
        
        layout.addStretch()
        scroll_area.setWidget(scroll_widget)
        
        main_layout = QVBoxLayout(parent)
        main_layout.addWidget(scroll_area)
        
    def setup_results_tab(self, parent):
        """Setup enhanced results display"""
        layout = QVBoxLayout(parent)
        
        # Analysis summary with enhanced metrics
        summary_group = QGroupBox("📊 Analysis Summary")
        summary_layout = QGridLayout(summary_group)
        
        self.total_chords_label = QLabel("Total Chords: 0")
        summary_layout.addWidget(self.total_chords_label, 0, 0)
        
        self.avg_duration_label = QLabel("Avg Duration: 0.0s")
        summary_layout.addWidget(self.avg_duration_label, 0, 1)
        
        self.key_confidence_label = QLabel("Key Confidence: 0%")
        summary_layout.addWidget(self.key_confidence_label, 1, 0)
        
        self.analysis_time_label = QLabel("Analysis Time: 0.0s")
        summary_layout.addWidget(self.analysis_time_label, 1, 1)
        
        self.cache_hit_label = QLabel("Cache Status: Miss")
        summary_layout.addWidget(self.cache_hit_label, 2, 0)
        
        self.complexity_label = QLabel("Harmonic Complexity: -")
        summary_layout.addWidget(self.complexity_label, 2, 1)
        
        layout.addWidget(summary_group)
        
        # Detailed results with tabs
        results_tabs = QTabWidget()
        
        # Chord progression tab
        prog_tab = QWidget()
        prog_layout = QVBoxLayout(prog_tab)
        
        self.results_text = QTextEdit()
        self.results_text.setReadOnly(True)
        self.results_text.setFont(QFont("Consolas", 10))
        prog_layout.addWidget(self.results_text)
        
        results_tabs.addTab(prog_tab, "Progression")
        
        # Statistics tab
        stats_tab = QWidget()
        stats_layout = QVBoxLayout(stats_tab)
        
        self.stats_text = QTextEdit()
        self.stats_text.setReadOnly(True)
        self.stats_text.setFont(QFont("Consolas", 10))
        stats_layout.addWidget(self.stats_text)
        
        results_tabs.addTab(stats_tab, "Statistics")
        
        layout.addWidget(results_tabs)
        
    def setup_performance_tab(self, parent):
        """Setup performance monitoring tab"""
        layout = QVBoxLayout(parent)
        
        # Performance metrics
        perf_group = QGroupBox("⚡ Performance Metrics")
        perf_layout = QGridLayout(perf_group)
        
        self.perf_total_label = QLabel("Total Analysis: -")
        perf_layout.addWidget(self.perf_total_label, 0, 0)
        
        self.perf_loading_label = QLabel("Audio Loading: -")
        perf_layout.addWidget(self.perf_loading_label, 0, 1)
        
        self.perf_features_label = QLabel("Feature Extraction: -")
        perf_layout.addWidget(self.perf_features_label, 1, 0)
        
        self.perf_matching_label = QLabel("Chord Matching: -")
        perf_layout.addWidget(self.perf_matching_label, 1, 1)
        
        layout.addWidget(perf_group)
        
        # Cache information
        cache_group = QGroupBox("💾 Cache Information")
        cache_layout = QVBoxLayout(cache_group)
        
        cache_info_layout = QHBoxLayout()
        self.btn_clear_cache = QPushButton("🗑️ Clear Cache")
        self.btn_clear_cache.clicked.connect(self.clear_cache)
        cache_info_layout.addWidget(self.btn_clear_cache)
        
        self.btn_cache_info = QPushButton("ℹ️ Cache Info")
        self.btn_cache_info.clicked.connect(self.show_cache_info)
        cache_info_layout.addWidget(self.btn_cache_info)
        
        cache_info_layout.addStretch()
        cache_layout.addLayout(cache_info_layout)
        
        self.cache_info_text = QTextEdit()
        self.cache_info_text.setReadOnly(True)
        self.cache_info_text.setMaximumHeight(150)
        cache_layout.addWidget(self.cache_info_text)
        
        layout.addWidget(cache_group)
        
        layout.addStretch()
        
    def apply_modern_theme(self):
        """Apply modern, professional theme"""
        # Enhanced color scheme
        self.setStyleSheet("""
            QWidget {
                font-family: 'Segoe UI', 'San Francisco', 'Helvetica Neue', Arial, sans-serif;
                font-size: 11px;
                background-color: #FAFAFA;
            }
            
            QGroupBox {
                font-weight: 600;
                border: 2px solid #E0E0E0;
                border-radius: 12px;
                margin-top: 12px;
                padding-top: 16px;
                background-color: #FFFFFF;
            }
            
            QGroupBox::title {
                subcontrol-origin: margin;
                left: 16px;
                padding: 0 12px 0 12px;
                color: #1976D2;
                font-size: 12px;
                font-weight: 700;
            }
            
            QPushButton {
                padding: 12px 24px;
                border-radius: 8px;
                background: qlineargradient(x1:0, y1:0, x2:0, y2:1,
                    stop:0 #2196F3, stop:1 #1976D2);
                color: white;
                border: none;
                font-weight: 600;
                font-size: 12px;
                min-height: 20px;
            }
            
            QPushButton:hover {
                background: qlineargradient(x1:0, y1:0, x2:0, y2:1,
                    stop:0 #1976D2, stop:1 #1565C0);
                transform: translateY(-1px);
            }
            
            QPushButton:pressed {
                background: qlineargradient(x1:0, y1:0, x2:0, y2:1,
                    stop:0 #1565C0, stop:1 #0D47A1);
            }
            
            QPushButton:disabled {
                background: #E0E0E0;
                color: #9E9E9E;
            }
            
            QProgressBar {
                border: 2px solid #E0E0E0;
                border-radius: 8px;
                text-align: center;
                background-color: #F5F5F5;
                font-weight: 600;
                font-size: 11px;
                min-height: 24px;
            }
            
            QProgressBar::chunk {
                background: qlineargradient(x1:0, y1:0, x2:1, y2:0,
                    stop:0 #4CAF50, stop:0.5 #66BB6A, stop:1 #81C784);
                border-radius: 6px;
            }
            
            QTabWidget::pane {
                border: 2px solid #E0E0E0;
                background-color: #FFFFFF;
                border-radius: 8px;
                margin-top: 4px;
            }
            
            QTabBar::tab {
                background: qlineargradient(x1:0, y1:0, x2:0, y2:1,
                    stop:0 #F5F5F5, stop:1 #E0E0E0);
                padding: 12px 20px;
                margin-right: 2px;
                border-top-left-radius: 8px;
                border-top-right-radius: 8px;
                font-weight: 600;
                color: #666666;
            }
            
            QTabBar::tab:selected {
                background: qlineargradient(x1:0, y1:0, x2:0, y2:1,
                    stop:0 #2196F3, stop:1 #1976D2);
                color: white;
            }
            
            QTabBar::tab:hover:!selected {
                background: qlineargradient(x1:0, y1:0, x2:0, y2:1,
                    stop:0 #EEEEEE, stop:1 #E0E0E0);
            }
            
            QListWidget {
                border: 2px solid #E0E0E0;
                border-radius: 8px;
                background-color: #FFFFFF;
                alternate-background-color: #F8F9FA;
                selection-background-color: #E3F2FD;
                font-family: 'Consolas', 'Monaco', monospace;
            }
            
            QListWidget::item {
                padding: 8px;
                border-bottom: 1px solid #F0F0F0;
            }
            
            QListWidget::item:selected {
                background-color: #2196F3;
                color: white;
            }
        """)
        
        # Enhanced chord display styling
        self.apply_chord_display_styling()
        
    def apply_chord_display_styling(self):
        """Apply enhanced styling to chord display elements"""
        # Main chord label styling
        chord_style = """
            QLabel {
                border: 4px solid #2196F3;
                border-radius: 20px;
                padding: 30px;
                margin: 8px;
                background: qlineargradient(x1:0, y1:0, x2:0, y2:1,
                    stop:0 #E3F2FD, stop:0.3 #BBDEFB, stop:0.7 #90CAF9, stop:1 #64B5F6);
                color: #0D47A1;
                font-size: 32px;
                font-weight: 700;
                text-align: center;
                letter-spacing: 2px;
            }
        """
        self.chord_label.setStyleSheet(chord_style)
        
        # Key signature styling
        key_style = """
            QLabel {
                background: qlineargradient(x1:0, y1:0, x2:1, y2:0,
                    stop:0 #FFF8E1, stop:0.5 #FFECB3, stop:1 #FFE082);
                border: 3px solid #FF8F00;
                border-radius: 15px;
                padding: 16px;
                color: #E65100;
                font-weight: 700;
                font-size: 14px;
            }
        """
        self.key_label.setStyleSheet(key_style)
        
        # Time display styling
        time_style = """
            QLabel {
                background: qlineargradient(x1:0, y1:0, x2:1, y2:0,
                    stop:0 #F3E5F5, stop:0.5 #E1BEE7, stop:1 #CE93D8);
                border: 3px solid #7B1FA2;
                border-radius: 15px;
                padding: 16px;
                color: #4A148C;
                font-weight: 700;
                font-size: 14px;
                font-family: 'Consolas', monospace;
            }
        """
        self.time_label.setStyleSheet(time_style)
        
        # Confidence styling
        confidence_style = """
            QLabel {
                background: qlineargradient(x1:0, y1:0, x2:1, y2:0,
                    stop:0 #E8F5E8, stop:1 #C8E6C9);
                border: 2px solid #4CAF50;
                border-radius: 12px;
                padding: 12px;
                color: #2E7D32;
                font-weight: 600;
                font-size: 12px;
            }
        """
        self.confidence_label.setStyleSheet(confidence_style)
        
        # Tempo styling
        tempo_style = """
            QLabel {
                background: qlineargradient(x1:0, y1:0, x2:1, y2:0,
                    stop:0 #FFF3E0, stop:1 #FFE0B2);
                border: 2px solid #FF9800;
                border-radius: 12px;
                padding: 12px;
                color: #F57C00;
                font-weight: 600;
                font-size: 12px;
            }
        """
        self.tempo_label.setStyleSheet(tempo_style)
        
        
        preset_name = (preset_name or '').lower()
        if preset_name == 'fast':
            self.performance_cb.setChecked(True)
            self.multi_feature_cb.setChecked(False)
            self.harmonic_cb.setChecked(False)
            self.smoothing_combo.setCurrentText('gaussian')
            self.threshold_spin.setValue(0.3)
            self.min_duration_spin.setValue(0.5)
            self.max_chord_types_spin.setValue(30)
        elif preset_name == 'balanced':
            self.performance_cb.setChecked(False)
            self.multi_feature_cb.setChecked(True)
            self.harmonic_cb.setChecked(True)
            self.smoothing_combo.setCurrentText('adaptive')
            self.threshold_spin.setValue(0.25)
            self.min_duration_spin.setValue(0.3)
            self.max_chord_types_spin.setValue(80)
        elif preset_name == 'accurate':
            self.performance_cb.setChecked(False)
            self.multi_feature_cb.setChecked(True)
            self.harmonic_cb.setChecked(True)
            self.smoothing_combo.setCurrentText('adaptive')
            self.threshold_spin.setValue(0.2)
            self.min_duration_spin.setValue(0.2)
            self.max_chord_types_spin.setValue(150)
        
        self.save_settings()
        QMessageBox.information(self, "Preset Applied", 
                              f"'{preset_name.title()}' preset has been applied.\n"
                              f"Settings will be used for the next analysis.")
        
        self.save_settings()
        QMessageBox.information(self, "Preset Applied", 
                              f"'{preset_name.title()}' preset has been applied.\n"
                              f"Settings will be used for the next analysis.")
        
    def load_settings(self):
        """Load user settings with new parameters"""
        self.performance_cb.setChecked(self.settings.value("performance_mode", False, type=bool))
        self.harmonic_cb.setChecked(self.settings.value("harmonic_separation", True, type=bool))
        self.multi_feature_cb.setChecked(self.settings.value("multi_features", True, type=bool))
        self.smoothing_combo.setCurrentText(self.settings.value("smoothing_method", "adaptive"))
        self.min_duration_spin.setValue(self.settings.value("min_duration", 0.3, type=float))
        self.threshold_spin.setValue(self.settings.value("threshold", 0.25, type=float))
        self.key_aware_cb.setChecked(self.settings.value("key_aware", True, type=bool))
        self.max_chord_types_spin.setValue(self.settings.value("max_chord_types", 100, type=int))
        
    def save_settings(self):
        """Save user settings with new parameters"""
        self.settings.setValue("performance_mode", self.performance_cb.isChecked())
        self.settings.setValue("harmonic_separation", self.harmonic_cb.isChecked())
        self.settings.setValue("multi_features", self.multi_feature_cb.isChecked())
        self.settings.setValue("smoothing_method", self.smoothing_combo.currentText())
        self.settings.setValue("min_duration", self.min_duration_spin.value())
        self.settings.setValue("threshold", self.threshold_spin.value())
        self.settings.setValue("key_aware", self.key_aware_cb.isChecked())
        self.settings.setValue("max_chord_types", self.max_chord_types_spin.value())
        
    def load_audio(self):
        """Enhanced audio loading with better error handling and feedback"""
        file_path, _ = QFileDialog.getOpenFileName(
            self, "Select Audio File", self.settings.value("last_directory", ""), 
            "Audio Files (*.wav *.mp3 *.flac *.aac *.m4a *.ogg *.opus *.wma);;All Files (*)"
        )
        if not file_path:
            return
            
        # Save directory for next time
        self.settings.setValue("last_directory", os.path.dirname(file_path))
        
        # Update file info
        file_name = os.path.basename(file_path)
        file_size = os.path.getsize(file_path) / (1024 * 1024)  # MB
        self.file_info_label.setText(f"File: {file_name} ({file_size:.1f} MB)")
        
        # Save current settings
        self.save_settings()
            
        # Disable controls during analysis
        self.btn_load.setEnabled(False)
        self.btn_stop.setEnabled(False)
        self.btn_pause.setEnabled(False)
        self.chord_label.setText("🔄 Analyzing audio...")
        self.status_label.setText("Initializing enhanced analysis...")
        self.progress_bar.setValue(0)
        
        # Clear previous results
        self.chord_list.clear()
        self.results_text.clear()
        self.stats_text.clear()
        
        # Get current settings
        settings = {
            'use_harmonic_separation': self.harmonic_cb.isChecked(),
            'use_multi_features': self.multi_feature_cb.isChecked(),
            'smoothing_method': self.smoothing_combo.currentText(),
            'min_chord_duration': self.min_duration_spin.value(),
            'template_threshold': self.threshold_spin.value(),
            'key_aware': self.key_aware_cb.isChecked(),
            'performance_mode': self.performance_cb.isChecked(),
            'max_chord_types': self.max_chord_types_spin.value()
        }
        
        # Update performance info
        mode_text = "Performance Mode" if settings['performance_mode'] else "Quality Mode"
        self.performance_label.setText(f"{mode_text} | {settings['smoothing_method'].title()} smoothing")
        
        # Start analysis in separate thread
        self.analysis_thread = QThread()
        self.worker = ChordAnalysisWorker(file_path, settings)
        self.worker.moveToThread(self.analysis_thread)
        
        # Connect signals
        self.analysis_thread.started.connect(self.worker.run)
        self.worker.progress.connect(self.update_progress)
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
            
            # Estimate tempo for display
            tempo, _ = librosa.beat.beat_track(y=y, sr=sr)
            self.estimated_tempo = tempo
            
        except Exception as e:
            QMessageBox.critical(self, "Audio Loading Error", 
                               f"Failed to load audio file:\n{str(e)}\n\n"
                               f"Please check:\n"
                               f"• File format is supported\n"
                               f"• File is not corrupted\n"
                               f"• Sufficient memory available")
            self.btn_load.setEnabled(True)
            
    def update_progress(self, value):
        """Enhanced progress update with time estimation"""
        self.progress_bar.setValue(value)
        if hasattr(self, 'analysis_start_time') and value > 0:
            elapsed = time.time() - self.analysis_start_time
            if value < 100:
                estimated_total = (elapsed / value) * 100
                remaining = estimated_total - elapsed
                self.progress_bar.setFormat(f"{value}% - ~{remaining:.0f}s remaining")
            else:
                self.progress_bar.setFormat(f"Complete in {elapsed:.1f}s")
        else:
            self.analysis_start_time = time.time()
            
    def on_analysis_finished(self, chords, key_signature, stats):
        """Enhanced analysis completion handler"""
        self.chords = chords
        self.key_signature = key_signature
        self.analysis_stats = stats
        
        # Update displays
        self.update_key_display(key_signature, stats.get('key_confidence', 0))
        self.display_enhanced_results()
        self.update_performance_display(stats)
        
        # Enable export buttons
        self.btn_export_txt.setEnabled(True)
        self.btn_export_json.setEnabled(True)
        self.btn_export_midi.setEnabled(True)
        
        # Update cache status
        cache_status = "Hit" if stats.get('cache_hit', False) else "Miss"
        self.cache_status_label.setText(f"Cache: {cache_status}")
        
        # Start playback if audio loaded successfully
        if hasattr(self, 'audio_data'):
            try:
                sd.stop()
                sd.play(self.audio_data, samplerate=self.audio_sr)
                
                self.start_time = time.time()
                self.is_paused = False
                self.timer.start()
                
                self.status_label.setText(f"✅ Analysis complete - Playback started ({len(chords)} chords detected)")
                self.btn_stop.setEnabled(True)
                self.btn_pause.setEnabled(True)
                
            except Exception as e:
                QMessageBox.warning(self, "Playback Error", 
                                  f"Analysis completed but playback failed:\n{str(e)}\n\n"
                                  f"You can still view and export the results.")
        
        # Re-enable load button
        self.btn_load.setEnabled(True)
        
        # Clean up thread
        self.analysis_thread.quit()
        self.analysis_thread.wait()
        
    def on_analysis_error(self, error_msg):
        """Enhanced error handling with better diagnostics"""
        error_details = f"Chord detection failed:\n{error_msg}\n\n"
        
        # Add diagnostic suggestions
        if "memory" in error_msg.lower():
            error_details += "💡 Suggestions:\n• Try enabling Performance Mode\n• Close other applications\n• Use a shorter audio file"
        elif "format" in error_msg.lower() or "codec" in error_msg.lower():
            error_details += "💡 Suggestions:\n• Convert to WAV or FLAC format\n• Check if file is corrupted\n• Try a different audio file"
        elif "permission" in error_msg.lower():
            error_details += "💡 Suggestions:\n• Check file permissions\n• Move file to a different location\n• Run as administrator"
        else:
            error_details += "💡 Suggestions:\n• Try Performance Mode for faster analysis\n• Reduce detection threshold\n• Check file format compatibility"
        
        QMessageBox.critical(self, "Analysis Error", error_details)
        
        self.chord_label.setText("❌ Analysis failed")
        self.status_label.setText("Ready - Try adjusting settings or different file")
        self.progress_bar.setValue(0)
        self.progress_bar.setFormat("")
        self.btn_load.setEnabled(True)
        
        # Clean up thread
        if self.analysis_thread:
            self.analysis_thread.quit()
            self.analysis_thread.wait()
            
    def update_key_display(self, key_signature, confidence):
        """Update key signature display with confidence"""
        confidence_pct = int(confidence * 100)
        self.key_label.setText(f"Key: {key_signature}")
        
        if confidence > 0:
            self.key_confidence_label.setText(f"Key Confidence: {confidence_pct}%")
            
            # Color-code confidence
            if confidence_pct >= 80:
                color = "#4CAF50"  # Green
            elif confidence_pct >= 60:
                color = "#FF9800"  # Orange
            else:
                color = "#F44336"  # Red
                
            self.key_confidence_label.setStyleSheet(f"""
                QLabel {{
                    background: qlineargradient(x1:0, y1:0, x2:1, y2:0,
                        stop:0 {color}30, stop:1 {color}20);
                    border: 2px solid {color};
                    border-radius: 12px;
                    padding: 8px;
                    color: {color};
                    font-weight: 600;
                }}
            """)
            
    def display_enhanced_results(self):
        """Display comprehensive analysis results"""
        if not self.chords:
            return
            
        # Update summary statistics
        total_chords = len(self.chords)
        self.total_chords_label.setText(f"Total Chords: {total_chords}")
        
        # Calculate metrics
        if total_chords > 1:
            durations = []
            for i in range(len(self.chords) - 1):
                duration = self.chords[i+1][0] - self.chords[i][0]
                durations.append(duration)
            
            avg_duration = np.mean(durations)
            self.avg_duration_label.setText(f"Avg Duration: {avg_duration:.1f}s")
            
            # Harmonic complexity (unique chords / total segments)
            unique_chords = len(set(chord for _, chord in self.chords if chord != "N.C."))
            complexity = (unique_chords / total_chords) * 100
            self.complexity_label.setText(f"Harmonic Complexity: {complexity:.1f}%")
        
        # Analysis time
        analysis_time = self.analysis_stats.get('total_analysis_time', 0)
        self.analysis_time_label.setText(f"Analysis Time: {analysis_time:.1f}s")
        
        # Cache status
        cache_hit = self.analysis_stats.get('cache_hit', False)
        cache_text = "Hit ✅" if cache_hit else "Miss ❌"
        self.cache_hit_label.setText(f"Cache Status: {cache_text}")
        
        # Detailed progression results
        self.update_progression_display()
        
        # Statistical analysis
        self.update_statistics_display()
        
    def update_progression_display(self):
        """Update detailed chord progression display"""
        results_text = f"🎼 Enhanced Chord Analysis Results\n"
        results_text += f"{'=' * 60}\n\n"
        results_text += f"📁 File: {os.path.basename(getattr(self, 'audio_file_path', 'Unknown'))}\n"
        results_text += f"🎵 Key Signature: {self.key_signature}\n"
        results_text += f"🎯 Total Segments: {len(self.chords)}\n"
        results_text += f"⏱️  Duration: {self.audio_duration:.1f} seconds\n"
        results_text += f"🚀 Performance Mode: {'✅' if self.analysis_stats.get('performance_mode', False) else '❌'}\n\n"
        
        results_text += f"🎶 Chord Progression:\n"
        results_text += f"{'-' * 60}\n"
        results_text += f"{'Time':>8} {'Chord':>15} {'Duration':>12} {'%':>8}\n"
        results_text += f"{'-' * 60}\n"
        
        total_duration = self.audio_duration
        for i, (time_stamp, chord) in enumerate(self.chords[:200]):  # Show first 200
            if i < len(self.chords) - 1:
                duration = self.chords[i+1][0] - time_stamp
                duration_str = f"{duration:8.1f}s"
                percentage = f"{(duration/total_duration)*100:5.1f}%"
            else:
                duration_str = "---"
                percentage = "---"
            
            # Add emoji for chord types
            chord_emoji = self.get_chord_emoji(chord)
            results_text += f"{time_stamp:8.1f}s {chord_emoji}{chord:>14} {duration_str:>12} {percentage:>8}\n"
            
        if len(self.chords) > 200:
            results_text += f"\n... and {len(self.chords) - 200} more segments\n"
            
        self.results_text.setText(results_text)
        
    def update_statistics_display(self):
        """Update statistical analysis display"""
        if not self.chords:
            return
            
        # Chord frequency analysis
        chord_counts = {}
        total_duration = 0
        chord_durations = {}
        
        for i, (time_stamp, chord) in enumerate(self.chords):
            chord_counts[chord] = chord_counts.get(chord, 0) + 1
            
            if i < len(self.chords) - 1:
                duration = self.chords[i+1][0] - time_stamp
                chord_durations[chord] = chord_durations.get(chord, 0) + duration
                total_duration += duration
        
        # Sort by frequency and duration
        most_common = sorted(chord_counts.items(), key=lambda x: x[1], reverse=True)[:15]
        longest_duration = sorted(chord_durations.items(), key=lambda x: x[1], reverse=True)[:10]
        
        stats_text = f"📊 Statistical Analysis\n"
        stats_text += f"{'=' * 50}\n\n"
        
        # Frequency analysis
        stats_text += f"🔢 Most Frequent Chords:\n"
        stats_text += f"{'-' * 40}\n"
        stats_text += f"{'Chord':>12} {'Count':>8} {'Freq %':>10}\n"
        stats_text += f"{'-' * 40}\n"
        for chord, count in most_common:
            percentage = (count / len(self.chords)) * 100
            emoji = self.get_chord_emoji(chord)
            stats_text += f"{emoji}{chord:>11} {count:>8} {percentage:>8.1f}%\n"
        
        # Duration analysis
        stats_text += f"\n⏱️  Longest Duration Chords:\n"
        stats_text += f"{'-' * 40}\n"
        stats_text += f"{'Chord':>12} {'Duration':>10} {'%':>8}\n"
        stats_text += f"{'-' * 40}\n"
        for chord, duration in longest_duration:
            percentage = (duration / total_duration) * 100 if total_duration > 0 else 0
            emoji = self.get_chord_emoji(chord)
            stats_text += f"{emoji}{chord:>11} {duration:>8.1f}s {percentage:>6.1f}%\n"
        
        # Key analysis
        if self.key_signature != "Unknown":
            stats_text += f"\n🎼 Key Analysis:\n"
            stats_text += f"{'-' * 30}\n"
            stats_text += f"Detected Key: {self.key_signature}\n"
            key_confidence = self.analysis_stats.get('key_confidence', 0) * 100
            stats_text += f"Confidence: {key_confidence:.1f}%\n"
            
            # Analyze chord-key relationships
            if 'Major' in self.key_signature:
                root = self.key_signature.replace(' Major', '')
                stats_text += f"Expected chords in {root} Major:\n"
                stats_text += f"  I: {root}, ii: {NOTE_NAMES[(NOTE_NAMES.index(root)+2)%12]}m, iii: {NOTE_NAMES[(NOTE_NAMES.index(root)+4)%12]}m\n"
                stats_text += f"  IV: {NOTE_NAMES[(NOTE_NAMES.index(root)+5)%12]}, V: {NOTE_NAMES[(NOTE_NAMES.index(root)+7)%12]}, vi: {NOTE_NAMES[(NOTE_NAMES.index(root)+9)%12]}m\n"
        
        # Performance metrics
        stats_text += f"\n🚀 Performance Metrics:\n"
        stats_text += f"{'-' * 30}\n"
        for key, value in self.analysis_stats.items():
            if 'time' in key and isinstance(value, (int, float)):
                stats_text += f"{key.replace('_', ' ').title()}: {value:.3f}s\n"
        
        self.stats_text.setText(stats_text)
        
    def get_chord_emoji(self, chord):
        """Get emoji representation for chord types"""
        if chord == "N.C.":
            return "🔇 "
        elif 'dim' in chord.lower():
            return "🔸 "
        elif 'aug' in chord.lower():
            return "🔹 "
        elif 'm' in chord and not any(x in chord for x in ['maj', 'M']):
            return "🎵 "
        elif any(x in chord for x in ['7', '9', '11', '13']):
            return "🎶 "
        elif 'sus' in chord.lower():
            return "🎭 "
        else:
            return "🎼 "
            
    def update_performance_display(self, stats):
        """Update performance metrics display"""
        self.perf_total_label.setText(f"Total Analysis: {stats.get('total_analysis_time', 0):.2f}s")
        
        # Individual timing components would need to be passed from the analysis
        # For now, show what we have
        cache_hit = stats.get('cache_hit', False)
        mode = "Performance" if stats.get('performance_mode', False) else "Quality"
        
        perf_text = f"Analysis completed in {mode} mode\n"
        perf_text += f"Cache status: {'Hit' if cache_hit else 'Miss'}\n"
        perf_text += f"Total frames processed: {stats.get('total_frames', 0)}\n"
        perf_text += f"Chord changes detected: {stats.get('chord_changes', 0)}\n"
        
        # Update cache info
        self.update_cache_info()
        
    def update_cache_info(self):
        """Update cache information display"""
        cache_dir = CHORDS_CACHE_DIR
        if os.path.exists(cache_dir):
            cache_files = [f for f in os.listdir(cache_dir) if f.endswith(CHORDS_CACHE_EXTENSION)]
            total_files = len(cache_files)
            
            # Calculate total cache size
            total_size = 0
            for file in cache_files:
                try:
                    file_path = os.path.join(cache_dir, file)
                    total_size += os.path.getsize(file_path)
                except:
                    pass
            
            cache_size_mb = total_size / (1024 * 1024)
            
            cache_info = f"Cache Directory: {cache_dir}\n"
            cache_info += f"Cached Analyses: {total_files}\n"
            cache_info += f"Total Size: {cache_size_mb:.1f} MB\n\n"
            cache_info += "Recent Cache Files:\n"
            
            # Show recent files
            recent_files = sorted(cache_files, key=lambda x: os.path.getmtime(os.path.join(cache_dir, x)), reverse=True)[:10]
            for file in recent_files:
                file_path = os.path.join(cache_dir, file)
                mod_time = time.ctime(os.path.getmtime(file_path))
                cache_info += f"• {file[:30]}... ({mod_time})\n"
            
            self.cache_info_text.setText(cache_info)
        else:
            self.cache_info_text.setText("No cache directory found.")
            
    def clear_cache(self):
        """Clear analysis cache"""
        reply = QMessageBox.question(self, "Clear Cache", 
                                   "Are you sure you want to clear the analysis cache?\n\n"
                                   "This will delete all cached chord analysis results, "
                                   "and future analyses will take longer until the cache is rebuilt.",
                                   QMessageBox.StandardButton.Yes | QMessageBox.StandardButton.No)
        
        if reply == QMessageBox.StandardButton.Yes:
            try:
                cache_dir = CHORDS_CACHE_DIR
                if os.path.exists(cache_dir):
                    import shutil
                    shutil.rmtree(cache_dir)
                    os.makedirs(cache_dir, exist_ok=True)
                    
                QMessageBox.information(self, "Cache Cleared", 
                                      "Analysis cache has been cleared successfully.")
                self.update_cache_info()
                self.cache_status_label.setText("Cache: Cleared")
                
            except Exception as e:
                QMessageBox.critical(self, "Error", f"Failed to clear cache:\n{str(e)}")
                
    def show_cache_info(self):
        """Show detailed cache information"""
        self.update_cache_info()
        
    def toggle_pause(self):
        """Toggle playback pause/resume"""
        if not hasattr(self, 'is_paused'):
            return
            
        if self.is_paused:
            # Resume
            if hasattr(self, 'audio_data'):
                try:
                    # Calculate remaining audio
                    elapsed = time.time() - self.start_time if self.start_time else 0
                    remaining_samples = int((self.audio_duration - elapsed) * self.audio_sr)
                    if remaining_samples > 0:
                        start_sample = len(self.audio_data) - remaining_samples
                        sd.play(self.audio_data[start_sample:], samplerate=self.audio_sr)
                        
                    self.timer.start()
                    self.is_paused = False
                    self.btn_pause.setText("⏸️ Pause")
                    self.status_label.setText("Playback resumed")
                except Exception as e:
                    QMessageBox.warning(self, "Resume Error", f"Could not resume playback:\n{str(e)}")
        else:
            # Pause
            sd.stop()
            self.timer.stop()
            self.is_paused = True
            self.btn_pause.setText("▶️ Resume")
            self.status_label.setText("Playback paused")
            
    def stop_playback(self):
        """Enhanced stop playback with cleanup"""
        sd.stop()
        self.timer.stop()
        self.start_time = None
        self.is_paused = False
        
        self.btn_stop.setEnabled(False)
        self.btn_pause.setEnabled(False)
        self.btn_pause.setText("⏸️ Pause")
        
        self.chord_label.setText("⏹️ Playback stopped")
        self.status_label.setText("Stopped - Load another file or replay current")
        self.progress_bar.setValue(0)
        self.progress_bar.setFormat("")
        self.chord_list.clear()
        
        # Reset displays
        self.confidence_label.setText("Confidence: -")
        self.tempo_label.setText("♩ = - BPM")
        
    def update_display(self):
        """Enhanced display update during playback"""
        if self.start_time is None or not self.chords or self.is_paused:
            return
            
        elapsed = time.time() - self.start_time
        
        # Update progress bar
        if self.audio_duration > 0:
            progress = min(100, int((elapsed / self.audio_duration) * 100))
            self.progress_bar.setValue(progress)
            self.progress_bar.setFormat(f"{progress}% - Playing")
        
        # Update time display
        elapsed_str = f"{int(elapsed//60):02d}:{int(elapsed%60):02d}"
        total_str = f"{int(self.audio_duration//60):02d}:{int(self.audio_duration%60):02d}"
        self.time_label.setText(f"{elapsed_str} / {total_str}")
        
        # Find current chord with confidence estimation
        current_chord = "♪"
        chord_confidence = 0
        chord_color_class = "default"
        
        for i, (t, chord) in enumerate(self.chords):
            if t <= elapsed:
                if chord != "N.C.":
                    current_chord = chord
                    chord_color_class = self.get_enhanced_chord_color_class(chord)
                    
                    # Estimate confidence based on chord duration and stability
                    if i < len(self.chords) - 1:
                        chord_duration = self.chords[i+1][0] - t
                        chord_confidence = min(1.0, chord_duration / 2.0)  # Longer chords = higher confidence
                else:
                    current_chord = "♪"
                    chord_color_class = "default"
                    chord_confidence = 0
            else:
                break
        
        # Update chord display with enhanced styling
        self.update_enhanced_chord_display(current_chord, chord_color_class)
        
        # Update confidence display
        if chord_confidence > 0:
            self.confidence_label.setText(f"Confidence: {int(chord_confidence * 100)}%")
        else:
            self.confidence_label.setText("Confidence: -")
        
        # Update tempo display
        if hasattr(self, 'estimated_tempo'):
            self.tempo_label.setText(f"♩ ≈ {self.estimated_tempo:.0f} BPM")
        
        # Update recent chords list
        self.update_chord_list_display(elapsed)
        
        # Stop when audio ends
        if elapsed >= self.audio_duration:
            self.stop_playback()
            
    def get_enhanced_chord_color_class(self, chord):
        """Enhanced chord color classification with more categories"""
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
        elif any(x in chord_lower for x in ['13']):
            return "thirteenth"
        elif any(x in chord_lower for x in ['11']):
            return "eleventh"
        elif any(x in chord_lower for x in ['9']):
            return "ninth"
        elif any(x in chord_lower for x in ['7']) and 'maj' not in chord_lower:
            return "dominant"
        elif any(x in chord_lower for x in ['maj7', 'm7']):
            return "major_seventh"
        elif 'sus' in chord_lower:
            return "suspended"
        elif '6' in chord_lower:
            return "sixth"
        else:
            return "major"
            
    def update_enhanced_chord_display(self, chord, color_class):
        """Update chord display with comprehensive color scheme"""
        styles = {
            "default": {"border": "#9E9E9E", "bg_start": "#F5F5F5", "bg_end": "#E0E0E0", "color": "#424242"},
            "major": {"border": "#4CAF50", "bg_start": "#E8F5E8", "bg_end": "#C8E6C9", "color": "#2E7D32"},
            "minor": {"border": "#2196F3", "bg_start": "#E3F2FD", "bg_end": "#BBDEFB", "color": "#1976D2"},
            "dominant": {"border": "#FF5722", "bg_start": "#FBE9E7", "bg_end": "#FFCCBC", "color": "#D84315"},
            "major_seventh": {"border": "#4CAF50", "bg_start": "#E8F5E8", "bg_end": "#A5D6A7", "color": "#1B5E20"},
            "minor_seventh": {"border": "#2196F3", "bg_start": "#E3F2FD", "bg_end": "#90CAF9", "color": "#0D47A1"},
            "ninth": {"border": "#9C27B0", "bg_start": "#F3E5F5", "bg_end": "#E1BEE7", "color": "#7B1FA2"},
            "eleventh": {"border": "#673AB7", "bg_start": "#EDE7F6", "bg_end": "#D1C4E9", "color": "#512DA8"},
            "thirteenth": {"border": "#3F51B5", "bg_start": "#E8EAF6", "bg_end": "#C5CAE9", "color": "#303F9F"},
            "diminished": {"border": "#795548", "bg_start": "#EFEBE9", "bg_end": "#D7CCC8", "color": "#5D4037"},
            "diminished7": {"border": "#6D4C41", "bg_start": "#D7CCC8", "bg_end": "#BCAAA4", "color": "#3E2723"},
            "half_diminished": {"border": "#8D6E63", "bg_start": "#EFEBE9", "bg_end": "#D7CCC8", "color": "#5D4037"},
            "augmented": {"border": "#FF9800", "bg_start": "#FFF3E0", "bg_end": "#FFE0B2", "color": "#F57C00"},
            "suspended": {"border": "#607D8B", "bg_start": "#ECEFF1", "bg_end": "#CFD8DC", "color": "#455A64"},
            "altered": {"border": "#E91E63", "bg_start": "#FCE4EC", "bg_end": "#F8BBD9", "color": "#C2185B"},
            "sixth": {"border": "#00BCD4", "bg_start": "#E0F7FA", "bg_end": "#B2EBF2", "color": "#00838F"}
        }
        
        style_info = styles.get(color_class, styles["default"])
        
        # Add subtle animation effect
        full_style = f"""
            QLabel {{
                border: 4px solid {style_info['border']};
                border-radius: 20px;
                padding: 30px;
                margin: 8px;
                background: qlineargradient(x1:0, y1:0, x2:0, y2:1,
                    stop:0 {style_info['bg_start']}, stop:0.5 {style_info['bg_end']}, 
                    stop:1 {style_info['bg_start']});
                color: {style_info['color']};
                font-size: 32px;
                font-weight: 700;
                text-align: center;
                letter-spacing: 2px;
            }}
        """
        
        self.chord_label.setStyleSheet(full_style)
        self.chord_label.setText(chord)
        
    def update_chord_list_display(self, elapsed_time=None):
        """Enhanced chord list display with better formatting"""
        if not self.chords:
            return
            
        self.chord_list.clear()
        max_chords = self.max_chords_spin.value()
        show_times = self.show_times_cb.isChecked()
        
        if elapsed_time is not None:
            # Show recent chords during playback
            recent_chords = []
            current_chord_idx = -1
            
            for i, (time_stamp, chord) in enumerate(self.chords):
                if time_stamp <= elapsed_time:
                    recent_chords.append((i, time_stamp, chord))
                    current_chord_idx = i
                else:
                    break
            
            # Show last N chords
            display_chords = recent_chords[-max_chords:]
            
            for idx, (orig_idx, time_stamp, chord) in enumerate(display_chords):
                if show_times:
                    time_str = f"{int(time_stamp//60):02d}:{int(time_stamp%60):02d}"
                    item_text = f"{time_str} - {chord}"
                else:
                    item_text = chord
                
                # Highlight current chord
                if orig_idx == current_chord_idx:
                    item_text = f"▶ {item_text}"
                
                self.chord_list.addItem(item_text)
        else:
            # Show all chords (when not playing)
            display_chords = self.chords[:max_chords]
            
            for time_stamp, chord in display_chords:
                if show_times:
                    time_str = f"{int(time_stamp//60):02d}:{int(time_stamp%60):02d}"
                    item_text = f"{time_str} - {chord}"
                else:
                    item_text = chord
                
                self.chord_list.addItem(item_text)
        
        # Auto-scroll to bottom during playback
        if elapsed_time is not None:
            self.chord_list.scrollToBottom()
            
    def export_results(self, format_type):
        """Enhanced export functionality with multiple formats"""
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
        elif format_type == 'midi':
            file_path, _ = QFileDialog.getSaveFileName(
                self, "Export as MIDI", "chord_progression.mid", "MIDI Files (*.mid)"
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
            elif format_type == 'midi':
                self.export_as_midi(file_path)
                
            QMessageBox.information(self, "Export Complete", 
                                  f"Analysis exported successfully to:\n{file_path}")
        except Exception as e:
            QMessageBox.critical(self, "Export Error", 
                               f"Failed to export analysis:\n{str(e)}")
            
    def export_as_text(self, file_path):
        """Export results as enhanced formatted text"""
        with open(file_path, 'w', encoding='utf-8') as f:
            f.write("🎼 Enhanced Chord Analysis Results v4.1\n")
            f.write("=" * 60 + "\n\n")
            f.write(f"📁 Source File: {getattr(self, 'audio_file_path', 'Unknown')}\n")
            f.write(f"🎵 Key Signature: {self.key_signature}\n")
            f.write(f"🎯 Total Chord Segments: {len(self.chords)}\n")
            f.write(f"⏱️  Analysis Duration: {self.audio_duration:.1f} seconds\n")
            f.write(f"🚀 Performance Mode: {'Enabled' if self.analysis_stats.get('performance_mode', False) else 'Disabled'}\n")
            f.write(f"📊 Analysis Time: {self.analysis_stats.get('total_analysis_time', 0):.2f} seconds\n\n")
            
            f.write("🎶 Detailed Chord Progression:\n")
            f.write("-" * 60 + "\n")
            f.write(f"{'Time':>8} {'Chord':>15} {'Duration':>12} {'Percentage':>12}\n")
            f.write("-" * 60 + "\n")
            
            total_duration = self.audio_duration
            for i, (time_stamp, chord) in enumerate(self.chords):
                if i < len(self.chords) - 1:
                    duration = self.chords[i+1][0] - time_stamp
                    duration_str = f"{duration:8.1f}s"
                    percentage = f"{(duration/total_duration)*100:8.1f}%"
                else:
                    duration_str = "---"
                    percentage = "---"
                    
                f.write(f"{time_stamp:8.1f}s {chord:>15} {duration_str:>12} {percentage:>12}\n")
            
            # Add statistics
            f.write(f"\n📊 Chord Statistics:\n")
            f.write("-" * 40 + "\n")
            
            chord_counts = {}
            for _, chord in self.chords:
                chord_counts[chord] = chord_counts.get(chord, 0) + 1
                
            most_common = sorted(chord_counts.items(), key=lambda x: x[1], reverse=True)[:10]
            f.write(f"{'Chord':>12} {'Count':>8} {'Frequency':>12}\n")
            f.write("-" * 40 + "\n")
            for chord, count in most_common:
                percentage = (count / len(self.chords)) * 100
                f.write(f"{chord:>12} {count:>8} {percentage:>10.1f}%\n")
                
    def export_as_json(self, file_path):
        """Export results as comprehensive JSON"""
        # Calculate additional statistics
        chord_counts = {}
        chord_durations = {}
        total_duration = 0
        
        for i, (time_stamp, chord) in enumerate(self.chords):
            chord_counts[chord] = chord_counts.get(chord, 0) + 1
            
            if i < len(self.chords) - 1:
                duration = self.chords[i+1][0] - time_stamp
                chord_durations[chord] = chord_durations.get(chord, 0) + duration
                total_duration += duration
        
        export_data = {
            "metadata": {
                "source_file": getattr(self, 'audio_file_path', 'Unknown'),
                "file_name": os.path.basename(getattr(self, 'audio_file_path', 'Unknown')),
                "key_signature": self.key_signature,
                "key_confidence": self.analysis_stats.get('key_confidence', 0),
                "total_segments": len(self.chords),
                "audio_duration": self.audio_duration,
                "analysis_time": self.analysis_stats.get('total_analysis_time', 0),
                "performance_mode": self.analysis_stats.get('performance_mode', False),
                "export_timestamp": time.time(),
                "analysis_version": "4.1",
                "estimated_tempo": getattr(self, 'estimated_tempo', None)
            },
            "analysis_settings": {
                "use_harmonic_separation": getattr(self, 'harmonic_cb', QCheckBox()).isChecked(),
                "use_multi_features": getattr(self, 'multi_feature_cb', QCheckBox()).isChecked(),
                "smoothing_method": getattr(self, 'smoothing_combo', QComboBox()).currentText(),
                "min_chord_duration": getattr(self, 'min_duration_spin', QDoubleSpinBox()).value(),
                "template_threshold": getattr(self, 'threshold_spin', QDoubleSpinBox()).value(),
                "performance_mode": getattr(self, 'performance_cb', QCheckBox()).isChecked()
            },
            "chord_progression": [
                {
                    "time": time_stamp,
                    "chord": chord,
                    "duration": (self.chords[i+1][0] - time_stamp 
                               if i < len(self.chords) - 1 else None),
                    "index": i
                }
                for i, (time_stamp, chord) in enumerate(self.chords)
            ],
            "statistics": {
                "chord_frequencies": chord_counts,
                "chord_durations": chord_durations,
                "most_common_chords": sorted(chord_counts.items(), key=lambda x: x[1], reverse=True)[:15],
                "unique_chords": len(set(chord for _, chord in self.chords if chord != "N.C.")),
                "total_chord_changes": len(self.chords),
                "harmonic_complexity": (len(set(chord for _, chord in self.chords if chord != "N.C.")) / len(self.chords)) * 100
            }
        }
        
        with open(file_path, 'w', encoding='utf-8') as f:
            json.dump(export_data, f, indent=2, ensure_ascii=False)
            
    def export_as_midi(self, file_path):
        """Export chord progression as MIDI file"""
        try:
            import mido
            from mido import MidiFile, MidiTrack, Message
        except ImportError:
            QMessageBox.critical(self, "Missing Dependency", 
                               "MIDI export requires the 'mido' library.\n\n"
                               "Install it with: pip install mido")
            return
        
        # Create MIDI file
        mid = MidiFile()
        track = MidiTrack()
        mid.tracks.append(track)
        
        # Set tempo (120 BPM default, or estimated tempo)
        tempo = getattr(self, 'estimated_tempo', 120)
        ticks_per_beat = mid.ticks_per_beat
        
        # Add tempo message
        tempo_message = mido.MetaMessage('set_tempo', tempo=mido.bpm2tempo(tempo))
        track.append(tempo_message)
        
        # Convert chords to MIDI notes
        chord_to_notes = {
            'C': [60], 'C#': [61], 'Db': [61], 'D': [62], 'D#': [63], 'Eb': [63],
            'E': [64], 'F': [65], 'F#': [66], 'Gb': [66], 'G': [67], 'G#': [68], 'Ab': [68],
            'A': [69], 'A#': [70], 'Bb': [70], 'B': [71]
        }
        
        def chord_to_midi_notes(chord_name):
            """Convert chord name to MIDI note numbers"""
            if chord_name == "N.C." or chord_name == "♪":
                return []
            
            # Extract root note
            root = chord_name[0]
            if len(chord_name) > 1 and chord_name[1] in ['#', 'b']:
                root = chord_name[:2]
                suffix = chord_name[2:]
            else:
                suffix = chord_name[1:]
            
            if root not in chord_to_notes:
                return []
            
            base_note = chord_to_notes[root][0]
            notes = [base_note]  # Root
            
            # Add chord tones based on suffix
            if suffix == '' or suffix == 'maj':  # Major
                notes.extend([base_note + 4, base_note + 7])  # 3rd, 5th
            elif suffix == 'm':  # Minor
                notes.extend([base_note + 3, base_note + 7])  # b3rd, 5th
            elif suffix == '7':  # Dominant 7th
                notes.extend([base_note + 4, base_note + 7, base_note + 10])
            elif suffix in ['M7', 'maj7']:  # Major 7th
                notes.extend([base_note + 4, base_note + 7, base_note + 11])
            elif suffix == 'm7':  # Minor 7th
                notes.extend([base_note + 3, base_note + 7, base_note + 10])
            elif suffix == 'dim':  # Diminished
                notes.extend([base_note + 3, base_note + 6])
            elif suffix == 'aug':  # Augmented
                notes.extend([base_note + 4, base_note + 8])
            elif suffix == 'sus4':  # Suspended 4th
                notes.extend([base_note + 5, base_note + 7])
            elif suffix == 'sus2':  # Suspended 2nd
                notes.extend([base_note + 2, base_note + 7])
            else:
                # Default to major if unknown
                notes.extend([base_note + 4, base_note + 7])
            
            return notes
        
        current_time = 0
        active_notes = []
        
        for i, (time_stamp, chord) in enumerate(self.chords):
            # Calculate duration in ticks
            if i < len(self.chords) - 1:
                duration = self.chords[i+1][0] - time_stamp
            else:
                duration = 1.0  # Default duration for last chord
            
            duration_ticks = int(duration * ticks_per_beat * (tempo / 60))
            
            # Turn off previous notes
            for note in active_notes:
                track.append(Message('note_off', note=note, velocity=64, time=0))
            active_notes = []
            
            # Turn on new notes
            notes = chord_to_midi_notes(chord)
            for j, note in enumerate(notes):
                time_delta = duration_ticks if j == 0 else 0
                track.append(Message('note_on', note=note, velocity=80, time=time_delta))
                active_notes.append(note)
        
        # Turn off final notes
        for note in active_notes:
            track.append(Message('note_off', note=note, velocity=64, time=duration_ticks))
        
        # Save MIDI file
        mid.save(file_path)
        
    def closeEvent(self, event):
        """Enhanced application close with cleanup"""
        # Stop playback
        sd.stop()
        
        # Save settings
        self.save_settings()
        
        # Clean up threads
        if self.analysis_thread and self.analysis_thread.isRunning():
            self.analysis_thread.quit()
            if not self.analysis_thread.wait(3000):  # Wait up to 3 seconds
                self.analysis_thread.terminate()
                
        event.accept()

# --- Main Application ---
def main():
    """Enhanced main entry point with better error handling"""
    import sys
    
    # Check dependencies
    try:
        import librosa
        import sounddevice as sd
        import numpy as np
        from scipy import signal
        from sklearn.preprocessing import normalize
    except ImportError as e:
        print(f"Missing required dependency: {e}")
        print("Please install required packages:")
        print("pip install librosa sounddevice numpy scipy scikit-learn PyQt6")
        sys.exit(1)
    
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
    app.setApplicationVersion("4.1")
    app.setOrganizationName("MusicTech Solutions")
    app.setOrganizationDomain("musictech.solutions")
    
    # Set application icon and metadata
    app.setApplicationDisplayName("Enhanced Chord Player Pro v4.1")
    
    # Apply global application style
    app.setStyleSheet("""
        QApplication {
            font-family: 'Segoe UI', 'San Francisco', 'Helvetica Neue', Arial, sans-serif;
        }
        QMainWindow {
            background-color: #FAFAFA;
        }
        QToolTip {
            background-color: #263238;
            color: #FFFFFF;
            border: 1px solid #37474F;
            border-radius: 6px;
            padding: 8px;
            font-size: 11px;
        }
        QScrollArea {
            border: none;
            background-color: transparent;
        }
        QScrollBar:vertical {
            background-color: #F5F5F5;
            width: 12px;
            border-radius: 6px;
        }
        QScrollBar::handle:vertical {
            background-color: #BDBDBD;
            border-radius: 6px;
            min-height: 20px;
        }
        QScrollBar::handle:vertical:hover {
            background-color: #9E9E9E;
        }
    """)
    
    # Create splash screen for loading
    splash_pixmap = QPixmap(400, 200)
    splash_pixmap.fill(QColor("#2196F3"))
    
    painter = QPainter(splash_pixmap)
    painter.setFont(QFont("Arial", 16, QFont.Weight.Bold))
    painter.setPen(QColor("white"))
    painter.drawText(splash_pixmap.rect(), Qt.AlignmentFlag.AlignCenter, 
                    "Enhanced Chord Player Pro v4.1\nLoading...")
    painter.end()
    
    from PyQt6.QtWidgets import QSplashScreen
    splash = QSplashScreen(splash_pixmap)
    splash.show()
    app.processEvents()
    
    # Initialize cache directory
    os.makedirs(CHORDS_CACHE_DIR, exist_ok=True)
    
    try:
        # Create and show main window
        player = EnhancedChordPlayer()
        splash.finish(player)
        player.show()
        
        # Show welcome message for first-time users
        settings = QSettings("MusicTech", "ChordPlayerPro")
        if not settings.value("first_run_complete", False, type=bool):
            QMessageBox.information(player, "Welcome to Enhanced Chord Player Pro v4.1!", 
                                  "🎼 Welcome to Enhanced Chord Player Pro v4.1!\n\n"
                                  "New features in this version:\n"
                                  "• 🚀 Performance mode for faster analysis\n"
                                  "• 🎹 MIDI export capability\n"
                                  "• 📊 Enhanced statistics and visualizations\n"
                                  "• 🎯 Improved chord detection accuracy\n"
                                  "• 🎨 Modern, responsive user interface\n\n"
                                  "Quick start:\n"
                                  "1. Click 'Load Audio File' to select your music\n"
                                  "2. Choose analysis settings or use presets\n"
                                  "3. Watch real-time chord detection during playback\n"
                                  "4. Export results in multiple formats\n\n"
                                  "Tip: Try the 'Balanced' preset for optimal results!")
            settings.setValue("first_run_complete", True)
        
        return app.exec()
        
    except Exception as e:
        splash.hide()
        QMessageBox.critical(None, "Startup Error", 
                           f"Failed to start Enhanced Chord Player Pro:\n\n{str(e)}\n\n"
                           f"Please check your Python environment and dependencies.")
        return 1

if __name__ == "__main__":
    sys.exit(main())
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
    QSplitter, QListWidget, QFrame, QScrollArea, QToolTip
)
from PyQt6.QtCore import QTimer, Qt, QThread, pyqtSignal, QObject, QSettings
from PyQt6.QtGui import QFont, QPalette, QColor, QPixmap, QPainter, QAction

# --- Enhanced Configuration with Optimizations ---
CHORDS_CACHE_DIR = "chord_cache_v41"
CHORDS_CACHE_EXTENSION = ".json"
SR = 22050
FFT_SIZE = 16384
HOP_LENGTH = 512
NOTE_NAMES = ['C','C#','D','D#','E','F','F#','G','G#','A','A#','B']

# Enhanced chord definitions with better organization
CHORD_DEFS = {
    # Basic triads (high priority)
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
    
    # 7th chords (very common)
    '7': [0,4,7,10],       # Dominant 7th
    'M7': [0,4,7,11],      # Major 7th
    'maj7': [0,4,7,11],    # Alternative notation
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
    
    # Jazz extensions
    'maj7#11': [0,4,7,11,18%12], # Major 7th sharp 11
    'm11': [0,3,7,10,14%12,17%12], # Minor 11th
    'm13': [0,3,7,10,14%12,21%12], # Minor 13th
}

# Chord priority weights for detection order
CHORD_PRIORITY = {
    '': 1.0,      # Major - highest priority
    'm': 1.0,     # Minor - highest priority
    '7': 0.9,     # Dominant 7th
    'm7': 0.9,    # Minor 7th
    'M7': 0.8,    # Major 7th
    'maj7': 0.8,  # Major 7th alt
    'dim': 0.7,   # Diminished
    'sus4': 0.6,  # Suspended 4th
    'sus2': 0.6,  # Suspended 2nd
    '6': 0.5,     # Sixth chords
    'add9': 0.4,  # Add 9th
    # All others get default 0.3
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
    'bass': 0.8
}

# --- Performance Optimizations ---
class PerformanceMonitor:
    """Monitor and optimize performance"""
    def __init__(self):
        self.timings = {}
        
    def start_timer(self, name):
        self.timings[name] = time.time()
        
    def end_timer(self, name):
        if name in self.timings:
            duration = time.time() - self.timings[name]
            print(f"[PERF] {name}: {duration:.3f}s")
            return duration
        return 0

# Global performance monitor
perf_monitor = PerformanceMonitor()

# --- Enhanced Analysis Worker ---
class ChordAnalysisWorker(QObject):
    progress = pyqtSignal(int)
    finished = pyqtSignal(list, str, dict)  # chords, key_signature, analysis_stats
    error = pyqtSignal(str)
    status_update = pyqtSignal(str)
    
    def __init__(self, file_path, settings):
        super().__init__()
        self.file_path = file_path
        self.settings = settings
        
    def run(self):
        try:
            self.status_update.emit("Initializing enhanced analysis...")
            perf_monitor.start_timer("total_analysis")
            
            chords, key_sig, stats = detect_chords_enhanced(
                self.file_path, 
                progress_callback=self.progress.emit,
                status_callback=self.status_update.emit,
                **self.settings
            )
            
            total_time = perf_monitor.end_timer("total_analysis")
            stats['total_analysis_time'] = total_time
            
            self.finished.emit(chords, key_sig, stats)
        except Exception as e:
            self.error.emit(str(e))

# --- Optimized Detection Functions ---
def create_enhanced_chord_templates():
    """Create optimized chord templates with better caching"""
    cache_file = os.path.join(CHORDS_CACHE_DIR, "templates_v41.json")
    
    # Try to load from cache
    if os.path.exists(cache_file):
        try:
            with open(cache_file, 'r') as f:
                cached_data = json.load(f)
                templates = {k: np.array(v) for k, v in cached_data.items()}
                print(f"[CACHE] Loaded {len(templates)} chord templates from cache")
                return templates
        except:
            pass
    
    print("[COMPUTE] Computing chord templates...")
    perf_monitor.start_timer("template_creation")
    
    templates = {}
    
    # Create templates with priority-based ordering
    chord_items = list(CHORD_DEFS.items())
    chord_items.sort(key=lambda x: CHORD_PRIORITY.get(x[0], 0.3), reverse=True)
    
    for root_idx, note in enumerate(NOTE_NAMES):
        for suffix, intervals in chord_items:
            name = note + suffix
            vec = np.zeros(12, dtype=np.float32)  # Use float32 for memory efficiency
            
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
    
    # Cache templates for future use
    try:
        os.makedirs(CHORDS_CACHE_DIR, exist_ok=True)
        cache_data = {k: v.tolist() for k, v in templates.items()}
        with open(cache_file, 'w') as f:
            json.dump(cache_data, f)
        print(f"[CACHE] Saved {len(templates)} templates to cache")
    except Exception as e:
        print(f"[WARNING] Could not cache templates: {e}")
    
    perf_monitor.end_timer("template_creation")
    return templates

def extract_enhanced_features(y, sr=SR):
    """Extract optimized audio features for chord detection"""
    perf_monitor.start_timer("feature_extraction")
    
    features = {}
    
    # Multi-resolution chroma with CQT (primary feature)
    features['chroma_cqt'] = librosa.feature.chroma_cqt(
        y=y, sr=sr, hop_length=HOP_LENGTH, fmin=librosa.note_to_hz('C1'),
        n_chroma=12, n_octaves=6
    )
    
    # STFT-based chroma for comparison (faster)
    features['chroma_stft'] = librosa.feature.chroma_stft(
        y=y, sr=sr, hop_length=HOP_LENGTH, n_fft=FFT_SIZE
    )
    
    # Harmonic-percussive separation (computationally expensive, make optional)
    try:
        y_harmonic, _ = librosa.effects.hpss(y, margin=(1.0, 5.0))
        features['chroma_harmonic'] = librosa.feature.chroma_cqt(
            y=y_harmonic, sr=sr, hop_length=HOP_LENGTH
        )
    except:
        # Fallback if HPSS fails
        features['chroma_harmonic'] = features['chroma_cqt']
    
    # Additional features for context
    features['spectral_centroid'] = librosa.feature.spectral_centroid(
        y=y, sr=sr, hop_length=HOP_LENGTH
    )
    
    features['rms'] = librosa.feature.rms(y=y, hop_length=HOP_LENGTH)
    
    perf_monitor.end_timer("feature_extraction")
    return features

def apply_intelligent_smoothing(chroma_sequence, features=None, method='adaptive'):
    """Optimized intelligent smoothing"""
    perf_monitor.start_timer("smoothing")
    
    if method == 'adaptive' and features is not None:
        # Use spectral flux to determine stability regions
        if 'spectral_centroid' in features:
            centroid = features['spectral_centroid'][0]
            centroid_diff = np.abs(np.diff(centroid))
            
            # Simplified adaptive sigma calculation
            sigma_base = 2.0
            stability = 1.0 / (1.0 + centroid_diff * 1000)
            sigma_adaptive = sigma_base * (0.5 + stability)
            
            # Apply gaussian smoothing with varying sigma
            smoothed = np.zeros_like(chroma_sequence)
            for i in range(12):
                # Use scipy's gaussian filter for efficiency
                smoothed[i, :] = gaussian_filter1d(
                    chroma_sequence[i, :], 
                    sigma=np.mean(sigma_adaptive)  # Use average sigma for simplicity
                )
        else:
            # Fallback to standard gaussian
            smoothed = gaussian_filter1d(chroma_sequence, sigma=2.0, axis=1)
    elif method == 'median':
        # Fast median filtering
        smoothed = np.zeros_like(chroma_sequence)
        for i in range(12):
            smoothed[i, :] = median_filter(chroma_sequence[i, :], size=5)
    else:
        # Simple gaussian smoothing (fastest)
        smoothed = gaussian_filter1d(chroma_sequence, sigma=1.5, axis=1)
    
    perf_monitor.end_timer("smoothing")
    return smoothed

def detect_key_enhanced(chroma_features, method='krumhansl'):
    """Enhanced key detection with confidence scoring"""
    perf_monitor.start_timer("key_detection")
    
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
    
    perf_monitor.end_timer("key_detection")
    
    if key_correlations:
        # Sort by correlation strength
        key_correlations.sort(key=lambda x: x[1], reverse=True)
        best_key = key_correlations[0]
        
        # Calculate confidence as difference between best and second-best
        confidence = best_key[1]
        if len(key_correlations) > 1:
            confidence = min(1.0, (best_key[1] - key_correlations[1][1]) * 2)
        
        return best_key[0], confidence
    else:
        return "Unknown", 0.0

def detect_chords_enhanced(path, progress_callback=None, status_callback=None, **settings):
    """Optimized chord detection with comprehensive analysis"""
    
    # Default settings with performance options
    default_settings = {
        'use_harmonic_separation': True,
        'smoothing_method': 'adaptive',
        'min_chord_duration': 0.3,
        'template_threshold': 0.25,
        'key_aware': True,
        'use_multi_features': True,
        'performance_mode': False,  # New setting for faster analysis
        'max_chord_types': 50       # Limit chord types for performance
    }
    
    # Merge with provided settings
    for key, value in default_settings.items():
        if key not in settings:
            settings[key] = value
    
    # Cache management
    os.makedirs(CHORDS_CACHE_DIR, exist_ok=True)
    fh = get_file_hash(path)
    cache_key = f"{fh}_enhanced_v41_{hash(str(sorted(settings.items())))}"
    cache_file = os.path.join(CHORDS_CACHE_DIR, cache_key + CHORDS_CACHE_EXTENSION)
    
    if os.path.exists(cache_file):
        try:
            with open(cache_file, 'r') as f:
                data = json.load(f)
                if status_callback:
                    status_callback("Loaded from cache!")
                return data['chords'], data.get('key_signature', 'Unknown'), data.get('stats', {})
        except:
            pass
    
    # Initialize analysis statistics
    analysis_stats = {
        'cache_hit': False,
        'audio_duration': 0,
        'total_frames': 0,
        'chord_changes': 0,
        'key_confidence': 0,
        'performance_mode': settings['performance_mode']
    }
    
    if progress_callback:
        progress_callback(5)
    if status_callback:
        status_callback("Loading audio file...")
    
    # Load audio
    perf_monitor.start_timer("audio_loading")
    y, sr = librosa.load(path, sr=SR)
    analysis_stats['audio_duration'] = len(y) / sr
    perf_monitor.end_timer("audio_loading")
    
    if progress_callback:
        progress_callback(15)
    if status_callback:
        status_callback("Extracting enhanced features...")
    
    # Extract enhanced features
    features = extract_enhanced_features(y, sr)
    
    if progress_callback:
        progress_callback(35)
    if status_callback:
        status_callback("Analyzing harmonic content...")
    
    # Choose primary chroma feature based on settings
    if settings['use_multi_features'] and not settings['performance_mode']:
        # Combine multiple chroma features with weighting
        chroma_combined = (
            0.4 * features['chroma_cqt'] +
            0.3 * features['chroma_harmonic'] +
            0.3 * features['chroma_stft']
        )
    else:
        # Use single feature for performance
        chroma_combined = features['chroma_cqt']
    
    # Apply intelligent smoothing
    smoothed_chroma = apply_intelligent_smoothing(
        chroma_combined, features, method=settings['smoothing_method']
    )
    
    analysis_stats['total_frames'] = smoothed_chroma.shape[1]
    
    if progress_callback:
        progress_callback(55)
    if status_callback:
        status_callback("Detecting key signature...")
    
    # Key detection
    key_signature = "Unknown"
    key_confidence = 0.0
    if settings['key_aware']:
        key_signature, key_confidence = detect_key_enhanced(smoothed_chroma)
        analysis_stats['key_confidence'] = key_confidence
    
    if progress_callback:
        progress_callback(65)
    if status_callback:
        status_callback("Building optimized chord templates...")
    
    # Normalize chroma
    chroma_norm = normalize(smoothed_chroma, axis=0, norm='l2')
    
    # Create enhanced templates
    templates = create_enhanced_chord_templates()
    
    # Limit templates for performance mode
    if settings['performance_mode']:
        # Keep only the most common chord types
        priority_chords = ['', 'm', '7', 'm7', 'M7', 'maj7', 'dim', 'sus4', 'sus2']
        filtered_templates = {}
        for note in NOTE_NAMES:
            for chord_type in priority_chords:
                chord_name = note + chord_type
                if chord_name in templates:
                    filtered_templates[chord_name] = templates[chord_name]
        templates = filtered_templates
        print(f"[PERF] Using {len(templates)} optimized templates")
    
    if progress_callback:
        progress_callback(75)
    if status_callback:
        status_callback("Performing chord matching...")
    
    # Enhanced chord detection with optimized matching
    perf_monitor.start_timer("chord_matching")
    chords = []
    times = librosa.frames_to_time(np.arange(chroma_norm.shape[1]), sr=sr, hop_length=HOP_LENGTH)
    
    # Pre-compute template matrix for vectorized operations
    template_names = list(templates.keys())
    template_matrix = np.array([templates[name] for name in template_names]).T
    
    frame_count = chroma_norm.shape[1]
    update_interval = max(10, frame_count // 50)  # More frequent updates
    
    # Vectorized chord detection for better performance
    batch_size = 100 if settings['performance_mode'] else 50
    
    for batch_start in range(0, frame_count, batch_size):
        batch_end = min(batch_start + batch_size, frame_count)
        batch_chroma = chroma_norm[:, batch_start:batch_end]
        
        # Vectorized similarity computation
        similarities = np.dot(template_matrix.T, batch_chroma)
        
        # Find best matches for each frame in batch
        best_indices = np.argmax(similarities, axis=0)
        best_scores = np.max(similarities, axis=0)
        
        for i, (frame_idx, best_idx, score) in enumerate(zip(
            range(batch_start, batch_end), best_indices, best_scores
        )):
            if frame_idx % update_interval == 0 and progress_callback:
                new_progress = min(95, 75 + int(20 * (frame_idx / frame_count)))
                progress_callback(new_progress)
            
            if score >= settings['template_threshold']:
                chord_name = template_names[best_idx]
            else:
                chord_name = "N.C."
            
            chords.append((times[frame_idx], chord_name))
    
    perf_monitor.end_timer("chord_matching")
    
    if progress_callback:
        progress_callback(90)
    if status_callback:
        status_callback("Post-processing results...")
    
    # Enhanced post-processing
    chords = post_process_enhanced(chords, settings['min_chord_duration'], key_signature)
    analysis_stats['chord_changes'] = len(chords)
    
    # Cache results
    try:
        cache_data = {
            'chords': chords,
            'key_signature': key_signature,
            'stats': analysis_stats,
            'analysis_settings': settings
        }
        with open(cache_file, 'w') as f:
            json.dump(cache_data, f)
        print(f"[CACHE] Saved analysis results to cache")
    except Exception as e:
        print(f"[WARNING] Could not cache results: {e}")
    
    if progress_callback:
        progress_callback(100)
    if status_callback:
        status_callback("Analysis complete!")
    
    return chords, key_signature, analysis_stats

def post_process_enhanced(chords, min_duration=0.3, key_signature=None):
    """Enhanced post-processing with better musical intelligence"""
    if not chords:
        return []
    
    perf_monitor.start_timer("post_processing")
    
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
    
    # Advanced filtering with musical context
    filtered_chords = []
    for i, (t, chord) in enumerate(merged_chords):
        keep = True
        
        # Skip very short chords unless they're harmonically important
        if i > 0 and i < len(merged_chords)-1:
            prev_chord = merged_chords[i-1][1]
            next_chord = merged_chords[i+1][1]
            
            # Check for musical patterns
            if chord == prev_chord and chord == next_chord:
                keep = False  # Remove redundant repetitions
            elif chord != prev_chord and chord != next_chord:
                duration = merged_chords[i+1][0] - t if i < len(merged_chords)-1 else min_duration
                if duration < min_duration and not is_harmonically_important(chord, merged_chords, i):
                    keep = False
        
        if keep:
            filtered_chords.append((t, chord))
    
    perf_monitor.end_timer("post_processing")
    return filtered_chords

def is_harmonically_important(chord, sequence, index):
    """Enhanced harmonic importance detection"""
    # Dominant chords are usually important
    if any(x in chord for x in ['7', 'V', 'dom']):
        return True
    
    # Diminished chords often serve as passing chords
    if 'dim' in chord:
        return True
    
    # Check for common progressions
    if index > 0 and index < len(sequence) - 1:
        prev_chord = sequence[index-1][1]
        next_chord = sequence[index+1][1]
        
        # V-I resolution patterns
        if '7' in chord and any(x in next_chord for x in ['', 'm']):
            return True
        
        # ii-V-I patterns
        if 'm7' in chord and '7' in next_chord:
            return True
    
    return False

def get_file_hash(path):
    """Generate SHA-256 hash of file for caching"""
    hasher = hashlib.sha256()
    with open(path, 'rb') as f:
        for chunk in iter(lambda: f.read(8192), b""):
            hasher.update(chunk)
    return hasher.hexdigest()

# --- Enhanced UI with Better UX ---
class EnhancedChordPlayer(QWidget):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("Enhanced Chord Player Pro v4.1")
        self.resize(1200, 800)
        
        # Initialize variables
        self.chords = []
        self.key_signature = "Unknown"
        self.analysis_stats = {}
        self.start_time = None
        self.audio_duration = 0
        self.analysis_thread = None
        self.worker = None
        self.settings = QSettings("MusicTech", "ChordPlayerPro")
        
        self.init_enhanced_ui()
        self.apply_modern_theme()
        self.load_settings()
        
    def init_enhanced_ui(self):
        """Initialize enhanced user interface with better organization"""
        main_layout = QVBoxLayout(self)
        
        # Create main splitter
        main_splitter = QSplitter(Qt.Orientation.Horizontal)
        main_layout.addWidget(main_splitter)
        
        # Left panel - Primary controls and display
        left_panel = self.create_left_panel()
        main_splitter.addWidget(left_panel)
        
        # Right panel - Advanced settings and analysis
        right_panel = self.create_right_panel()
        main_splitter.addWidget(right_panel)
        
        # Set splitter proportions (60% left, 40% right)
        main_splitter.setSizes([720, 480])
        
        # Status bar
        self.create_status_bar(main_layout)
        
        # Timer for updates
        self.timer = QTimer(self)
        self.timer.setInterval(50)
        self.timer.timeout.connect(self.update_display)
        
    def create_left_panel(self):
        """Create the main control and display panel"""
        left_widget = QWidget()
        left_layout = QVBoxLayout(left_widget)
        
        # File controls with enhanced styling
        file_group = QGroupBox("🎵 Audio File Controls")
        file_layout = QHBoxLayout(file_group)
        
        self.btn_load = QPushButton("📁 Load Audio File")
        self.btn_load.clicked.connect(self.load_audio)
        self.btn_load.setToolTip("Load an audio file for chord analysis\nSupported formats: WAV, MP3, FLAC, AAC, M4A, OGG")
        file_layout.addWidget(self.btn_load)
        
        self.btn_stop = QPushButton("⏹️ Stop Playback")
        self.btn_stop.clicked.connect(self.stop_playback)
        self.btn_stop.setEnabled(False)
        self.btn_stop.setToolTip("Stop audio playback and analysis")
        file_layout.addWidget(self.btn_stop)
        
        self.btn_pause = QPushButton("⏸️ Pause")
        self.btn_pause.clicked.connect(self.toggle_pause)
        self.btn_pause.setEnabled(False)
        self.btn_pause.setToolTip("Pause/Resume playback")
        file_layout.addWidget(self.btn_pause)
        
        left_layout.addWidget(file_group)
        
        # Progress and status with enhanced display
        progress_group = QGroupBox("📊 Analysis Progress")
        progress_layout = QVBoxLayout(progress_group)
        
        self.progress_bar = QProgressBar()
        self.progress_bar.setTextVisible(True)
        progress_layout.addWidget(self.progress_bar)
        
        # Status with performance indicators
        status_layout = QHBoxLayout()
        self.status_label = QLabel("Ready to analyze audio")
        self.performance_label = QLabel("")
        self.performance_label.setStyleSheet("color: #666; font-size: 10px;")
        status_layout.addWidget(self.status_label)
        status_layout.addStretch()
        status_layout.addWidget(self.performance_label)
        progress_layout.addLayout(status_layout)
        
        left_layout.addWidget(progress_group)
        
        # Enhanced current chord display
        chord_group = QGroupBox("🎼 Current Analysis")
        chord_layout = QVBoxLayout(chord_group)
        
        # Main chord display
        self.chord_label = QLabel("Load an audio file to begin")
        self.chord_label.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self.chord_label.setMinimumHeight(120)
        chord_layout.addWidget(self.chord_label)
        
        # Info row with key and time
        info_layout = QHBoxLayout()
        
        self.key_label = QLabel("Key: Unknown")
        self.key_label.setAlignment(Qt.AlignmentFlag.AlignCenter)
        info_layout.addWidget(self.key_label)
        
        self.time_label = QLabel("00:00 / 00:00")
        self.time_label.setAlignment(Qt.AlignmentFlag.AlignCenter)
        info_layout.addWidget(self.time_label)
        
        chord_layout.addLayout(info_layout)
        
        # Confidence and additional info
        detail_layout = QHBoxLayout()
        self.confidence_label = QLabel("Confidence: -")
        self.confidence_label.setAlignment(Qt.AlignmentFlag.AlignCenter)
        detail_layout.addWidget(self.confidence_label)
        
        self.tempo_label = QLabel("♩ = - BPM")
        self.tempo_label.setAlignment(Qt.AlignmentFlag.AlignCenter)
        detail_layout.addWidget(self.tempo_label)
        
        chord_layout.addLayout(detail_layout)
        left_layout.addWidget(chord_group)
        
        # Chord progression display with enhanced features
        progression_group = QGroupBox("🎶 Chord Progression")
        progression_layout = QVBoxLayout(progression_group)
        
        # Controls for progression view
        prog_controls = QHBoxLayout()
        self.show_times_cb = QCheckBox("Show Times")
        self.show_times_cb.setChecked(True)
        self.show_times_cb.stateChanged.connect(self.update_chord_list_display)
        prog_controls.addWidget(self.show_times_cb)
        
        self.max_chords_spin = QSpinBox()
        self.max_chords_spin.setRange(5, 50)
        self.max_chords_spin.setValue(12)
        self.max_chords_spin.setPrefix("Show last ")
        self.max_chords_spin.setSuffix(" chords")
        self.max_chords_spin.valueChanged.connect(self.update_chord_list_display)
        prog_controls.addWidget(self.max_chords_spin)
        
        prog_controls.addStretch()
        progression_layout.addLayout(prog_controls)
        
        self.chord_list = QListWidget()
        self.chord_list.setMaximumHeight(180)
        self.chord_list.setAlternatingRowColors(True)
        progression_layout.addWidget(self.chord_list)
        
        left_layout.addWidget(progression_group)
        
        return left_widget
        
    def create_right_panel(self):
        """Create the settings and analysis panel"""
        right_widget = QWidget()
        tabs = QTabWidget()
        right_widget_layout = QVBoxLayout(right_widget)
        right_widget_layout.addWidget(tabs)
        
        # Enhanced Settings tab
        settings_tab = QWidget()
        tabs.addTab(settings_tab, "⚙️ Settings")
        self.setup_enhanced_settings_tab(settings_tab)
        
        # Analysis Results tab
        results_tab = QWidget()
        tabs.addTab(results_tab, "📈 Analysis")
        self.setup_results_tab(results_tab)
        
        # Performance tab
        performance_tab = QWidget()
        tabs.addTab(performance_tab, "🚀 Performance")
        self.setup_performance_tab(performance_tab)
        
        return right_widget
        
    def create_status_bar(self, layout):
        """Create enhanced status bar"""
        status_frame = QFrame()
        status_frame.setFrameStyle(QFrame.Shape.StyledPanel)
        status_layout = QHBoxLayout(status_frame)
        status_layout.setContentsMargins(5, 2, 5, 2)
        
        self.file_info_label = QLabel("No file loaded")
        self.file_info_label.setStyleSheet("color: #666;")
        status_layout.addWidget(self.file_info_label)
        
        status_layout.addStretch()
        
        self.cache_status_label = QLabel("Cache: Ready")
        self.cache_status_label.setStyleSheet("color: #666;")
        status_layout.addWidget(self.cache_status_label)
        
        layout.addWidget(status_frame)
        
    def setup_enhanced_settings_tab(self, parent):
        """Setup enhanced settings interface with better organization"""
        scroll_area = QScrollArea()
        scroll_widget = QWidget()
        layout = QVBoxLayout(scroll_widget)
        
        # Analysis Quality Settings
        quality_group = QGroupBox("🎯 Analysis Quality")
        quality_layout = QGridLayout(quality_group)
        
        # Performance mode toggle
        quality_layout.addWidget(QLabel("Performance Mode:"), 0, 0)
        self.performance_cb = QCheckBox("Enable fast analysis")
        self.performance_cb.setToolTip("Faster analysis with reduced accuracy\nUses fewer chord types and simplified algorithms")
        quality_layout.addWidget(self.performance_cb, 0, 1)
        
        # Multi-feature analysis
        quality_layout.addWidget(QLabel("Multi-Feature Analysis:"), 1, 0)
        self.multi_feature_cb = QCheckBox("Use multiple algorithms")
        self.multi_feature_cb.setChecked(True)
        self.multi_feature_cb.setToolTip("Combines CQT, STFT, and harmonic features\nMore accurate but slower")
        quality_layout.addWidget(self.multi_feature_cb, 1, 1)
        
        # Harmonic separation
        quality_layout.addWidget(QLabel("Harmonic Separation:"), 2, 0)
        self.harmonic_cb = QCheckBox("Separate harmonic content")
        self.harmonic_cb.setChecked(True)
        self.harmonic_cb.setToolTip("Isolates harmonic content from percussive\nImproves chord detection accuracy")
        quality_layout.addWidget(self.harmonic_cb, 2, 1)
        
        layout.addWidget(quality_group)
        
        # Signal Processing Settings
        signal_group = QGroupBox("🎛️ Signal Processing")
        signal_layout = QGridLayout(signal_group)
        
        # Smoothing method
        signal_layout.addWidget(QLabel("Smoothing Method:"), 0, 0)
        self.smoothing_combo = QComboBox()
        self.smoothing_combo.addItems(['adaptive', 'gaussian', 'median'])
        self.smoothing_combo.setToolTip("adaptive: Context-aware smoothing\ngaussian: Standard smoothing\nmedian: Noise-resistant smoothing")
        signal_layout.addWidget(self.smoothing_combo, 0, 1)
        
        # Detection threshold
        signal_layout.addWidget(QLabel("Detection Threshold:"), 1, 0)
        self.threshold_spin = QDoubleSpinBox()
        self.threshold_spin.setRange(0.1, 0.9)
        self.threshold_spin.setValue(0.25)
        self.threshold_spin.setSingleStep(0.05)
        self.threshold_spin.setToolTip("Minimum confidence for chord detection\nLower = more chords detected\nHigher = only confident detections")
        signal_layout.addWidget(self.threshold_spin, 1, 1)
        
        layout.addWidget(signal_group)
        
        # Musical Analysis Settings
        musical_group = QGroupBox("🎼 Musical Analysis")
        musical_layout = QGridLayout(musical_group)
        
        # Minimum chord duration
        musical_layout.addWidget(QLabel("Min Chord Duration:"), 0, 0)
        self.min_duration_spin = QDoubleSpinBox()
        self.min_duration_spin.setRange(0.1, 5.0)
        self.min_duration_spin.setValue(0.3)
        self.min_duration_spin.setSingleStep(0.1)
        self.min_duration_spin.setSuffix(" sec")
        self.min_duration_spin.setToolTip("Minimum duration for a chord to be recognized\nFilters out very brief chord changes")
        musical_layout.addWidget(self.min_duration_spin, 0, 1)
        
        # Key-aware analysis
        musical_layout.addWidget(QLabel("Key-Aware Analysis:"), 1, 0)
        self.key_aware_cb = QCheckBox("Use key context")
        self.key_aware_cb.setChecked(True)
        self.key_aware_cb.setToolTip("Considers detected key for chord recognition\nImproves accuracy for tonal music")
        musical_layout.addWidget(self.key_aware_cb, 1, 1)
        
        # Maximum chord types
        musical_layout.addWidget(QLabel("Max Chord Types:"), 2, 0)
        self.max_chord_types_spin = QSpinBox()
        self.max_chord_types_spin.setRange(20, 200)
        self.max_chord_types_spin.setValue(100)
        self.max_chord_types_spin.setToolTip("Maximum number of chord types to consider\nLower values = faster analysis")
        musical_layout.addWidget(self.max_chord_types_spin, 2, 1)
        
        layout.addWidget(musical_group)
        
        # Export and Presets
        export_group = QGroupBox("💾 Export & Presets")
        export_layout = QVBoxLayout(export_group)
        
        # Preset buttons
        preset_layout = QHBoxLayout()
        
        self.btn_preset_fast = QPushButton("⚡ Fast")
        self.btn_preset_fast.clicked.connect(lambda: self.apply_preset('fast'))
        self.btn_preset_fast.setToolTip("Optimized for speed\nGood for real-time analysis")
        preset_layout.addWidget(self.btn_preset_fast)
        
        self.btn_preset_balanced = QPushButton("⚖️ Balanced")
        self.btn_preset_balanced.clicked.connect(lambda: self.apply_preset('balanced'))
        self.btn_preset_balanced.setToolTip("Balance of speed and accuracy\nRecommended for most users")
        preset_layout.addWidget(self.btn_preset_balanced)
        
        self.btn_preset_accurate = QPushButton("🎯 Accurate")
        self.btn_preset_accurate.clicked.connect(lambda: self.apply_preset('accurate'))
        self.btn_preset_accurate.setToolTip("Maximum accuracy\nBest for detailed analysis")
        preset_layout.addWidget(self.btn_preset_accurate)
        
        export_layout.addLayout(preset_layout)
        
        # Export buttons
        export_buttons = QHBoxLayout()
        
        self.btn_export_txt = QPushButton("📄 Export Text")
        self.btn_export_txt.clicked.connect(lambda: self.export_results('txt'))
        self.btn_export_txt.setEnabled(False)
        export_buttons.addWidget(self.btn_export_txt)
        
        self.btn_export_json = QPushButton("📊 Export JSON")
        self.btn_export_json.clicked.connect(lambda: self.export_results('json'))
        self.btn_export_json.setEnabled(False)
        export_buttons.addWidget(self.btn_export_json)
        
        self.btn_export_midi = QPushButton("🎹 Export MIDI")
        self.btn_export_midi.clicked.connect(lambda: self.export_results('midi'))
        self.btn_export_midi.setEnabled(False)
        self.btn_export_midi.setToolTip("Export chord progression as MIDI file")
        export_buttons.addWidget(self.btn_export_midi)
        
        export_layout.addLayout(export_buttons)
        layout.addWidget(export_group)
        
        layout.addStretch()
        scroll_area.setWidget(scroll_widget)
        
        main_layout = QVBoxLayout(parent)
        main_layout.addWidget(scroll_area)
        
    def setup_results_tab(self, parent):
        """Setup enhanced results display"""
        layout = QVBoxLayout(parent)
        
        # Analysis summary with enhanced metrics
        summary_group = QGroupBox("📊 Analysis Summary")
        summary_layout = QGridLayout(summary_group)
        
        self.total_chords_label = QLabel("Total Chords: 0")
        summary_layout.addWidget(self.total_chords_label, 0, 0)
        
        self.avg_duration_label = QLabel("Avg Duration: 0.0s")
        summary_layout.addWidget(self.avg_duration_label, 0, 1)
        
        self.key_confidence_label = QLabel("Key Confidence: 0%")
        summary_layout.addWidget(self.key_confidence_label, 1, 0)
        
        self.analysis_time_label = QLabel("Analysis Time: 0.0s")
        summary_layout.addWidget(self.analysis_time_label, 1, 1)
        
        self.cache_hit_label = QLabel("Cache Status: Miss")
        summary_layout.addWidget(self.cache_hit_label, 2, 0)
        
        self.complexity_label = QLabel("Harmonic Complexity: -")
        summary_layout.addWidget(self.complexity_label, 2, 1)
        
        layout.addWidget(summary_group)
        
        # Detailed results with tabs
        results_tabs = QTabWidget()
        
        # Chord progression tab
        prog_tab = QWidget()
        prog_layout = QVBoxLayout(prog_tab)
        
        self.results_text = QTextEdit()
        self.results_text.setReadOnly(True)
        self.results_text.setFont(QFont("Consolas", 10))
        prog_layout.addWidget(self.results_text)
        
        results_tabs.addTab(prog_tab, "Progression")
        
        # Statistics tab
        stats_tab = QWidget()
        stats_layout = QVBoxLayout(stats_tab)
        
        self.stats_text = QTextEdit()
        self.stats_text.setReadOnly(True)
        self.stats_text.setFont(QFont("Consolas", 10))
        stats_layout.addWidget(self.stats_text)
        
        results_tabs.addTab(stats_tab, "Statistics")
        
        layout.addWidget(results_tabs)
        
    def setup_performance_tab(self, parent):
        """Setup performance monitoring tab"""
        layout = QVBoxLayout(parent)
        
        # Performance metrics
        perf_group = QGroupBox("⚡ Performance Metrics")
        perf_layout = QGridLayout(perf_group)
        
        self.perf_total_label = QLabel("Total Analysis: -")
        perf_layout.addWidget(self.perf_total_label, 0, 0)
        
        self.perf_loading_label = QLabel("Audio Loading: -")
        perf_layout.addWidget(self.perf_loading_label, 0, 1)
        
        self.perf_features_label = QLabel("Feature Extraction: -")
        perf_layout.addWidget(self.perf_features_label, 1, 0)
        
        self.perf_matching_label = QLabel("Chord Matching: -")
        perf_layout.addWidget(self.perf_matching_label, 1, 1)
        
        layout.addWidget(perf_group)
        
        # Cache information
        cache_group = QGroupBox("💾 Cache Information")
        cache_layout = QVBoxLayout(cache_group)
        
        cache_info_layout = QHBoxLayout()
        self.btn_clear_cache = QPushButton("🗑️ Clear Cache")
        self.btn_clear_cache.clicked.connect(self.clear_cache)
        cache_info_layout.addWidget(self.btn_clear_cache)
        
        self.btn_cache_info = QPushButton("ℹ️ Cache Info")
        self.btn_cache_info.clicked.connect(self.show_cache_info)
        cache_info_layout.addWidget(self.btn_cache_info)
        
        cache_info_layout.addStretch()
        cache_layout.addLayout(cache_info_layout)
        
        self.cache_info_text = QTextEdit()
        self.cache_info_text.setReadOnly(True)
        self.cache_info_text.setMaximumHeight(150)
        cache_layout.addWidget(self.cache_info_text)
        
        layout.addWidget(cache_group)
        
        layout.addStretch()
        
    def apply_modern_theme(self):
        """Apply modern, professional theme"""
        # Enhanced color scheme
        self.setStyleSheet("""
            QWidget {
                font-family: 'Segoe UI', 'San Francisco', 'Helvetica Neue', Arial, sans-serif;
                font-size: 11px;
                background-color: #FAFAFA;
            }
            
            QGroupBox {
                font-weight: 600;
                border: 2px solid #E0E0E0;
                border-radius: 12px;
                margin-top: 12px;
                padding-top: 16px;
                background-color: #FFFFFF;
            }
            
            QGroupBox::title {
                subcontrol-origin: margin;
                left: 16px;
                padding: 0 12px 0 12px;
                color: #1976D2;
                font-size: 12px;
                font-weight: 700;
            }
            
            QPushButton {
                padding: 12px 24px;
                border-radius: 8px;
                background: qlineargradient(x1:0, y1:0, x2:0, y2:1,
                    stop:0 #2196F3, stop:1 #1976D2);
                color: white;
                border: none;
                font-weight: 600;
                font-size: 12px;
                min-height: 20px;
            }
            
            QPushButton:hover {
                background: qlineargradient(x1:0, y1:0, x2:0, y2:1,
                    stop:0 #1976D2, stop:1 #1565C0);
                transform: translateY(-1px);
            }
            
            QPushButton:pressed {
                background: qlineargradient(x1:0, y1:0, x2:0, y2:1,
                    stop:0 #1565C0, stop:1 #0D47A1);
            }
            
            QPushButton:disabled {
                background: #E0E0E0;
                color: #9E9E9E;
            }
            
            QProgressBar {
                border: 2px solid #E0E0E0;
                border-radius: 8px;
                text-align: center;
                background-color: #F5F5F5;
                font-weight: 600;
                font-size: 11px;
                min-height: 24px;
            }
            
            QProgressBar::chunk {
                background: qlineargradient(x1:0, y1:0, x2:1, y2:0,
                    stop:0 #4CAF50, stop:0.5 #66BB6A, stop:1 #81C784);
                border-radius: 6px;
            }
            
            QTabWidget::pane {
                border: 2px solid #E0E0E0;
                background-color: #FFFFFF;
                border-radius: 8px;
                margin-top: 4px;
            }
            
            QTabBar::tab {
                background: qlineargradient(x1:0, y1:0, x2:0, y2:1,
                    stop:0 #F5F5F5, stop:1 #E0E0E0);
                padding: 12px 20px;
                margin-right: 2px;
                border-top-left-radius: 8px;
                border-top-right-radius: 8px;
                font-weight: 600;
                color: #666666;
            }
            
            QTabBar::tab:selected {
                background: qlineargradient(x1:0, y1:0, x2:0, y2:1,
                    stop:0 #2196F3, stop:1 #1976D2);
                color: white;
            }
            
            QTabBar::tab:hover:!selected {
                background: qlineargradient(x1:0, y1:0, x2:0, y2:1,
                    stop:0 #EEEEEE, stop:1 #E0E0E0);
            }
            
            QListWidget {
                border: 2px solid #E0E0E0;
                border-radius: 8px;
                background-color: #FFFFFF;
                alternate-background-color: #F8F9FA;
                selection-background-color: #E3F2FD;
                font-family: 'Consolas', 'Monaco', monospace;
            }
            
            QListWidget::item {
                padding: 8px;
                border-bottom: 1px solid #F0F0F0;
            }
            
            QListWidget::item:selected {
                background-color: #2196F3;
                color: white;
            }
        """)
        
        # Enhanced chord display styling
        self.apply_chord_display_styling()
        
    def apply_chord_display_styling(self):
        """Apply enhanced styling to chord display elements"""
        # Main chord label styling
        chord_style = """
            QLabel {
                border: 4px solid #2196F3;
                border-radius: 20px;
                padding: 30px;
                margin: 8px;
                background: qlineargradient(x1:0, y1:0, x2:0, y2:1,
                    stop:0 #E3F2FD, stop:0.3 #BBDEFB, stop:0.7 #90CAF9, stop:1 #64B5F6);
                color: #0D47A1;
                font-size: 32px;
                font-weight: 700;
                text-align: center;
                letter-spacing: 2px;
            }
        """
        self.chord_label.setStyleSheet(chord_style)
        
        # Key signature styling
        key_style = """
            QLabel {
                background: qlineargradient(x1:0, y1:0, x2:1, y2:0,
                    stop:0 #FFF8E1, stop:0.5 #FFECB3, stop:1 #FFE082);
                border: 3px solid #FF8F00;
                border-radius: 15px;
                padding: 16px;
                color: #E65100;
                font-weight: 700;
                font-size: 14px;
            }
        """
        self.key_label.setStyleSheet(key_style)
        
        # Time display styling
        time_style = """
            QLabel {
                background: qlineargradient(x1:0, y1:0, x2:1, y2:0,
                    stop:0 #F3E5F5, stop:0.5 #E1BEE7, stop:1 #CE93D8);
                border: 3px solid #7B1FA2;
                border-radius: 15px;
                padding: 16px;
                color: #4A148C;
                font-weight: 700;
                font-size: 14px;
                font-family: 'Consolas', monospace;
            }
        """
        self.time_label.setStyleSheet(time_style)
        
        # Confidence styling
        confidence_style = """
            QLabel {
                background: qlineargradient(x1:0, y1:0, x2:1, y2:0,
                    stop:0 #E8F5E8, stop:1 #C8E6C9);
                border: 2px solid #4CAF50;
                border-radius: 12px;
                padding: 12px;
                color: #2E7D32;
                font-weight: 600;
                font-size: 12px;
            }
        """
        self.confidence_label.setStyleSheet(confidence_style)
        
        # Tempo styling
        tempo_style = """
            QLabel {
                background: qlineargradient(x1:0, y1:0, x2:1, y2:0,
                    stop:0 #FFF3E0, stop:1 #FFE0B2);
                border: 2px solid #FF9800;
                border-radius: 12px;
                padding: 12px;
                color: #F57C00;
                font-weight: 600;
                font-size: 12px;
            }
        """
        self.tempo_label.setStyleSheet(tempo_style)
