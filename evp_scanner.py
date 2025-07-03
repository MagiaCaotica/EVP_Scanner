import asyncio
import platform
import numpy as np
import pyaudio
import librosa
from scipy import signal
import pygame
import os
import time
from datetime import datetime
import scipy.io.wavfile as wavfile
from PIL import Image, ImageDraw, ImageFont
from moviepy import ImageClip, AudioFileClip # Corrected import from moviepy.editor for clarity

# Configuration parameters
SAMPLE_RATE = 44100  # Hz, standard audio sampling rate
CHUNK = 2048  # Buffer size for better frequency resolution
MAX_DURATION = 300  # Max seconds per session (5 minutes)
# SPIRICOM Mark IV tones, crucial for energy input and voice tonal inflections
TONE_FREQUENCIES = [131, 192, 241, 296, 364, 422, 483, 534, 587, 643, 704, 767, 871]
BASE_NOISE_AMPLITUDE = 0.01  # Reduced further to emphasize SPIRICOM tones as primary "energy source"
EVP_FREQ_RANGE = (200, 4000)  # Frequency range for EVP and human voice, aligning with audibility
LOW_PASS_CUTOFF = 4500  # Hz, slightly above EVP range to cut off problematic high frequencies
FPS = 60  # Frame rate for async loop control
NOTCH_FREQ = 60  # Hz, for mains hum removal
SILENCE_THRESHOLD = 0.005  # Reduced threshold for more sensitive EVP capture
DYNAMIC_GAIN = 2.0  # Increased gain to amplify potential EVP signals, crucial for audibility
NOISE_GATE_THRESHOLD = 0.008  # Adjusted noise gate threshold to be less strict
NOISE_PROFILE_DURATION = 0.5  # Seconds for noise profile estimation
SPECTROGRAM_WIDTH = 600  # Pixels for spectrogram display
SPECTROGRAM_HEIGHT = 200  # Pixels for spectrogram display
SPECTROGRAM_X = 100  # X position of spectrogram
SPECTROGRAM_Y = 250  # Y position of spectrogram

# --- NEW PARAMETERS FOR ANALOG TAPE EMULATION ---
TAPE_HISS_AMPLITUDE = 0.003 # Subtle, constant background noise like tape hiss
SATURATION_THRESHOLD = 0.6 # Audio level above which soft clipping starts
SATURATION_FACTOR = 0.8 # Controls the "aggressiveness" of soft clipping
WOW_FLUTTER_RATE = 0.5 # Hz, frequency of pitch modulation for wow/flutter
WOW_FLUTTER_DEPTH = 0.002 # Depth of pitch modulation (e.g., 0.2% change)
# --- END NEW PARAMETERS ---

# --- NEW PARAMETER FOR NOISE REDUCTION STRENGTH ---
# Controls the aggressiveness of noise reduction. 0.0 means no spectral subtraction, 1.0 means full.
# A value like 0.5 is a good starting point for less aggressive reduction, preserving potential EVPs.
NOISE_REDUCTION_STRENGTH = 0.5
# --- END NEW PARAMETER ---

# --- NEW PARAMETERS FOR ANOMALY DETECTION AND VISUALIZATION ---
ANOMALY_THRESHOLD_MULTIPLIER = 3.0  # Multiplier for median intensity to detect anomalies
ANOMALY_MIN_INTENSITY = 100 # Minimum intensity for a pixel to be considered for anomaly (0-255 scale)
ANOMALY_COLOR_HIGH = (255, 0, 0)    # Red for very strong anomalies
ANOMALY_COLOR_MEDIUM = (255, 165, 0) # Orange for medium anomalies
ANOMALY_COLOR_LOW = (255, 255, 0)   # Yellow for subtle anomalies
# --- END NEW PARAMETERS ---

# Initialize PyAudio
try:
    p = pyaudio.PyAudio()
    stream = p.open(format=pyaudio.paFloat32,
                     channels=1,
                     rate=SAMPLE_RATE,
                     input=True,
                     output=True, # Keep output=True for playing tones
                     frames_per_buffer=CHUNK)
except Exception as e:
    print(f"Error initializing audio stream: {e}")
    exit(1)

# Initialize Pygame
pygame.init()
WINDOW_SIZE = (800, 600)
screen = pygame.display.set_mode(WINDOW_SIZE)
pygame.display.set_caption("EVP Capture Interface with Spectrogram (SPIRICOM Adapted)")
font = pygame.font.SysFont("arial", 24)
clock = pygame.time.Clock()

# Colors
BLACK = (0, 0, 0)
WHITE = (255, 255, 255)
GRAY = (100, 100, 100)
GREEN = (0, 255, 0)
RED = (255, 0, 0)
BLUE = (0, 0, 255)

# Button class
class Button:
    def __init__(self, x, y, width, height, text, color, hover_color):
        self.rect = pygame.Rect(x, y, width, height)
        self.text = text
        self.color = color
        self.hover_color = hover_color
        self.font = font

    def draw(self, screen):
        mouse_pos = pygame.mouse.get_pos()
        color = self.hover_color if self.rect.collidepoint(mouse_pos) else self.color
        pygame.draw.rect(screen, color, self.rect)
        text_surf = self.font.render(self.text, True, WHITE)
        text_rect = text_surf.get_rect(center=self.rect.center)
        screen.blit(text_surf, text_rect)

    def is_clicked(self, event):
        if event.type == pygame.MOUSEBUTTONDOWN and event.button == 1:
            return self.rect.collidepoint(event.pos)
        return False

# Generate SPIRICOM-like tones with modulation
def generate_tones(duration, sample_rate):
    t = np.linspace(0, duration, int(sample_rate * duration), False)
    tones = np.zeros_like(t)
    for freq in TONE_FREQUENCIES:
        tone = np.sin(2 * np.pi * freq * t)
        # Adding a slight modulation as an interpretation of "complex series" and potential environmental interaction
        modulation = 0.8 + 0.2 * np.sin(2 * np.pi * 0.5 * t)
        tones += tone * modulation / len(TONE_FREQUENCIES)  # Normalize amplitude
    return tones

# Generate subtle "supplemental energy" noise
def generate_base_noise(duration, sample_rate, amplitude=BASE_NOISE_AMPLITUDE):
    return amplitude * np.random.normal(0, 1, int(sample_rate * duration))

# --- NEW FUNCTION: Generate Tape Hiss ---
def generate_tape_hiss(duration, sample_rate, amplitude=TAPE_HISS_AMPLITUDE):
    return amplitude * np.random.normal(0, 1, int(sample_rate * duration)).astype(np.float32)

# Bandpass filter for EVP frequency range
def bandpass_filter(data, lowcut, highcut, fs, order=6):
    nyq = 0.5 * fs
    low = lowcut / nyq
    high = highcut / nyq
    b, a = signal.butter(order, [low, high], btype='band')
    return signal.lfilter(b, a, data)

# Low-pass filter to remove unwanted high frequencies (new function)
def lowpass_filter(data, cutoff, fs, order=6):
    nyq = 0.5 * fs
    normal_cutoff = cutoff / nyq
    b, a = signal.butter(order, normal_cutoff, btype='low', analog=False)
    return signal.lfilter(b, a, data)

# Notch filter for mains hum
def notch_filter(data, freq, fs, quality=30.0):
    nyq = 0.5 * fs
    freq = freq / nyq
    b, a = signal.iirnotch(freq, quality)
    return signal.lfilter(b, a, data)

# --- NEW FUNCTION: Soft Clipping (Saturation) ---
def soft_clip(audio_data, threshold=SATURATION_THRESHOLD, factor=SATURATION_FACTOR):
    # Simple soft clipping function (e.g., using tanh or arctan)
    # This brings up quieter signals without harsh clipping
    return np.tanh(audio_data * factor / threshold) * threshold

# --- NEW FUNCTION: Wow/Flutter Emulation (Subtle Pitch Modulation) ---
# This is simplified; true wow/flutter involves variable delay lines.
# Here we'll do a very slight resampling effect.
def apply_wow_flutter(audio_data, sample_rate, current_time):
    # Sinusoidal modulation of playback rate
    mod_signal = WOW_FLUTTER_DEPTH * np.sin(2 * np.pi * WOW_FLUTTER_RATE * current_time)
    
    # Apply a very subtle variable delay or resampling based on the modulation
    # This is a conceptual simplification. For a true effect, consider a variable delay line.
    # For now, we'll slightly adjust amplitude to simulate "stretching" or "compressing"
    # the waveform. This is less accurate for pitch but introduces a subtle instability.
    # A more accurate approach would use `librosa.effects.time_stretch` or a custom resampler.
    # For real-time, this is a heavy operation. We'll simplify for performance.
    
    # A very simple amplitude modulation for subtle "wow"
    modulated_audio = audio_data * (1 + mod_signal)
    return np.clip(modulated_audio, -1.0, 1.0) # Ensure no overflow


# Analyze frequency content
def analyze_frequencies(data, sample_rate):
    try:
        stft = np.abs(librosa.stft(data, n_fft=CHUNK, hop_length=256))
        freqs = librosa.fft_frequencies(sr=sample_rate, n_fft=CHUNK)
        mask = (freqs >= EVP_FREQ_RANGE[0]) & (freqs <= EVP_FREQ_RANGE[1])
        power = np.mean(stft[mask, :], axis=1)
        peak_freq = freqs[mask][np.argmax(power)] if np.any(power) else 0
        return peak_freq
    except Exception as e:
        print(f"Frequency analysis error: {e}")
        return 0

# Calculate RMS for silence detection
def calculate_rms(data):
    return np.sqrt(np.mean(data ** 2))

# Noise reduction: spectral subtraction and noise gate, crucial for clearer audibility
def reduce_noise(audio_data, sample_rate):
    try:
        noise_samples = int(NOISE_PROFILE_DURATION * sample_rate)
        if len(audio_data) < noise_samples:
            noise_samples = len(audio_data)
        noise_segment = audio_data[:noise_samples]
        noise_stft = librosa.stft(noise_segment, n_fft=2048, hop_length=512)
        noise_magnitude = np.mean(np.abs(noise_stft), axis=1)
        
        signal_stft = librosa.stft(audio_data, n_fft=2048, hop_length=512)
        signal_magnitude, signal_phase = librosa.magphase(signal_stft)
        
        # --- MODIFICATION: Less aggressive noise reduction based on NOISE_REDUCTION_STRENGTH ---
        # Multiplies the estimated noise magnitude by a configurable strength factor (0.0 to 1.0).
        # A lower strength allows more of the original "noise" (where EVPs might be) to pass through.
        clean_magnitude = signal_magnitude - noise_magnitude[:, np.newaxis] * NOISE_REDUCTION_STRENGTH
        clean_magnitude = np.maximum(clean_magnitude, 0.0) # Ensure non-negative magnitudes
        
        clean_stft = clean_magnitude * signal_phase
        clean_audio = librosa.istft(clean_stft, hop_length=512, length=len(audio_data))
        
        # Noise gate application
        window_size = int(0.05 * sample_rate)
        rms_windows = [calculate_rms(clean_audio[i:i+window_size])
                         for i in range(0, len(clean_audio), window_size)]
        # The noise gate threshold is now less strict (0.008) to allow fainter signals
        gate_mask = np.repeat([r > NOISE_GATE_THRESHOLD for r in rms_windows],
                                  window_size)[:len(clean_audio)]
        clean_audio *= gate_mask
        
        clean_audio = bandpass_filter(clean_audio, EVP_FREQ_RANGE[0], EVP_FREQ_RANGE[1], sample_rate)
        
        clean_audio *= DYNAMIC_GAIN # Amplifying potential EVP signals
        clean_audio = np.clip(clean_audio, -1.0, 1.0)
        
        return clean_audio, "Noise reduction applied successfully"
    except Exception as e:
        return audio_data, f"Noise reduction failed: {e}"

# Create MP4 from spectrogram PNG and WAV
def create_mp4(wav_path, spectrogram_path, mp4_path):
    try:
        # Load the spectrogram image
        spectrogram_clip = ImageClip(spectrogram_path)
        
        # Load the audio
        audio_clip = AudioFileClip(wav_path)
        
        # Set the duration of the video to match the audio
        spectrogram_clip = spectrogram_clip.set_duration(audio_clip.duration)
        
        # Set the audio to the spectrogram clip
        video_clip = spectrogram_clip.set_audio(audio_clip)
        
        # Write the final video file
        video_clip.write_videofile(mp4_path, fps=24, codec='libx264', audio_codec='aac')
        
        # Close clips to free memory
        video_clip.close()
        audio_clip.close()
        spectrogram_clip.close()
        
        return f"Saved MP4 to {mp4_path}", True
    except Exception as e:
        return f"Error creating MP4: {e}", False

# Save audio to WAV, spectrogram to PNG, and create MP4
def save_audio(output_audio_raw, sample_rate, filename, full_spectrogram_data_for_save=None, anomaly_data_for_save=None, spectrogram_freqs=None): # Adjusted arguments
    try:
        if not os.path.exists("evp_sessions"):
            os.makedirs("evp_sessions")
        
        # Concatenate raw audio and then apply all processing
        audio_data_raw = np.concatenate(output_audio_raw)
        
        # --- NEW: Apply analog-like processing before saving ---
        # Add tape hiss
        hiss_chunk = generate_tape_hiss(len(audio_data_raw) / sample_rate, sample_rate)
        processed_audio_for_save = audio_data_raw + hiss_chunk[:len(audio_data_raw)]
        
        # Apply soft clipping
        processed_audio_for_save = soft_clip(processed_audio_for_save)

        # Apply filters and noise reduction *before* saving
        processed_audio_for_save = lowpass_filter(processed_audio_for_save, LOW_PASS_CUTOFF, sample_rate)
        processed_audio_for_save = bandpass_filter(processed_audio_for_save, EVP_FREQ_RANGE[0], EVP_FREQ_RANGE[1], sample_rate)
        processed_audio_for_save = notch_filter(processed_audio_for_save, NOTCH_FREQ, sample_rate)
        
        clean_audio, noise_status = reduce_noise(processed_audio_for_save, sample_rate)
        
        audio_int16 = np.int16(clean_audio * 32767)
        wav_path = os.path.join("evp_sessions", filename)
        wavfile.write(wav_path, sample_rate, audio_int16)
        
        status = f"Saved WAV to {wav_path}, {noise_status}"
        success = True
        
        if full_spectrogram_data_for_save is not None and spectrogram_freqs is not None:
            spectrogram_filename = filename.replace('.wav', '_spectrogram.png')
            spec_status, spec_success = save_spectrogram(full_spectrogram_data_for_save, anomaly_data_for_save, spectrogram_freqs, spectrogram_filename)
            status += f"; {spec_status}"
            success = success and spec_success
            
            if spec_success:
                mp4_filename = filename.replace('.wav', '.mp4')
                mp4_path = os.path.join("evp_sessions", mp4_filename)
                mp4_status, mp4_success = create_mp4(wav_path, os.path.join("evp_sessions", spectrogram_filename), mp4_path)
                status += f"; {mp4_status}"
                success = success and mp4_success
        
        return status, success
    except Exception as e:
        return f"Error saving WAV: {e}", False

# Compute spectrogram data for visualization
def compute_spectrogram_data(audio_data, sample_rate):
    try:
        n_fft = 2048
        hop_length = n_fft // 4
        stft = np.abs(librosa.stft(audio_data, n_fft=n_fft, hop_length=hop_length))
        freqs = librosa.fft_frequencies(sr=sample_rate, n_fft=n_fft)
        mask = (freqs >= EVP_FREQ_RANGE[0]) & (freqs <= EVP_FREQ_RANGE[1])
        stft = stft[mask, :]
        freqs = freqs[mask]
        
        # Normalize for display and apply log scale
        if np.max(stft) > 0:
            stft_normalized = np.log1p(stft)
            stft_scaled = (stft_normalized / np.max(stft_normalized) * 255).astype(np.uint8)
        else:
            stft_scaled = np.zeros_like(stft, dtype=np.uint8)

        # Anomaly detection
        anomaly_map = np.zeros_like(stft_scaled, dtype=bool)
        if stft_scaled.size > 0:
            median_intensity = np.median(stft_scaled[stft_scaled > 0]) if np.any(stft_scaled > 0) else 0
            threshold = median_intensity * ANOMALY_THRESHOLD_MULTIPLIER
            anomaly_map = (stft_scaled > threshold) & (stft_scaled > ANOMALY_MIN_INTENSITY)

        return stft_scaled, anomaly_map, freqs
    except Exception as e:
        print(f"Spectrogram computation error: {e}")
        return None, None, None

# Save spectrogram as PNG
def save_spectrogram(spectrogram_data, anomaly_data, freqs, filename):
    try:
        if spectrogram_data.size == 0 or freqs.size == 0:
            return "No spectrogram data to save or invalid frequencies array", False
        
        if not os.path.exists("evp_sessions"):
            os.makedirs("evp_sessions")
        
        original_height, original_width = spectrogram_data.shape
        
        # Create an RGB image directly
        image_rgb_data = np.zeros((original_height, original_width, 3), dtype=np.uint8)
        
        for y in range(original_height):
            for x in range(original_width):
                intensity = spectrogram_data[y, x]
                if anomaly_data[y, x]:
                    # Map intensity to color within the anomaly range
                    # Higher intensity anomalies are red, lower are yellow
                    norm_intensity = intensity / 255.0
                    if norm_intensity > 0.7:
                        color = ANOMALY_COLOR_HIGH
                    elif norm_intensity > 0.4:
                        color = ANOMALY_COLOR_MEDIUM
                    else:
                        color = ANOMALY_COLOR_LOW
                    image_rgb_data[y, x] = color
                else:
                    image_rgb_data[y, x] = (intensity, intensity, intensity) # Grayscale for normal parts
        
        img_data = Image.fromarray(image_rgb_data, 'RGB')
        
        resized_img_data = img_data.resize((SPECTROGRAM_WIDTH, SPECTROGRAM_HEIGHT), Image.Resampling.LANCZOS)
        
        image = Image.new('RGB', (SPECTROGRAM_WIDTH + 80, SPECTROGRAM_HEIGHT + 60), 'black')
        draw = ImageDraw.Draw(image)
        
        try:
            font = ImageFont.truetype("arial.ttf", 12)
        except IOError:
            font = ImageFont.load_default()

        # Spectrograms are typically plotted with low frequencies at the bottom, so flip for display
        flipped_img_data = resized_img_data.transpose(Image.FLIP_TOP_BOTTOM)
        image.paste(flipped_img_data, (40, 20))
        
        min_freq = freqs[0]
        max_freq = freqs[-1]
        
        if freqs.size > 1:
            for i in range(0, SPECTROGRAM_HEIGHT + 1, 50):
                norm_i = i / SPECTROGRAM_HEIGHT
                actual_freq = min_freq + (1 - norm_i) * (max_freq - min_freq)
                draw.text((5, 20 + i - 8), f"{int(actual_freq)} Hz", fill='white', font=font)
        elif freqs.size == 1:
            draw.text((5, 20 + SPECTROGRAM_HEIGHT // 2 - 8), f"{int(freqs[0])} Hz", fill='white', font=font)

        # Calculate total duration based on the actual spectrogram data's time frames
        # Assuming hop_length used in compute_spectrogram_data is CHUNK // 4 (512)
        total_time_duration = original_width * ( (2048 // 4) / SAMPLE_RATE ) # Using n_fft=2048, hop_length=512 for save
        
        for i in range(0, SPECTROGRAM_WIDTH + 1, SPECTROGRAM_WIDTH // 5):
            if SPECTROGRAM_WIDTH > 0:
                time_sec = (i / SPECTROGRAM_WIDTH) * total_time_duration
                draw.text((40 + i - 15, SPECTROGRAM_HEIGHT + 20), f"{time_sec:.1f}s", fill='white', font=font)
        
        spec_path = os.path.join("evp_sessions", filename)
        image.save(spec_path, 'PNG')
        return f"Saved spectrogram to {spec_path}", True
    except Exception as e:
        return f"Error saving spectrogram: {e}", False


async def main():
    # Setup
    recorded_audio_chunks = [] # Store raw audio chunks from microphone
    recording = False
    start_time = 0
    tones = generate_tones(MAX_DURATION + 5, SAMPLE_RATE) # Generate slightly more tones than max duration
    save_status = ""
    
    # Spectrogram setup
    spectrogram_surface = pygame.Surface((SPECTROGRAM_WIDTH, SPECTROGRAM_HEIGHT))
    spectrogram_data_slices = [] # Stores columns of spectrogram data (time slices for live display)
    anomaly_data_slices = [] # Stores anomaly maps corresponding to spectrogram_data_slices
    full_spectrogram_data_for_save = np.array([]) # Store full spectrogram data for saving
    full_anomaly_data_for_save = np.array([]) # Store full anomaly data for saving
    current_freqs = np.array([]) # Store frequencies for spectrogram display and saving
    
    # GUI elements
    start_button = Button(50, 500, 200, 50, "Start Recording", GREEN, (0, 200, 0))
    stop_button = Button(300, 500, 200, 50, "Stop Recording", RED, (200, 0, 0))
    export_button = Button(550, 500, 200, 50, "Export Spectrogram", BLUE, (0, 0, 200))
    
    running = True
    while running:
        # Handle Pygame events
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                running = False
            elif start_button.is_clicked(event) and not recording:
                recording = True
                recorded_audio_chunks = [] # Clear recorded audio on new recording
                spectrogram_data_slices = [] # Clear spectrogram data on new recording
                anomaly_data_slices = [] # Clear anomaly data
                full_spectrogram_data_for_save = np.array([]) # Clear full spectrogram data
                full_anomaly_data_for_save = np.array([]) # Clear full anomaly data
                current_freqs = np.array([]) # Clear frequencies
                start_time = time.time()
                save_status = "Recording started..."
                print(save_status)
            elif stop_button.is_clicked(event) and recording:
                recording = False
                if recorded_audio_chunks and calculate_rms(np.concatenate(recorded_audio_chunks)) > SILENCE_THRESHOLD:
                    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
                    filename = f"evp_session_{timestamp}.wav"
                    # Pass the raw recorded audio for saving and subsequent processing
                    save_status, success = save_audio(recorded_audio_chunks, SAMPLE_RATE, filename, full_spectrogram_data_for_save, full_anomaly_data_for_save, current_freqs)
                else:
                    save_status = "Recording stopped (silence detected or too short, not saved)"
                print(save_status)
            elif export_button.is_clicked(event) and full_spectrogram_data_for_save.size > 0:
                timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
                filename = f"spectrogram_{timestamp}.png"
                save_status, success = save_spectrogram(full_spectrogram_data_for_save, full_anomaly_data_for_save, current_freqs, filename)
                print(save_status)

        # Audio processing and playback of tones
        if recording and (time.time() - start_time) < MAX_DURATION:
            try:
                # Read raw input audio
                data_input = stream.read(CHUNK, exception_on_overflow=False)
                audio_chunk_input = np.frombuffer(data_input, dtype=np.float32).copy()
                
                # --- NEW: Add tape hiss to the input audio for recording ---
                hiss_chunk = generate_tape_hiss(CHUNK / SAMPLE_RATE, SAMPLE_RATE)
                audio_chunk_input += hiss_chunk[:len(audio_chunk_input)] # Ensure length matches

                # --- NEW: Apply soft clipping to the input audio for recording ---
                audio_chunk_input = soft_clip(audio_chunk_input)

                # --- NEW: Apply subtle wow/flutter to the input audio for recording ---
                audio_chunk_input = apply_wow_flutter(audio_chunk_input, SAMPLE_RATE, time.time() - start_time)
                
                # Store raw input audio (now with analog-like characteristics) for later saving and processing
                recorded_audio_chunks.append(audio_chunk_input)

                # Generate base noise dynamically based on the input RMS (for output only)
                input_rms = calculate_rms(audio_chunk_input)
                noise_amplitude = BASE_NOISE_AMPLITUDE * max(0.5, min(2.0, 0.01 / (input_rms + 1e-6)))
                base_noise_chunk = generate_base_noise(CHUNK / SAMPLE_RATE, SAMPLE_RATE, noise_amplitude)
                
                # Get the current chunk of tones for playback
                current_tone_idx = len(recorded_audio_chunks) * CHUNK
                tone_chunk_to_play = tones[current_tone_idx : current_tone_idx + CHUNK]
                
                # Mix only tones and base noise for immediate output
                output_to_speaker = tone_chunk_to_play[:len(audio_chunk_input)] + base_noise_chunk
                
                # Play back only the generated tones and base noise
                stream.write(output_to_speaker.astype(np.float32).tobytes())
                
                # --- For live spectrogram display and peak frequency analysis:
                # Process a copy of the *input audio* for real-time visualization and analysis.
                # This processing does NOT affect the audio being saved, only the display.
                display_audio_chunk = lowpass_filter(audio_chunk_input, LOW_PASS_CUTOFF, SAMPLE_RATE)
                display_audio_chunk = bandpass_filter(display_audio_chunk, EVP_FREQ_RANGE[0], EVP_FREQ_RANGE[1], SAMPLE_RATE)
                display_audio_chunk = notch_filter(display_audio_chunk, NOTCH_FREQ, SAMPLE_RATE)
                
                peak_freq = analyze_frequencies(display_audio_chunk, SAMPLE_RATE)
                
                spec_data_slice_2d, anomaly_map_slice, freqs_for_spec = compute_spectrogram_data(display_audio_chunk, SAMPLE_RATE)
                if spec_data_slice_2d is not None and spec_data_slice_2d.shape[1] > 0:
                    spectrogram_data_slices.append(spec_data_slice_2d[:, -1])
                    anomaly_data_slices.append(anomaly_map_slice[:, -1])
                    current_freqs = freqs_for_spec 

                    # Accumulate full spectrogram data for saving (based on processed *input* for display)
                    if full_spectrogram_data_for_save.size == 0:
                        full_spectrogram_data_for_save = spec_data_slice_2d
                        full_anomaly_data_for_save = anomaly_map_slice
                    else:
                        # Ensure dimensions match before concatenating
                        if spec_data_slice_2d.shape[0] < full_spectrogram_data_for_save.shape[0]:
                            padding_spec = np.zeros((full_spectrogram_data_for_save.shape[0] - spec_data_slice_2d.shape[0], spec_data_slice_2d.shape[1]), dtype=spec_data_slice_2d.dtype)
                            spec_data_slice_2d = np.vstack((spec_data_slice_2d, padding_spec))
                            padding_anomaly = np.zeros((full_anomaly_data_for_save.shape[0] - anomaly_map_slice.shape[0], anomaly_map_slice.shape[1]), dtype=anomaly_map_slice.dtype)
                            anomaly_map_slice = np.vstack((anomaly_map_slice, padding_anomaly))
                        elif spec_data_slice_2d.shape[0] > full_spectrogram_data_for_save.shape[0]:
                            full_spectrogram_data_for_save = np.vstack((full_spectrogram_data_for_save, np.zeros((spec_data_slice_2d.shape[0] - full_spectrogram_data_for_save.shape[0], full_spectrogram_data_for_save.shape[1]), dtype=full_spectrogram_data_for_save.dtype)))
                            full_anomaly_data_for_save = np.vstack((full_anomaly_data_for_save, np.zeros((anomaly_map_slice.shape[0] - full_anomaly_data_for_save.shape[0], full_anomaly_data_for_save.shape[1]), dtype=full_anomaly_data_for_save.dtype)))
                        
                        full_spectrogram_data_for_save = np.concatenate((full_spectrogram_data_for_save, spec_data_slice_2d), axis=1)
                        full_anomaly_data_for_save = np.concatenate((full_anomaly_data_for_save, anomaly_map_slice), axis=1)


                    if len(spectrogram_data_slices) > SPECTROGRAM_WIDTH:
                        spectrogram_data_slices.pop(0)
                        anomaly_data_slices.pop(0)
                
                # Update GUI
                screen.fill(BLACK)
                start_button.draw(screen)
                stop_button.draw(screen)
                export_button.draw(screen)
                
                # Draw spectrogram
                spectrogram_surface.fill(BLACK)
                if len(spectrogram_data_slices) > 0 and len(anomaly_data_slices) > 0:
                    current_spectrogram_view = np.array(spectrogram_data_slices).T 
                    current_anomaly_view = np.array(anomaly_data_slices).T
                    
                    spec_height_for_display = current_spectrogram_view.shape[0]
                    spec_width_for_display = current_spectrogram_view.shape[1]
                    
                    for x in range(spec_width_for_display):
                        for y in range(spec_height_for_display):
                            if y < SPECTROGRAM_HEIGHT and x < SPECTROGRAM_WIDTH:
                                intensity = current_spectrogram_view[y, x]
                                if current_anomaly_view[y, x]:
                                    # Map intensity to color within the anomaly range
                                    norm_intensity = intensity / 255.0
                                    if norm_intensity > 0.7:
                                        color = ANOMALY_COLOR_HIGH
                                    elif norm_intensity > 0.4:
                                        color = ANOMALY_COLOR_MEDIUM
                                    else:
                                        color = ANOMALY_COLOR_LOW
                                    spectrogram_surface.set_at((x, SPECTROGRAM_HEIGHT - 1 - y), color)
                                else:
                                    color = (intensity, intensity, intensity) # Grayscale for normal parts
                                    spectrogram_surface.set_at((x, SPECTROGRAM_HEIGHT - 1 - y), color)
                screen.blit(spectrogram_surface, (SPECTROGRAM_X, SPECTROGRAM_Y))
                
                # Display status
                status_text = f"Status: {'Recording' if recording else 'Idle'}"
                elapsed_text = f"Elapsed: {int(time.time() - start_time)}s" if recording else ""
                freq_text = f"Peak Freq: {peak_freq:.1f} Hz" if peak_freq > 0 else "Peak Freq: N/A"
                
                screen.blit(font.render(status_text, True, WHITE), (50, 50))
                screen.blit(font.render(elapsed_text, True, WHITE), (50, 100))
                screen.blit(font.render(freq_text, True, WHITE), (50, 150))
                screen.blit(font.render(save_status, True, WHITE), (50, 200))
                screen.blit(font.render("Spectrogram (200-4000 Hz) - SPIRICOM Focus", True, WHITE), (SPECTROGRAM_X, SPECTROGRAM_Y - 30))
                
                pygame.display.flip()
                
            except Exception as e:
                save_status = f"Audio processing error: {e}"
                print(save_status)
                recording = False
        
        # Stop recording if max duration reached
        if recording and (time.time() - start_time) >= MAX_DURATION:
            recording = False
            if recorded_audio_chunks and calculate_rms(np.concatenate(recorded_audio_chunks)) > SILENCE_THRESHOLD:
                timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
                filename = f"evp_session_{timestamp}.wav"
                save_status, success = save_audio(recorded_audio_chunks, SAMPLE_RATE, filename, full_spectrogram_data_for_save, full_anomaly_data_for_save, current_freqs)
            else:
                save_status = "Max duration reached (silence detected or too short, not saved)"
            print(save_status)
        
        # Update display when idle
        if not recording:
            screen.fill(BLACK)
            start_button.draw(screen)
            stop_button.draw(screen)
            export_button.draw(screen)
            screen.blit(spectrogram_surface, (SPECTROGRAM_X, SPECTROGRAM_Y)) # Keep last spectrogram visible
            screen.blit(font.render("Status: Idle", True, WHITE), (50, 50))
            screen.blit(font.render(save_status, True, WHITE), (50, 200))
            screen.blit(font.render("Spectrogram (200-4000 Hz) - SPIRICOM Focus", True, WHITE), (SPECTROGRAM_X, SPECTROGRAM_Y - 30))
            pygame.display.flip()
        
        clock.tick(FPS)
        await asyncio.sleep(1.0 / FPS)
        
    # Cleanup on exit
    if recorded_audio_chunks and recording and calculate_rms(np.concatenate(recorded_audio_chunks)) > SILENCE_THRESHOLD:
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        filename = f"evp_session_{timestamp}.wav"
        save_status, success = save_audio(recorded_audio_chunks, SAMPLE_RATE, filename, full_spectrogram_data_for_save, full_anomaly_data_for_save, current_freqs)
        print(save_status)
        
    stream.stop_stream()
    stream.close()
    p.terminate()
    pygame.quit()

if platform.system() == "Emscripten":
    asyncio.ensure_future(main())
else:
    if __name__ == "__main__":
        asyncio.run(main())
