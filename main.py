import os
import gradio as gr
import warnings
import numpy as np
import torch
import time
import json
import re
from datetime import datetime
from kokoro import KModel, KPipeline

# Suppress warnings for cleaner output
warnings.filterwarnings("ignore")

class KokoroTTSApp:
    """
    A simplified Gradio application for Kokoro TTS with style-based voice blending
    and improved user experience.
    """
    
    MODEL = None
    PIPELINES = {}
    VOICES = {}
    MODEL_LOCK = None
    USER_PRESETS_FILE = "user_presets.json"
    HISTORY_FILE = "generation_history.json"
    MAX_HISTORY_ENTRIES = 20
    
    def __init__(self):
        """Initialize Kokoro TTS models and pipelines."""
        self.history = self._load_history()
        self.user_presets = self._load_user_presets()
        self._initialize()
    
    def _initialize(self):
        """Initialize the Kokoro TTS models and pipelines."""
        print("Initializing Kokoro TTS...")
        
        # Initialize voice options
        self.VOICES = {
            '🇺🇸 🚺 Heart ❤️': 'af_heart', '🇺🇸 🚺 Bella 🔥': 'af_bella',
            '🇺🇸 🚺 Nicole 🎧': 'af_nicole', '🇺🇸 🚺 Aoede': 'af_aoede',
            '🇺🇸 🚺 Kore': 'af_kore', '🇺🇸 🚺 Sarah': 'af_sarah',
            '🇺🇸 🚺 Nova': 'af_nova', '🇺🇸 🚺 Sky': 'af_sky',
            '🇺🇸 🚺 Alloy': 'af_alloy', '🇺🇸 🚺 Jessica': 'af_jessica',
            '🇺🇸 🚺 River': 'af_river', '🇺🇸 🚹 Michael': 'am_michael',
            '🇺🇸 🚹 Fenrir': 'am_fenrir', '🇺🇸 🚹 Puck': 'am_puck',
            '🇺🇸 🚹 Echo': 'am_echo', '🇺🇸 🚹 Eric': 'am_eric',
            '🇺🇸 🚹 Liam': 'am_liam', '🇺🇸 🚹 Onyx': 'am_onyx',
            '🇺🇸 🚹 Adam': 'am_adam', '🇬🇧 🚺 Emma': 'bf_emma',
            '🇬🇧 🚺 Isabella': 'bf_isabella', '🇬🇧 🚺 Alice': 'bf_alice',
            '🇬🇧 🚺 Lily': 'bf_lily', '🇬🇧 🚹 George': 'bm_george',
            '🇬🇧 🚹 Fable': 'bm_fable', '🇬🇧 🚹 Lewis': 'bm_lewis',
            '🇬🇧 🚹 Daniel': 'bm_daniel',
        }
        
        # Create pipelines for US and UK English
        for code in ['a', 'b']:
            self.PIPELINES[code] = KPipeline(lang_code=code, model=False)
        
        # Add custom pronunciations
        self.PIPELINES['a'].g2p.lexicon.golds['kokoro'] = 'kˈOkəɹO'
        self.PIPELINES['b'].g2p.lexicon.golds['kokoro'] = 'kˈQkəɹQ'
        
        # Load all voice models
        for voice_code in self.VOICES.values():
            self.PIPELINES[voice_code[0]].load_voice(voice_code)
        
        # Initialize the model
        if self.MODEL_LOCK is None:
            self.MODEL_LOCK = torch.multiprocessing.Lock()
        
        with self.MODEL_LOCK:
            if self.MODEL is None:
                self.MODEL = {}
                self.MODEL[False] = KModel().to('cpu').eval()
                if torch.cuda.is_available():
                    print("CUDA available. GPU model on demand.")
        
        print("Initialization complete.")
    
    def _load_user_presets(self):
        """Load user presets from file or return empty dict if file doesn't exist."""
        if os.path.exists(self.USER_PRESETS_FILE):
            try:
                with open(self.USER_PRESETS_FILE, 'r') as f:
                    return json.load(f)
            except Exception as e:
                print(f"Error loading user presets: {e}")
                return {}
        return {}
    
    def _save_user_presets(self):
        """Save user presets to file."""
        try:
            with open(self.USER_PRESETS_FILE, 'w') as f:
                json.dump(self.user_presets, f)
        except Exception as e:
            print(f"Error saving user presets: {e}")
    
    def _load_history(self):
        """Load generation history from file or return empty list if file doesn't exist."""
        if os.path.exists(self.HISTORY_FILE):
            try:
                with open(self.HISTORY_FILE, 'r') as f:
                    return json.load(f)
            except Exception as e:
                print(f"Error loading history: {e}")
                return []
        return []
    
    def _save_history(self):
        """Save generation history to file."""
        try:
            with open(self.HISTORY_FILE, 'w') as f:
                json.dump(self.history, f)
        except Exception as e:
            print(f"Error saving history: {e}")
    
    def _add_to_history(self, text, voice, speed, enable_blending=False, second_voice=None, blend_ratio=0.5):
        """Add an entry to the generation history."""
        entry = {
            "text": text,
            "timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
            "voice": voice,
            "speed": speed,
            "enable_blending": enable_blending,
            "second_voice": second_voice,
            "blend_ratio": blend_ratio
        }
        
        # Add to start of list to show newest first
        self.history.insert(0, entry)
        
        # Trim history if needed
        if len(self.history) > self.MAX_HISTORY_ENTRIES:
            self.history = self.history[:self.MAX_HISTORY_ENTRIES]
        
        self._save_history()
        return self.history
    
    def _normalize_text(self, text):
        """
        Normalize text to improve TTS quality:
        - Convert numbers to words
        - Expand common abbreviations
        - Handle special characters
        """
        if not text:
            return text
            
        # Simple number conversion for demonstration
        # In production, use a proper number-to-word library
        def replace_number(match):
            number = match.group(0)
            try:
                if '.' in number:
                    # Handle decimals specifically
                    return number  # Let the TTS system handle pronunciation
                num = int(number)
                if 0 <= num <= 20:  # Simple conversion for small numbers
                    words = ['zero', 'one', 'two', 'three', 'four', 'five', 
                             'six', 'seven', 'eight', 'nine', 'ten', 'eleven', 
                             'twelve', 'thirteen', 'fourteen', 'fifteen', 
                             'sixteen', 'seventeen', 'eighteen', 'nineteen', 'twenty']
                    return words[num]
                return number  # Keep larger numbers as is
            except:
                return number
                
        # Convert standalone numbers
        text = re.sub(r'\b\d+\b', replace_number, text)
        
        # Handle common abbreviations
        abbreviations = {
            r'\bMr\.': 'Mister',
            r'\bDr\.': 'Doctor',
            r'\bSt\.': 'Street',
            r'\bAve\.': 'Avenue',
            r'\bJan\.': 'January',
            r'\bFeb\.': 'February',
            r'\bMar\.': 'March',
            r'\bApr\.': 'April',
            r'\bJun\.': 'June',
            r'\bJul\.': 'July',
            r'\bAug\.': 'August',
            r'\bSep\.': 'September',
            r'\bOct\.': 'October',
            r'\bNov\.': 'November',
            r'\bDec\.': 'December',
        }
        
        for pattern, replacement in abbreviations.items():
            text = re.sub(pattern, replacement, text)
            
        return text
    
    def _process_text_chunks(self, text, pipeline, voice_code, speed, use_gpu, max_chunk_size=400, ref_s=None):
        """
        Process text in chunks with improved handling and crossfading
        Optional ref_s for blended voices
        """
        # Apply text normalization
        processed_text = self._normalize_text(text.replace('\n', ' ').strip())
        if not processed_text:
            print("Warning: Empty text after preprocessing.")
            return None, 24000
        
        # Split into chunks, preserving sentence boundaries
        chunks = []
        sentences = processed_text.split('.')
        current_chunk, current_size = [], 0
        
        for sentence in sentences:
            sentence = sentence.strip()
            if not sentence:
                continue
            sentence += '.'
            sentence_size = len(sentence)
            
            if sentence_size > max_chunk_size:
                words = sentence.split()
                word_chunk, word_size = [], 0
                for word in words:
                    word_size_curr = len(word) + 1
                    if word_size + word_size_curr > max_chunk_size and word_chunk:
                        chunks.append(' '.join(word_chunk))
                        word_chunk, word_size = [], 0
                    word_chunk.append(word)
                    word_size += word_size_curr
                if word_chunk:
                    chunks.append(' '.join(word_chunk))
            elif current_size + sentence_size > max_chunk_size and current_chunk:
                chunks.append(' '.join(current_chunk))
                current_chunk, current_size = [sentence], sentence_size
            else:
                current_chunk.append(sentence)
                current_size += sentence_size
        
        if current_chunk:
            chunks.append(' '.join(current_chunk))
        if not chunks:
            chunks = [processed_text]
        
        print(f"Processing {len(chunks)} chunks...")
        all_audio = []
        sample_rate = 24000
        
        for i, chunk in enumerate(chunks):
            if not chunk.strip():
                continue
            print(f"Chunk {i+1}/{len(chunks)}: {chunk[:50]}...")
            phoneme_sequences = list(pipeline(chunk, voice_code, speed))
            if not phoneme_sequences:
                print(f"Warning: No phonemes for chunk {i+1}")
                continue
            
            _, ps, _ = phoneme_sequences[0]
            # Use provided ref_s for blending, otherwise load from pipeline
            ref_s_chunk = ref_s if ref_s is not None else pipeline.load_voice(voice_code)[len(ps)-1]
            try:
                with self.MODEL_LOCK:
                    model_key = use_gpu and True in self.MODEL
                    if use_gpu and model_key not in self.MODEL and torch.cuda.is_available():
                        self.MODEL[True] = KModel().to('cuda').eval()
                        model_key = True
                    audio = self.MODEL[model_key](ps, ref_s_chunk, speed)
                audio_np = audio.cpu().numpy() if isinstance(audio, torch.Tensor) else audio
                all_audio.append(audio_np)
            except Exception as e:
                print(f"Chunk {i+1} failed: {e}")
                if use_gpu:
                    try:
                        with self.MODEL_LOCK:
                            audio = self.MODEL[False](ps, ref_s_chunk, speed)
                        audio_np = audio.cpu().numpy() if isinstance(audio, torch.Tensor) else audio
                        all_audio.append(audio_np)
                    except Exception as e2:
                        print(f"CPU fallback failed: {e2}")
        
        if not all_audio:
            print("No audio generated from chunks.")
            return None, sample_rate
        
        if len(all_audio) == 1:
            return all_audio[0], sample_rate
        
        # Crossfade between chunks
        crossfade_ms = 30  # 30ms
        crossfade_samples = int(crossfade_ms / 1000 * sample_rate)
        total_length = sum(len(a) for a in all_audio) - crossfade_samples * (len(all_audio) - 1)
        result = np.zeros(total_length, dtype=np.float32)
        pos = 0
        
        for i, chunk in enumerate(all_audio):
            chunk_len = len(chunk)
            if i == 0:
                result[:chunk_len] = chunk
                pos += chunk_len - crossfade_samples
            else:
                fade_out = np.linspace(1, 0, crossfade_samples)
                fade_in = np.linspace(0, 1, crossfade_samples)
                result[pos:pos+crossfade_samples] = (
                    result[pos:pos+crossfade_samples] * fade_out +
                    chunk[:crossfade_samples] * fade_in
                )
                if crossfade_samples < chunk_len:
                    result[pos+crossfade_samples:pos+chunk_len] = chunk[crossfade_samples:]
                pos += chunk_len - crossfade_samples
        
        return result, sample_rate
    
    def generate_speech(self, text, voice, speed, use_gpu, enable_blending=False, 
                        second_voice=None, blend_ratio=0.5, save_to_history=True):
        """
        Generate speech from text with the specified voice settings
        
        Args:
            text (str): The text to convert to speech
            voice (str): The voice to use for TTS
            speed (float): Speed multiplier (0.5-2.0)
            use_gpu (bool): Whether to use GPU acceleration
            enable_blending (bool): Whether to blend two voices
            second_voice (str): The second voice to blend with
            blend_ratio (float): Blend ratio between primary and secondary voice (0.0-1.0)
            save_to_history (bool): Whether to save this generation to history
        
        Returns:
            tuple: (sample_rate, audio_array), processed_text - Formatted for Gradio's Audio component
        """
        try:
            if not self.VOICES:
                self._initialize()
            
            # Check if GPU is requested but not loaded
            if use_gpu and True not in self.MODEL and torch.cuda.is_available():
                try:
                    with self.MODEL_LOCK:
                        self.MODEL[True] = KModel().to('cuda').eval()
                except Exception as e:
                    print(f"GPU load failed: {e}. Using CPU.")
                    use_gpu = False
            
            voice_code = self.VOICES[voice]
            pipeline = self.PIPELINES[voice_code[0]]
            start_time = time.time()
            
            if enable_blending and second_voice and second_voice != voice:
                second_code = self.VOICES[second_voice]
                pipeline2 = self.PIPELINES[second_code[0]]
                
                # Process text into phoneme sequences
                processed_text = self._normalize_text(text.replace('\n', ' ').strip())
                phoneme_sequences = list(pipeline(processed_text, voice_code, speed))
                if not phoneme_sequences:
                    print("No phoneme sequences generated.")
                    return (24000, np.zeros(1000, dtype=np.float32)), text
                
                # Get reference styles for both voices
                _, ps, _ = phoneme_sequences[0]
                ref_s1 = pipeline.load_voice(voice_code)[len(ps)-1]
                ref_s2 = pipeline2.load_voice(second_code)[len(ps)-1]
                
                # Blend styles
                print(f"Blending '{voice}' ({blend_ratio*100}%) and '{second_voice}' ({(1-blend_ratio)*100}%)")
                if isinstance(ref_s1, torch.Tensor) and isinstance(ref_s2, torch.Tensor):
                    blended_ref_s = ref_s1 * blend_ratio + ref_s2 * (1 - blend_ratio)
                else:
                    blended_ref_s = np.add(ref_s1 * blend_ratio, ref_s2 * (1 - blend_ratio))
                
                # Process chunks with blended style
                final_audio, sample_rate = self._process_text_chunks(
                    text, pipeline, voice_code, speed, use_gpu, ref_s=blended_ref_s
                )
                
                if final_audio is None:
                    print("Blended audio generation failed, falling back to primary voice.")
                    final_audio, sample_rate = self._process_text_chunks(text, pipeline, voice_code, speed, use_gpu)
            else:
                # Single voice, no blending
                final_audio, sample_rate = self._process_text_chunks(text, pipeline, voice_code, speed, use_gpu)
            
            # Ensure we have a valid numpy array for the audio output
            if final_audio is None:
                print("Audio generation returned None, creating empty audio.")
                final_audio = np.zeros(1000, dtype=np.float32)
            elif isinstance(final_audio, int):
                print(f"Audio generation returned integer {final_audio}, creating empty audio.")
                final_audio = np.zeros(1000, dtype=np.float32)
            elif isinstance(final_audio, torch.Tensor):
                final_audio = final_audio.cpu().numpy()
            
            # Make sure it's a 1D array for Gradio's Audio component
            if hasattr(final_audio, 'ndim') and final_audio.ndim > 1:
                final_audio = final_audio.squeeze()
            
            # Pad if too short
            if hasattr(final_audio, 'size') and final_audio.size < 1000:
                final_audio = np.pad(final_audio, (0, 1000 - len(final_audio)), 'constant')
            
            print(f"Completed in {time.time() - start_time:.2f} seconds")
            
            # Add to history if requested
            if save_to_history:
                self._add_to_history(text, voice, speed, enable_blending, second_voice, blend_ratio)
            
            # Ensure sample_rate is a valid integer
            if not isinstance(sample_rate, int) or sample_rate <= 0:
                sample_rate = 24000
            
            # Return with sample_rate first, as required by Gradio's Audio component
            return (sample_rate, final_audio), text
            
        except Exception as e:
            # Catch any unexpected errors and return empty audio
            print(f"Error in speech generation: {e}")
            return (24000, np.zeros(1000, dtype=np.float32)), text
    
    def generate_voice_preview(self, voice, use_gpu=True):
        """Generate a short voice preview"""
        preview_text = f"This is a preview of the {voice} voice."
        return self.generate_speech(preview_text, voice, 1.0, use_gpu, save_to_history=False)[0]
    
    def save_preset(self, preset_name, voice, speed, enable_blending, second_voice, blend_ratio):
        """Save current voice settings as a preset"""
        if not preset_name.strip():
            return gr.update(), gr.update(value="Please enter a preset name")
        
        preset = {
            "voice": voice,
            "speed": speed,
            "enable_blending": enable_blending,
            "second_voice": second_voice,
            "blend_ratio": blend_ratio
        }
        
        self.user_presets[preset_name] = preset
        self._save_user_presets()
        
        return gr.update(choices=list(self.user_presets.keys())), gr.update(value=f"Preset '{preset_name}' saved successfully")
    
    def load_preset(self, preset_name):
        """Load a saved preset"""
        if not preset_name or preset_name not in self.user_presets:
            return [gr.update()] * 5 + [gr.update(value="Preset not found")]
            
        preset = self.user_presets[preset_name]
        
        return [
            gr.update(value=preset["voice"]),
            gr.update(value=preset["speed"]),
            gr.update(value=preset["enable_blending"]),
            gr.update(value=preset["second_voice"]),
            gr.update(value=preset["blend_ratio"]),
            gr.update(value=f"Preset '{preset_name}' loaded")
        ]
    
    def delete_preset(self, preset_name):
        """Delete a saved preset"""
        if not preset_name or preset_name not in self.user_presets:
            return gr.update(), gr.update(value="Preset not found")
            
        del self.user_presets[preset_name]
        self._save_user_presets()
        
        return gr.update(choices=list(self.user_presets.keys())), gr.update(value=f"Preset '{preset_name}' deleted")
    
    def load_history_entry(self, selected_index):
        """Load settings from a history entry"""
        if not self.history or selected_index < 0 or selected_index >= len(self.history):
            return [gr.update()] * 6 + [gr.update(value="Invalid history selection")]
            
        entry = self.history[selected_index]
        
        return [
            gr.update(value=entry["text"]),
            gr.update(value=entry["voice"]),
            gr.update(value=entry["speed"]),
            gr.update(value=entry["enable_blending"]),
            gr.update(value=entry["second_voice"]),
            gr.update(value=entry["blend_ratio"]),
            gr.update(value=f"Loaded entry from {entry['timestamp']}")
        ]
    
    def batch_process_texts(self, texts, voice, speed, use_gpu):
        """Process multiple texts and return a list of audio outputs"""
        audio_outputs = []
        status_msg = ""
        
        try:
            texts_list = texts.strip().split("\n\n")
            total_texts = len(texts_list)
            
            for i, text in enumerate(texts_list):
                if not text.strip():
                    continue
                    
                status_msg = f"Processing {i+1}/{total_texts}..."
                audio, _ = self.generate_speech(text, voice, speed, use_gpu)
                audio_outputs.append(audio)
                
            status_msg = f"Completed batch processing of {len(audio_outputs)} texts"
            return audio_outputs, status_msg
            
        except Exception as e:
            status_msg = f"Error in batch processing: {e}"
            return audio_outputs, status_msg

    def create_ui(self):
        """Create a Gradio UI for the Kokoro TTS system"""
        with gr.Blocks(title="Kokoro TTS") as demo:
            gr.Markdown("# 🔊 Kokoro TTS")
            gr.Markdown("Generate natural-sounding speech with voice customization and blending")
            
            # Status message for feedback
            status_msg = gr.Textbox(label="Status", value="Ready", visible=True)
            
            # TTS Interface
            with gr.Row():
                with gr.Column(scale=2):
                    text_input = gr.Textbox(
                        label="Text to speak",
                        placeholder="Enter the text you want to convert to speech...",
                        lines=5,
                        value="Welcome to Kokoro TTS. This advanced text-to-speech system creates natural-sounding voices with voice blending."
                    )
                    
                    with gr.Row():
                        voice_selector = gr.Dropdown(
                            choices=list(self.VOICES.keys()),
                            value="🇺🇸 🚺 Heart ❤️",
                            label="Voice"
                        )
                        
                        speed_slider = gr.Slider(
                            minimum=0.5,
                            maximum=2.0,
                            value=1.0,
                            step=0.1,
                            label="Speed"
                        )
                        
                        use_gpu = gr.Checkbox(
                            value=torch.cuda.is_available(),
                            label="Use GPU",
                            visible=torch.cuda.is_available()
                        )
                    
                    with gr.Accordion("Voice Blending", open=False):
                        enable_blending = gr.Checkbox(
                            value=False,
                            label="Enable Voice Blending"
                        )
                        
                        with gr.Row():
                            second_voice = gr.Dropdown(
                                choices=list(self.VOICES.keys()),
                                value="🇺🇸 🚹 Michael",
                                label="Second Voice"
                            )
                            
                            blend_ratio = gr.Slider(
                                minimum=0.0,
                                maximum=1.0,
                                value=0.7,
                                step=0.1,
                                label="Primary Voice Ratio"
                            )
                    
                    generate_btn = gr.Button("Generate Speech", variant="primary")
                
                with gr.Column(scale=1):
                    audio_output = gr.Audio(
                        label="Generated Speech",
                        type="numpy",
                        autoplay=True
                    )
                    
                    text_output = gr.Textbox(
                        label="Processed Text",
                        lines=5,
                        visible=False
                    )
            
            # Preset Management
            with gr.Accordion("Presets & History", open=False):
                with gr.Row():
                    with gr.Column():
                        preset_name = gr.Textbox(label="Preset Name", placeholder="Enter a name for this preset")
                        save_preset_btn = gr.Button("Save Current Settings as Preset")
                        preset_list = gr.Dropdown(choices=list(self.user_presets.keys()), label="Saved Presets")
                        with gr.Row():
                            load_preset_btn = gr.Button("Load Preset")
                            delete_preset_btn = gr.Button("Delete Preset")
                    
                    with gr.Column():
                        history_entries = gr.Dropdown(
                            choices=[f"{entry['timestamp']} - {entry['text'][:30]}..." for entry in self.history],
                            label="History",
                            type="index"
                        )
                        load_history_btn = gr.Button("Load from History")
                
                preset_status = gr.Textbox(label="Preset Status", value="")
            
            # Add example buttons at the bottom
            with gr.Row():
                gr.Markdown("## Example Phrases")
            
            with gr.Row():
                example1_btn = gr.Button("Example 1: Basic Introduction")
                example2_btn = gr.Button("Example 2: British Accent")
                example3_btn = gr.Button("Example 3: Different Style")
            
            with gr.Row():
                example4_btn = gr.Button("Example 4: Voice Blending")
                example5_btn = gr.Button("Example 5: British Blending")
            
            # Create a wrapper function to properly handle the audio format
            def generate_speech_wrapper(text, voice, speed, use_gpu, enable_blending, second_voice, blend_ratio):
                # Call the original function
                result, processed_text = self.generate_speech(
                    text, voice, speed, use_gpu, enable_blending, second_voice, blend_ratio
                )
                # Return the result directly - it's already in the format (sample_rate, audio_array)
                return result, processed_text, "Speech generated successfully."
            
            # Event handlers for TTS tab
            generate_btn.click(
                fn=generate_speech_wrapper,
                inputs=[
                    text_input, voice_selector, speed_slider, use_gpu,
                    enable_blending, second_voice, blend_ratio
                ],
                outputs=[audio_output, text_output, status_msg]
            )
            
            # Preset management
            save_preset_btn.click(
                fn=self.save_preset,
                inputs=[
                    preset_name, voice_selector, speed_slider,
                    enable_blending, second_voice, blend_ratio
                ],
                outputs=[preset_list, preset_status]
            )
            
            load_preset_btn.click(
                fn=self.load_preset,
                inputs=[preset_list],
                outputs=[
                    voice_selector, speed_slider, enable_blending,
                    second_voice, blend_ratio, preset_status
                ]
            )
            
            delete_preset_btn.click(
                fn=self.delete_preset,
                inputs=[preset_list],
                outputs=[preset_list, preset_status]
            )
            
            # History management
            load_history_btn.click(
                fn=self.load_history_entry,
                inputs=[history_entries],
                outputs=[
                    text_input, voice_selector, speed_slider, enable_blending,
                    second_voice, blend_ratio, preset_status
                ]
            )
            
            # Example button handlers
            def set_example1():
                return {
                    text_input: "Hello, my name is Kokoro. I am a text-to-speech system.",
                    voice_selector: "🇺🇸 🚺 Heart ❤️",
                    speed_slider: 1.0,
                    use_gpu: True,
                    enable_blending: False,
                    second_voice: "🇺🇸 🚹 Michael",
                    blend_ratio: 0.7
                }
                
            def set_example2():
                return {
                    text_input: "The quick brown fox jumps over the lazy dog.",
                    voice_selector: "🇬🇧 🚹 George",
                    speed_slider: 1.0,
                    use_gpu: True,
                    enable_blending: False,
                    second_voice: "🇬🇧 🚺 Emma",
                    blend_ratio: 0.7
                }
                
            def set_example3():
                return {
                    text_input: "I can speak with different voices and styles. Let me demonstrate that for you.",
                    voice_selector: "🇺🇸 🚺 Nicole 🎧",
                    speed_slider: 1.2,
                    use_gpu: True,
                    enable_blending: False,
                    second_voice: "🇺🇸 🚹 Fenrir",
                    blend_ratio: 0.7
                }
                
            def set_example4():
                return {
                    text_input: "Blending voices can create unique speaking styles and personalities.",
                    voice_selector: "🇺🇸 🚺 Bella 🔥",
                    speed_slider: 1.0,
                    use_gpu: True,
                    enable_blending: True,
                    second_voice: "🇺🇸 🚹 Puck",
                    blend_ratio: 0.6
                }
                
            def set_example5():
                return {
                    text_input: "This is an example of a British accent with voice blending.",
                    voice_selector: "🇬🇧 🚺 Isabella",
                    speed_slider: 0.9,
                    use_gpu: True,
                    enable_blending: True,
                    second_voice: "🇬🇧 🚹 Lewis",
                    blend_ratio: 0.5
                }
            
            # Connect example buttons to their handlers
            example1_btn.click(
                fn=set_example1,
                inputs=None,
                outputs=[text_input, voice_selector, speed_slider, use_gpu, enable_blending, second_voice, blend_ratio]
            )
            
            example2_btn.click(
                fn=set_example2,
                inputs=None,
                outputs=[text_input, voice_selector, speed_slider, use_gpu, enable_blending, second_voice, blend_ratio]
            )
            
            example3_btn.click(
                fn=set_example3,
                inputs=None,
                outputs=[text_input, voice_selector, speed_slider, use_gpu, enable_blending, second_voice, blend_ratio]
            )
            
            example4_btn.click(
                fn=set_example4,
                inputs=None,
                outputs=[text_input, voice_selector, speed_slider, use_gpu, enable_blending, second_voice, blend_ratio]
            )
            
            example5_btn.click(
                fn=set_example5,
                inputs=None,
                outputs=[text_input, voice_selector, speed_slider, use_gpu, enable_blending, second_voice, blend_ratio]
            )
            
            # Footer
            gr.Markdown("""
            ## About Kokoro TTS
            
            Kokoro TTS is a high-quality text-to-speech system that generates natural-sounding voices.
            It supports multiple languages, voice styles, and voice blending.
            
            - **Multiple Languages**: Currently supports US and UK English
            - **Voice Blending**: Combine two voices to create unique speaking styles
            - **User Presets**: Save your favorite voice configurations for quick access
            - **History**: Easily recall and reuse previous text-to-speech generations
            """)
        
        return demo

# Create and launch the application
if __name__ == "__main__":
    app = KokoroTTSApp()
    demo = app.create_ui()
    demo.launch()