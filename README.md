# Geeky Ghost Kokoro TTS Webui

A high-quality Text-to-Speech application with voice customization and blending capabilities, built using Gradio.

![Kokoro TTS](https://raw.githubusercontent.com/username/kokoro-tts/main/images/screenshot.png)

## Features

- **Natural-sounding TTS**: Convert text to high-quality, natural-sounding speech
- **Multiple Voices**: Choose from a wide range of US and UK English voices (both male and female)
- **Voice Blending**: Create unique speaking styles by blending two voices together
- **Speed Control**: Adjust speaking speed from 0.5x to 2.0x
- **GPU Acceleration**: Utilize GPU for faster audio generation (when available)
- **User Presets**: Save and load your favorite voice configurations
- **Generation History**: Easily recall and reuse previous text-to-speech generations
- **Batch Processing**: Convert multiple text entries in one go

## Requirements

- Python 3.8+
- CUDA-compatible GPU (recommended but not required)
- Internet connection for initial model download

## Installation

### Option 1: Using the launcher (Windows)

1. Make sure you have Python 3.8+ installed and added to your PATH
2. Clone this repository:
   ```bash
   git clone https://github.com/GeekyGhost/Geeky-Ghost-Kokoro-TTS-Webui.git
   cd Geeky-Ghost-Kokoro-TTS-Webui
   ```
3. Run `run.bat` to automatically set up the environment and launch the application

### Option 2: Manual installation

1. Make sure you have Python 3.8+ installed
2. Clone this repository:
   ```bash
   git clone https://github.com/GeekyGhost/Geeky-Ghost-Kokoro-TTS-Webui.git
   cd Geeky-Ghost-Kokoro-TTS-Webui
   ```
3. Create a virtual environment:
   ```bash
   python -m venv venv
   ```
4. Activate the virtual environment:
   - Windows: `venv\Scripts\activate`
   - Linux/Mac: `source venv/bin/activate`
5. Install the dependencies:
   ```bash
   pip install -r requirements.txt
   ```
   
   Note: For better audio processing, these additional packages are recommended:
   ```bash
   pip install resampy==0.4.2
   pip install librosa>=0.10.0
   ```
   
6. Run the application:
   ```bash
   python main.py
   ```

### First Run Information

On first run, the application will:
1. Create necessary directories if they don't exist
2. Download model files from Hugging Face if needed
3. Set up default configuration

The model will automatically download voice files as needed. The Kokoro TTS system uses the kokoro version 0.8.4 which avoids pickle-related issues found in some other versions.

## Usage

### Basic Usage

1. Enter the text you want to convert to speech in the text input area
2. Select a voice from the dropdown menu
3. Adjust the speed using the slider if needed
4. Click "Generate Speech" to create the audio
5. The generated audio will automatically play and can be downloaded

### Voice Blending

1. Expand the "Voice Blending" section
2. Enable voice blending by checking the box
3. Select a second voice from the dropdown
4. Adjust the blend ratio slider to control the mix between the two voices
5. Click "Generate Speech" to create audio with the blended voice

### Using Presets

1. Expand the "Presets & History" section
2. To save current settings:
   - Enter a name for the preset
   - Click "Save Current Settings as Preset"
3. To load a preset:
   - Select a saved preset from the dropdown
   - Click "Load Preset"
4. To delete a preset:
   - Select the preset to delete
   - Click "Delete Preset"

### Using History

1. Expand the "Presets & History" section
2. Select an entry from the history dropdown
3. Click "Load from History" to restore those settings
4. The text and voice settings will be loaded from the selected history entry

## Voice Options

### US English Voices
- Heart, Bella, Nicole, Aoede, Kore, Sarah, Nova, Sky, Alloy, Jessica, River (Female)
- Michael, Fenrir, Puck, Echo, Eric, Liam, Onyx, Adam (Male)

### UK English Voices
- Emma, Isabella, Alice, Lily (Female)
- George, Fable, Lewis, Daniel (Male)

## Configuration

The application uses a `config.json` file to store default settings:

```json
{
  "use_gpu": true,
  "default_voice": "af_heart",
  "max_chunk_size": 400,
  "crossfade_ms": 30,
  "max_history_entries": 20,
  "theme": "auto",
  "sample_rate": 24000
}
```

You can modify these settings to customize the default behavior of the application.

## Troubleshooting

### Audio Generation Issues

- If GPU acceleration is causing problems, try disabling the "Use GPU" option
- For long texts, the application automatically breaks them into chunks for better processing
- If a specific voice is not working well, try an alternative voice or adjust the speed

### Installation Issues

- Make sure you have Python 3.8 or higher installed
- Ensure you have pip installed and updated to the latest version
- If you encounter CUDA-related errors but have a compatible GPU, make sure you have the appropriate CUDA toolkit installed
- The virtual environment setup in `run.bat` handles most installation issues automatically

## Contributing

Contributions are welcome! Please feel free to submit a Pull Request.

## License

This project is licensed under the Apache License 2.0 - see the LICENSE.txt file for details.

## Acknowledgments

- Built using the [Kokoro](https://github.com/username/kokoro) TTS framework
- User interface created with [Gradio](https://gradio.app/)
