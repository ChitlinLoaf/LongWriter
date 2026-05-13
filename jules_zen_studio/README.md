# JULES ZEN STUDIO

JULES ZEN STUDIO is a first-pass static prototype for a dark, neon-green AI co-producer dashboard. It is intentionally self-contained so it can be opened locally in a desktop browser or embedded later in a Pythonista `ui.WebView` without needing a backend service.

## What is included

- Dark charcoal dashboard with neon green accents.
- Header and runtime subtitle for **JULES ZEN STUDIO** / **Jules Co-Producer Runtime**.
- Mode indicators for **ZEN MODE**, **Hugging Face Audio Engineers**, and **Llama Gopher**.
- Chat panel with mocked Jules replies.
- System status, transport controls, and animated level meters.
- 7-band EQ visualization.
- Fake vocal clip list and editable producer notes.
- Animated waveform canvas, fake pitch curve canvas, synth lanes, clip info, and pitch detection visual.
- Footer status readouts for sample rate, buffer size, latency, HF models, Llama inference, and Zen state.

## How to open locally

No build step is required.

1. Open this folder:
   ```text
   jules_zen_studio/
   ```
2. Double-click `index.html`, or open it from a browser with **File → Open**.
3. Optional local static server from the repository root:
   ```bash
   python -m http.server 8000
   ```
   Then visit:
   ```text
   http://localhost:8000/jules_zen_studio/
   ```

## Mocked prototype behavior

This version does not connect to live audio, AI inference, or DSP engines yet. The following items are simulated in `app.js`:

- Vocal clips and clip metadata.
- Jules co-producer chat responses.
- Waveform animation.
- Input/output/monitor meters.
- Pitch curve and detected notes.
- Synth lane arrangement blocks.
- DSP load and latency readouts.

## Future integration map

### Pythonista

The UI is plain HTML/CSS/JS so it can be loaded into a Pythonista `ui.WebView`. Future Python code can inject JSON into the page for clips, meters, notes, and transport state. A likely bridge shape is:

- Python records or imports audio.
- Python runs lightweight analysis or sends work to local/network services.
- Python calls JavaScript functions in the WebView to update meters, waveform buffers, pitch points, and clip metadata.

### Hugging Face

Future Hugging Face integration can power audio engineering tools such as source separation, transcription, voice activity detection, and music generation. Do not put secrets in the static app. Use environment variables or a local backend proxy with placeholder names only:

- `HF_TOKEN`

### Ollama / local Llama

The **Llama Gopher** indicator is currently visual only. A later local service can connect to Ollama for co-producer reasoning, prompt routing, notes, and arrangement suggestions. Use a local environment variable or config value such as:

- `OLLAMA_HOST`

### Gemini / multimodal assistant expansion

If a future assistant layer uses Gemini for text, image, or multimodal analysis, keep the browser app keyless and route requests through a local or server-side adapter configured with:

- `GEMINI_API_KEY`

### DSP and audio engine

The transport, EQ, meters, waveform, and pitch views are UI shells. Future DSP work can add:

- Microphone or file input.
- Web Audio API playback and routing.
- Real FFT-based meters.
- Pitch detection from live buffers.
- EQ and compressor nodes.
- Clip export and session persistence.
