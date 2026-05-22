# SMART DAW v2 — Architecture Analysis (Pythonista + Objective-C)

## 0) Scope and assumptions
- **Execution target:** Pythonista on iOS (Python runtime + `ui` + `objc_util`).
- **Primary mission:** Producer-intelligence DAW shell with real-time telemetry and analysis overlays.
- **Current phase:** Architecture-only blueprint (no full implementation).
- **Design goal:** Modular, touch-first, low-latency aware, graceful degradation when optional modules/providers are unavailable.

---

## 1) System identity and operating model
SMART DAW v2 should be treated as an **audio operating system layer** with three parallel loops:

1. **Audio loop**: transport, buffering, playback/record state.
2. **Analysis loop**: RMS/FFT/spectrogram/features + recommendation precompute.
3. **UI loop**: composited rendering, gestures, event visualization, co-producer feedback.

This prevents UI and AI orchestration from interfering with timing-sensitive audio paths.

---

## 2) Python vs Objective-C responsibility split

### Python-native (orchestration + analytics)
- Session/state model and project graph.
- DSP feature extraction (NumPy, `audioop`, `wave`).
- AI routing/orchestration policies.
- Telemetry/event schema + JSONL persistence.
- Annotation logic, confidence/rationale aggregation.
- Export/import (JSON metadata, timeline markers).

### Objective-C bridged (UIKit/AVFoundation-critical)
- High-frequency gesture handling and kinetic scrolling.
- `UIView`/`CALayer` backed draw surfaces for smooth visual updates.
- `CADisplayLink` frame clock integration.
- `AVAudioSession` setup/interruption management.
- `AVAudioEngine` transport plumbing and node graph ownership.
- Low-latency status indicators mirrored into Python state.

**Rule of thumb:** anything requiring deterministic per-frame responsiveness or strict audio callback behavior stays native-facing; everything semantic/planning-heavy stays Python.

---

## 3) Layered module architecture

```text
┌─────────────────────────────────────────────────────────────┐
│                   Presentation Layer                        │
│ ui Views + objc_util UIKit wrappers + touch gestures        │
└───────────────┬─────────────────────────────────────────────┘
                │ ViewModel/state events
┌───────────────▼─────────────────────────────────────────────┐
│                 Interaction & State Layer                   │
│ panel layout manager, transport state, selection model      │
└───────────────┬─────────────────────────────────────────────┘
                │ commands / pub-sub events
┌───────────────▼─────────────────────────────────────────────┐
│              Audio & Analysis Runtime Layer                 │
│ AVAudio bridge, buffers, RMS/FFT, spectrogram tiles         │
└───────────────┬─────────────────────────────────────────────┘
                │ feature vectors / clip descriptors
┌───────────────▼─────────────────────────────────────────────┐
│               Intelligence Orchestration Layer              │
│ provider router, fallback matrix, recommendation engine     │
└───────────────┬─────────────────────────────────────────────┘
                │ append-only events
┌───────────────▼─────────────────────────────────────────────┐
│               Persistence & Telemetry Layer                 │
│ JSONL event log, session snapshots, asset/clip metadata     │
└─────────────────────────────────────────────────────────────┘
```

---

## 4) Objective-C / UIKit bridge design

## Core bridge adapters
- `objc_bridge/uikit_views.py` (wrapper constructors for `UIView`, `UILabel`, `UIButton`, `UIScrollView`, `UICollectionView`).
- `objc_bridge/layers.py` (`CALayer` creation, tile-backed update hooks).
- `objc_bridge/display_link.py` (`CADisplayLink` callbacks routed into throttled Python dispatch).
- `objc_bridge/audio_session.py` (`AVAudioSession` category, sample rate, interruption route changes).
- `objc_bridge/audio_engine.py` (`AVAudioEngine`, player/mixer node setup, meter taps).

## Bridging rules
- Python never performs heavy math inside native callback context.
- Native callbacks only queue lightweight events into lock-free or minimal-lock queues.
- Python workers consume queues asynchronously for analysis and AI logic.

---

## 5) GUI architecture plan (DAW layout)

## View hierarchy
1. **Root Studio View**
   - Top Bar (session, BPM, transport, CPU/latency/provider indicators)
   - Main Split Container
     - Left Panel: clip browser/library/tags/search
     - Center Panel: arrangement timeline + waveform/spectrogram stack
     - Right Panel: metadata/pitch role/AI recommendations
   - Lower Panel: synth rack + FX rack + routing matrix + harmony controls
   - Floating AI Panel: co-producer chat + diagnostics + task plans

## Panel behavior
- Collapsible side panels with snap points.
- Gesture zones:
  - pinch = timeline zoom,
  - two-finger drag = spectral viewport pan,
  - long press = marker/effect suggestion context menu.
- Scroll optimization via clipping regions and dirty-rect invalidation.

## Rendering priorities
1. Transport cursor + playhead alignment.
2. Waveform envelope + active selection.
3. Spectrogram tiles (incremental updates).
4. Overlay markers (AI cues/effect regions/notes).
5. Non-critical UI chrome and secondary stats.

## Update frequencies
- Waveform/playhead: 30–60 FPS target.
- FFT analyzer: 10–20 FPS target.
- Spectrogram: tiled streaming updates at adaptive cadence.
- AI panel/provider status: event-driven, not frame-driven.

---

## 6) Audio engine architecture (real-time aware)

## Pipeline
```text
Input/File Source
    -> AVAudioEngine graph (player nodes / mixer)
    -> monitoring taps (RMS, peak, windowed samples)
    -> ring buffer bridge
    -> Python DSP workers (FFT/features/transients)
    -> recommendation engine
    -> overlay/event outputs to UI timeline
```

## Feature extraction blocks
- RMS envelope + peak + crest factor.
- Spectral centroid + rolloff + band-energy buckets.
- Transient detector (energy flux + short-window diff).
- Cadence pulse estimation (onset intervals + tempo proximity).
- Silence/trim candidate finder.
- Pitch contour estimator (placeholder abstraction for future model).

## Threading model
- **Audio thread (native):** transport + node graph + buffer taps only.
- **Analysis worker pool (Python threads):** FFT/spectrogram/features.
- **UI thread:** render + gestures + lightweight state diff apply.
- **AI/router worker:** provider calls, recommendation synthesis.

No blocking network, no disk I/O, and no matplotlib plotting on audio callback path.

---

## 7) Effect intelligence engine design

## Inputs
- waveform windows, RMS, FFT bins, spectrogram tiles, transient maps, vocal-energy lanes.

## Decision outputs
- Compression watch zones.
- Delay throw timing markers.
- Reverb tail suggestions.
- Gate/de-ess candidate regions.
- Stereo widening/saturation hints.
- Harmonic doubling opportunities.

## Frequency-zone heuristics (baseline)
- 20–80 Hz: rumble/sub containment cues.
- 80–250 Hz: mud/body control cues.
- 250–1000 Hz: intelligibility balance cues.
- 1–4 kHz: presence/harshness cues.
- 5–12 kHz: air/sibilance cues.

## Output schema
Each recommendation:
- `time_start`, `time_end`, `type`, `confidence`, `rationale`, `feature_refs`, `suggested_params`.

Plus exportable JSON for later ML training and provider inference.

---

## 8) AI orchestration and provider routing

## Provider roles
- **HuggingFace:** embeddings/audio-model inference jobs.
- **Gemini Studio:** long-context planning + multimodal interpretation.
- **Jules API:** orchestration/delegation/cross-provider planning.
- **Ollama:** offline/private fallback.
- **GitHub:** project memory/changelog sync.
- **Freesound:** sample/ambience discovery.

## Routing hierarchy
1. Classify task (`analysis`, `generation`, `planning`, `retrieval`, `offline`).
2. Score providers by capability, latency, cost, privacy constraints.
3. Select primary + fallback chain.
4. Emit telemetry events for decision and response latency.

## Fallback matrix (example)
- HF unavailable -> Ollama local model for approximate semantic tasks.
- Gemini unavailable -> Jules or local planner with reduced context depth.
- Network degraded -> offline mode with queued jobs.

---

## 9) Scientific visualization system

## Visual subsystems
- Waveform + RMS overlay.
- FFT magnitude view.
- Rolling spectrogram (tile cache).
- Cadence grid and pulse markers.
- Pitch contour + harmonic density lane.
- Clipping heatmap + stereo field map + loudness timeline.

## Optimization strategy
- Tile-based redraw for spectrogram to avoid full-canvas repaints.
- Multi-resolution caches for zoomed waveform levels.
- Adaptive frame throttling under CPU pressure.
- Defer non-critical annotations when transport is active.

---

## 10) Harmony / vocal stack engine (design phase)

## Core model
- Assign vocal roles (root/third/fifth/octaves/doubles/whisper).
- Compute harmonic compatibility score per section.
- Generate pan/formant placement suggestions.
- Attach trigger events to timeline segments.

## UI representation
- Harmony matrix grid.
- Lane color coding by role.
- Pan arc visualization.
- Trigger buttons for apply/audition/commit.

---

## 11) Synth architecture (design phase)

## Engine blocks
- OSC1/OSC2/sub, FM amount, filter/resonance, ADSR, glide, arp, mod matrix.
- Future: wavetable/granular/AI patch generation.

## UI blocks
- Touch knobs and XY modulation pad.
- Compact keyboard strip.
- Mod overlays tied to gesture automation lanes.

---

## 12) HPET-style telemetry/event system

## Event schema
- `event_id`, `timestamp`, `monotonic_time`, `source`, `action`, `latency_ms`,
- `provider`, `cpu_load`, `memory_estimate`, `audio_state`, `session_id`.

## Storage and replay
- Append-only JSONL.
- Session replay reconstructs transport state, UI actions, AI decisions.
- Enables workflow analytics and future autonomous optimization.

---

## 13) Minimal stable skeleton plan (first build target)

## Phase A (must-have)
- App shell with panel placeholders and transport state.
- AV session bootstrap bridge stubs.
- Event bus + JSONL logger.
- Mock waveform timeline + synthetic FFT data path.
- Provider registry with no-secret config validation.

## Phase B (functional prototype)
- Real audio file load/play + RMS/FFT worker.
- Spectrogram tile updates.
- Effect marker overlays with heuristic suggestions.
- Floating AI panel using mock provider responses.

## Phase C (expansion)
- External provider adapters and fallback chains.
- Harmony assist and synth rack controls.
- Advanced recommendation confidence tuning.

---

## 14) Risk controls and constraints
- Keep audio callback path minimal and deterministic.
- Decouple AI/network operations from transport-critical loops.
- Enforce bounded queues + backpressure to prevent UI stalls.
- Validate module availability and degrade gracefully (disable unavailable subsystems with visible status).
- Never store raw secrets in logs or exports.

---

## 15) Immediate next steps
1. Scaffold folder layout and interface contracts only.
2. Add compatibility probes for Pythonista modules and bridge availability.
3. Build telemetry/event spine first; attach all subsystem stubs.
4. Add synthetic test generators for waveform/FFT/spectrogram so UI and timing can be tuned before full audio capture.

This architecture establishes a stable base for implementation while honoring Pythonista/iOS constraints and preserving a professional DAW interaction model.
