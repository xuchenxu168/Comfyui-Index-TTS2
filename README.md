# 🎙️ ComfyUI IndexTTS2 Plugin

<div align="center">

![IndexTTS2 Logo](https://img.shields.io/badge/IndexTTS2-ComfyUI%20Plugin-blue?style=for-the-badge&logo=audio&logoColor=white)
![Version](https://img.shields.io/badge/Version-2.1-green?style=for-the-badge)
![License](https://img.shields.io/badge/License-Apache%202.0-orange?style=for-the-badge)
![Python](https://img.shields.io/badge/Python-3.8%2B-blue?style=for-the-badge&logo=python)
![PyTorch](https://img.shields.io/badge/PyTorch-2.1%2B-red?style=for-the-badge&logo=pytorch)

**🚀 The Most Advanced Text-to-Speech System for ComfyUI**

*Breakthrough emotionally expressive and duration-controlled autoregressive zero-shot text-to-speech synthesis*

[🎯 Quick Start](#-quick-start) • [📦 Installation](#-installation) • [🎨 Features](#-features) • [📚 Documentation](#-documentation) • [🤝 Community](#-community)

</div>

---

## 🌟 What Makes IndexTTS2 Special?

### 🏆 Industry-First Innovations

<table>
<tr>
<td width="50%">

**🎯 Duration Control**
- First autoregressive TTS with precise timing control
- Speed adjustment (0.5x - 2.0x)
- Target duration specification
- Token-level precision control

</td>
<td width="50%">

**🎭 Speaker-Emotion Disentanglement**
- Independent control of voice and emotion
- Cross-speaker emotion transfer
- Emotion preservation across speakers
- Advanced feature separation

</td>
</tr>
<tr>
<td width="50%">

**🎨 Multi-Modal Emotion Control**
- Audio-based emotion reference
- 8-dimensional emotion vectors
- Natural language emotion descriptions
- Real-time emotion adjustment
- **⚠️ Mode-specific configuration** (vector/text/audio)

</td>
<td width="50%">

**🗣️ Multi-Speaker Conversations**
- 2-4 speaker support with individual emotions
- Classroom discussions, meetings, dialogues
- Modular emotion configuration system
- Custom speaker personality design
- **🆕 Voice consistency control** for better speaker similarity
- **🆕 Reference audio enhancement** for improved cloning quality

</td>
</tr>
</table>

### 🔥 Core Capabilities

- **🎤 Zero-Shot Speaker Cloning**: Clone any voice with just one audio sample
- **🌍 Multi-Language Support**: Chinese, English, and seamless code-switching
- **⚡ Real-Time Performance**: Optimized for fast, high-quality synthesis
- **🧠 GPT Latent Integration**: Enhanced stability and naturalness
- **🎛️ Professional Control**: Prosody, timing, and emotion fine-tuning

## 📦 Installation

<details>
<summary><b>🚀 Quick Installation (Recommended)</b></summary>

### Prerequisites
- ✅ ComfyUI installed and working
- ✅ Python 3.8+ (3.10-3.11 recommended)
- ✅ CUDA-capable GPU (recommended for performance)
- ✅ 10GB+ free disk space

### One-Command Installation
```bash
# Clone and install everything
cd ComfyUI/custom_nodes
git clone https://github.com/your-repo/comfyui-index-tts2.git
cd comfyui-index-tts2
pip install -r requirements.txt
python download_models.py
```

</details>

<details>
<summary><b>📋 Detailed Installation Steps</b></summary>

### Step 1: Clone the Repository
```bash
cd ComfyUI/custom_nodes
git clone https://github.com/your-repo/comfyui-index-tts2.git
cd comfyui-index-tts2
```

### Step 2: Choose Your Installation Method

#### 🌟 Standard Installation (Recommended)
```bash
pip install -r requirements.txt
```

#### ⚡ Minimal Installation (Faster)
```bash
pip install -r requirements_minimal.txt
```

#### 🔧 Full Installation (All Features)
```bash
pip install -r requirements_full.txt
```

#### 🐍 Python 3.12+ Support
```bash
# Quick fix for Python 3.12
python quick_fix_py312.py

# Or full installation
python install_py312.py
```

### Step 3: Install Core Package
```bash
pip install -e index-tts/
```

### Step 4: Download Models
```bash
# Automatic download (recommended)
python download_models.py

# Alternative methods:
# huggingface-cli download IndexTeam/IndexTTS-2 --local-dir=checkpoints
# modelscope download --model IndexTeam/IndexTTS-2 --local_dir checkpoints
```

### Step 5: Verify Installation
```bash
python check_all_dependencies.py
```

</details>

<details>
<summary><b>🔤 Optional: Advanced Text Processing with Pynini</b></summary>

**Pynini** provides professional-grade text normalization for TTS applications, handling complex text formats like numbers, dates, currencies, and abbreviations.

#### 🌟 What Pynini Adds
- **📞 Phone Numbers**: `123-456-7890` → `one two three four five six seven eight nine zero`
- **💰 Currency**: `$29.99` → `twenty nine dollars and ninety nine cents`
- **📅 Dates**: `2024年3月15日` → `二零二四年三月十五日`
- **🔢 Numbers**: `Dr. Smith` → `Doctor Smith`
- **🌍 Mixed Languages**: Better Chinese-English text processing

#### 📦 Installation by Platform

**🐧 Linux (Recommended - Has Pre-compiled Wheels)**
```bash
# Easy installation with pre-compiled wheels (~150MB)
pip install pynini==2.1.6
```

**🍎 macOS**
```bash
# Method 1: Conda (Recommended)
conda install -c conda-forge pynini=2.1.6

# Method 2: Pip (may require compilation)
pip install pynini==2.1.6
```

**🪟 Windows**
```bash
# Method 1: Conda (Recommended)
conda install -c conda-forge pynini=2.1.6

# Method 2: WSL (Windows Subsystem for Linux)
# Install WSL, then use Linux instructions
```

#### ⚠️ Important Notes
- **📦 Large Package**: ~150MB download size
- **🐧 Best Support**: Linux x86_64 with pre-compiled wheels
- **🔧 Compilation**: Other platforms may require C++ compiler
- **🎯 Optional**: Core TTS functionality works without Pynini
- **🚀 Performance**: Significantly improves text processing quality

#### 🚀 Automated Installation Script
```bash
# Use our automated installer (recommended)
python install_pynini.py

# With testing
python install_pynini.py --test

# Force reinstall
python install_pynini.py --force
```

#### 🧪 Manual Test Installation
```bash
# Verify Pynini installation
python -c "import pynini; print('✅ Pynini installed successfully!')"

# Simple test
python -c "
import pynini
rule = pynini.string_map([('$', 'dollar')])
print('✅ Pynini test passed!')
"
```

#### 🎯 When to Install Pynini
**✅ Recommended for:**
- Professional/commercial applications
- Complex text with numbers, dates, currencies
- Multi-language content (Chinese-English)
- High-quality text normalization needs

**❌ Skip if:**
- Simple text-only content
- Quick prototyping/testing
- Limited storage/bandwidth
- Basic personal use

</details>

<details>
<summary><b>🎵 Audio File Setup</b></summary>

### Smart Audio File Management

IndexTTS2 features **intelligent audio file scanning** across your entire ComfyUI installation:

#### 📁 Supported Locations (Auto-Detected)
```
ComfyUI/
├── input/audio/          🌟 Highest Priority
├── input/                🌟 High Priority
├── output/               ✓ Supported
├── audio/                ✓ Supported
├── user/audio/           ✓ Supported
└── [any other location]  ✓ Supported
```

#### 🎯 Two-Level Audio Selection System
1. **Level 1**: Select directory from dropdown
2. **Level 2**: Enter filename in text field

This eliminates long dropdown menus while supporting flexible file organization!

#### 📋 Audio Requirements
- **Formats**: WAV, MP3, FLAC, OGG, M4A, AAC, WMA
- **Duration**: 3-10 seconds recommended
- **Quality**: Clear, noise-free, single speaker
- **Sample Rate**: 16kHz+ (22.05kHz optimal)

#### 🚀 Quick Setup
```bash
python setup_audio_files.py
python test_two_level_selection.py
```

</details>

### ✅ Verification

After installation, restart ComfyUI and look for **IndexTTS2** nodes in the node browser under the `IndexTTS2` category.

## 🎯 Node Architecture

<div align="center">

### 🧩 Complete Node Ecosystem

</div>

<table>
<tr>
<th width="25%">Category</th>
<th width="25%">Node</th>
<th width="50%">Description</th>
</tr>

<tr>
<td rowspan="2"><b>🎤 Core TTS</b></td>
<td><code>IndexTTS2 Basic TTS</code></td>
<td>Foundation zero-shot speaker cloning with high-quality synthesis</td>
</tr>
<tr>
<td><code>IndexTTS2 Basic TTS V2</code></td>
<td>Enhanced version with two-level audio selection system</td>
</tr>

<tr>
<td rowspan="3"><b>🎛️ Control</b></td>
<td><code>IndexTTS2 Duration Control</code></td>
<td>Precise timing control: speed, duration, token-level precision</td>
</tr>
<tr>
<td><code>IndexTTS2 Emotion Control</code></td>
<td>Multi-modal emotion control: audio, vector, text descriptions</td>
</tr>
<tr>
<td><code>IndexTTS2 Advanced Control</code></td>
<td>Combined duration + emotion control with GPT latents</td>
</tr>

<tr>
<td rowspan="2"><b>🗣️ Multi-Speaker</b></td>
<td><code>IndexTTS2 Multi-Talk</code></td>
<td>2-4 speaker conversations with individual emotion control</td>
</tr>
<tr>
<td><code>IndexTTS2 Speaker Emotion Config</code></td>
<td>Modular emotion configuration for complex workflows</td>
</tr>

<tr>
<td rowspan="3"><b>🔧 Utilities</b></td>
<td><code>IndexTTS2 Model Manager</code></td>
<td>Efficient model loading, caching, and memory management</td>
</tr>
<tr>
<td><code>IndexTTS2 Audio Utils</code></td>
<td>Audio processing, analysis, and quality enhancement</td>
</tr>
<tr>
<td><code>IndexTTS2 Audio Browser</code></td>
<td>Smart audio file discovery and management</td>
</tr>

</table>

### 🌟 Key Features by Node

<details>
<summary><b>🎤 IndexTTS2 Basic TTS</b></summary>

**Perfect for**: First-time users, simple voice cloning

**Features**:
- ✅ Zero-shot speaker cloning
- ✅ Multi-language support (Chinese, English, mixed)
- ✅ High-quality synthesis
- ✅ Configurable quality settings
- ✅ GPU/CPU optimization

**Inputs**: Text, speaker audio, output filename
**Outputs**: Synthesized audio, file path, synthesis info

</details>

<details>
<summary><b>🎛️ IndexTTS2 Duration Control</b></summary>

**Perfect for**: Precise timing requirements, video dubbing

**Features**:
- ✅ Speed control (0.5x - 2.0x)
- ✅ Target duration specification
- ✅ Token-level precision
- ✅ Prosody preservation
- ✅ Natural timing adaptation

**Control Modes**:
- `speed_control`: Adjust synthesis speed
- `token_control`: Specify exact token count
- `target_duration`: Set precise output duration
- `auto`: Natural duration with prosody preservation

</details>

<details>
<summary><b>🎭 IndexTTS2 Emotion Control</b></summary>

**Perfect for**: Expressive speech, character voices

**Features**:
- ✅ Audio-based emotion reference
- ✅ 8-dimensional emotion vectors
- ✅ Natural language emotion descriptions
- ✅ Cross-speaker emotion transfer
- ✅ Emotion intensity adjustment

**Emotion Dimensions**:
- Happy, Angry, Sad, Fear, Hate, Low, Surprise, Neutral

**Control Methods**:
- `audio_prompt`: Use emotion reference audio
- `emotion_vector`: 8D emotion control
- `text_description`: Natural language emotions

</details>

<details>
<summary><b>🗣️ IndexTTS2 Multi-Talk</b></summary>

**Perfect for**: Conversations, dialogues, classroom discussions

**Features**:
- ✅ 2-4 speaker support
- ✅ Individual emotion control per speaker
- ✅ Automatic conversation parsing
- ✅ Configurable silence intervals
- ✅ Modular emotion configuration

**Use Cases**:
- Classroom discussions
- Business meetings
- Character dialogues
- Podcast conversations

</details>

## 🚀 Quick Start

<div align="center">

### 🎯 Get Started in 3 Minutes!

</div>

<details>
<summary><b>🎤 Basic Text-to-Speech (Beginner)</b></summary>

**Perfect for**: First-time users, simple voice cloning

### Step-by-Step Guide

1. **Add the node**: `IndexTTS2 Basic TTS`
2. **Set your text**:
   ```
   "Hello! This is my first IndexTTS2 synthesis."
   ```
3. **Choose speaker audio**: Use the two-level selection:
   - Directory: `input/audio`
   - File: `my_voice.wav`
4. **Set output**: `my_first_tts.wav`
5. **Execute**: Click "Queue Prompt"

### Expected Result
High-quality speech in your chosen speaker's voice!

</details>

<details>
<summary><b>🎛️ Duration-Controlled Synthesis (Intermediate)</b></summary>

**Perfect for**: Video dubbing, precise timing

### Step-by-Step Guide

1. **Add the node**: `IndexTTS2 Duration Control`
2. **Choose duration mode**:
   - `speed_control`: Adjust speed (0.8x for slower)
   - `target_duration`: Set exact duration (5.0 seconds)
3. **Configure text and audio** (same as basic)
4. **Execute and compare** timing

### Pro Tips
- Use `speed_control` for natural speed changes
- Use `target_duration` for exact timing requirements
- Enable `prosody_preservation` for better naturalness

</details>

<details>
<summary><b>🎭 Emotion-Controlled Synthesis (Advanced)</b></summary>

**Perfect for**: Character voices, expressive speech

### Method 1: Emotion Vector Control
```
Happy: 0.8, Surprise: 0.2, Neutral: 0.0
Result: Excited, joyful speech
```

### Method 2: Text Description
```
Emotion Text: "devastated and heartbroken"
Result: Deep sadness and emotional pain
```

### Method 3: Audio Reference
```
Emotion Audio: "angry_reference.wav"
Result: Transfers anger from reference to your speaker
```

</details>

<details>
<summary><b>🗣️ Multi-Speaker Conversations (Expert)</b></summary>

**Perfect for**: Dialogues, meetings, classroom discussions

### Quick Setup for 2-Speaker Dialogue

1. **Create emotion configs**:
   ```
   Speaker1 Config: Happy (0.8), Confident
   Speaker2 Config: Sad (0.6), Worried
   ```

2. **⚠️ IMPORTANT: Set correct emotion mode**:
   ```
   For emotion sliders (happy: 0.8) → Use "emotion_vector" mode
   For text descriptions → Use "text_description" mode
   ```

3. **Format conversation**:
   ```
   Speaker1: I'm so excited about this project!
   Speaker2: I'm worried we won't finish on time...
   Speaker1: Don't worry, we've got this!
   ```

4. **Connect everything**:
   - Audio inputs → Speaker audio files
   - Emotion configs → Multi-talk node
   - Set silence duration: 0.5 seconds

5. **Execute**: Get natural conversation with distinct emotions!

### 🎯 Pro Example: Classroom Discussion
Check out `workflow-examples/classroom_4speakers_fixed.json` for a complete 4-speaker classroom discussion with Teacher, Student1, Student2, and Student3!

</details>

### 🎨 Ready-to-Use Examples

| Example | Speakers | Complexity | Use Case |
|---------|----------|------------|----------|
| `simple_2speaker_example.json` | 2 | ⭐ | Basic dialogue |
| `classroom_4speakers_fixed.json` | 4 | ⭐⭐⭐ | Educational content |
| `business_meeting.json` | 3 | ⭐⭐ | Professional scenarios |
| `mixed_emotion_modes.json` | 3 | ⭐⭐⭐ | Advanced emotion control |

### 🚀 Next Steps

1. **Try the examples**: Import a workflow from `workflow-examples/`
2. **Customize emotions**: Experiment with different emotion settings
3. **Create your content**: Build your own conversations and scenarios
4. **Join the community**: Share your creations and get help!

## 📋 Parameter Reference

<div align="center">

### 🎛️ Complete Parameter Guide

</div>

<details>
<summary><b>📝 Text & Audio Parameters</b></summary>

| Parameter | Type | Description | Example |
|-----------|------|-------------|---------|
| `text` | String | Text to synthesize (Chinese, English, mixed) | `"Hello! 你好！"` |
| `speaker_audio` | Audio Path | Speaker reference audio file | `"input/audio/speaker.wav"` |
| `output_filename` | String | Generated audio filename | `"output.wav"` |
| `language` | Dropdown | Language mode | `auto`, `zh`, `en` |

</details>

<details>
<summary><b>⏱️ Duration Control Parameters</b></summary>

| Parameter | Type | Range | Description |
|-----------|------|-------|-------------|
| `duration_mode` | Dropdown | - | Control method |
| `speed_multiplier` | Float | 0.5 - 2.0 | Speed adjustment |
| `target_duration` | Float | 1.0 - 60.0 | Desired duration (seconds) |
| `token_count` | Integer | 10 - 1000 | Specific token count |
| `prosody_preservation` | Float | 0.0 - 1.0 | Maintain natural rhythm |

**Duration Modes**:
- `speed_control`: Adjust synthesis speed
- `token_control`: Specify exact tokens
- `target_duration`: Set precise duration
- `auto`: Natural duration

</details>

<details>
<summary><b>🎭 Emotion Control Parameters</b></summary>

| Parameter | Type | Range | Description |
|-----------|------|-------|-------------|
| `emotion_mode` | Dropdown | - | Control method |
| `emotion_alpha` | Float | 0.0 - 2.0 | Emotion intensity |
| `emotion_audio` | Audio Path | - | Emotion reference audio |
| `emotion_text` | String | - | Natural language description |

**Emotion Vector (8 Dimensions)**:
- `happy` | Float | 0.0 - 1.0 | Joy, excitement
- `angry` | Float | 0.0 - 1.0 | Anger, frustration
- `sad` | Float | 0.0 - 1.0 | Sadness, melancholy
- `fear` | Float | 0.0 - 1.0 | Fear, anxiety
- `hate` | Float | 0.0 - 1.0 | Disgust, hatred
- `low` | Float | 0.0 - 1.0 | Low energy, calm
- `surprise` | Float | 0.0 - 1.0 | Surprise, wonder
- `neutral` | Float | 0.0 - 1.0 | Neutral, balanced

**Emotion Modes**:
- `audio_prompt`: Use emotion reference audio
- `emotion_vector`: 8-dimensional control
- `text_description`: Natural language
- `auto`: Automatic emotion detection

**⚠️ Important: Emotion Mode Configuration**
Make sure your `emotion_mode` setting matches your intended control method:
- If you set emotion vector values (happy, angry, etc.), use `emotion_vector` mode
- If you provide emotion text description, use `text_description` mode
- If you provide emotion audio file, use `audio_prompt` mode

**🔧 Common Issue**: Setting emotion vector values but leaving mode as `text_description` will result in neutral emotion output.

</details>

<details>
<summary><b>🔧 Advanced Settings</b></summary>

| Parameter | Type | Range | Description |
|-----------|------|-------|-------------|
| `use_gpt_latents` | Boolean | - | Enhanced stability |
| `stability_enhancement` | Boolean | - | Improve consistency |
| `clarity_boost` | Boolean | - | Enhance audio clarity |
| `fp16` | Boolean | - | Half-precision (faster) |
| `device` | Dropdown | - | GPU/CPU selection |
| `batch_size` | Integer | 1 - 8 | Batch processing size |

</details>

<details>
<summary><b>🗣️ Multi-Speaker Parameters</b></summary>

| Parameter | Type | Range | Description |
|-----------|------|-------|-------------|
| `num_speakers` | Integer | 2 - 4 | Number of speakers |
| `conversation_text` | Text | - | Formatted dialogue |
| `silence_duration` | Float | 0.1 - 2.0 | Pause between speakers |
| **`voice_consistency`** | **Float** | **0.1 - 2.0** | **🆕 Voice similarity to reference (higher = more similar)** |
| **`reference_boost`** | **Boolean** | **-** | **🆕 Enable reference audio enhancement** |
| `auto_normalize` | Boolean | - | Volume normalization |
| `speaker_balance` | Boolean | - | Balance speaker volumes |

**Conversation Format**:
```
Speaker1: Hello everyone!
Speaker2: Hi there!
Speaker3: Good to see you all!
```

**🎯 Voice Consistency Tips**:
- **High Similarity (1.5-2.0)**: Professional dubbing, character voices
- **Balanced (1.0-1.3)**: General conversations, podcasts
- **Natural (0.8-1.0)**: Casual dialogues, creative content

</details>

<details>
<summary><b>🎯 Voice Consistency Improvement Guide</b></summary>

### 🔧 New Features for Better Voice Similarity

#### 1. Voice Consistency Control
- **Parameter**: `voice_consistency` (0.1 - 2.0)
- **Default**: 1.0 (balanced)
- **Higher values**: More similar to reference audio
- **Lower values**: More natural variation

#### 2. Reference Audio Enhancement
- **Parameter**: `reference_boost` (True/False)
- **Function**: Automatically optimizes reference audio quality
- **Benefits**: Clearer voice cloning, reduced noise

### 📊 Recommended Settings

| Use Case | Voice Consistency | Reference Boost | Temperature | Top P |
|----------|------------------|-----------------|-------------|-------|
| **Professional Dubbing** | 1.6 | ✅ True | 0.6 | 0.8 |
| **Casual Conversation** | 1.1 | ✅ True | 0.7 | 0.9 |
| **Character Role-play** | 1.4 | ✅ True | 0.65 | 0.85 |
| **Natural Dialogue** | 1.0 | ❌ False | 0.8 | 0.9 |

### 🎯 Troubleshooting Voice Issues

**Problem**: Generated voice differs too much from reference
**Solution**:
- Increase `voice_consistency` to 1.5-2.0
- Enable `reference_boost`
- Lower `temperature` to 0.5-0.6
- Use high-quality reference audio (2+ seconds, clear speech)

**Problem**: Voice sounds too mechanical
**Solution**:
- Decrease `voice_consistency` to 0.8-1.0
- Disable `reference_boost`
- Increase `temperature` to 0.8-1.0

### 📋 Reference Audio Best Practices

**Quality Requirements**:
- **Duration**: 2-10 seconds
- **Format**: WAV/FLAC recommended
- **Content**: Clear speech, no background noise
- **Speaker**: Single person, consistent voice

**Optimization Tips**:
- Use speech with varied intonation
- Avoid music, echo, or noise
- Include different emotions/tones
- Ensure good audio quality (16kHz+)

For detailed guidance, see: [docs/voice_consistency_guide.md](docs/voice_consistency_guide.md)

</details>

## 🎨 Usage Examples & Showcase

<div align="center">

### 🌟 Real-World Applications

</div>

<details>
<summary><b>🎤 Example 1: Basic Speaker Cloning</b></summary>

**Scenario**: Clone your voice for audiobook narration

```yaml
Input:
  Text: "Welcome to Chapter One of our exciting adventure story."
  Speaker Audio: "my_voice_sample.wav"
  Language: "auto"

Output:
  High-quality speech in your exact voice
  Perfect for: Audiobooks, podcasts, voice-overs
```

**Pro Tips**:
- Use 5-10 second clear audio samples
- Ensure single speaker, no background noise
- Test with different text lengths

</details>

<details>
<summary><b>🎭 Example 2: Character Voice Creation</b></summary>

**Scenario**: Create distinct character voices for animation

```yaml
Character 1 - Hero:
  Text: "I will protect this city with my life!"
  Emotion Vector: {happy: 0.7, surprise: 0.2, neutral: 0.1}
  Result: Confident, heroic voice

Character 2 - Villain:
  Text: "You cannot stop my evil plan!"
  Emotion Vector: {angry: 0.8, hate: 0.3, neutral: 0.0}
  Result: Menacing, threatening voice
```

</details>

<details>
<summary><b>⏱️ Example 3: Video Dubbing with Precise Timing</b></summary>

**Scenario**: Dub a 5-second video clip exactly

```yaml
Input:
  Text: "This line must match the video timing perfectly."
  Duration Mode: "target_duration"
  Target Duration: 5.0
  Prosody Preservation: 0.8

Output:
  Exactly 5.0 seconds of natural-sounding speech
  Perfect for: Video dubbing, lip-sync, presentations
```

</details>

<details>
<summary><b>🗣️ Example 4: Multi-Speaker Podcast</b></summary>

**Scenario**: Create a 3-person business podcast

```yaml
Host: "Welcome to TechTalk! Today we're discussing AI."
  Emotion: Professional, confident

Guest1: "Thanks for having me! I'm excited to share our research."
  Emotion: Enthusiastic, happy

Guest2: "The implications are concerning though..."
  Emotion: Thoughtful, slightly worried

Result: Natural conversation with distinct personalities
```

</details>

<details>
<summary><b>🎓 Example 5: Educational Content</b></summary>

**Scenario**: Create engaging classroom discussion

```yaml
Teacher: "Today we'll explore quantum physics. Who knows about superposition?"
  Emotion: {happy: 0.3, neutral: 0.7} - Encouraging educator

Student1: "Oh! I know this! Particles can be in multiple states!"
  Emotion: {happy: 0.8, surprise: 0.2} - Excited learner

Student2: "I'm confused... how is that possible?"
  Emotion: {sad: 0.4, fear: 0.3, neutral: 0.3} - Uncertain

Student3: "Wait, does this mean reality isn't what we think?"
  Emotion Text: "philosophical and contemplative"
```

</details>

<details>
<summary><b>🌍 Example 6: Cross-Language Content</b></summary>

**Scenario**: Bilingual customer service

```yaml
Text: "Hello! 您好！Welcome to our international support.
       How can I help you today? 我可以帮助您吗？"

Features:
- Seamless language switching
- Maintains speaker identity across languages
- Natural pronunciation in both languages
```

</details>

### 🎯 Use Case Gallery

| Industry | Application | IndexTTS2 Advantage |
|----------|-------------|-------------------|
| **🎬 Entertainment** | Character voices, dubbing | Emotion control + speaker cloning |
| **📚 Education** | Interactive lessons, audiobooks | Multi-speaker conversations |
| **🏢 Business** | Training materials, presentations | Professional quality + timing control |
| **🎮 Gaming** | Dynamic dialogue, NPCs | Real-time emotion adaptation |
| **♿ Accessibility** | Text-to-speech, voice restoration | Personalized voice synthesis |
| **🎙️ Content Creation** | Podcasts, YouTube, social media | Consistent voice across content |

## 🔧 Troubleshooting & Support

<div align="center">

### 🆘 Common Issues & Solutions

</div>

<details>
<summary><b>❌ Installation Issues</b></summary>

### "No module named 'indextts'" Error
```bash
# Solution 1: Install the package
pip install -e index-tts/

# Solution 2: Check Python path
python -c "import sys; print(sys.path)"

# Solution 3: Reinstall dependencies
pip install -r requirements.txt --force-reinstall
```

### Model Loading Fails
```bash
# Check model files
ls checkpoints/
# Should contain: config.yaml, model files

# Download models again
python download_models.py

# Check GPU memory
nvidia-smi  # For NVIDIA GPUs
```

### Python 3.12 Compatibility
```bash
# Quick fix
python quick_fix_py312.py

# Full installation
python install_py312.py

# Check compatibility
python check_all_dependencies.py
```

</details>

<details>
<summary><b>🎵 Audio Issues</b></summary>

### Audio File Not Found
- ✅ Use the two-level audio selection system
- ✅ Check file exists in selected directory
- ✅ Verify supported formats: WAV, MP3, FLAC, OGG
- ✅ Test with absolute paths if needed

### Poor Audio Quality
**Speaker Audio Requirements**:
- 📏 Duration: 3-10 seconds
- 🔊 Quality: Clear, noise-free
- 👤 Content: Single speaker only
- 📊 Sample Rate: 16kHz+ (22.05kHz optimal)

**Quality Improvement Tips**:
```bash
# Test audio quality
python test_audio_quality.py your_audio.wav

# Audio preprocessing
python fix_audio_quality.py input.wav output.wav
```

</details>

<details>
<summary><b>⚡ Performance Issues</b></summary>

### Memory Optimization
```python
# Use Model Manager for efficient caching
Model Manager → Enable FP16 → Set device to GPU

# Clear cache when needed
Model Manager → Clear Cache
```

### Speed Optimization
- 🚀 Enable CUDA if available
- ⚡ Use FP16 precision
- 🧠 Cache models with Model Manager
- 📦 Use batch processing for multiple files

### GPU Memory Issues
```bash
# Check GPU memory
nvidia-smi

# Reduce memory usage
- Enable FP16
- Reduce batch size
- Use CPU for large models
```

</details>

<details>
<summary><b>🎭 Emotion Control Issues</b></summary>

### ❌ Problem: Emotion Vector Values Ignored (Shows "neutral: 1.0")

**Symptoms**:
- You set `happy: 0.8` in emotion config
- Console shows `neutral: 1.0` instead
- Generated voice sounds neutral despite emotion settings

**Root Cause**: Emotion mode mismatch

**✅ Solution**:
1. **Check your emotion config node**
2. **Change `emotion_mode` from `text_description` to `emotion_vector`**
3. **Reconnect the emotion config to multi-talk node**
4. **Re-run the workflow**

**Step-by-Step Fix**:
```
1. Click on your Speaker Emotion Config node
2. Find the "emotion_mode" dropdown
3. Change from "text_description" → "emotion_vector"
4. Verify your emotion values are set (e.g., happy: 0.8)
5. Make sure "enabled" is checked (True)
6. Execute workflow again
```

### ❌ Problem: Wrong Emotion Mode Selected

**Common Mistakes**:
- Setting emotion vector values but using `text_description` mode
- Providing emotion text but using `emotion_vector` mode
- Using `audio_prompt` mode without emotion audio file

**✅ Mode Selection Guide**:
| Your Input | Correct Mode | Example |
|------------|--------------|---------|
| **Emotion sliders** (happy: 0.8, etc.) | `emotion_vector` | happy: 0.8, angry: 0.2 |
| **Text description** | `text_description` | "excited and joyful" |
| **Audio file** | `audio_prompt` | emotion_audio.wav |
| **Auto-detect** | `auto` | Let system decide |

### ❌ Problem: Emotions Not Working
1. **Check emotion config enabled**: `enabled = true`
2. **Verify connections**: Emotion config → Multi-talk node
3. **Test emotion values**: At least one dimension > 0
4. **Check emotion mode**: Matches your input type

### ❌ Problem: Emotion Vector All Zeros
```python
# System auto-fix: Sets neutral = 0.1
# Manual fix: Set at least one emotion > 0
happy: 0.5, neutral: 0.0  # Good
all zeros → neutral: 0.1  # Auto-corrected
```

### 🔍 Debug Console Messages

**What to Look For**:
```bash
# ✅ Correct (emotion_vector mode):
[MultiTalk] Speaker1 emotion mode: emotion_vector
[MultiTalk] Speaker1 emotions: Happy: 0.80

# ❌ Wrong (text_description mode with no text):
[MultiTalk] Speaker1 emotion mode: text_description
{'neutral': 1.0}  # Falls back to neutral
```

**Expected Console Output**:
```bash
# When working correctly:
[MultiTalk] Emotion mode: emotion_vector
['愤怒:0.00', '高兴:0.80', '恐惧:0.00', '反感:0.00', '悲伤:0.00', '低落:0.00', '惊讶:0.00', '自然:0.20']
{'happy': 0.8, 'angry': 0.0, 'sad': 0.0, 'fear': 0.0, 'hate': 0.0, 'low': 0.0, 'surprise': 0.0, 'neutral': 0.2}
Use the specified emotion vector
```

</details>

### 🔍 Diagnostic Tools

| Tool | Purpose | Command |
|------|---------|---------|
| **Dependency Checker** | Verify all packages | `python check_all_dependencies.py` |
| **Audio Tester** | Test audio files | `python test_audio_quality.py` |
| **Model Validator** | Check model files | `python test_model_loading.py` |
| **System Info** | Hardware compatibility | `python system_info.py` |

### 📞 Getting Help

<table>
<tr>
<td width="50%">

**🐛 Bug Reports**
- GitHub Issues
- Include error logs
- System information
- Reproduction steps

</td>
<td width="50%">

**💬 Community Support**
- Discord: ComfyUI community
- Reddit: r/ComfyUI
- GitHub Discussions
- Documentation wiki

</td>
</tr>
</table>

### 🚀 Performance Benchmarks

| Hardware | Speed | Memory | Quality |
|----------|-------|--------|---------|
| **RTX 4090** | 10x real-time | 8GB | Excellent |
| **RTX 3080** | 6x real-time | 6GB | Excellent |
| **RTX 2080** | 3x real-time | 4GB | Very Good |
| **CPU Only** | 0.5x real-time | 8GB RAM | Good |

## 📚 Documentation & Resources

<div align="center">

### 📖 Complete Documentation Library

</div>

<details>
<summary><b>📋 Technical Documentation</b></summary>

### Core Documentation Files
- 📘 **[FEATURES.md](FEATURES.md)** - Complete feature overview
- 🔧 **[DEPENDENCY_INSTALLATION_GUIDE.md](DEPENDENCY_INSTALLATION_GUIDE.md)** - Installation guide
- 🎭 **[MODULAR_EMOTION_CONTROL_GUIDE.md](MODULAR_EMOTION_CONTROL_GUIDE.md)** - Emotion control
- 🗣️ **[MULTI_TALK_GUIDE.md](MULTI_TALK_GUIDE.md)** - Multi-speaker conversations
- 🎵 **[docs/classroom_discussion_guide.md](docs/classroom_discussion_guide.md)** - Classroom scenarios

### Workflow Examples
- 📁 **[workflow-examples/](workflow-examples/)** - Ready-to-use workflows
- 🚀 **[workflow-examples/QUICK_START.md](workflow-examples/QUICK_START.md)** - Quick start guide
- 📖 **[workflow-examples/README.md](workflow-examples/README.md)** - Example documentation

</details>

<details>
<summary><b>🔬 Technical Specifications</b></summary>

### Model Architecture
**IndexTTS2** is built on cutting-edge autoregressive transformer technology:

- 🧠 **Base Architecture**: GPT-style autoregressive transformer
- 🎯 **Training Paradigm**: Three-stage training with GPT latents
- 📏 **Context Length**: Up to 2048 tokens
- 🔊 **Sample Rate**: 22.05 kHz (configurable)
- 💾 **Model Size**: ~1.5GB (optimized)

### Key Innovations
1. **🎛️ Duration Control**: First autoregressive TTS with precise timing
2. **🎭 Emotion Disentanglement**: Independent voice and emotion control
3. **🌐 Multi-Modal Control**: Audio, vector, and text emotion input
4. **⚡ GPT Latents**: Enhanced stability and naturalness

### Performance Specifications
- **🚀 Synthesis Speed**: Real-time to 10x faster
- **💾 Memory Usage**: 4-8GB GPU (with optimization)
- **🎯 Quality**: High speaker similarity, natural prosody
- **⏱️ Latency**: Sub-second for short texts

</details>

<details>
<summary><b>🛠️ Development Resources</b></summary>

### Testing & Validation Tools
```bash
# Comprehensive dependency check
python check_all_dependencies.py

# Audio quality testing
python test_audio_quality.py

# Model validation
python test_model_loading.py

# Workflow validation
python workflow-examples/validate_workflows.py
```

### Development Scripts
- 🔧 **Setup**: `setup_audio_files.py`, `download_models.py`
- 🧪 **Testing**: `test_*.py` files for various components
- 🔍 **Debugging**: `debug_*.py` files for troubleshooting
- ⚡ **Optimization**: `fix_*.py` files for performance

</details>

## 🤝 Community & Contributing

<div align="center">

### 🌟 Join Our Growing Community!

</div>

<table>
<tr>
<td width="50%">

**🤝 How to Contribute**
1. 🍴 Fork the repository
2. 🌿 Create a feature branch
3. ✨ Make your improvements
4. 🧪 Add tests if applicable
5. 📤 Submit a pull request

**🎯 Contribution Areas**
- 🐛 Bug fixes and improvements
- 📚 Documentation enhancements
- 🎨 New workflow examples
- 🔧 Performance optimizations
- 🌍 Translations and localization

</td>
<td width="50%">

**💬 Community Channels**
- 🐙 **GitHub**: Issues, discussions, PRs
- 💬 **Discord**: ComfyUI community server
- 📱 **Reddit**: r/ComfyUI subreddit
- 📧 **Email**: For commercial inquiries
- 📖 **Wiki**: Community documentation

**🏆 Recognition**
- Contributors listed in CONTRIBUTORS.md
- Special thanks in release notes
- Community showcase features

</td>
</tr>
</table>

## 📄 License & Legal

<div align="center">

**📜 Apache 2.0 License**

This project is licensed under the Apache 2.0 License - see the [LICENSE](LICENSE) file for details.

</div>

### 🙏 Acknowledgments

<table>
<tr>
<td width="33%">

**🔬 Research Team**
- IndexTTS2 research team
- Original model developers
- Academic contributors

</td>
<td width="33%">

**🛠️ Technical Community**
- ComfyUI framework team
- PyTorch community
- Open source contributors

</td>
<td width="33%">

**👥 User Community**
- Beta testers
- Documentation contributors
- Workflow creators

</td>
</tr>
</table>

---

<div align="center">

### 🚀 Ready to Create Amazing Voice Content?

**[⬆️ Back to Top](#-comfyui-indextts2-plugin)** • **[📦 Install Now](#-installation)** • **[🎯 Quick Start](#-quick-start)** • **[🤝 Join Community](#-community--contributing)**

---

**🎙️ IndexTTS2 ComfyUI Plugin** - *Bringing Professional Voice Synthesis to Everyone*

![Made with ❤️](https://img.shields.io/badge/Made%20with-❤️-red?style=for-the-badge)
![Open Source](https://img.shields.io/badge/Open%20Source-💚-green?style=for-the-badge)
![Community Driven](https://img.shields.io/badge/Community%20Driven-🤝-blue?style=for-the-badge)

</div>
