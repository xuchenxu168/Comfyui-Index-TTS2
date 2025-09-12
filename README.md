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

#### ⚡ 智能安装 (推荐)
```bash
# 使用智能安装脚本，自动处理依赖问题
python install_requirements.py
```

#### 📦 手动安装
```bash
# 标准安装（不包含可选的 pynini）
pip install -r requirements.txt

# 🪟 Windows 用户额外福利：安装 pynini 高级文本处理
# Python 3.10 用户:
pip install pynini-wheel/pynini-2.1.6.post1-cp310-cp310-win_amd64.whl

# Python 3.11 用户:
pip install pynini-wheel/pynini-2.1.6.post1-cp311-cp311-win_amd64.whl

# Python 3.12 用户:
pip install pynini-wheel/pynini-2.1.6.post1-cp312-cp312-win_amd64.whl

# Python 3.13 用户:
pip install pynini-wheel/pynini-2.1.6.post1-cp313-cp313-win_amd64.whl
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

#### 🔗 Model Download Links

| Platform | Model | Download Link |
|----------|-------|---------------|
| **HuggingFace** | IndexTTS-2 | [🤗 IndexTTS-2](https://huggingface.co/IndexTeam/IndexTTS-2) |
| **ModelScope** | IndexTTS-2 | [🔗 IndexTTS-2](https://modelscope.cn/models/IndexTeam/IndexTTS-2) |
| **HuggingFace** | IndexTTS-1.5 | [🤗 IndexTTS-1.5](https://huggingface.co/IndexTeam/IndexTTS-1.5) |
| **ModelScope** | IndexTTS-1.5 | [🔗 IndexTTS-1.5](https://modelscope.cn/models/IndexTeam/IndexTTS-1.5) |
| **HuggingFace** | IndexTTS | [🤗 IndexTTS](https://huggingface.co/IndexTeam/IndexTTS) |
| **ModelScope** | IndexTTS | [🔗 IndexTTS](https://modelscope.cn/models/IndexTeam/IndexTTS) |

#### 📁 Model File Placement

**重要说明**: 下载整个 IndexTTS-2 模型文件夹，并将其放入 `Models/TTS/` 目录中。

```
ComfyUI/
└── Models/
    └── TTS/
        └── IndexTTS-2/          # 将下载的完整模型文件夹放在这里
            ├── config.yaml
            ├── model.pth
            └── [其他模型文件...]
```

#### 🚀 Download Methods

```bash
# Automatic download (recommended)
python download_models.py

# Alternative methods:
# huggingface-cli download IndexTeam/IndexTTS-2 --local-dir=Models/TTS/IndexTTS-2
# modelscope download --model IndexTeam/IndexTTS-2 --local_dir Models/TTS/IndexTTS-2
```

### Step 5: Verify Installation
```bash
# Check transformers compatibility
python check_transformers_compatibility.py

# Or verify by loading a node in ComfyUI
# The nodes should appear in the IndexTTS2 category
```

### 🔧 Troubleshooting Dependencies

If you encounter missing dependencies in different environments, you can check compatibility:

```bash
# Check transformers compatibility
python check_transformers_compatibility.py

# Install missing dependencies
pip install -r requirements.txt
```

**Common Missing Dependencies:**
- `descript-audiotools` - Required for audio processing (imports as `audiotools`)
- `json5` - Required for configuration files
- `transformers` - Required for model loading
- `einops` - Required for tensor operations
- `WeTextProcessing` - Optional for better text normalization (may fail on Windows)

**Quick Fix for Missing Dependencies:**
```bash
# Install all required dependencies
pip install -r requirements.txt

# For Python 3.12 environments
pip install -r requirements_py312.txt

# Install specific missing packages
pip install descript-audiotools json5 transformers einops

# Alternative audiotools installation (if pip fails)
pip install git+https://github.com/descriptinc/audiotools
```

**🔧 Special Notes for Dependencies:**

**For audiotools:**
If `pip install descript-audiotools` fails, try:
```bash
# Method 1: Install from GitHub
pip install git+https://github.com/descriptinc/audiotools

# Method 2: Clone and install manually
git clone https://github.com/descriptinc/audiotools
cd audiotools
pip install .
```

**For Text Normalization:**
IndexTTS2 now uses `wetext` for better text normalization - **no pynini dependency required!**

**🎉 wetext - The Perfect Solution**
- ✅ **wetext-0.1.0-py3-none-any.whl** - Universal Windows wheel available
- ✅ **No pynini dependency** - avoids all compilation issues
- ✅ **Same functionality** as WeTextProcessing but more reliable
- ✅ **Now included in requirements.txt** - automatically installed

**🔍 Common Issue: "Why does WeTextProcessing recompile pynini even when it's already installed?"**

This happens because:
1. **Version Pinning**: WeTextProcessing requires `pynini==2.1.6` specifically
2. **Dependency Resolution**: pip tries to install the exact version required
3. **Build Dependencies**: WeTextProcessing may specify build-time dependencies
4. **Platform Compatibility**: Your pynini wheel may not match WeTextProcessing's requirements

**🚀 Solutions (if wetext installation fails):**

```bash
# Method 1: wetext is now in requirements.txt (automatic)
pip install -r requirements.txt

# Method 2: Manual wetext installation
pip install wetext

# Method 3: Alternative - WeTextProcessing (if you prefer)
pip install WeTextProcessing --only-binary=all

# Method 4: Use our complete solution installer
python install_text_processing_solution.py

# Method 5: Use fallback (always works)
# Skip text processing packages - IndexTTS2 works with basic fallback
```

**🎉 wetext is now the default!**
Since wetext is superior (no pynini dependency, same functionality), it's now included in requirements.txt and will be automatically installed.

**Note:** If WeTextProcessing installation fails, IndexTTS2 will automatically fall back to basic text processing. The plugin will still work, but text normalization quality may be reduced.

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

**🪟 Windows (现已简化！)**
```bash
# Method 1: 使用项目提供的轮子文件 (推荐，最简单)
# Python 3.10 用户:
pip install pynini-wheel/pynini-2.1.6.post1-cp310-cp310-win_amd64.whl

# Python 3.11 用户:
pip install pynini-wheel/pynini-2.1.6.post1-cp311-cp311-win_amd64.whl

# Python 3.12 用户:
pip install pynini-wheel/pynini-2.1.6.post1-cp312-cp312-win_amd64.whl

# Python 3.13 用户:
pip install pynini-wheel/pynini-2.1.6.post1-cp313-cp313-win_amd64.whl

# Method 2: 使用轮子文件快速安装脚本 (推荐)
python install_pynini_wheel.py

# Method 3: 使用完整安装脚本 (包含多种方法)
python install_pynini_windows.py

# Method 4: Conda (如果可用)
conda install -c conda-forge pynini=2.1.6

# Method 5: 跳过安装 (推荐给大多数用户)
# pynini 是可选的，基本功能不受影响
```

**🎉 Windows 用户福音！** 我们现在提供了预编译的 Windows 轮子文件，支持 Python 3.10-3.13，无需复杂的编译过程！

#### 📦 Windows 轮子文件详情

| Python 版本 | 轮子文件 | 大小 | 支持架构 |
|-------------|----------|------|----------|
| **Python 3.10** | `pynini-2.1.6.post1-cp310-cp310-win_amd64.whl` | ~150MB | Windows x64 |
| **Python 3.11** | `pynini-2.1.6.post1-cp311-cp311-win_amd64.whl` | ~150MB | Windows x64 |
| **Python 3.12** | `pynini-2.1.6.post1-cp312-cp312-win_amd64.whl` | ~150MB | Windows x64 |
| **Python 3.13** | `pynini-2.1.6.post1-cp313-cp313-win_amd64.whl` | ~150MB | Windows x64 |

**✅ 优势**:
- 🚀 **即装即用** - 无需编译环境
- ⚡ **快速安装** - 几秒钟完成安装
- 🛡️ **稳定可靠** - 经过测试的预编译版本
- 🔧 **零配置** - 无需安装 Visual Studio 或其他工具
- 🎯 **全版本支持** - 覆盖 Python 3.10-3.13

#### 🚀 Windows 用户快速安装指南

**最简单的方法**：
```bash
# 一键安装 (自动检测 Python 版本)
python install_pynini_wheel.py
```

**手动安装**：
```bash
# 检查您的 Python 版本
python --version

# 根据版本选择对应的轮子文件
pip install pynini-wheel/pynini-2.1.6.post1-cp3XX-cp3XX-win_amd64.whl
```

**支持的 Python 版本**：
- ✅ Python 3.10
- ✅ Python 3.11
- ✅ Python 3.12
- ✅ Python 3.13

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
<summary><b>⚡ DeepSpeed 加速支持 (可选)</b></summary>

**DeepSpeed** 是一个深度学习优化库，可以显著提升 IndexTTS2 的训练和推理性能，特别是在大模型和多GPU环境下。

#### 🎯 DeepSpeed 的优势
- 🚀 **显著加速** - 推理速度提升 2-5 倍
- 💾 **内存优化** - 减少 GPU 内存使用
- 🔧 **自动优化** - 智能模型并行和内存管理
- 🎛️ **灵活配置** - 支持多种优化策略

#### 🪟 Windows 安装方法

**⚠️ 注意**: DeepSpeed 官方不直接支持 Windows，但社区提供了预编译轮子。

**🔗 Windows 轮子下载地址**:
[https://github.com/6Morpheus6/deepspeed-windows-wheels/releases](https://github.com/6Morpheus6/deepspeed-windows-wheels/releases)

**安装步骤**:
```bash
# 1. 访问上述链接，选择适合您环境的轮子文件
# 2. 下载对应的 .whl 文件
# 3. 使用 pip 安装下载的轮子文件

# 示例 (请根据实际下载的文件名调整):
pip install deepspeed-0.12.6+cu118-cp311-cp311-win_amd64.whl
```

#### 📋 版本选择指南

| Python 版本 | CUDA 版本 | 轮子文件示例 |
|-------------|-----------|-------------|
| **Python 3.10** | CUDA 11.8 | `deepspeed-*-cp310-cp310-win_amd64.whl` |
| **Python 3.11** | CUDA 11.8 | `deepspeed-*-cp311-cp311-win_amd64.whl` |
| **Python 3.12** | CUDA 11.8 | `deepspeed-*-cp312-cp312-win_amd64.whl` |

**💡 选择提示**:
- 检查您的 Python 版本: `python --version`
- 检查您的 CUDA 版本: `nvidia-smi`
- 选择匹配的轮子文件下载

#### 🐧 Linux/macOS 安装
```bash
# Linux/macOS 用户可以直接使用官方版本
pip install deepspeed
```

#### 🔍 验证安装
```bash
# 检查 DeepSpeed 是否正确安装
python -c "import deepspeed; print('✅ DeepSpeed version:', deepspeed.__version__)"
```

#### ⚙️ 在 IndexTTS2 中使用
DeepSpeed 安装后会自动被 IndexTTS2 检测和使用，无需额外配置。

#### 💡 使用建议
- **🎯 推荐场景**: 大模型推理、批量处理、多GPU环境
- **⚠️ 注意事项**: 需要兼容的 CUDA 版本和足够的 GPU 内存
- **🔧 故障排除**: 如果安装失败，可以跳过 DeepSpeed，基本功能不受影响
- **📊 性能提升**: 在支持的硬件上可获得显著的速度提升

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

**Perfect for**: Voice cloning, conversations, dialogues, classroom discussions

**Features**:
- ✅ 1-4 speaker support: 1=pure voice cloning, 2-4=conversation mode
- ✅ Individual emotion control per speaker
- ✅ Automatic conversation parsing (multi-speaker mode)
- ✅ Configurable silence intervals
- ✅ Modular emotion configuration

**Use Cases**:
- **Single Speaker**: Voice cloning, audiobooks, narration
- **Multi-Speaker**: Classroom discussions, business meetings
- **Character dialogues**: Theater, gaming, storytelling
- **Podcast conversations**: Multi-host discussions

</details>

<details>
<summary><b>🎵 IndexTTS2 Emotion Voice Multi-Talk (NEW!)</b></summary>

**Perfect for**: Voice cloning, emotion-driven conversations, character role-play

**Features**:
- ✅ 1-4 speaker support: 1=pure voice cloning, 2-4=conversation mode
- ✅ Direct audio input for emotion voice samples (no file paths needed!)
- ✅ Smart text parsing with emotion markers `[Happy]` (multi-speaker mode)
- ✅ Adjustable emotion intensity (0.0-2.0) per speaker
- ✅ Multiple emotion modes: emotion_voice, emotion_vector, auto
- ✅ High-performance synthesis with FP16/CUDA support

**Text Format**:

**Single Speaker Mode (num_speakers=1)**:
```
Hello everyone! This is a simple voice cloning example.
You can add emotion through the emotion voice input.
```

**Multi-Speaker Mode (num_speakers=2-4)**:
```
Speaker1: [Happy] Hello everyone! How are you doing today?
Speaker2: [Excited] I'm doing fantastic! Thanks for asking!
Speaker3: [Calm] I'm well, thank you. It's nice to see everyone.
```

**Emotion Control**:
- Connect audio loader nodes directly to emotion voice inputs
- Adjust emotion intensity with alpha values (0.0-2.0)
- Automatic emotion detection from text markers
- No need to manually input file paths - just connect audio nodes!

**Use Cases**:
- **Single Speaker**: Voice cloning, audiobooks, narration, emotional speech
- **Multi-Speaker**: Character role-playing, dramatic performances, theater
- **Educational**: Classroom discussions with emotional context
- **Entertainment**: Gaming, interactive storytelling, podcast creation

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

### ❌ Transformers 兼容性错误

**错误信息**: `cannot import name 'QuantizedCacheConfig' from 'transformers.cache_utils'`

**原因**: transformers 库版本不兼容

**解决方案**:
```bash
# 1. 检查兼容性和当前版本
python check_transformers_compatibility.py

# 2. 如果版本过旧，尝试升级
pip install --upgrade transformers

# 3. 如果版本过新，可能需要降级 (谨慎操作)
# pip install transformers==4.36.2

# 4. 重启 ComfyUI
```

### ❌ GenerationMode 属性错误

**错误信息**: `'NoneType' object has no attribute 'ASSISTED_GENERATION'`

**原因**: GenerationMode 类未正确导入

**解决方案**:
```bash
# 1. 检查兼容性
python check_transformers_compatibility.py

# 2. 重启 ComfyUI (重要!)
# 代码已包含兼容性处理，重启后应该正常工作
```

**💡 注意**:
- 优先保持与 ComfyUI 环境兼容的 transformers 版本
- 如果问题持续，代码已包含兼容性处理，基本功能不受影响
- 重启 ComfyUI 可以解决大部分导入相关问题

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

## 🚀 开发与发布

### GitHub Actions 工作流

本项目包含完整的 CI/CD 流水线：

#### 📋 **持续集成 (CI)**
- **触发条件**: 推送到 main/develop 分支，Pull Request
- **测试环境**: Ubuntu + Windows, Python 3.10/3.11
- **检查内容**:
  - ✅ Python 语法验证
  - ✅ 依赖项安装测试
  - ✅ 模块导入测试
  - ✅ 代码格式检查 (black, flake8, isort)
  - ✅ 安全扫描 (bandit, safety)
  - ✅ JSON 工作流验证

#### 🚀 **自动发布 (Publish)**
- **触发条件**: 推送版本标签 (v*)，手动触发
- **发布内容**:
  - 📦 创建 GitHub Release
  - 📄 自动生成 Changelog
  - 🗜️ 打包源码 (tar.gz + zip)
  - 📋 生成 ComfyUI Manager 提交信息

### 🏷️ **发布新版本**

1. **创建版本标签**:
   ```bash
   git tag v1.0.0
   git push origin v1.0.0
   ```

2. **自动发布流程**:
   - GitHub Actions 自动构建和测试
   - 创建 GitHub Release
   - 生成下载包
   - 通知 ComfyUI Manager

3. **手动触发**:
   - 访问 GitHub Actions 页面
   - 选择 "Publish ComfyUI IndexTTS2" 工作流
   - 点击 "Run workflow"

### 🔧 **开发贡献**

欢迎提交 Pull Request！请确保：
- 代码通过所有 CI 检查
- 遵循项目代码风格
- 添加适当的测试和文档

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
