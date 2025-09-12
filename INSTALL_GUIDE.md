# 🚀 IndexTTS2 Complete Installation Guide

## 📋 Quick Installation for Different Environments

### 🎯 For Your ComfyUI_311 Environment

```bash
# Navigate to your ComfyUI_311 environment
cd ComfyUI_311/ComfyUI/custom_nodes/comfyui-Index-TTS2

# Step 1: Install required dependencies
pip install -r requirements.txt

# Step 2: Install audiotools (if needed)
python install_audiotools.py

# Step 3: Check compatibility
python check_transformers_compatibility.py

# Step 4: Verify installation by loading ComfyUI
# IndexTTS2 nodes should appear in the node menu
```

## 🔧 Dependency Issues and Solutions

### ❌ Problem: `No module named 'audiotools'`
**Solution:**
```bash
# Method 1: Use our installer
python install_audiotools.py

# Method 2: Install correct package name
pip install descript-audiotools

# Method 3: Install from GitHub
pip install git+https://github.com/descriptinc/audiotools
```

### ❌ Problem: `No module named 'tn'`
**Solution:**
```bash
# Method 1: Try our installer (may fail on Windows)
python install_wetextprocessing.py

# Method 2: Manual installation (may fail)
pip install WeTextProcessing

# Method 3: Use fallback (recommended for Windows)
# IndexTTS2 will automatically use basic text processing
# No action needed - the plugin will work without WeTextProcessing
```

### ❌ Problem: `No module named 'json5'`
**Solution:**
```bash
pip install json5
```

## 🎮 Verifying Your Installation

### 1. Check Compatibility
```bash
python check_transformers_compatibility.py
```

### 2. Verify in ComfyUI
- Start ComfyUI
- Look for IndexTTS2 nodes in the node menu
- Try loading a basic workflow

### 3. Expected Behavior
```
✅ IndexTTS2 nodes appear in ComfyUI
✅ No import errors in console

⚠️  3 optional dependencies missing:
   - pynini (optional)
   - deepspeed (optional)
   - WeTextProcessing (optional - better text normalization)

✅ TextNormalizer is working with fallback mechanism
✅ IndexTTS2 should work even without WeTextProcessing
```

## 📝 Dependency Categories

### ✅ Required Dependencies (Must Install)
- `torch` - Deep learning framework
- `torchaudio` - Audio processing
- `librosa` - Audio analysis
- `soundfile` - Audio I/O
- `descript-audiotools` - Audio tools (imports as `audiotools`)
- `jieba` - Chinese text segmentation
- `cn2an` - Chinese number conversion
- `g2p-en` - English phoneme conversion
- `wetext` - Text normalization (no pynini dependency)
- `transformers` - Model loading
- `json5` - Configuration files
- `einops` - Tensor operations
- `scipy` - Scientific computing
- `omegaconf` - Configuration management
- `munch` - Dictionary utilities

### ⚠️ Optional Dependencies (Nice to Have)
- `WeTextProcessing` - Alternative text normalization (wetext is preferred)
- `pynini` - Text processing rules (required by WeTextProcessing)
- `deepspeed` - Model acceleration

## 🌍 Platform-Specific Notes

### 🪟 Windows
- ✅ `wetext` works perfectly (no compilation needed)
- `WeTextProcessing` may fail due to pynini compilation issues
- Fallback mechanism available if both fail (automatic)
- All other dependencies should install normally

### 🐧 Linux
- All dependencies should install without issues
- Both `wetext` and `WeTextProcessing` work well on Linux
- `wetext` is still recommended for consistency

### 🍎 macOS
- Most dependencies work well
- May need Xcode command line tools for some packages

## 🎯 Troubleshooting

### If Installation Fails
1. **Update pip**: `python -m pip install --upgrade pip`
2. **Use fresh environment**: Create new virtual environment
3. **Check Python version**: Ensure Python 3.8-3.12
4. **Install Visual Studio Build Tools** (Windows only)

### If Nodes Don't Load
1. **Restart ComfyUI** after installing dependencies
2. **Check ComfyUI logs** for specific error messages
3. **Verify installation** with test scripts

### If Audio Processing Fails
1. **Check audiotools**: `python -c "import audiotools; print('OK')"`
2. **Reinstall audiotools**: `python install_audiotools.py`
3. **Check audio file formats**: Use WAV files for testing

## 🎉 Success Indicators

You know the installation is successful when:
- ✅ `python check_transformers_compatibility.py` shows no errors
- ✅ IndexTTS2 nodes appear in ComfyUI
- ✅ You can load and run IndexTTS2 workflows
- ✅ No import errors in ComfyUI console

## 📞 Getting Help

If you still have issues:
1. Run `python check_transformers_compatibility.py` and share the output
2. Check ComfyUI console for error messages
3. Provide your Python version and operating system
4. Share any specific error messages you encounter

The IndexTTS2 plugin is designed to be robust and work even with missing optional dependencies!
