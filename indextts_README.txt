# comfyui-index-tts2/indextts 目录结构说明
# Directory structure for comfyui-index-tts2/indextts

本插件需将 IndexTTS2 推理相关源码（如 indextts/infer_v2.py 及依赖）完整复制到如下目录：
You need to copy all IndexTTS2 inference code (e.g. indextts/infer_v2.py and dependencies) to:

comfyui-index-tts2/
├── indextts/
│   ├── __init__.py
│   ├── infer_v2.py
│   ├── ... (其它依赖文件 other dependency files)

建议直接从 https://github.com/index-tts/index-tts 下载 indextts 目录，复制到插件 indextts 文件夹。
建议保留原始结构，确保 infer_v2.py 可用。

依赖说明：
- 需包含 infer_v2.py 及其所需的所有 Python 文件和模块。
- 若有 C/C++/CUDA 扩展或特殊依赖，请一并复制。
- 可选：examples/ 目录用于测试音频样例。

如需自动化复制脚本，可进一步补充。
