# indextts 源码自动复制脚本
# Auto copy script for indextts source code
# 用法 Usage: python copy_indextts.py <index-tts项目路径>
#
# 注意：index-tts 目录已被清理，此脚本用于从外部 IndexTTS 项目更新源码
# Note: index-tts directory has been cleaned up, this script is for updating source from external IndexTTS project

import os
import shutil
import sys

# 获取源 index-tts 路径 Get source index-tts repo path
if len(sys.argv) < 2:
    print("用法: python copy_indextts.py <index-tts项目路径>\nUsage: python copy_indextts.py <path_to_index-tts_repo>")
    print("示例 Example: python copy_indextts.py /path/to/IndexTTS")
    sys.exit(1)

src_repo = sys.argv[1]
src_indextts = os.path.join(src_repo, "indextts")  # 源项目下的 indextts 目录
if not os.path.exists(src_indextts):
    print(f"未找到 indextts 目录: {src_indextts}\nindextts directory not found: {src_indextts}")
    print("请确保指向正确的 IndexTTS 项目根目录\nPlease ensure you're pointing to the correct IndexTTS project root directory")
    sys.exit(1)

# 目标 indextts 路径 Target indextts path
# 插件目录下 indextts 文件夹
plugin_dir = os.path.dirname(__file__)
dst_indextts = os.path.join(plugin_dir, "indextts")

if os.path.exists(dst_indextts):
    print(f"目标 indextts 已存在，将覆盖: {dst_indextts}\nTarget indextts exists, will overwrite: {dst_indextts}")
    shutil.rmtree(dst_indextts)

print(f"复制 indextts 源码到插件...\nCopying indextts source to plugin...")
shutil.copytree(src_indextts, dst_indextts)
print("复制完成 Copy done.")
print("请重启 ComfyUI 以使更改生效\nPlease restart ComfyUI for changes to take effect")
