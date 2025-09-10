#!/usr/bin/env python3
"""
IndexTTS2 模型工具函数
Model utility functions for IndexTTS2
"""

import os

def get_indextts2_model_path():
    """
    获取IndexTTS2模型路径
    Get IndexTTS2 model path
    
    Returns:
        tuple: (model_dir, config_path)
    """
    # 获取ComfyUI根目录
    plugin_dir = os.path.dirname(os.path.dirname(__file__))
    comfyui_root = os.path.dirname(os.path.dirname(plugin_dir))
    
    # 模型路径：ComfyUI/models/TTS/IndexTTS-2
    model_dir = os.path.join(comfyui_root, "models", "TTS", "IndexTTS-2")
    config_path = os.path.join(model_dir, "config.yaml")
    
    return model_dir, config_path

def validate_model_path(model_dir, config_path):
    """
    验证模型路径和配置文件是否存在
    Validate model path and config file existence
    
    Args:
        model_dir (str): 模型目录路径
        config_path (str): 配置文件路径
        
    Returns:
        bool: 是否有效
        
    Raises:
        FileNotFoundError: 如果路径不存在
    """
    if not os.path.exists(model_dir):
        raise FileNotFoundError(f"Model directory not found: {model_dir}")
    
    if not os.path.exists(config_path):
        raise FileNotFoundError(f"Config file not found: {config_path}")
    
    # 检查关键模型文件
    required_files = [
        "gpt.pth",
        "s2mel.pth", 
        "bpe.model",
        "wav2vec2bert_stats.pt",
        "feat1.pt",
        "feat2.pt"
    ]
    
    missing_files = []
    for file in required_files:
        file_path = os.path.join(model_dir, file)
        if not os.path.exists(file_path):
            missing_files.append(file)
    
    if missing_files:
        print(f"⚠️  Missing model files: {missing_files}")
        print(f"⚠️  缺少模型文件: {missing_files}")
    
    return len(missing_files) == 0

def get_model_info(model_dir):
    """
    获取模型信息
    Get model information
    
    Args:
        model_dir (str): 模型目录路径
        
    Returns:
        dict: 模型信息
    """
    info = {
        "model_dir": model_dir,
        "exists": os.path.exists(model_dir),
        "files": []
    }
    
    if info["exists"]:
        try:
            files = os.listdir(model_dir)
            info["files"] = files
            info["file_count"] = len(files)
            
            # 检查模型文件大小
            model_files = ["gpt.pth", "s2mel.pth"]
            for file in model_files:
                file_path = os.path.join(model_dir, file)
                if os.path.exists(file_path):
                    size_mb = os.path.getsize(file_path) / (1024 * 1024)
                    info[f"{file}_size_mb"] = round(size_mb, 2)
                    
        except Exception as e:
            info["error"] = str(e)
    
    return info

if __name__ == "__main__":
    # 测试模型路径
    model_dir, config_path = get_indextts2_model_path()
    
    print("IndexTTS2 模型路径测试")
    print("=" * 50)
    print(f"模型目录: {model_dir}")
    print(f"配置文件: {config_path}")
    print(f"目录存在: {os.path.exists(model_dir)}")
    print(f"配置存在: {os.path.exists(config_path)}")
    
    if os.path.exists(model_dir):
        info = get_model_info(model_dir)
        print(f"文件数量: {info.get('file_count', 0)}")
        print("文件列表:")
        for file in info.get('files', []):
            print(f"  - {file}")
            
        # 验证路径
        try:
            is_valid = validate_model_path(model_dir, config_path)
            print(f"路径验证: {'✓ 通过' if is_valid else '❌ 失败'}")
        except Exception as e:
            print(f"路径验证: ❌ {e}")
    else:
        print("❌ 模型目录不存在")
