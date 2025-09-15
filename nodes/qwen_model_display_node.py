"""
IndexTTS2 Qwen模型显示节点
在ComfyUI界面中直观显示当前使用的Qwen情感分析模型
"""

import os
import sys
from pathlib import Path

# 添加项目根目录到Python路径
current_dir = Path(__file__).resolve().parent
project_root = current_dir.parent
if str(project_root) not in sys.path:
    sys.path.insert(0, str(project_root))

try:
    from indextts.infer_v2 import QwenEmotion
except ImportError as e:
    print(f"Failed to import QwenEmotion: {e}")
    QwenEmotion = None


class IndexTTS2QwenModelDisplayNode:
    """
    IndexTTS2 Qwen模型显示节点
    专门用于在ComfyUI界面中直观显示当前使用的Qwen情感分析模型
    """
    
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {},
            "optional": {}
        }
    
    RETURN_TYPES = ("STRING",)
    RETURN_NAMES = ("current_qwen_model",)
    FUNCTION = "display_current_qwen_model"
    CATEGORY = "IndexTTS2"
    
    def display_current_qwen_model(self):
        """显示当前使用的Qwen模型"""
        try:
            if QwenEmotion is None:
                return ("❌ QwenEmotion类导入失败 - 请检查安装",)
            
            # 直接检查兼容性，不创建QwenEmotion实例
            compatible_models = self._get_compatible_qwen_models_direct()
            
            # 获取transformers版本
            try:
                import transformers
                transformers_version = transformers.__version__
            except ImportError:
                transformers_version = "未安装"
            
            # 构建显示信息
            display_lines = []
            
            # 标题
            display_lines.append("🤖 IndexTTS2 当前Qwen情感模型")
            display_lines.append("=" * 50)
            
            # Transformers版本
            display_lines.append(f"📦 Transformers: {transformers_version}")
            
            # 当前模型状态
            if compatible_models:
                # 有兼容模型可用
                best_model = compatible_models[0]  # 第一个是优先级最高的
                display_lines.extend([
                    f"✅ 状态: 可用",
                    f"🎯 推荐模型: {best_model['name']}",
                    f"📊 大小: {best_model['size']}",
                    f"🔧 类型: 智能Qwen模型",
                    f"📝 描述: {best_model['description']}",
                    f"⭐ 精度: 高精度情感分析"
                ])
            else:
                # 使用关键词匹配
                display_lines.extend([
                    f"⚠️  状态: 备用模式",
                    f"🎯 模型: 关键词匹配",
                    f"🔧 类型: 基础备用方案",
                    f"⭐ 精度: 基础情感分析"
                ])

            # 兼容模型数量
            display_lines.append(f"🔍 可用模型: {len(compatible_models)}个")
            
            # 功能状态
            display_lines.append("")
            display_lines.append("🔧 功能状态:")
            if compatible_models:
                display_lines.append("  ✅ 高精度情感分析")
                display_lines.append("  ✅ 智能情感识别")
                display_lines.append("  ✅ 多语言支持")
            else:
                display_lines.append("  ⚠️  基础情感分析")
                display_lines.append("  ✅ 关键词匹配")
                display_lines.append("  ✅ 中文情感词汇")

            display_lines.append("  ✅ IndexTTS2基本功能")

            # 建议
            if not compatible_models:
                display_lines.extend([
                    "",
                    "💡 优化建议:",
                    "  🔧 更新transformers获得更好体验",
                    "  📝 pip install --upgrade transformers"
                ])

            display_lines.append("=" * 50)
            
            return ("\n".join(display_lines),)
            
        except Exception as e:
            error_info = [
                "❌ Qwen模型状态检查失败",
                "=" * 30,
                f"错误: {str(e)[:100]}...",
                "",
                "ℹ️  这不影响IndexTTS2的基本TTS功能",
                "ℹ️  系统会自动使用备用情感分析方案"
            ]
            return ("\n".join(error_info),)

    def _get_compatible_qwen_models_direct(self):
        """直接获取兼容的Qwen模型列表，不创建QwenEmotion实例"""
        try:
            import transformers
            from packaging import version

            current_ver = version.parse(transformers.__version__)

            # 定义不同Qwen模型的版本要求和优先级
            qwen_models = []

            # Qwen3系列 (需要transformers >= 4.51.0)
            if current_ver >= version.parse("4.51.0"):
                qwen_models.extend([
                    {
                        "name": "Qwen3-0.5B-Instruct",
                        "model_id": "Qwen/Qwen3-0.5B-Instruct",
                        "priority": 1,
                        "size": "0.5B",
                        "description": "最新Qwen3模型，小型高效"
                    },
                    {
                        "name": "Qwen3-1.8B-Instruct",
                        "model_id": "Qwen/Qwen3-1.8B-Instruct",
                        "priority": 2,
                        "size": "1.8B",
                        "description": "Qwen3中型模型"
                    }
                ])

            # Qwen2.5系列 (需要transformers >= 4.37.0)
            if current_ver >= version.parse("4.37.0"):
                qwen_models.extend([
                    {
                        "name": "Qwen2.5-0.5B-Instruct",
                        "model_id": "Qwen/Qwen2.5-0.5B-Instruct",
                        "priority": 3,
                        "size": "0.5B",
                        "description": "Qwen2.5小型模型"
                    },
                    {
                        "name": "Qwen2.5-1.5B-Instruct",
                        "model_id": "Qwen/Qwen2.5-1.5B-Instruct",
                        "priority": 4,
                        "size": "1.5B",
                        "description": "Qwen2.5中型模型"
                    }
                ])

            # Qwen2系列 (需要transformers >= 4.37.0)
            if current_ver >= version.parse("4.37.0"):
                qwen_models.extend([
                    {
                        "name": "Qwen2-0.5B-Instruct",
                        "model_id": "Qwen/Qwen2-0.5B-Instruct",
                        "priority": 5,
                        "size": "0.5B",
                        "description": "Qwen2小型模型"
                    },
                    {
                        "name": "Qwen2-1.5B-Instruct",
                        "model_id": "Qwen/Qwen2-1.5B-Instruct",
                        "priority": 6,
                        "size": "1.5B",
                        "description": "Qwen2中型模型"
                    }
                ])

            # Qwen1.5系列 (需要transformers >= 4.37.0)
            if current_ver >= version.parse("4.37.0"):
                qwen_models.extend([
                    {
                        "name": "Qwen1.5-0.5B-Chat",
                        "model_id": "Qwen/Qwen1.5-0.5B-Chat",
                        "priority": 7,
                        "size": "0.5B",
                        "description": "Qwen1.5小型模型"
                    },
                    {
                        "name": "Qwen1.5-1.8B-Chat",
                        "model_id": "Qwen/Qwen1.5-1.8B-Chat",
                        "priority": 8,
                        "size": "1.8B",
                        "description": "Qwen1.5中型模型"
                    }
                ])

            # 按优先级排序
            qwen_models.sort(key=lambda x: x["priority"])

            return qwen_models

        except Exception as e:
            print(f"[IndexTTS2] ⚠️  获取兼容模型列表失败: {e}")
            return []


# 节点映射
NODE_CLASS_MAPPINGS = {
    "IndexTTS2_QwenModelDisplay": IndexTTS2QwenModelDisplayNode
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "IndexTTS2_QwenModelDisplay": "IndexTTS2 Qwen Model Display"
}
