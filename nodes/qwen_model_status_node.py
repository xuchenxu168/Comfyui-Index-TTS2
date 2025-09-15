"""
IndexTTS2 Qwen模型状态节点
显示当前使用的Qwen情感分析模型信息
"""

import os
import sys
import traceback
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


class IndexTTS2QwenModelStatusNode:
    """
    IndexTTS2 Qwen模型状态节点
    显示当前使用的Qwen情感分析模型的详细信息
    """
    
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {},
            "optional": {}
        }
    
    RETURN_TYPES = ("STRING", "STRING", "STRING")
    RETURN_NAMES = ("current_model", "model_status", "detailed_info")
    FUNCTION = "get_qwen_model_status"
    CATEGORY = "IndexTTS2"
    
    def get_qwen_model_status(self):
        """获取Qwen模型状态信息"""
        try:
            if QwenEmotion is None:
                return ("❌ QwenEmotion类导入失败", "❌ 导入失败", "QwenEmotion类导入失败")

            # 直接检查兼容性，不创建QwenEmotion实例
            compatible_models = self._get_compatible_qwen_models_direct()

            # 获取当前使用的模型信息（简洁版）
            current_model = self._get_current_model_simple(compatible_models)

            # 获取模型状态（简洁版）
            model_status = self._get_model_status_simple(compatible_models)

            # 获取详细信息
            detailed_info = self._get_detailed_info()

            return (current_model, model_status, detailed_info)

        except Exception as e:
            error_msg = f"获取Qwen模型状态时发生错误: {str(e)}"
            return ("❌ 错误", "❌ 状态获取失败", error_msg)

    def _get_current_model_simple(self, compatible_models) -> str:
        """获取当前模型的简洁信息"""
        try:
            import transformers
            current_version = transformers.__version__

            if compatible_models:
                best_model = compatible_models[0]  # 第一个是优先级最高的
                return f"🤖 {best_model['name']} ({best_model['size']}) | Transformers {current_version}"
            else:
                return f"🔤 关键词匹配备用方案 | Transformers {current_version}"

        except Exception as e:
            return f"❌ 模型信息获取失败: {str(e)[:30]}..."

    def _get_model_status_simple(self, compatible_models) -> str:
        """获取模型状态的简洁信息"""
        try:
            if compatible_models:
                return "✅ Qwen模型可用"
            else:
                return "⚠️  关键词匹配模式"

        except Exception as e:
            return f"❌ 状态检查失败"

    def _get_detailed_info(self) -> str:
        """获取详细信息"""
        try:
            status_info = []
            status_info.append("=" * 60)
            status_info.append("IndexTTS2 Qwen情感分析模型详细状态")
            status_info.append("IndexTTS2 Qwen Emotion Analysis Model Detailed Status")
            status_info.append("=" * 60)

            # 检查transformers版本
            try:
                import transformers
                from packaging import version

                current_version = transformers.__version__
                status_info.append(f"📦 Transformers版本: {current_version}")
                status_info.append(f"📦 Transformers Version: {current_version}")

                # 获取兼容的模型列表
                compatible_models = self._get_compatible_qwen_models_direct()
                status_info.append(f"🔍 兼容模型数量: {len(compatible_models)}")
                status_info.append(f"🔍 Compatible Models Count: {len(compatible_models)}")

                if compatible_models:
                    status_info.append("\n📋 兼容的Qwen模型列表:")
                    status_info.append("📋 Compatible Qwen Models List:")
                    for i, model in enumerate(compatible_models[:5], 1):  # 只显示前5个
                        status_info.append(f"  {i}. {model['name']} ({model['size']}) - {model['description']}")

                    if len(compatible_models) > 5:
                        status_info.append(f"  ... 还有 {len(compatible_models) - 5} 个模型")
                        status_info.append(f"  ... and {len(compatible_models) - 5} more models")

            except ImportError:
                status_info.append("❌ Transformers未安装")
                status_info.append("❌ Transformers not installed")
            except Exception as e:
                status_info.append(f"⚠️  版本检查失败: {e}")
                status_info.append(f"⚠️  Version check failed: {e}")

            # 检查当前模型状态
            status_info.append("\n🤖 当前模型状态:")
            status_info.append("🤖 Current Model Status:")

            if temp_qwen.is_available:
                if hasattr(temp_qwen, 'fallback_model_info'):
                    model_info = temp_qwen.fallback_model_info
                    status_info.append(f"✅ 使用备用模型: {model_info['name']}")
                    status_info.append(f"✅ Using fallback model: {model_info['name']}")
                    status_info.append(f"📊 模型大小: {model_info['size']}")
                    status_info.append(f"📊 Model size: {model_info['size']}")
                    status_info.append(f"📝 描述: {model_info['description']}")
                    status_info.append(f"📝 Description: {model_info['description']}")
                else:
                    status_info.append("✅ 使用原始Qwen模型")
                    status_info.append("✅ Using original Qwen model")
            else:
                status_info.append("⚠️  Qwen模型不可用，使用关键词匹配")
                status_info.append("⚠️  Qwen model unavailable, using keyword matching")

            # 功能状态
            status_info.append("\n🔧 功能状态:")
            status_info.append("🔧 Functionality Status:")

            if temp_qwen.is_available:
                status_info.append("✅ 高精度情感分析: 可用")
                status_info.append("✅ High-precision emotion analysis: Available")
            else:
                status_info.append("⚠️  高精度情感分析: 不可用")
                status_info.append("⚠️  High-precision emotion analysis: Unavailable")

            status_info.append("✅ 关键词匹配备用方案: 始终可用")
            status_info.append("✅ Keyword matching fallback: Always available")
            status_info.append("✅ IndexTTS2基本功能: 不受影响")
            status_info.append("✅ IndexTTS2 basic functionality: Unaffected")

            # 建议
            status_info.append("\n💡 建议:")
            status_info.append("💡 Recommendations:")

            if not temp_qwen.is_available:
                status_info.append("🔧 考虑更新transformers以获得更好的情感分析")
                status_info.append("🔧 Consider updating transformers for better emotion analysis")
                status_info.append("📝 命令: pip install --upgrade transformers")
                status_info.append("📝 Command: pip install --upgrade transformers")
            else:
                status_info.append("✅ 当前配置良好，无需额外操作")
                status_info.append("✅ Current configuration is good, no additional action needed")

            status_info.append("=" * 60)

            return "\n".join(status_info)

        except Exception as e:
            return f"获取详细信息时发生错误: {str(e)}"

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
    "IndexTTS2_QwenModelStatus": IndexTTS2QwenModelStatusNode
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "IndexTTS2_QwenModelStatus": "IndexTTS2 Qwen Model Status"
}
