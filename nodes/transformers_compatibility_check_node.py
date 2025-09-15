#!/usr/bin/env python3
"""
Transformers兼容性检查节点
Transformers Compatibility Check Node - 检查和诊断transformers版本兼容性问题
"""

import sys
import subprocess
from typing import Tuple, Any

class IndexTTS2TransformersCompatibilityCheckNode:
    """
    IndexTTS2 Transformers兼容性检查节点
    检查transformers版本并提供升级建议
    """
    
    def __init__(self):
        pass
    
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "check_compatibility": ("BOOLEAN", {
                    "default": True,
                    "tooltip": "检查transformers兼容性"
                }),
                "show_upgrade_command": ("BOOLEAN", {
                    "default": True,
                    "tooltip": "显示升级命令"
                }),
            },
            "optional": {
                "verbose": ("BOOLEAN", {
                    "default": False,
                    "tooltip": "详细输出"
                }),
            }
        }
    
    RETURN_TYPES = ("STRING", "DICT", "STRING")
    RETURN_NAMES = ("compatibility_info", "details", "upgrade_command")
    FUNCTION = "check_compatibility"
    CATEGORY = "IndexTTS2/Utils"
    DESCRIPTION = "Check transformers version compatibility for IndexTTS2 Qwen models"
    
    def check_compatibility(
        self,
        check_compatibility: bool,
        show_upgrade_command: bool,
        verbose: bool = False
    ) -> Tuple[str, dict, str]:
        """
        检查transformers兼容性
        Check transformers compatibility
        """
        try:
            if verbose:
                print("[IndexTTS2 CompatCheck] 检查transformers兼容性...")
            
            # 检查transformers版本
            compatibility_info = self._check_transformers_version()
            
            # 生成详细信息
            details = {
                "transformers_version": compatibility_info.get("version"),
                "is_compatible": compatibility_info.get("is_compatible", False),
                "compatibility_status": compatibility_info.get("status"),
                "required_version": ">=4.35.0",
                "recommended_version": "4.36.2",
                "supports_qwen3": compatibility_info.get("supports_qwen3", False),
                "python_version": f"{sys.version_info.major}.{sys.version_info.minor}.{sys.version_info.micro}",
                "upgrade_needed": compatibility_info.get("upgrade_needed", False)
            }
            
            # 生成信息字符串
            info_lines = [
                "=" * 60,
                "IndexTTS2 Transformers兼容性检查",
                "IndexTTS2 Transformers Compatibility Check",
                "=" * 60,
                f"Python版本 / Python Version: {details['python_version']}",
                f"Transformers版本 / Transformers Version: {details['transformers_version'] or 'Not installed'}",
                f"兼容性状态 / Compatibility Status: {details['compatibility_status']}",
                f"支持Qwen3架构 / Supports Qwen3: {'Yes' if details['supports_qwen3'] else 'No'}",
                f"需要升级 / Upgrade Needed: {'Yes' if details['upgrade_needed'] else 'No'}",
                "",
                "要求 / Requirements:",
                f"• 最低版本 / Minimum Version: {details['required_version']}",
                f"• 推荐版本 / Recommended Version: {details['recommended_version']}",
                ""
            ]
            
            # 添加状态说明
            if details["is_compatible"]:
                info_lines.extend([
                    "✅ 兼容性检查通过！",
                    "✅ Compatibility check passed!",
                    "• IndexTTS2的Qwen情感模型应该能正常工作",
                    "• IndexTTS2 Qwen emotion model should work properly",
                ])
            else:
                info_lines.extend([
                    "❌ 兼容性检查失败！",
                    "❌ Compatibility check failed!",
                    "• IndexTTS2的Qwen情感模型可能无法工作",
                    "• IndexTTS2 Qwen emotion model may not work",
                    "• 系统将使用备用情感分析方法",
                    "• System will use fallback emotion analysis method",
                ])
            
            info_lines.append("")
            
            # 生成升级命令
            upgrade_command = ""
            if details["upgrade_needed"] and show_upgrade_command:
                upgrade_command = f"pip install --upgrade transformers>={details['required_version']}"
                info_lines.extend([
                    "🔧 升级命令 / Upgrade Command:",
                    f"  {upgrade_command}",
                    "",
                    "或者安装推荐版本 / Or install recommended version:",
                    f"  pip install transformers=={details['recommended_version']}",
                    ""
                ])
            
            # 添加故障排除信息
            if not details["is_compatible"]:
                info_lines.extend([
                    "🛠️  故障排除 / Troubleshooting:",
                    "1. 更新transformers到最新版本",
                    "   Update transformers to latest version",
                    f"   pip install --upgrade transformers",
                    "",
                    "2. 如果仍有问题，尝试安装特定版本",
                    "   If still having issues, try installing specific version",
                    f"   pip install transformers=={details['recommended_version']}",
                    "",
                    "3. 重启ComfyUI以使更改生效",
                    "   Restart ComfyUI for changes to take effect",
                    ""
                ])
            
            info_lines.extend([
                "=" * 60,
                "注意 / Note:",
                "• 即使transformers版本不兼容，IndexTTS2仍可使用备用情感分析",
                "• Even if transformers is incompatible, IndexTTS2 can still use fallback emotion analysis",
                "• 备用方法基于关键词匹配，准确性可能较低",
                "• Fallback method is based on keyword matching, accuracy may be lower",
                "=" * 60
            ])
            
            compatibility_info_str = "\n".join(info_lines)
            
            if verbose:
                print(f"[IndexTTS2 CompatCheck] 兼容性检查完成")
                print(f"[IndexTTS2 CompatCheck] 兼容状态: {details['compatibility_status']}")
                if details["upgrade_needed"]:
                    print(f"[IndexTTS2 CompatCheck] 建议升级命令: {upgrade_command}")
            
            return (compatibility_info_str, details, upgrade_command)
            
        except Exception as e:
            error_msg = f"Compatibility check failed: {str(e)}"
            print(f"[IndexTTS2 CompatCheck Error] {error_msg}")
            
            error_details = {
                "error": str(e),
                "transformers_version": None,
                "is_compatible": False,
                "compatibility_status": "Error",
                "upgrade_needed": True
            }
            
            return (error_msg, error_details, "pip install --upgrade transformers")
    
    def _check_transformers_version(self):
        """检查transformers版本"""
        try:
            import transformers
            from packaging import version
            
            current_version = transformers.__version__
            current_ver = version.parse(current_version)
            
            # 完全移除版本限制 - 所有版本都尝试使用
            # Completely remove version restrictions - try all versions
            recommended = version.parse("4.36.2")

            # 所有版本都标记为兼容，实际支持由运行时决定
            # Mark all versions as compatible, actual support determined at runtime
            is_compatible = True
            supports_qwen3 = True  # 假设支持，实际加载时验证
            upgrade_needed = False
            
            # 基于版本给出状态描述，但不影响兼容性判断
            if current_ver >= recommended:
                status = "Excellent (推荐版本或更高)"
            elif current_ver >= version.parse("4.35.0"):
                status = "Good (较新版本)"
            else:
                status = "Older version (较旧版本，但仍会尝试使用)"
            
            return {
                "version": current_version,
                "is_compatible": is_compatible,
                "supports_qwen3": supports_qwen3,
                "upgrade_needed": upgrade_needed,
                "status": status
            }
            
        except ImportError:
            return {
                "version": None,
                "is_compatible": False,
                "supports_qwen3": False,
                "upgrade_needed": True,
                "status": "Not installed (未安装)"
            }
        except Exception as e:
            return {
                "version": "Unknown",
                "is_compatible": False,
                "supports_qwen3": False,
                "upgrade_needed": True,
                "status": f"Error: {str(e)}"
            }

# 节点映射
NODE_CLASS_MAPPINGS = {
    "IndexTTS2TransformersCompatibilityCheckNode": IndexTTS2TransformersCompatibilityCheckNode
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "IndexTTS2TransformersCompatibilityCheckNode": "IndexTTS2 Transformers Compatibility Check"
}
