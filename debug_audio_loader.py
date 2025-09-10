#!/usr/bin/env python3
"""
调试音频加载节点
Debug audio loader node
"""

import sys
import os

# 添加当前目录到Python路径
current_dir = os.path.dirname(__file__)
if current_dir not in sys.path:
    sys.path.insert(0, current_dir)

def debug_audio_loader():
    """调试音频加载节点"""
    print("=" * 60)
    print("调试IndexTTS2音频加载节点")
    print("Debugging IndexTTS2 Audio Loader Node")
    print("=" * 60)
    
    try:
        # 导入必要的模块
        print("\n1. 导入模块...")
        from nodes.audio_loader_node import IndexTTS2LoadAudio
        import folder_paths
        
        # 创建节点实例
        print("\n2. 创建节点实例...")
        loader = IndexTTS2LoadAudio()
        
        # 获取输入类型
        print("\n3. 获取输入类型...")
        input_types = IndexTTS2LoadAudio.INPUT_TYPES()
        print(f"INPUT_TYPES: {input_types}")
        
        # 检查可用文件
        required_inputs = input_types.get('required', {})
        if 'audio' in required_inputs:
            audio_files = required_inputs['audio'][0]
            print(f"\n4. 可用音频文件数量: {len(audio_files)}")
            if audio_files:
                print(f"前5个文件: {audio_files[:5]}")
                
                # 尝试加载一个wav文件
                wav_files = [f for f in audio_files if f.endswith('.wav')]
                if wav_files:
                    test_file = wav_files[0]
                else:
                    test_file = audio_files[0]
                print(f"\n5. 尝试加载测试文件: {test_file}")
                
                try:
                    result = loader.load(test_file)
                    print(f"加载成功! 结果类型: {type(result)}")
                    if isinstance(result, tuple) and len(result) > 0:
                        audio_file = result[0]
                        if isinstance(audio_file, dict):
                            print(f"AUDIO_FILE对象键: {list(audio_file.keys())}")
                            print(f"文件名: {audio_file.get('filename', 'N/A')}")
                            print(f"路径: {audio_file.get('path', 'N/A')}")
                            print(f"采样率: {audio_file.get('sample_rate', 'N/A')}")
                            print(f"时长: {audio_file.get('duration', 'N/A')}")
                        else:
                            print(f"意外的结果类型: {type(audio_file)}")
                    else:
                        print(f"意外的结果格式: {result}")
                        
                except Exception as e:
                    print(f"加载失败: {e}")
                    import traceback
                    traceback.print_exc()
            else:
                print("没有找到音频文件!")
        else:
            print("没有找到audio输入!")
            
    except Exception as e:
        print(f"\n调试失败: {e}")
        import traceback
        traceback.print_exc()

def test_folder_paths():
    """测试folder_paths模块"""
    print("\n" + "=" * 60)
    print("测试folder_paths模块")
    print("Testing folder_paths module")
    print("=" * 60)
    
    try:
        import folder_paths
        
        # 测试基本方法
        print(f"\n1. get_input_directory: {folder_paths.get_input_directory()}")
        
        # 测试文件过滤方法
        input_dir = folder_paths.get_input_directory()
        all_files = os.listdir(input_dir)
        print(f"\n2. input目录中的文件总数: {len(all_files)}")
        
        # 测试filter_files_content_types方法
        try:
            filtered_files = folder_paths.filter_files_content_types(all_files, ["audio", "video"])
            print(f"3. 过滤后的音频/视频文件数量: {len(filtered_files)}")
            if filtered_files:
                print(f"前5个过滤后的文件: {filtered_files[:5]}")
        except Exception as e:
            print(f"3. filter_files_content_types失败: {e}")
            
            # 使用备用方法
            audio_extensions = {'.wav', '.mp3', '.flac', '.ogg', '.m4a', '.aac', '.wma', 
                              '.mp4', '.avi', '.mov', '.mkv', '.webm', '.opus'}
            manual_filtered = [f for f in all_files if os.path.splitext(f)[1].lower() in audio_extensions]
            print(f"备用方法过滤的文件数量: {len(manual_filtered)}")
            if manual_filtered:
                print(f"前5个手动过滤的文件: {manual_filtered[:5]}")
        
        # 测试get_annotated_filepath方法
        if all_files:
            test_file = all_files[0]
            try:
                annotated_path = folder_paths.get_annotated_filepath(test_file)
                print(f"\n4. get_annotated_filepath测试成功: {annotated_path}")
            except Exception as e:
                print(f"4. get_annotated_filepath失败: {e}")
                
    except Exception as e:
        print(f"\nfolder_paths测试失败: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    test_folder_paths()
    debug_audio_loader()
