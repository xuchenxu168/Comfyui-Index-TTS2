/**
 * 音频加载器Web扩展 - 模仿LoadImage的实现
 * Audio Loader Web Extension - LoadImage-like Implementation
 * 
 * 提供与LoadImage完全相同的用户体验：
 * - 文件下拉选择器
 * - "选择文件上传"按钮
 * - 音频预览播放器
 */

import { app } from "../../scripts/app.js";
import { api } from "../../scripts/api.js";

// 注册音频加载器节点的Web扩展
app.registerExtension({
    name: "IndexTTS2.AudioLoader",
    
    async beforeRegisterNodeDef(nodeType, nodeData, app) {
        // 只处理我们的音频加载器节点
        if (nodeData.name === "IndexTTS2LoadAudio") {
            
            // 重写节点的onNodeCreated方法
            const onNodeCreated = nodeType.prototype.onNodeCreated;
            nodeType.prototype.onNodeCreated = function() {
                const result = onNodeCreated?.apply(this, arguments);
                
                // 添加音频上传功能
                this.addAudioUploadWidget();
                
                return result;
            };
            
            // 添加音频上传小部件方法
            nodeType.prototype.addAudioUploadWidget = function() {
                const audioWidget = this.widgets.find(w => w.name === "audio");
                if (!audioWidget) return;
                
                // 创建文件上传按钮
                const uploadButton = document.createElement("button");
                uploadButton.textContent = "选择文件上传";
                uploadButton.style.cssText = `
                    width: 100%;
                    padding: 8px;
                    margin-top: 5px;
                    background: #444;
                    color: white;
                    border: 1px solid #666;
                    border-radius: 4px;
                    cursor: pointer;
                    font-size: 12px;
                `;
                
                // 创建隐藏的文件输入
                const fileInput = document.createElement("input");
                fileInput.type = "file";
                fileInput.accept = ".wav,.mp3,.flac,.ogg,.m4a,.aac";
                fileInput.style.display = "none";
                
                // 文件选择处理
                fileInput.addEventListener("change", async (e) => {
                    const file = e.target.files[0];
                    if (!file) return;
                    
                    try {
                        // 显示上传进度
                        uploadButton.textContent = "上传中...";
                        uploadButton.disabled = true;
                        
                        // 上传文件到ComfyUI
                        const formData = new FormData();
                        formData.append("image", file);  // ComfyUI使用"image"字段名
                        formData.append("type", "input");
                        formData.append("subfolder", "audio");
                        
                        const response = await api.fetchApi("/upload/image", {
                            method: "POST",
                            body: formData
                        });
                        
                        if (response.ok) {
                            const result = await response.json();
                            
                            // 更新音频选择器
                            audioWidget.value = result.name;
                            
                            // 刷新音频文件列表
                            await this.refreshAudioList();
                            
                            // 创建音频预览
                            this.createAudioPreview(result.name);
                            
                            uploadButton.textContent = "选择文件上传";
                            uploadButton.disabled = false;
                            
                        } else {
                            throw new Error("Upload failed");
                        }
                        
                    } catch (error) {
                        console.error("Audio upload error:", error);
                        uploadButton.textContent = "上传失败，请重试";
                        uploadButton.disabled = false;
                        
                        setTimeout(() => {
                            uploadButton.textContent = "选择文件上传";
                        }, 2000);
                    }
                });
                
                // 按钮点击处理
                uploadButton.addEventListener("click", () => {
                    fileInput.click();
                });
                
                // 将按钮添加到节点
                if (this.audioUploadButton) {
                    this.audioUploadButton.remove();
                }
                this.audioUploadButton = uploadButton;
                
                // 添加到DOM
                const nodeElement = this.graph?.canvas?.canvas?.parentElement;
                if (nodeElement) {
                    nodeElement.appendChild(uploadButton);
                }
            };
            
            // 刷新音频文件列表方法
            nodeType.prototype.refreshAudioList = async function() {
                try {
                    // 获取最新的音频文件列表
                    const response = await api.fetchApi("/object_info");
                    const objectInfo = await response.json();
                    
                    if (objectInfo.IndexTTS2LoadAudio?.input?.required?.audio?.[0]) {
                        const audioFiles = objectInfo.IndexTTS2LoadAudio.input.required.audio[0];
                        
                        // 更新音频选择器的选项
                        const audioWidget = this.widgets.find(w => w.name === "audio");
                        if (audioWidget) {
                            audioWidget.options.values = audioFiles;
                        }
                    }
                } catch (error) {
                    console.error("Failed to refresh audio list:", error);
                }
            };
            
            // 创建音频预览方法
            nodeType.prototype.createAudioPreview = function(filename) {
                // 移除现有预览
                if (this.audioPreview) {
                    this.audioPreview.remove();
                }
                
                // 创建音频预览元素
                const audioPreview = document.createElement("audio");
                audioPreview.controls = true;
                audioPreview.style.cssText = `
                    width: 100%;
                    margin-top: 5px;
                    background: #333;
                `;
                
                // 设置音频源
                const audioUrl = `/view?filename=${encodeURIComponent(filename)}&type=input&subfolder=audio`;
                audioPreview.src = audioUrl;
                
                this.audioPreview = audioPreview;
                
                // 添加到DOM
                const nodeElement = this.graph?.canvas?.canvas?.parentElement;
                if (nodeElement) {
                    nodeElement.appendChild(audioPreview);
                }
            };
            
            // 监听音频选择变化
            const onWidgetChange = nodeType.prototype.onWidgetChanged;
            nodeType.prototype.onWidgetChanged = function(widget, value) {
                const result = onWidgetChange?.apply(this, arguments);
                
                if (widget.name === "audio" && value) {
                    // 当音频选择改变时，创建预览
                    this.createAudioPreview(value);
                }
                
                return result;
            };
        }
    }
});

// 添加CSS样式
const style = document.createElement("style");
style.textContent = `
    .comfy-audio-loader {
        background: #2a2a2a;
        border: 1px solid #444;
        border-radius: 4px;
        padding: 10px;
        margin: 5px 0;
    }
    
    .comfy-audio-loader audio {
        width: 100%;
        height: 32px;
        background: #333;
    }
    
    .comfy-audio-loader button {
        transition: background-color 0.2s;
    }
    
    .comfy-audio-loader button:hover {
        background: #555 !important;
    }
    
    .comfy-audio-loader button:disabled {
        background: #666 !important;
        cursor: not-allowed;
    }
`;
document.head.appendChild(style);
