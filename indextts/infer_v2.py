import os
from subprocess import CalledProcessError

os.environ['HF_HUB_CACHE'] = './checkpoints/hf_cache'
import time
import librosa
import torch
import torchaudio
from torch.nn.utils.rnn import pad_sequence

import warnings

warnings.filterwarnings("ignore", category=FutureWarning)
warnings.filterwarnings("ignore", category=UserWarning)

from omegaconf import OmegaConf

from indextts.gpt.model_v2 import UnifiedVoice
from indextts.utils.maskgct_utils import build_semantic_model, build_semantic_codec
from indextts.utils.checkpoint import load_checkpoint
from indextts.utils.front import TextNormalizer, TextTokenizer

from indextts.s2mel.modules.commons import load_checkpoint2, MyModel
from indextts.s2mel.modules.bigvgan import bigvgan
from indextts.s2mel.modules.campplus.DTDNN import CAMPPlus
from indextts.s2mel.modules.audio import mel_spectrogram

# 使用简化兼容层导入 transformers 组件
from indextts.compat.simple_imports import AutoTokenizer, SeamlessM4TFeatureExtractor
from modelscope import AutoModelForCausalLM
from huggingface_hub import hf_hub_download
import safetensors
import random
import torch.nn.functional as F

class IndexTTS2:
    def __init__(
            self, cfg_path="checkpoints/config.yaml", model_dir="checkpoints", is_fp16=False, device=None,
            use_cuda_kernel=None,
    ):
        """
        Args:
            cfg_path (str): path to the config file.
            model_dir (str): path to the model directory.
            is_fp16 (bool): whether to use fp16.
            device (str): device to use (e.g., 'cuda:0', 'cpu'). If None, it will be set automatically based on the availability of CUDA or MPS.
            use_cuda_kernel (None | bool): whether to use BigVGan custom fused activation CUDA kernel, only for CUDA device.
        """
        if device is not None:
            self.device = torch.device(device)
            self.is_fp16 = False if device == "cpu" else is_fp16
            self.use_cuda_kernel = use_cuda_kernel is not None and use_cuda_kernel and device.startswith("cuda")
        elif torch.cuda.is_available():
            self.device = torch.device("cuda:0")
            self.is_fp16 = is_fp16
            self.use_cuda_kernel = use_cuda_kernel is None or use_cuda_kernel
        elif hasattr(torch, "mps") and torch.backends.mps.is_available():
            self.device = torch.device("mps")
            self.is_fp16 = False  # Use float16 on MPS is overhead than float32
            self.use_cuda_kernel = False
        else:
            self.device = torch.device("cpu")
            self.is_fp16 = False
            self.use_cuda_kernel = False
            print(">> Be patient, it may take a while to run in CPU mode.")

        self.cfg = OmegaConf.load(cfg_path)
        self.model_dir = model_dir
        self.dtype = torch.float16 if self.is_fp16 else None
        self.stop_mel_token = self.cfg.gpt.stop_mel_token

        # ========== 系统性属性初始化 - 避免所有AttributeError ==========
        # 进度引用显示（可选）
        self.gr_progress = None

        # 缓存参考音频
        self.cache_spk_cond = None
        self.cache_s2mel_style = None
        self.cache_s2mel_prompt = None
        self.cache_spk_audio_prompt = None
        self.cache_emo_cond = None
        self.cache_emo_audio_prompt = None
        self.cache_mel = None

        # 模型相关属性
        self.semantic_model = None
        self.semantic_codec = None
        self.gpt = None
        self.s2mel = None
        self.bigvgan = None
        self.campplus_model = None
        self.extract_features = None
        self.normalizer = None
        self.tokenizer = None

        # 统计和矩阵属性
        self.semantic_mean = None
        self.semantic_std = None
        self.emo_matrix = None
        self.spk_matrix = None
        self.emo_num = None

        # mel_fn函数 - 使用正确的导入路径
        try:
            from indextts.s2mel.modules.audio import mel_spectrogram
            mel_fn_args = {
                "n_fft": self.cfg.s2mel['preprocess_params']['spect_params']['n_fft'],
                "win_size": self.cfg.s2mel['preprocess_params']['spect_params']['win_length'],
                "hop_size": self.cfg.s2mel['preprocess_params']['spect_params']['hop_length'],
                "num_mels": self.cfg.s2mel['preprocess_params']['spect_params']['n_mels'],
                "sampling_rate": self.cfg.s2mel["preprocess_params"]["sr"],
                "fmin": self.cfg.s2mel['preprocess_params']['spect_params'].get('fmin', 0),
                "fmax": None if self.cfg.s2mel['preprocess_params']['spect_params'].get('fmax', "None") == "None" else 8000,
                "center": False
            }
            self.mel_fn = lambda x: mel_spectrogram(x, **mel_fn_args)
        except ImportError as e:
            print(f"[WARNING] mel_spectrogram导入失败: {e}")
            print("[WARNING] 将在后续初始化中重试")
            self.mel_fn = None

        # 模型版本
        self.model_version = self.cfg.version if hasattr(self.cfg, "version") else None

        print("[IndexTTS2] ✓ 所有关键属性已提前初始化")
        # ========== 属性初始化完成 ==========

        # 检查qwen_emo模型路径是否存在
        qwen_emo_path = os.path.join(self.model_dir, self.cfg.qwen_emo_path)
        if os.path.exists(qwen_emo_path):
            self.qwen_emo = QwenEmotion(qwen_emo_path)
        else:
            print(f"⚠️  Qwen emotion model not found at: {qwen_emo_path}")
            print("⚠️  Emotion analysis will be disabled")
            self.qwen_emo = None

        self.gpt = UnifiedVoice(**self.cfg.gpt)
        self.gpt_path = os.path.join(self.model_dir, self.cfg.gpt_checkpoint)
        load_checkpoint(self.gpt, self.gpt_path)
        self.gpt = self.gpt.to(self.device)
        if self.is_fp16:
            self.gpt.eval().half()
        else:
            self.gpt.eval()
        print(">> GPT weights restored from:", self.gpt_path)
        # 使用兼容性模块检查DeepSpeed可用性
        try:
            from indextts.compat.deepspeed_compat import DEEPSPEED_AVAILABLE, check_deepspeed_availability
            use_deepspeed, _ = check_deepspeed_availability()
        except ImportError:
            # 如果兼容性模块不可用，回退到原始检查
            use_deepspeed = False
            try:
                import deepspeed
                if hasattr(deepspeed, 'init_inference'):
                    use_deepspeed = True
                    print(">> DeepSpeed可用，启用加速推理")
                else:
                    print(">> DeepSpeed模块不完整，使用标准PyTorch推理")
            except (ImportError, OSError, CalledProcessError, FileNotFoundError, AttributeError, ModuleNotFoundError) as e:
                use_deepspeed = False
                print(f">> DeepSpeed不可用，使用标准PyTorch推理: {e}")
                if "deepspeed.utils.torch" in str(e):
                    print(">> 检测到DeepSpeed版本兼容性问题，建议更新DeepSpeed或使用标准推理模式")

        if self.is_fp16:
            self.gpt.post_init_gpt2_config(use_deepspeed=use_deepspeed, kv_cache=True, half=True)
        else:
            self.gpt.post_init_gpt2_config(use_deepspeed=use_deepspeed, kv_cache=True, half=False)

        if self.use_cuda_kernel:
            # preload the CUDA kernel for BigVGAN
            try:
                from indextts.BigVGAN.alias_free_activation.cuda import load

                anti_alias_activation_cuda = load.load()
                print(">> Preload custom CUDA kernel for BigVGAN", anti_alias_activation_cuda)
            except:
                print(">> Failed to load custom CUDA kernel for BigVGAN. Falling back to torch.")
                self.use_cuda_kernel = False

        # 检查本地是否已有w2v-bert模型文件
        from indextts.utils.model_cache_manager import get_indextts2_cache_dir
        cache_dir = get_indextts2_cache_dir()

        # 初始化local_w2v_path变量
        local_w2v_path = None

        # 检查所有可能的HuggingFace缓存格式
        hf_cache_paths = [
            # 标准external_models缓存
            cache_dir / "w2v_bert" / "models--facebook--w2v-bert-2.0",
            cache_dir / "w2v_bert" / "facebook_w2v-bert-2.0",
            # HuggingFace Hub缓存
            cache_dir / "huggingface" / "hub" / "models--facebook--w2v-bert-2.0",
            cache_dir / "huggingface" / "transformers" / "models--facebook--w2v-bert-2.0",
            # 其他可能的格式
            cache_dir / "models--facebook--w2v-bert-2.0",
            cache_dir.parent / "w2v_bert" / "models--facebook--w2v-bert-2.0",
        ]

        # 查找HuggingFace缓存中的snapshots目录
        for hf_path in hf_cache_paths:
            if hf_path.exists():
                snapshots_dir = hf_path / "snapshots"
                if snapshots_dir.exists():
                    for snapshot in snapshots_dir.iterdir():
                        if snapshot.is_dir():
                            config_file = snapshot / "config.json"
                            model_file = snapshot / "model.safetensors"
                            preprocessor_file = snapshot / "preprocessor_config.json"
                            if config_file.exists() and model_file.exists():
                                local_w2v_path = snapshot
                                print(f"[IndexTTS2] 发现本地w2v-bert模型 (HuggingFace缓存): {local_w2v_path}")
                                break
                    if local_w2v_path:
                        break

        # 如果HuggingFace缓存中没有找到，检查直接路径
        if not local_w2v_path:
            direct_paths = [
                cache_dir / "w2v_bert",  # 标准缓存路径
                cache_dir,  # 直接在external_models目录
                cache_dir.parent / "w2v_bert",  # 上一级目录的w2v_bert文件夹
            ]
            for path in direct_paths:
                config_file = path / "config.json"
                model_file = path / "model.safetensors"
                if config_file.exists() and model_file.exists():
                    local_w2v_path = path
                    print(f"[IndexTTS2] 发现本地w2v-bert模型 (直接路径): {local_w2v_path}")
                    break

        # 加载SeamlessM4TFeatureExtractor
        if local_w2v_path:
            print(f"[IndexTTS2] 使用本地w2v-bert模型: {local_w2v_path}")
            self.extract_features = SeamlessM4TFeatureExtractor.from_pretrained(
                str(local_w2v_path),
                local_files_only=True
            )
        else:
            print(f"[IndexTTS2] 本地未找到w2v-bert模型，尝试从远程下载...")
            from indextts.utils.model_cache_manager import get_hf_download_kwargs
            w2v_kwargs = get_hf_download_kwargs("facebook/w2v-bert-2.0")
            self.extract_features = SeamlessM4TFeatureExtractor.from_pretrained(
                "facebook/w2v-bert-2.0", **w2v_kwargs
            )
        self.semantic_model, self.semantic_mean, self.semantic_std = build_semantic_model(
            os.path.join(self.model_dir, self.cfg.w2v_stat))
        self.semantic_model = self.semantic_model.to(self.device)
        self.semantic_model.eval()
        self.semantic_mean = self.semantic_mean.to(self.device)
        self.semantic_std = self.semantic_std.to(self.device)

        semantic_codec = build_semantic_codec(self.cfg.semantic_codec)

        # 检查本地是否已有MaskGCT语义编解码器
        local_maskgct_path = None

        # 检查所有可能的HuggingFace缓存格式
        maskgct_cache_paths = [
            # 标准external_models缓存
            cache_dir / "maskgct" / "models--amphion--MaskGCT",
            cache_dir / "maskgct" / "amphion_MaskGCT",
            # HuggingFace Hub缓存
            cache_dir / "huggingface" / "hub" / "models--amphion--MaskGCT",
            cache_dir / "huggingface" / "transformers" / "models--amphion--MaskGCT",
            # 其他可能的格式
            cache_dir / "models--amphion--MaskGCT",
            cache_dir.parent / "maskgct" / "models--amphion--MaskGCT",
        ]

        # 查找HuggingFace缓存中的snapshots目录
        for hf_path in maskgct_cache_paths:
            if hf_path.exists():
                snapshots_dir = hf_path / "snapshots"
                if snapshots_dir.exists():
                    for snapshot in snapshots_dir.iterdir():
                        if snapshot.is_dir():
                            semantic_codec_file = snapshot / "semantic_codec" / "model.safetensors"
                            if semantic_codec_file.exists():
                                local_maskgct_path = semantic_codec_file
                                print(f"[IndexTTS2] 发现本地MaskGCT语义编解码器 (HuggingFace缓存): {local_maskgct_path}")
                                break
                    if local_maskgct_path:
                        break

        # 加载MaskGCT语义编解码器
        if local_maskgct_path:
            print(f"[IndexTTS2] 使用本地MaskGCT语义编解码器: {local_maskgct_path}")
            semantic_code_ckpt = str(local_maskgct_path)
        else:
            print(f"[IndexTTS2] 本地未找到MaskGCT语义编解码器，尝试从远程下载...")
            maskgct_kwargs = get_hf_download_kwargs("amphion/MaskGCT")
            semantic_code_ckpt = hf_hub_download(
                "amphion/MaskGCT",
                filename="semantic_codec/model.safetensors",
                **maskgct_kwargs
            )
        safetensors.torch.load_model(semantic_codec, semantic_code_ckpt)
        self.semantic_codec = semantic_codec.to(self.device)
        self.semantic_codec.eval()
        print('>> semantic_codec weights restored from: {}'.format(semantic_code_ckpt))

        s2mel_path = os.path.join(self.model_dir, self.cfg.s2mel_checkpoint)
        s2mel = MyModel(self.cfg.s2mel, use_gpt_latent=True)
        s2mel, _, _, _ = load_checkpoint2(
            s2mel,
            None,
            s2mel_path,
            load_only_params=True,
            ignore_modules=[],
            is_distributed=False,
        )
        self.s2mel = s2mel.to(self.device)
        self.s2mel.models['cfm'].estimator.setup_caches(max_batch_size=1, max_seq_length=8192)
        self.s2mel.eval()
        print(">> s2mel weights restored from:", s2mel_path)

        # 检查本地是否已有CAMPPlus模型
        local_campplus_path = None

        # 检查所有可能的HuggingFace缓存格式
        campplus_cache_paths = [
            # 标准external_models缓存
            cache_dir / "campplus" / "models--funasr--campplus",
            cache_dir / "campplus" / "funasr_campplus",
            # HuggingFace Hub缓存
            cache_dir / "huggingface" / "hub" / "models--funasr--campplus",
            cache_dir / "huggingface" / "transformers" / "models--funasr--campplus",
            # 其他可能的格式
            cache_dir / "models--funasr--campplus",
            cache_dir.parent / "campplus" / "models--funasr--campplus",
        ]

        # 查找HuggingFace缓存中的snapshots目录
        for hf_path in campplus_cache_paths:
            if hf_path.exists():
                snapshots_dir = hf_path / "snapshots"
                if snapshots_dir.exists():
                    for snapshot in snapshots_dir.iterdir():
                        if snapshot.is_dir():
                            campplus_file = snapshot / "campplus_cn_common.bin"
                            if campplus_file.exists():
                                local_campplus_path = campplus_file
                                print(f"[IndexTTS2] 发现本地CAMPPlus模型 (HuggingFace缓存): {local_campplus_path}")
                                break
                    if local_campplus_path:
                        break

        # 加载CAMPPlus模型
        if local_campplus_path:
            print(f"[IndexTTS2] 使用本地CAMPPlus模型: {local_campplus_path}")
            campplus_ckpt_path = str(local_campplus_path)
        else:
            print(f"[IndexTTS2] 本地未找到CAMPPlus模型，尝试从远程下载...")
            campplus_kwargs = get_hf_download_kwargs("funasr/campplus")
            campplus_ckpt_path = hf_hub_download(
                "funasr/campplus",
                filename="campplus_cn_common.bin",
                **campplus_kwargs
            )
        campplus_model = CAMPPlus(feat_dim=80, embedding_size=192)
        campplus_model.load_state_dict(torch.load(campplus_ckpt_path, map_location="cpu"))
        self.campplus_model = campplus_model.to(self.device)
        self.campplus_model.eval()
        print(">> campplus_model weights restored from:", campplus_ckpt_path)

        bigvgan_name = self.cfg.vocoder.name
        # 下载BigVGAN到ComfyUI模型目录
        from indextts.utils.model_cache_manager import get_bigvgan_download_kwargs
        bigvgan_kwargs = get_bigvgan_download_kwargs(bigvgan_name)
        
        # 检查本地是否已有BigVGAN模型文件
        from indextts.utils.model_cache_manager import get_indextts2_cache_dir
        cache_dir = get_indextts2_cache_dir()

        # 初始化local_bigvgan_path变量
        local_bigvgan_path = None

        # 检查多个可能的本地路径
        local_bigvgan_paths = [
            cache_dir / "bigvgan",  # 标准缓存路径
            cache_dir,  # 直接在external_models目录
            cache_dir.parent / "bigvgan",  # 上一级目录的bigvgan文件夹
        ]
        
        # 检查所有可能的HuggingFace缓存格式
        hf_cache_paths = [
            # 标准external_models缓存
            cache_dir / "bigvgan" / "models--nvidia--bigvgan_v2_22khz_80band_256x",
            cache_dir / "bigvgan" / "nvidia_bigvgan_v2_22khz_80band_256x",
            # HuggingFace Hub缓存
            cache_dir / "huggingface" / "hub" / "models--nvidia--bigvgan_v2_22khz_80band_256x",
            cache_dir / "huggingface" / "transformers" / "models--nvidia--bigvgan_v2_22khz_80band_256x",
            # 其他可能的格式
            cache_dir / "models--nvidia--bigvgan_v2_22khz_80band_256x",
            cache_dir.parent / "bigvgan" / "models--nvidia--bigvgan_v2_22khz_80band_256x",
        ]
        
        # 查找HuggingFace缓存中的snapshots目录
        for hf_path in hf_cache_paths:
            if hf_path.exists():
                snapshots_dir = hf_path / "snapshots"
                if snapshots_dir.exists():
                    for snapshot in snapshots_dir.iterdir():
                        if snapshot.is_dir():
                            config_file = snapshot / "config.json"
                            model_file = snapshot / "bigvgan_generator.pt"
                            if config_file.exists() and model_file.exists():
                                local_bigvgan_path = snapshot
                                print(f"[IndexTTS2] 发现本地BigVGAN模型 (HuggingFace缓存): {local_bigvgan_path}")
                                break
                    if local_bigvgan_path:
                        break
        
        # 如果HuggingFace缓存中没有找到，检查直接路径
        if not local_bigvgan_path:
            for path in local_bigvgan_paths:
                config_file = path / "config.json"
                model_file = path / "bigvgan_generator.pt"
                if config_file.exists() and model_file.exists():
                    local_bigvgan_path = path
                    print(f"[IndexTTS2] 发现本地BigVGAN模型 (直接路径): {local_bigvgan_path}")
                    break
        
        # 添加超时和错误处理的BigVGAN加载
        print(f"[IndexTTS2] 开始加载BigVGAN模型: {bigvgan_name}")
        print(f"[IndexTTS2] 缓存目录: {bigvgan_kwargs['cache_dir']}")
        
        # 检查系统是否支持signal.SIGALRM (Windows不支持)
        import threading
        import platform
        import signal
        
        # 检查是否在主线程中运行
        import threading
        is_main_thread = threading.current_thread() is threading.main_thread()
        
        if platform.system() == "Windows" or not hasattr(signal, 'SIGALRM') or not is_main_thread:
            # Windows系统、没有SIGALRM或不在主线程中，使用threading超时机制
            print("[IndexTTS2] 使用threading超时机制 (跨平台兼容)")
            
            def load_bigvgan_with_timeout():
                try:
                    # 优先使用本地路径
                    if local_bigvgan_path:
                        print(f"[IndexTTS2] 使用本地BigVGAN模型: {local_bigvgan_path}")
                        self.bigvgan = bigvgan.BigVGAN.from_pretrained(
                            str(local_bigvgan_path),  # 使用本地路径
                            use_cuda_kernel=False,
                            cache_dir=bigvgan_kwargs["cache_dir"]
                        )
                    else:
                        print(f"[IndexTTS2] 从HuggingFace下载BigVGAN模型: {bigvgan_name}")
                        self.bigvgan = bigvgan.BigVGAN.from_pretrained(
                            bigvgan_name,  # 使用远程ID
                            use_cuda_kernel=False,
                            cache_dir=bigvgan_kwargs["cache_dir"]
                        )

                    print("[IndexTTS2] 开始后处理BigVGAN模型...")

                    # 检查GPU内存
                    if self.device.type == 'cuda':
                        torch.cuda.empty_cache()  # 清理GPU缓存
                        print(f"[IndexTTS2] GPU内存清理完成")

                    # 移动模型到设备
                    print(f"[IndexTTS2] 将BigVGAN模型移动到设备: {self.device}")
                    self.bigvgan = self.bigvgan.to(self.device)
                    print("[IndexTTS2] ✓ 模型移动完成")

                    # 移除权重归一化
                    print("[IndexTTS2] 移除权重归一化...")
                    self.bigvgan.remove_weight_norm()
                    print("[IndexTTS2] ✓ 权重归一化移除完成")

                    # 设置为评估模式
                    print("[IndexTTS2] 设置模型为评估模式...")
                    self.bigvgan.eval()
                    print("[IndexTTS2] ✓ 评估模式设置完成")

                    return True
                except Exception as e:
                    print(f"[ERROR] BigVGAN模型加载失败: {e}")
                    return False
            
            # 使用线程和超时
            result = [False]
            exception = [None]
            
            def target():
                try:
                    result[0] = load_bigvgan_with_timeout()
                except Exception as e:
                    exception[0] = e
            
            thread = threading.Thread(target=target)
            thread.daemon = True
            thread.start()
            thread.join(timeout=300)  # 5分钟超时
            
            if thread.is_alive():
                print("[ERROR] BigVGAN模型加载超时，可能是网络问题")
                print("[ERROR] 请检查网络连接或尝试使用代理")
                raise TimeoutError("BigVGAN模型加载超时")
            
            if exception[0]:
                raise exception[0]
            
            if not result[0]:
                raise RuntimeError("BigVGAN模型加载失败")

            print(">> bigvgan weights restored from:", local_bigvgan_path if local_bigvgan_path else bigvgan_name)
            
        else:
            # Unix/Linux系统且在主线程中，使用signal超时机制
            print("[IndexTTS2] 使用signal超时机制 (Unix/Linux主线程)")
            
            import signal
            
            def timeout_handler(signum, frame):
                raise TimeoutError("BigVGAN模型加载超时")
            
            # 设置超时（5分钟）
            signal.signal(signal.SIGALRM, timeout_handler)
            signal.alarm(300)  # 5分钟超时
            
            try:
                # 优先使用本地路径
                if local_bigvgan_path:
                    print(f"[IndexTTS2] 使用本地BigVGAN模型: {local_bigvgan_path}")
                    self.bigvgan = bigvgan.BigVGAN.from_pretrained(
                        str(local_bigvgan_path),  # 使用本地路径
                        use_cuda_kernel=False,
                        cache_dir=bigvgan_kwargs["cache_dir"]
                    )
                else:
                    print(f"[IndexTTS2] 从HuggingFace下载BigVGAN模型: {bigvgan_name}")
                    self.bigvgan = bigvgan.BigVGAN.from_pretrained(
                        bigvgan_name,  # 使用远程ID
                        use_cuda_kernel=False,
                        cache_dir=bigvgan_kwargs["cache_dir"]
                    )

                print("[IndexTTS2] 开始后处理BigVGAN模型...")

                # 检查GPU内存
                if self.device.type == 'cuda':
                    torch.cuda.empty_cache()  # 清理GPU缓存
                    print(f"[IndexTTS2] GPU内存清理完成")

                # 移动模型到设备
                print(f"[IndexTTS2] 将BigVGAN模型移动到设备: {self.device}")
                self.bigvgan = self.bigvgan.to(self.device)
                print("[IndexTTS2] ✓ 模型移动完成")

                # 移除权重归一化
                print("[IndexTTS2] 移除权重归一化...")
                self.bigvgan.remove_weight_norm()
                print("[IndexTTS2] ✓ 权重归一化移除完成")

                # 设置为评估模式
                print("[IndexTTS2] 设置模型为评估模式...")
                self.bigvgan.eval()
                print("[IndexTTS2] ✓ 评估模式设置完成")

                signal.alarm(0)  # 取消超时

                print(">> bigvgan weights restored from:", local_bigvgan_path if local_bigvgan_path else bigvgan_name)
                
            except TimeoutError:
                signal.alarm(0)
                print("[ERROR] BigVGAN模型加载超时，可能是网络问题")
                print("[ERROR] 请检查网络连接或尝试使用代理")
                raise
            except Exception as e:
                signal.alarm(0)
                print(f"[ERROR] BigVGAN模型加载失败: {e}")
                print(f"[ERROR] 模型名称: {bigvgan_name}")
                print(f"[ERROR] 缓存目录: {bigvgan_kwargs['cache_dir']}")
                if local_bigvgan_path:
                    print(f"[ERROR] 本地路径: {local_bigvgan_path}")
                raise

        # 检查BPE模型文件路径
        bpe_filename = self.cfg.dataset["bpe_model"]

        # 构建可能的BPE文件路径，使用多种方法确保兼容性
        possible_bpe_paths = []

        # 方法1: 直接使用os.path.join（保持原有兼容性）
        possible_bpe_paths.append(os.path.join(self.model_dir, bpe_filename))
        possible_bpe_paths.append(os.path.join(self.model_dir, "bpe_model.model"))

        # 方法2: 如果model_dir指向checkpoints，尝试上一级目录
        parent_dir = os.path.dirname(self.model_dir)
        possible_bpe_paths.append(os.path.join(parent_dir, bpe_filename))

        # 方法3: 相对于当前脚本的路径
        script_dir = os.path.dirname(__file__)
        possible_bpe_paths.append(os.path.join(script_dir, "..", "bpe_model.model"))

        # 方法4: 在当前工作目录查找
        possible_bpe_paths.append(bpe_filename)
        possible_bpe_paths.append("bpe_model.model")

        self.bpe_path = None
        for path in possible_bpe_paths:
            if os.path.exists(path):
                self.bpe_path = path
                print(f"[IndexTTS2] 发现BPE模型文件: {self.bpe_path}")
                break

        if not self.bpe_path:
            print(f"[ERROR] 未找到BPE模型文件，尝试的路径:")
            for path in possible_bpe_paths:
                print(f"  {path} - {'存在' if os.path.exists(path) else '不存在'}")
            print(f"[ERROR] 当前model_dir: {self.model_dir}")
            print(f"[ERROR] 配置中的BPE文件名: {bpe_filename}")
            raise FileNotFoundError(f"BPE模型文件未找到: {bpe_filename}")

        print("[IndexTTS2] 开始创建TextNormalizer...")
        print("[IndexTTS2] 由于TextNormalizer经常卡住，直接使用简化版本...")

        # 直接使用简化的TextNormalizer，避免卡住问题
        try:
            self.normalizer = self._create_fallback_normalizer()
            print("[IndexTTS2] ✓ 使用简化TextNormalizer（跳过可能卡住的正常加载）")
            print(">> TextNormalizer loaded")
        except Exception as e:
            print(f"[ERROR] 创建简化TextNormalizer失败: {e}")
            raise RuntimeError(f"TextNormalizer初始化失败: {e}")

    def _create_fallback_normalizer(self):
        """创建一个增强的TextNormalizer作为回退方案，包含数字转换功能"""
        class EnhancedFallbackTextNormalizer:
            def __init__(self):
                self.zh_normalizer = self._create_enhanced_normalizer()
                self.en_normalizer = self._create_simple_normalizer()

            def _create_enhanced_normalizer(self):
                class EnhancedNormalizer:
                    def __init__(self):
                        # 中文数字映射
                        self.digit_map = {
                            '0': '零', '1': '一', '2': '二', '3': '三', '4': '四',
                            '5': '五', '6': '六', '7': '七', '8': '八', '9': '九'
                        }
                        self.unit_map = ['', '十', '百', '千', '万', '十万', '百万', '千万', '亿']

                    def number_to_chinese(self, num_str):
                        """将数字字符串转换为中文"""
                        try:
                            num = int(num_str)
                            if num == 0:
                                return '零'

                            # 扩展的数字转换（支持0-99999999）
                            if num < 10:
                                return self.digit_map[str(num)]
                            elif num < 100:
                                tens = num // 10
                                ones = num % 10
                                if tens == 1:
                                    result = '十'
                                else:
                                    result = self.digit_map[str(tens)] + '十'
                                if ones > 0:
                                    result += self.digit_map[str(ones)]
                                return result
                            elif num < 1000:
                                hundreds = num // 100
                                remainder = num % 100
                                result = self.digit_map[str(hundreds)] + '百'
                                if remainder > 0:
                                    if remainder < 10:
                                        result += '零' + self.digit_map[str(remainder)]
                                    else:
                                        result += self.number_to_chinese(str(remainder))
                                return result
                            elif num < 10000:
                                thousands = num // 1000
                                remainder = num % 1000
                                result = self.digit_map[str(thousands)] + '千'
                                if remainder > 0:
                                    if remainder < 100:
                                        result += '零' + self.number_to_chinese(str(remainder))
                                    else:
                                        result += self.number_to_chinese(str(remainder))
                                return result
                            elif num < 100000:
                                # 万级别处理
                                wan = num // 10000
                                remainder = num % 10000
                                if wan == 1:
                                    result = '一万'
                                else:
                                    result = self.number_to_chinese(str(wan)) + '万'
                                if remainder > 0:
                                    if remainder < 1000:
                                        result += '零' + self.number_to_chinese(str(remainder))
                                    else:
                                        result += self.number_to_chinese(str(remainder))
                                return result
                            elif num < 1000000:
                                # 十万级别
                                wan = num // 10000
                                remainder = num % 10000
                                result = self.number_to_chinese(str(wan)) + '万'
                                if remainder > 0:
                                    if remainder < 1000:
                                        result += '零' + self.number_to_chinese(str(remainder))
                                    else:
                                        result += self.number_to_chinese(str(remainder))
                                return result
                            else:
                                # 对于更大的数字，简化处理
                                if num < 100000000:  # 一亿以下
                                    wan = num // 10000
                                    remainder = num % 10000
                                    result = self.number_to_chinese(str(wan)) + '万'
                                    if remainder > 0:
                                        if remainder < 1000:
                                            result += '零' + self.number_to_chinese(str(remainder))
                                        else:
                                            result += self.number_to_chinese(str(remainder))
                                    return result
                                else:
                                    # 超大数字，逐位转换
                                    return ''.join(self.digit_map.get(d, d) for d in num_str)
                        except:
                            # 如果转换失败，逐位转换
                            return ''.join(self.digit_map.get(d, d) for d in num_str)

                    def normalize(self, text):
                        import re

                        # 基本的文本清理
                        text = re.sub(r'["""]', '"', text)
                        text = re.sub(r"[''']", "'", text)
                        text = re.sub(r'[…]', '...', text)
                        text = re.sub(r'[—–]', '-', text)

                        # 数字转换（匹配连续的数字）
                        def replace_numbers(match):
                            number = match.group()
                            return self.number_to_chinese(number)

                        # 转换连续的数字
                        text = re.sub(r'\d+', replace_numbers, text)

                        # 标准化空白字符
                        text = re.sub(r'\s+', ' ', text)
                        text = text.strip()
                        return text

                return EnhancedNormalizer()

            def _create_simple_normalizer(self):
                class SimpleNormalizer:
                    def normalize(self, text):
                        import re
                        # 英文数字保持不变，只做基本清理
                        text = re.sub(r'["""]', '"', text)
                        text = re.sub(r"[''']", "'", text)
                        text = re.sub(r'[…]', '...', text)
                        text = re.sub(r'[—–]', '-', text)
                        text = re.sub(r'\s+', ' ', text)
                        text = text.strip()
                        return text
                return SimpleNormalizer()

            def normalize(self, text, lang="zh"):
                if lang == "zh":
                    return self.zh_normalizer.normalize(text)
                else:
                    return self.en_normalizer.normalize(text)

        return EnhancedFallbackTextNormalizer()

        print(f"[IndexTTS2] 开始创建TextTokenizer，BPE路径: {self.bpe_path}")
        self.tokenizer = TextTokenizer(self.bpe_path, self.normalizer)
        print("[IndexTTS2] ✓ TextTokenizer创建完成")
        print(">> bpe model loaded from:", self.bpe_path)

        print(f"[IndexTTS2] 开始加载情感矩阵: {self.cfg.emo_matrix}")
        emo_matrix = torch.load(os.path.join(self.model_dir, self.cfg.emo_matrix))
        self.emo_matrix = emo_matrix.to(self.device)
        self.emo_num = list(self.cfg.emo_num)
        print("[IndexTTS2] ✓ 情感矩阵加载完成")

        print(f"[IndexTTS2] 开始加载说话人矩阵: {self.cfg.spk_matrix}")
        spk_matrix = torch.load(os.path.join(self.model_dir, self.cfg.spk_matrix))
        self.spk_matrix = spk_matrix.to(self.device)
        print("[IndexTTS2] ✓ 说话人矩阵加载完成")

        self.emo_matrix = torch.split(self.emo_matrix, self.emo_num)
        self.spk_matrix = torch.split(self.spk_matrix, self.emo_num)

        # 后备mel_fn初始化（如果前面失败了）
        if self.mel_fn is None:
            try:
                from indextts.s2mel.modules.audio import mel_spectrogram
                mel_fn_args = {
                    "n_fft": self.cfg.s2mel['preprocess_params']['spect_params']['n_fft'],
                    "win_size": self.cfg.s2mel['preprocess_params']['spect_params']['win_length'],
                    "hop_size": self.cfg.s2mel['preprocess_params']['spect_params']['hop_length'],
                    "num_mels": self.cfg.s2mel['preprocess_params']['spect_params']['n_mels'],
                    "sampling_rate": self.cfg.s2mel["preprocess_params"]["sr"],
                    "fmin": self.cfg.s2mel['preprocess_params']['spect_params'].get('fmin', 0),
                    "fmax": None if self.cfg.s2mel['preprocess_params']['spect_params'].get('fmax', "None") == "None" else 8000,
                    "center": False
                }
                self.mel_fn = lambda x: mel_spectrogram(x, **mel_fn_args)
                print("[IndexTTS2] ✓ mel_fn后备初始化成功")
            except Exception as e:
                print(f"[ERROR] mel_fn后备初始化也失败: {e}")
                raise RuntimeError(f"无法初始化mel_fn函数: {e}")

        print("[IndexTTS2] 🎉 IndexTTS2初始化完成！所有属性和模型已就绪")

    @torch.no_grad()
    def get_emb(self, input_features, attention_mask):
        vq_emb = self.semantic_model(
            input_features=input_features,
            attention_mask=attention_mask,
            output_hidden_states=True,
        )
        feat = vq_emb.hidden_states[17]  # (B, T, C)
        feat = (feat - self.semantic_mean) / self.semantic_std
        return feat

    def remove_long_silence(self, codes: torch.Tensor, silent_token=52, max_consecutive=30):
        """
        Shrink special tokens (silent_token and stop_mel_token) in codes
        codes: [B, T]
        """
        code_lens = []
        codes_list = []
        device = codes.device
        dtype = codes.dtype
        isfix = False
        for i in range(0, codes.shape[0]):
            code = codes[i]
            if not torch.any(code == self.stop_mel_token).item():
                len_ = code.size(0)
            else:
                stop_mel_idx = (code == self.stop_mel_token).nonzero(as_tuple=False)
                len_ = stop_mel_idx[0].item() if len(stop_mel_idx) > 0 else code.size(0)

            count = torch.sum(code == silent_token).item()
            if count > max_consecutive:
                # code = code.cpu().tolist()
                ncode_idx = []
                n = 0
                for k in range(len_):
                    assert code[
                               k] != self.stop_mel_token, f"stop_mel_token {self.stop_mel_token} should be shrinked here"
                    if code[k] != silent_token:
                        ncode_idx.append(k)
                        n = 0
                    elif code[k] == silent_token and n < 10:
                        ncode_idx.append(k)
                        n += 1
                    # if (k == 0 and code[k] == 52) or (code[k] == 52 and code[k-1] == 52):
                    #    n += 1
                # new code
                len_ = len(ncode_idx)
                codes_list.append(code[ncode_idx])
                isfix = True
            else:
                # shrink to len_
                codes_list.append(code[:len_])
            code_lens.append(len_)
        if isfix:
            if len(codes_list) > 1:
                codes = pad_sequence(codes_list, batch_first=True, padding_value=self.stop_mel_token)
            else:
                codes = codes_list[0].unsqueeze(0)
        else:
            # unchanged
            pass
        # clip codes to max length
        max_len = max(code_lens)
        if max_len < codes.shape[1]:
            codes = codes[:, :max_len]
        code_lens = torch.tensor(code_lens, dtype=torch.long, device=device)
        return codes, code_lens

    def insert_interval_silence(self, wavs, sampling_rate=22050, interval_silence=200):
        """
        Insert silences between sentences.
        wavs: List[torch.tensor]
        """

        if not wavs or interval_silence <= 0:
            return wavs

        # get channel_size
        channel_size = wavs[0].size(0)
        # get silence tensor
        sil_dur = int(sampling_rate * interval_silence / 1000.0)
        sil_tensor = torch.zeros(channel_size, sil_dur)

        wavs_list = []
        for i, wav in enumerate(wavs):
            wavs_list.append(wav)
            if i < len(wavs) - 1:
                wavs_list.append(sil_tensor)

        return wavs_list

    def _set_gr_progress(self, value, desc):
        if self.gr_progress is not None:
            self.gr_progress(value, desc=desc)

    # 原始推理模式
    def infer(self, spk_audio_prompt, text, output_path,
              emo_audio_prompt=None, emo_alpha=1.0,
              emo_vector=None,
              use_emo_text=False, emo_text=None, use_random=False, interval_silence=200,
              verbose=False, max_text_tokens_per_sentence=120, **generation_kwargs):
        print(">> start inference...")
        self._set_gr_progress(0, "start inference...")
        if verbose:
            print(f"origin text:{text}, spk_audio_prompt:{spk_audio_prompt},"
                  f" emo_audio_prompt:{emo_audio_prompt}, emo_alpha:{emo_alpha}, "
                  f"emo_vector:{emo_vector}, use_emo_text:{use_emo_text}, "
                  f"emo_text:{emo_text}")
        start_time = time.perf_counter()

        if use_emo_text:
            emo_audio_prompt = None
            emo_alpha = 1.0
            # assert emo_audio_prompt is None
            # assert emo_alpha == 1.0
            if emo_text is None:
                emo_text = text

            if self.qwen_emo is not None:
                emo_dict, content = self.qwen_emo.inference(emo_text)
                print(emo_dict)
                emo_vector = list(emo_dict.values())
            else:
                print("⚠️  Emotion model not available, using default emotion vector")
                # 使用默认的情感向量
                emo_vector = [0.5] * 8  # 假设有8个情感维度

        if emo_vector is not None:
            emo_audio_prompt = None
            emo_alpha = 1.0
            # assert emo_audio_prompt is None
            # assert emo_alpha == 1.0

        if emo_audio_prompt is None:
            emo_audio_prompt = spk_audio_prompt
            emo_alpha = 1.0
            # assert emo_alpha == 1.0

        # 如果参考音频改变了，才需要重新生成, 提升速度
        if self.cache_spk_cond is None or self.cache_spk_audio_prompt != spk_audio_prompt:
            audio, sr = librosa.load(spk_audio_prompt)
            audio = torch.tensor(audio).unsqueeze(0)
            audio_22k = torchaudio.transforms.Resample(sr, 22050)(audio)
            audio_16k = torchaudio.transforms.Resample(sr, 16000)(audio)

            inputs = self.extract_features(audio_16k, sampling_rate=16000, return_tensors="pt")
            input_features = inputs["input_features"]
            attention_mask = inputs["attention_mask"]
            input_features = input_features.to(self.device)
            attention_mask = attention_mask.to(self.device)
            spk_cond_emb = self.get_emb(input_features, attention_mask)

            _, S_ref = self.semantic_codec.quantize(spk_cond_emb)
            ref_mel = self.mel_fn(audio_22k.to(spk_cond_emb.device).float())
            ref_target_lengths = torch.LongTensor([ref_mel.size(2)]).to(ref_mel.device)
            feat = torchaudio.compliance.kaldi.fbank(audio_16k.to(ref_mel.device),
                                                     num_mel_bins=80,
                                                     dither=0,
                                                     sample_frequency=16000)
            feat = feat - feat.mean(dim=0, keepdim=True)  # feat2另外一个滤波器能量组特征[922, 80]
            style = self.campplus_model(feat.unsqueeze(0))  # 参考音频的全局style2[1,192]

            prompt_condition = self.s2mel.models['length_regulator'](S_ref,
                                                                     ylens=ref_target_lengths,
                                                                     n_quantizers=3,
                                                                     f0=None)[0]

            self.cache_spk_cond = spk_cond_emb
            self.cache_s2mel_style = style
            self.cache_s2mel_prompt = prompt_condition
            self.cache_spk_audio_prompt = spk_audio_prompt
            self.cache_mel = ref_mel
        else:
            style = self.cache_s2mel_style
            prompt_condition = self.cache_s2mel_prompt
            spk_cond_emb = self.cache_spk_cond
            ref_mel = self.cache_mel

        if emo_vector is not None:
            print(f"[IndexTTS2] Processing emotion vector: {emo_vector}")
            weight_vector = torch.tensor(emo_vector).to(self.device)

            # 验证情感向量的有效性
            weight_sum = torch.sum(weight_vector)
            print(f"[IndexTTS2] Emotion vector sum: {weight_sum:.6f}")
            print(f"[IndexTTS2] Individual values: {[f'{v:.6f}' for v in emo_vector]}")

            if weight_sum <= 0.001:
                print("[IndexTTS2] Warning: emotion vector sum is near zero, using default neutral emotion")
                # 设置默认的中性情感
                weight_vector = torch.zeros_like(weight_vector)
                weight_vector[7] = 0.2  # Neutral emotion
            elif weight_sum > 2.0:
                print(f"[IndexTTS2] Warning: emotion vector sum is {weight_sum:.3f}, normalizing")
                weight_vector = weight_vector / weight_sum * 1.0  # 归一化到合理范围

            print(f"[IndexTTS2] Final weight_vector: {weight_vector.tolist()}")

            if use_random:
                random_index = [random.randint(0, x - 1) for x in self.emo_num]
            else:
                random_index = [find_most_similar_cosine(style, tmp) for tmp in self.spk_matrix]

            # 验证索引的有效性，防止索引超出范围
            validated_indices = []
            for i, (index, tmp, emo_dim_size) in enumerate(zip(random_index, self.emo_matrix, self.emo_num)):
                # 确保索引在有效范围内
                if index >= tmp.shape[0]:
                    print(f"[IndexTTS2] Warning: emotion index {index} >= matrix size {tmp.shape[0]} for dimension {i}, using 0")
                    index = 0
                elif index < 0:
                    print(f"[IndexTTS2] Warning: emotion index {index} < 0 for dimension {i}, using 0")
                    index = 0
                validated_indices.append(index)

            try:
                emo_matrix = [tmp[index].unsqueeze(0) for index, tmp in zip(validated_indices, self.emo_matrix)]
                emo_matrix = torch.cat(emo_matrix, 0)
                emovec_mat = weight_vector.unsqueeze(1) * emo_matrix
                emovec_mat = torch.sum(emovec_mat, 0)
                emovec_mat = emovec_mat.unsqueeze(0)
            except Exception as e:
                print(f"[IndexTTS2] Error in emotion matrix processing: {e}")
                print(f"[IndexTTS2] weight_vector shape: {weight_vector.shape}")
                print(f"[IndexTTS2] validated_indices: {validated_indices}")
                print(f"[IndexTTS2] emo_matrix shapes: {[tmp.shape for tmp in self.emo_matrix]}")
                # 创建一个安全的默认情感矩阵
                default_emovec = torch.zeros((1, self.emo_matrix[0].shape[1]), device=self.device)
                emovec_mat = default_emovec

        if self.cache_emo_cond is None or self.cache_emo_audio_prompt != emo_audio_prompt:
            emo_audio, _ = librosa.load(emo_audio_prompt, sr=16000)
            emo_inputs = self.extract_features(emo_audio, sampling_rate=16000, return_tensors="pt")
            emo_input_features = emo_inputs["input_features"]
            emo_attention_mask = emo_inputs["attention_mask"]
            emo_input_features = emo_input_features.to(self.device)
            emo_attention_mask = emo_attention_mask.to(self.device)
            emo_cond_emb = self.get_emb(emo_input_features, emo_attention_mask)

            self.cache_emo_cond = emo_cond_emb
            self.cache_emo_audio_prompt = emo_audio_prompt
        else:
            emo_cond_emb = self.cache_emo_cond

        self._set_gr_progress(0.1, "text processing...")
        text_tokens_list = self.tokenizer.tokenize(text)
        sentences = self.tokenizer.split_sentences(text_tokens_list, max_text_tokens_per_sentence)
        if verbose:
            print("text_tokens_list:", text_tokens_list)
            print("sentences count:", len(sentences))
            print("max_text_tokens_per_sentence:", max_text_tokens_per_sentence)
            print(*sentences, sep="\n")
        do_sample = generation_kwargs.pop("do_sample", True)
        top_p = generation_kwargs.pop("top_p", 0.8)
        top_k = generation_kwargs.pop("top_k", 30)
        temperature = generation_kwargs.pop("temperature", 0.8)
        autoregressive_batch_size = 1
        length_penalty = generation_kwargs.pop("length_penalty", 0.0)
        num_beams = generation_kwargs.pop("num_beams", 3)
        repetition_penalty = generation_kwargs.pop("repetition_penalty", 10.0)
        max_mel_tokens = generation_kwargs.pop("max_mel_tokens", 1500)
        sampling_rate = 22050

        wavs = []
        gpt_gen_time = 0
        gpt_forward_time = 0
        s2mel_time = 0
        bigvgan_time = 0
        progress = 0
        has_warned = False
        for sent in sentences:
            text_tokens = self.tokenizer.convert_tokens_to_ids(sent)
            text_tokens = torch.tensor(text_tokens, dtype=torch.int32, device=self.device).unsqueeze(0)
            if verbose:
                print(text_tokens)
                print(f"text_tokens shape: {text_tokens.shape}, text_tokens type: {text_tokens.dtype}")
                # debug tokenizer
                text_token_syms = self.tokenizer.convert_ids_to_tokens(text_tokens[0].tolist())
                print("text_token_syms is same as sentence tokens", text_token_syms == sent)

            m_start_time = time.perf_counter()
            with torch.no_grad():
                with torch.amp.autocast(text_tokens.device.type, enabled=self.dtype is not None, dtype=self.dtype):
                    emovec = self.gpt.merge_emovec(
                        spk_cond_emb,
                        emo_cond_emb,
                        torch.tensor([spk_cond_emb.shape[-1]], device=text_tokens.device),
                        torch.tensor([emo_cond_emb.shape[-1]], device=text_tokens.device),
                        alpha=emo_alpha
                    )

                    if emo_vector is not None:
                        # 确保权重向量的和在合理范围内
                        weight_sum = torch.sum(weight_vector)
                        weight_sum = torch.clamp(weight_sum, 0.0, 1.0)  # 限制在[0,1]范围内

                        # 混合情感向量和原始向量
                        emovec = emovec_mat + (1 - weight_sum) * emovec
                        # emovec = emovec_mat

                    codes, speech_conditioning_latent = self.gpt.inference_speech(
                        spk_cond_emb,
                        text_tokens,
                        emo_cond_emb,
                        cond_lengths=torch.tensor([spk_cond_emb.shape[-1]], device=text_tokens.device),
                        emo_cond_lengths=torch.tensor([emo_cond_emb.shape[-1]], device=text_tokens.device),
                        emo_vec=emovec,
                        do_sample=True,
                        top_p=top_p,
                        top_k=top_k,
                        temperature=temperature,
                        num_return_sequences=autoregressive_batch_size,
                        length_penalty=length_penalty,
                        num_beams=num_beams,
                        repetition_penalty=repetition_penalty,
                        max_generate_length=max_mel_tokens,
                        **generation_kwargs
                    )

                gpt_gen_time += time.perf_counter() - m_start_time
                if not has_warned and (codes[:, -1] != self.stop_mel_token).any():
                    warnings.warn(
                        f"WARN: generation stopped due to exceeding `max_mel_tokens` ({max_mel_tokens}). "
                        f"Input text tokens: {text_tokens.shape[1]}. "
                        f"Consider reducing `max_text_tokens_per_sentence`({max_text_tokens_per_sentence}) or increasing `max_mel_tokens`.",
                        category=RuntimeWarning
                    )
                    has_warned = True

                code_lens = torch.tensor([codes.shape[-1]], device=codes.device, dtype=codes.dtype)
                #                 if verbose:
                #                     print(codes, type(codes))
                #                     print(f"codes shape: {codes.shape}, codes type: {codes.dtype}")
                #                     print(f"code len: {code_lens}")

                code_lens = []
                for code in codes:
                    if self.stop_mel_token not in code:
                        code_lens.append(len(code))
                        code_len = len(code)
                    else:
                        len_ = (code == self.stop_mel_token).nonzero(as_tuple=False)[0] + 1
                        code_len = len_ - 1
                    code_lens.append(code_len)
                codes = codes[:, :code_len]
                code_lens = torch.LongTensor(code_lens)
                code_lens = code_lens.to(self.device)
                if verbose:
                    print(codes, type(codes))
                    print(f"fix codes shape: {codes.shape}, codes type: {codes.dtype}")
                    print(f"code len: {code_lens}")

                m_start_time = time.perf_counter()
                use_speed = torch.zeros(spk_cond_emb.size(0)).to(spk_cond_emb.device).long()
                with torch.amp.autocast(text_tokens.device.type, enabled=self.dtype is not None, dtype=self.dtype):
                    latent = self.gpt(
                        speech_conditioning_latent,
                        text_tokens,
                        torch.tensor([text_tokens.shape[-1]], device=text_tokens.device),
                        codes,
                        torch.tensor([codes.shape[-1]], device=text_tokens.device),
                        emo_cond_emb,
                        cond_mel_lengths=torch.tensor([spk_cond_emb.shape[-1]], device=text_tokens.device),
                        emo_cond_mel_lengths=torch.tensor([emo_cond_emb.shape[-1]], device=text_tokens.device),
                        emo_vec=emovec,
                        use_speed=use_speed,
                    )
                    gpt_forward_time += time.perf_counter() - m_start_time

                dtype = None
                with torch.amp.autocast(text_tokens.device.type, enabled=dtype is not None, dtype=dtype):
                    m_start_time = time.perf_counter()
                    diffusion_steps = 25
                    inference_cfg_rate = 0.7
                    latent = self.s2mel.models['gpt_layer'](latent)
                    S_infer = self.semantic_codec.quantizer.vq2emb(codes.unsqueeze(1))
                    S_infer = S_infer.transpose(1, 2)
                    S_infer = S_infer + latent
                    target_lengths = (code_lens * 1.72).long()

                    cond = self.s2mel.models['length_regulator'](S_infer,
                                                                 ylens=target_lengths,
                                                                 n_quantizers=3,
                                                                 f0=None)[0]
                    cat_condition = torch.cat([prompt_condition, cond], dim=1)
                    vc_target = self.s2mel.models['cfm'].inference(cat_condition,
                                                                   torch.LongTensor([cat_condition.size(1)]).to(
                                                                       cond.device),
                                                                   ref_mel, style, None, diffusion_steps,
                                                                   inference_cfg_rate=inference_cfg_rate)
                    vc_target = vc_target[:, :, ref_mel.size(-1):]
                    s2mel_time += time.perf_counter() - m_start_time

                    m_start_time = time.perf_counter()
                    wav = self.bigvgan(vc_target.float()).squeeze().unsqueeze(0)
                    print(wav.shape)
                    bigvgan_time += time.perf_counter() - m_start_time
                    wav = wav.squeeze(1)

                wav = torch.clamp(32767 * wav, -32767.0, 32767.0)
                if verbose:
                    print(f"wav shape: {wav.shape}", "min:", wav.min(), "max:", wav.max())
                # wavs.append(wav[:, :-512])
                wavs.append(wav.cpu())  # to cpu before saving
        end_time = time.perf_counter()
        self._set_gr_progress(0.9, "save audio...")
        wavs = self.insert_interval_silence(wavs, sampling_rate=sampling_rate, interval_silence=interval_silence)
        wav = torch.cat(wavs, dim=1)
        wav_length = wav.shape[-1] / sampling_rate
        print(f">> gpt_gen_time: {gpt_gen_time:.2f} seconds")
        print(f">> gpt_forward_time: {gpt_forward_time:.2f} seconds")
        print(f">> s2mel_time: {s2mel_time:.2f} seconds")
        print(f">> bigvgan_time: {bigvgan_time:.2f} seconds")
        print(f">> Total inference time: {end_time - start_time:.2f} seconds")
        print(f">> Generated audio length: {wav_length:.2f} seconds")
        print(f">> RTF: {(end_time - start_time) / wav_length:.4f}")

        # save audio
        wav = wav.cpu()  # to cpu
        if output_path:
            # 直接保存音频到指定路径中
            if os.path.isfile(output_path):
                os.remove(output_path)
                print(">> remove old wav file:", output_path)
            if os.path.dirname(output_path) != "":
                os.makedirs(os.path.dirname(output_path), exist_ok=True)
            torchaudio.save(output_path, wav.type(torch.int16), sampling_rate)
            print(">> wav file saved to:", output_path)
            return output_path
        else:
            # 返回以符合Gradio的格式要求
            wav_data = wav.type(torch.int16)
            wav_data = wav_data.numpy().T
            return (sampling_rate, wav_data)


def find_most_similar_cosine(query_vector, matrix):
    try:
        query_vector = query_vector.float()
        matrix = matrix.float()

        # 检查输入的有效性
        if matrix.shape[0] == 0:
            print("[IndexTTS2] Warning: empty matrix in find_most_similar_cosine, returning 0")
            return 0

        if torch.isnan(query_vector).any() or torch.isinf(query_vector).any():
            print("[IndexTTS2] Warning: invalid query_vector in find_most_similar_cosine, returning 0")
            return 0

        similarities = F.cosine_similarity(query_vector, matrix, dim=1)

        # 检查相似度计算结果
        if torch.isnan(similarities).any() or torch.isinf(similarities).any():
            print("[IndexTTS2] Warning: invalid similarities in find_most_similar_cosine, returning 0")
            return 0

        most_similar_index = torch.argmax(similarities)

        # 确保索引在有效范围内
        index_value = most_similar_index.item()
        if index_value >= matrix.shape[0]:
            print(f"[IndexTTS2] Warning: computed index {index_value} >= matrix size {matrix.shape[0]}, using 0")
            return 0

        return index_value

    except Exception as e:
        print(f"[IndexTTS2] Error in find_most_similar_cosine: {e}")
        return 0

class QwenEmotion:
    def __init__(self, model_dir):
        # 首先设置所有必要的属性，确保即使初始化失败也不会出现AttributeError
        self.model_dir = model_dir
        self.model = None
        self.tokenizer = None
        self.is_available = False

        # 设置默认属性
        self._initialize_default_attributes()

        # 智能加载策略：先检查transformers版本兼容性
        # Smart loading strategy: check transformers version compatibility first
        print(f"[IndexTTS2] 尝试加载Qwen情感模型: {model_dir}")
        print(f"[IndexTTS2] Attempting to load Qwen emotion model: {model_dir}")

        # 检查是否应该跳过初始模型加载
        should_skip_initial_load = self._should_skip_initial_model_load(model_dir)

        if should_skip_initial_load:
            print(f"[IndexTTS2] 🔄 检测到版本兼容性问题，直接使用备用方案")
            print(f"[IndexTTS2] 🔄 Version compatibility issue detected, using fallback directly")
            # 不抛出异常，而是直接跳到备用方案
            self._handle_fallback_loading()
            return

        try:
            # 直接尝试加载，让transformers自己处理兼容性
            if os.path.exists(model_dir):
                # 本地路径，使用local_files_only=True
                print(f"[IndexTTS2] 从本地路径加载模型...")
                print(f"[IndexTTS2] Loading model from local path...")

                self.tokenizer = AutoTokenizer.from_pretrained(
                    self.model_dir,
                    local_files_only=True,
                    trust_remote_code=True
                )
                self.model = AutoModelForCausalLM.from_pretrained(
                    self.model_dir,
                    torch_dtype="float16",
                    device_map="auto",
                    local_files_only=True,
                    trust_remote_code=True
                )
            else:
                # 远程repo，正常加载
                print(f"[IndexTTS2] 从远程仓库加载模型...")
                print(f"[IndexTTS2] Loading model from remote repository...")

                self.tokenizer = AutoTokenizer.from_pretrained(
                    self.model_dir,
                    trust_remote_code=True
                )
                self.model = AutoModelForCausalLM.from_pretrained(
                    self.model_dir,
                    torch_dtype="float16",
                    device_map="auto",
                    trust_remote_code=True
                )

            self.is_available = True
            print(f"[IndexTTS2] ✅ Qwen情感模型加载成功！")
            print(f"[IndexTTS2] ✅ Qwen emotion model loaded successfully!")

        except Exception as e:
            # 任何加载失败都使用备用方案，不管具体原因
            print(f"[IndexTTS2] ⚠️  Qwen情感模型加载失败: {e}")
            print(f"[IndexTTS2] ⚠️  Failed to load Qwen emotion model: {e}")

            # 提供具体的错误分析和建议
            self._analyze_loading_error(e)

            print(f"[IndexTTS2] 🔄 将使用备用情感分析方法")
            print(f"[IndexTTS2] 🔄 Will use fallback emotion analysis method")

            # 尝试智能备用方案：根据transformers版本加载兼容的Qwen模型
            print(f"[IndexTTS2] 🔄 尝试智能备用方案...")
            print(f"[IndexTTS2] 🔄 Trying intelligent fallback...")

            fallback_success = self._try_fallback_qwen_models()

            if not fallback_success:
                print(f"[IndexTTS2] 🔄 所有Qwen模型都无法加载，使用关键词匹配备用方案")
                print(f"[IndexTTS2] 🔄 All Qwen models failed to load, using keyword matching fallback")
                self.is_available = False
                self.model = None
                self.tokenizer = None

    def _initialize_default_attributes(self):
        """初始化默认属性，确保所有方法都能正常调用"""
        # 设置情感分析相关的默认属性
        self.prompt = """你是一个情感分析专家。请分析以下文本的情感，并给出8个维度的情感分数（0-1之间的浮点数）：
        happy（开心）、angry（愤怒）、sad（悲伤）、fear（恐惧）、hate（厌恶）、low（低落）、surprise（惊讶）、neutral（中性）。

        请直接返回JSON格式的结果，例如：
        {"happy": 0.8, "angry": 0.0, "sad": 0.1, "fear": 0.0, "hate": 0.0, "low": 0.0, "surprise": 0.1, "neutral": 0.0}

        文本："""

        # 设置备用情感字典
        self.backup_dict = {
            "happy": 0, "angry": 0, "sad": 0, "fear": 0,
            "hate": 0, "low": 0, "surprise": 0, "neutral": 1.0
        }

        # 设置分数范围
        self.max_score = 1.2
        self.min_score = 0.0
        # 设置转换字典
        self.convert_dict = {
            "愤怒": "angry",
            "高兴": "happy",
            "恐惧": "fear",
            "反感": "hate",
            "悲伤": "sad",
            "低落": "low",
            "惊讶": "surprise",
            "自然": "neutral",
        }

    def _analyze_loading_error(self, error):
        """分析加载错误并提供具体的解决建议"""
        error_str = str(error).lower()

        if "qwen3" in error_str and "transformers does not recognize" in error_str:
            print(f"[IndexTTS2] 💡 错误分析: Qwen3模型需要更新的transformers版本")
            print(f"[IndexTTS2] 💡 Error analysis: Qwen3 model requires newer transformers version")
            print(f"[IndexTTS2] 🔧 建议解决方案:")
            print(f"[IndexTTS2] 🔧 Suggested solutions:")
            print(f"[IndexTTS2]    1. 升级transformers: pip install --upgrade transformers")
            print(f"[IndexTTS2]    2. 或安装开发版本: pip install git+https://github.com/huggingface/transformers.git")
            print(f"[IndexTTS2]    3. 当前将尝试使用兼容的备用模型")
        elif "keyerror" in error_str:
            print(f"[IndexTTS2] 💡 错误分析: 模型架构不被当前transformers版本支持")
            print(f"[IndexTTS2] 💡 Error analysis: Model architecture not supported by current transformers version")
        elif "no module named" in error_str:
            print(f"[IndexTTS2] 💡 错误分析: 缺少必要的依赖包")
            print(f"[IndexTTS2] 💡 Error analysis: Missing required dependencies")
        elif "out of memory" in error_str or "cuda out of memory" in error_str:
            print(f"[IndexTTS2] 💡 错误分析: GPU内存不足")
            print(f"[IndexTTS2] 💡 Error analysis: Insufficient GPU memory")
            print(f"[IndexTTS2] 🔧 建议: 将尝试使用更小的模型")
        else:
            print(f"[IndexTTS2] 💡 错误分析: 通用加载错误，将尝试备用方案")
            print(f"[IndexTTS2] 💡 Error analysis: General loading error, trying fallback options")

    def _should_skip_initial_model_load(self, model_dir):
        """
        检查是否应该跳过初始模型加载
        基于模型路径和transformers版本进行智能判断
        """
        try:
            import transformers
            from packaging import version

            current_ver = version.parse(transformers.__version__)
            print(f"[IndexTTS2] 检查版本兼容性 - transformers: {transformers.__version__}")
            print(f"[IndexTTS2] Checking version compatibility - transformers: {transformers.__version__}")

            # 检查模型路径中是否包含已知的版本敏感关键词
            model_path_lower = model_dir.lower()

            # Qwen3相关模型需要transformers >= 4.51.0
            if any(keyword in model_path_lower for keyword in ['qwen3', 'qwen-3', 'qwen_3']):
                if current_ver < version.parse("4.51.0"):
                    print(f"[IndexTTS2] ⚠️  检测到Qwen3模型，但transformers版本 {transformers.__version__} < 4.51.0")
                    print(f"[IndexTTS2] ⚠️  Detected Qwen3 model, but transformers version {transformers.__version__} < 4.51.0")
                    return True

            # 检查配置文件中的特定模型名称
            if 'qwen0.6bemo4-merge' in model_path_lower:
                # 这个模型很可能是Qwen3架构，需要更新的transformers
                # 对于4.49.0+版本，我们可以尝试加载，但仍然准备备用方案
                if current_ver < version.parse("4.49.0"):
                    print(f"[IndexTTS2] ⚠️  检测到qwen0.6bemo4-merge模型，transformers版本 {transformers.__version__} 可能不兼容")
                    print(f"[IndexTTS2] ⚠️  Detected qwen0.6bemo4-merge model, transformers version {transformers.__version__} may not be compatible")
                    return True
                else:
                    print(f"[IndexTTS2] 💡 transformers版本 {transformers.__version__} >= 4.49.0，尝试加载qwen0.6bemo4-merge模型")
                    print(f"[IndexTTS2] 💡 transformers version {transformers.__version__} >= 4.49.0, attempting to load qwen0.6bemo4-merge model")

            return False

        except Exception as e:
            print(f"[IndexTTS2] ⚠️  版本兼容性检查失败: {e}")
            print(f"[IndexTTS2] ⚠️  Version compatibility check failed: {e}")
            return False

    def _handle_fallback_loading(self):
        """处理备用加载逻辑"""
        print(f"[IndexTTS2] 🔄 将使用备用情感分析方法")
        print(f"[IndexTTS2] 🔄 Will use fallback emotion analysis method")

        # 尝试智能备用方案：根据transformers版本加载兼容的Qwen模型
        print(f"[IndexTTS2] 🔄 尝试智能备用方案...")
        print(f"[IndexTTS2] 🔄 Trying intelligent fallback...")

        fallback_success = self._try_fallback_qwen_models()

        if not fallback_success:
            print(f"[IndexTTS2] 🔄 所有Qwen模型都无法加载，使用关键词匹配备用方案")
            print(f"[IndexTTS2] 🔄 All Qwen models failed to load, using keyword matching fallback")
            self.is_available = False
            self.model = None
            self.tokenizer = None

    def _get_compatible_qwen_models(self):
        """根据transformers版本获取兼容的Qwen模型列表"""
        try:
            import transformers
            from packaging import version

            current_ver = version.parse(transformers.__version__)
            print(f"[IndexTTS2] 检测transformers版本: {transformers.__version__}")
            print(f"[IndexTTS2] Detecting transformers version: {transformers.__version__}")

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

            print(f"[IndexTTS2] 找到 {len(qwen_models)} 个兼容的Qwen模型")
            print(f"[IndexTTS2] Found {len(qwen_models)} compatible Qwen models")

            return qwen_models

        except Exception as e:
            print(f"[IndexTTS2] ⚠️  获取兼容模型列表失败: {e}")
            print(f"[IndexTTS2] ⚠️  Failed to get compatible model list: {e}")
            return []

    def _try_fallback_qwen_models(self):
        """尝试加载备用Qwen模型"""
        compatible_models = self._get_compatible_qwen_models()

        if not compatible_models:
            print(f"[IndexTTS2] ⚠️  没有找到兼容的Qwen模型")
            print(f"[IndexTTS2] ⚠️  No compatible Qwen models found")
            return False

        for model_info in compatible_models:
            try:
                print(f"[IndexTTS2] 🔄 尝试加载备用模型: {model_info['name']} ({model_info['size']})")
                print(f"[IndexTTS2] 🔄 Trying fallback model: {model_info['name']} ({model_info['size']})")
                print(f"[IndexTTS2] 📝 模型描述: {model_info['description']}")
                print(f"[IndexTTS2] 📝 Model description: {model_info['description']}")

                # 尝试加载备用模型
                self.tokenizer = AutoTokenizer.from_pretrained(
                    model_info['model_id'],
                    trust_remote_code=True
                )

                self.model = AutoModelForCausalLM.from_pretrained(
                    model_info['model_id'],
                    torch_dtype="float16",
                    device_map="auto",
                    trust_remote_code=True
                )

                self.is_available = True
                self.fallback_model_info = model_info

                print(f"[IndexTTS2] ✅ 备用模型加载成功: {model_info['name']}")
                print(f"[IndexTTS2] ✅ Fallback model loaded successfully: {model_info['name']}")
                print(f"[IndexTTS2] 💡 使用 {model_info['size']} 参数的 {model_info['name']} 进行情感分析")
                print(f"[IndexTTS2] 💡 Using {model_info['size']} parameter {model_info['name']} for emotion analysis")

                return True

            except Exception as e:
                print(f"[IndexTTS2] ⚠️  备用模型 {model_info['name']} 加载失败: {e}")
                print(f"[IndexTTS2] ⚠️  Fallback model {model_info['name']} failed to load: {e}")
                continue

        print(f"[IndexTTS2] ❌ 所有备用Qwen模型都加载失败")
        print(f"[IndexTTS2] ❌ All fallback Qwen models failed to load")
        return False

    def convert(self, content):
        content = content.replace("\n", " ")
        content = content.replace(" ", "")
        content = content.replace("{", "")
        content = content.replace("}", "")
        content = content.replace('"', "")
        parts = content.strip().split(',')
        print(parts)
        parts_dict = {}
        desired_order = ["高兴", "愤怒", "悲伤", "恐惧", "反感", "低落", "惊讶", "自然"]
        for part in parts:
            key_value = part.strip().split(':')
            if len(key_value) == 2:
                parts_dict[key_value[0].strip()] = part
        # 按照期望顺序重新排列
        ordered_parts = [parts_dict[key] for key in desired_order if key in parts_dict]
        parts = ordered_parts
        if len(parts) != len(self.convert_dict):
            return self.backup_dict

        emotion_dict = {}
        for part in parts:
            key_value = part.strip().split(':')
            if len(key_value) == 2:
                try:
                    key = self.convert_dict[key_value[0].strip()]
                    value = float(key_value[1].strip())
                    value = max(self.min_score, min(self.max_score, value))
                    emotion_dict[key] = value
                except Exception:
                    continue

        for key in self.backup_dict:
            if key not in emotion_dict:
                emotion_dict[key] = 0.0

        if sum(emotion_dict.values()) <= 0:
            return self.backup_dict

        return emotion_dict

    def inference(self, text_input):
        """
        进行情感推理
        如果模型不可用，返回备用情感字典
        """
        # 检查模型是否可用
        if not self.is_available or self.model is None or self.tokenizer is None:
            print(f"[IndexTTS2] ⚠️  Qwen emotion model not available, using keyword-based fallback")
            print(f"[IndexTTS2] ⚠️  Qwen情感模型不可用，使用关键词匹配备用方案")

            # 使用简单的关键词匹配作为备用方案
            fallback_emotion = self._fallback_emotion_analysis(text_input)
            return fallback_emotion, f"Keyword fallback for: {text_input[:50]}..."

        # 显示使用的模型信息
        if hasattr(self, 'fallback_model_info'):
            model_info = self.fallback_model_info
            print(f"[IndexTTS2] 🤖 使用备用模型进行情感分析: {model_info['name']} ({model_info['size']})")
            print(f"[IndexTTS2] 🤖 Using fallback model for emotion analysis: {model_info['name']} ({model_info['size']})")
        else:
            print(f"[IndexTTS2] 🤖 使用原始Qwen模型进行情感分析")
            print(f"[IndexTTS2] 🤖 Using original Qwen model for emotion analysis")

        try:
            start = time.time()
            messages = [
                {"role": "system", "content": f"{self.prompt}"},
                {"role": "user", "content": f"{text_input}"}
            ]
            text = self.tokenizer.apply_chat_template(
                messages,
                tokenize=False,
                add_generation_prompt=True,
                enable_thinking=False,
            )
            model_inputs = self.tokenizer([text], return_tensors="pt").to(self.model.device)

            # conduct text completion
            generated_ids = self.model.generate(
                **model_inputs,
                max_new_tokens=32768,
                pad_token_id=self.tokenizer.eos_token_id
            )
            output_ids = generated_ids[0][len(model_inputs.input_ids[0]):].tolist()

            # parsing thinking content
            try:
                # rindex finding 151668 (</think>)
                index = len(output_ids) - output_ids[::-1].index(151668)
            except ValueError:
                index = 0

            content = self.tokenizer.decode(output_ids[index:], skip_special_tokens=True).strip("\n")
            emotion_dict = self.convert(content)
            return emotion_dict, content

        except Exception as e:
            print(f"[IndexTTS2] ⚠️  Qwen emotion inference failed: {e}")
            print(f"[IndexTTS2] ⚠️  Qwen情感推理失败，使用备用分析")

            # 发生错误时使用备用方案
            fallback_emotion = self._fallback_emotion_analysis(text_input)
            return fallback_emotion, f"Error fallback for: {text_input[:50]}..."

    def _fallback_emotion_analysis(self, text_input):
        """
        增强的备用情感分析方法
        使用更智能的关键词匹配和语义分析来分析情感
        Enhanced fallback emotion analysis method using smarter keyword matching and semantic analysis
        """
        print(f"[IndexTTS2] 🔍 使用增强关键词匹配进行情感分析")
        print(f"[IndexTTS2] 🔍 Using enhanced keyword matching for emotion analysis")

        text_lower = text_input.lower()

        # 定义更全面的情感关键词库，包含权重
        emotion_keywords = {
            "happy": {
                "high": ["太好了", "超开心", "非常高兴", "特别兴奋", "狂欢", "欣喜若狂"],
                "medium": ["开心", "高兴", "快乐", "兴奋", "愉快", "欢乐", "喜悦", "好棒", "棒极了"],
                "low": ["哈哈", "笑", "呵呵", "嘿嘿", "不错", "挺好"]
            },
            "angry": {
                "high": ["气死了", "愤怒至极", "火冒三丈", "暴怒", "狂怒"],
                "medium": ["生气", "愤怒", "气愤", "恼火", "烦躁", "愤慨", "火大"],
                "low": ["讨厌", "烦", "怒", "不爽", "郁闷"]
            },
            "sad": {
                "high": ["心痛", "痛不欲生", "悲痛欲绝", "绝望", "崩溃"],
                "medium": ["伤心", "难过", "悲伤", "沮丧", "失望", "痛苦", "难受"],
                "low": ["哭", "眼泪", "唉", "可惜", "遗憾"]
            },
            "fear": {
                "high": ["恐怖", "惊慌失措", "吓死了", "恐惧至极"],
                "medium": ["害怕", "恐惧", "担心", "紧张", "焦虑", "不安", "惊慌"],
                "low": ["可怕", "吓", "担忧", "忧虑", "不放心"]
            },
            "hate": {
                "high": ["憎恨", "厌恶至极", "深恶痛绝", "恨死了"],
                "medium": ["讨厌", "厌恶", "反感", "恶心", "嫌弃", "受不了"],
                "low": ["烦人", "不喜欢", "反对", "拒绝"]
            },
            "low": {
                "high": ["消沉", "颓废", "绝望", "无助", "空虚"],
                "medium": ["低落", "郁闷", "无聊", "疲惫", "没劲", "无力"],
                "low": ["累", "懒", "困", "倦", "乏"]
            },
            "surprise": {
                "high": ["震惊", "惊呆了", "不敢相信", "太意外了"],
                "medium": ["惊讶", "意外", "吃惊", "惊奇", "想不到"],
                "low": ["天哪", "哇", "真的吗", "是吗", "咦"]
            },
            "neutral": {
                "high": ["明白了", "了解了", "知道了"],
                "medium": ["好的", "明白", "了解", "是的", "对"],
                "low": ["嗯", "哦", "这样", "那样", "好吧"]
            }
        }

        # 权重设置
        weight_map = {"high": 3.0, "medium": 2.0, "low": 1.0}

        # 计算每种情感的加权匹配分数
        emotion_scores = {}
        matched_keywords = {}

        for emotion, levels in emotion_keywords.items():
            score = 0
            matches = []
            for level, keywords in levels.items():
                weight = weight_map[level]
                for keyword in keywords:
                    if keyword in text_lower:
                        score += weight
                        matches.append(f"{keyword}({level})")
            emotion_scores[emotion] = score
            if matches:
                matched_keywords[emotion] = matches

        # 显示匹配的关键词（用于调试）
        if matched_keywords:
            print(f"[IndexTTS2] 🔍 匹配的情感关键词: {matched_keywords}")
            print(f"[IndexTTS2] 🔍 Matched emotion keywords: {matched_keywords}")

        # 如果没有匹配到任何关键词，返回中性情感
        if sum(emotion_scores.values()) == 0:
            return self.backup_dict.copy()

        # 归一化分数
        total_score = sum(emotion_scores.values())
        normalized_scores = {}
        for emotion, score in emotion_scores.items():
            if total_score > 0:
                normalized_scores[emotion] = min(self.max_score, (score / total_score) * 1.0)
            else:
                normalized_scores[emotion] = 0.0

        # 确保至少有一个情感有分数
        if sum(normalized_scores.values()) == 0:
            normalized_scores["neutral"] = 0.8

        return normalized_scores


if __name__ == "__main__":
    prompt_wav = "examples/voice_01.wav"
    text = '欢迎大家来体验indextts2，并给予我们意见与反馈，谢谢大家。'

    tts = IndexTTS2(cfg_path="checkpoints/config.yaml", model_dir="checkpoints", use_cuda_kernel=False)
    tts.infer(spk_audio_prompt=prompt_wav, text=text, output_path="gen.wav", verbose=True)
