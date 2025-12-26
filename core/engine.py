# -*- coding: utf-8 -*-
"""
推理引擎 (API适配版)
负责模型的加载、显存优化及图片生成。
返回结构化数据而非 UI 字符串。
"""
import torch
from diffusers import DiffusionPipeline # type: ignore
import gc
import time
import os
from core.utils import detect_device, get_torch_dtype
from core.lora_manager import LoRAMerger
import config

class ZImageEngine:
    def __init__(self):
        self.pipe = None
        self.device = None
        self.dtype = None
        self.lora_merger = None
        self.current_lora_applied = False
        self.current_lora_configs = []

    def is_loaded(self):
        return self.pipe is not None

    def load_model(self):
        """加载模型 (自动检测设备)"""
        # --- 自动下载模型检测 ---
        if not os.path.exists(config.MODEL_PATH) or not os.listdir(config.MODEL_PATH):
            print(f"⚠️ [Engine] 未检测到模型，正在从 ModelScope 下载 Tongyi-MAI/Z-Image-Turbo...")
            print(f"   目标路径: {config.MODEL_PATH}")
            try:
                # 优先尝试使用 Python API
                from modelscope import snapshot_download
                snapshot_download('Tongyi-MAI/Z-Image-Turbo', local_dir=config.MODEL_PATH)
                print("✅ [Engine] 模型下载完成。")
            except ImportError:
                print("⚠️ [Engine] 未检测到 modelscope 库，尝试使用命令行工具...")
                try:
                    subprocess.run(
                        ["modelscope", "download", "--model", "Tongyi-MAI/Z-Image-Turbo", "--local_dir", config.MODEL_PATH],
                        check=True
                    )
                except Exception as cmd_e:
                     return False, f"模型缺失且无法自动下载: {str(cmd_e)}"
            except Exception as e:
                return False, f"模型下载过程中出错: {str(e)}"

        self.device = detect_device()
        self.dtype = torch.bfloat16
        
        print(f"🚀 [Engine] 正在加载模型... 设备: {self.device.upper()}, 精度: {self.dtype}")
        
        # 清理旧显存
        if self.pipe:
            del self.pipe
            self.pipe = None
            gc.collect()
            if torch.cuda.is_available(): torch.cuda.empty_cache()
            if torch.backends.mps.is_available(): torch.mps.empty_cache()

        try:
            self.pipe = DiffusionPipeline.from_pretrained(
                config.MODEL_PATH,
                torch_dtype=self.dtype,
                trust_remote_code=True,
            )
            self.pipe.to(self.device)
            
            self.lora_merger = LoRAMerger(self.pipe)
            self.current_lora_applied = False
            self.current_lora_configs = []
            
            self._apply_optimizations()
            
            print("✅ [Engine] 模型加载完毕。")
            return True, f"就绪 ({self.device.upper()})"
            
        except Exception as e:
            print(f"❌ [Engine] 加载失败: {e}")
            return False, str(e)

    def _apply_optimizations(self):
        """应用优化策略"""
        # VAE 强制 FP32
        if hasattr(self.pipe, "vae"):
            self.pipe.vae.to(dtype=torch.float32) # pyright: ignore[reportOptionalMemberAccess]
            self.pipe.vae.config.force_upcast = True # pyright: ignore[reportOptionalMemberAccess]

        # 硬件特定优化
        if self.device == "mps":
            # MPS 显存足够时关闭 Tiling 以获得最佳画质
            pass 
        elif self.device == "cuda":
            self.pipe.enable_model_cpu_offload() # pyright: ignore[reportOptionalMemberAccess]
            if hasattr(self.pipe, "enable_vae_tiling"):
                self.pipe.enable_vae_tiling() # pyright: ignore[reportOptionalMemberAccess]

    def update_lora(self, enable, lora_configs):
        """
        更新 LoRA 状态 (增量更新，无需重载模型)
        lora_configs: list of dict {'name': str, 'scale': float}
        """
        if not self.is_loaded(): return
        
        target_configs = lora_configs if enable else []
        
        # 转换为字典以便比较: {name: scale}
        current_map = {c['name']: c['scale'] for c in self.current_lora_configs}
        target_map = {c['name']: c['scale'] for c in target_configs}
        
        # 快速检查是否完全一致
        if current_map == target_map:
            return

        print("🔄 [Engine] 检测到 LoRA 变更，正在应用增量...")
        
        # 获取所有涉及的 LoRA 名称
        all_names = set(current_map.keys()) | set(target_map.keys())
        
        try:
            changes_count = 0
            for name in all_names:
                old_scale = current_map.get(name, 0.0)
                new_scale = target_map.get(name, 0.0)
                diff = new_scale - old_scale
                
                if abs(diff) > 1e-4: # 忽略微小浮点误差
                    lora_path = os.path.join(config.LORA_DIR, name)
                    self.lora_merger.apply_lora_weights(lora_path, diff)
                    changes_count += 1
            
            if changes_count > 0:
                print(f"✅ [Engine] LoRA 更新完成 ({changes_count} 个变动)")
            
            # 更新当前状态
            self.current_lora_configs = target_configs
            self.current_lora_applied = bool(target_configs)
            
        except Exception as e:
            print(f"❌ [Engine] LoRA 增量更新失败: {e}")
            print("⚠️ [Engine] 正在尝试回退到全量重载模式...")
            
            # 回退机制：重新加载模型并应用目标配置
            self.load_model()
            if target_configs:
                for config_item in target_configs:
                     lora_path = os.path.join(config.LORA_DIR, config_item['name'])
                     if os.path.exists(lora_path):
                         self.lora_merger.load_lora_weights(lora_path, config_item['scale'])
                self.current_lora_configs = target_configs
                self.current_lora_applied = True
            else:
                self.current_lora_configs = []
                self.current_lora_applied = False

    def generate(self, prompt, neg_prompt, steps, cfg, width, height, seed, seed_mode, progress_callback=None):
        """
        生成图片
        Returns:
        dict: { "image": PIL_Image, "seed": int, "duration": float }
        """
        start_time = time.time()
        
        # 显存清理
        gc.collect()
        if self.device == "mps": torch.mps.empty_cache()
        if self.device == "cuda": torch.cuda.empty_cache()

        # 种子处理
        if seed_mode == "random" or seed == -1:
            actual_seed = torch.randint(0, 2**32 - 1, (1,)).item()
        else:
            actual_seed = int(seed)
            
        gen_device = "cpu" if self.device == "mps" else self.device
        generator = torch.Generator(gen_device).manual_seed(actual_seed) # pyright: ignore[reportArgumentType]

        # 打印当前生效的 LoRA 信息
        if self.current_lora_configs:
            lora_info = ", ".join([f"{c['name']}({c['scale']})" for c in self.current_lora_configs])
            print(f"🎨 [Generate] 正在使用 LoRA: {lora_info}")
        else:
            print(f"🎨 [Generate] 未启用 LoRA")

        print(f"🎨 [Generate] 尺寸: {width}x{height} | 步数: {steps} | 种子: {actual_seed}")

        # 回调函数封装
        def step_callback(pipe, step_index, timestep, callback_kwargs):
            if progress_callback:
                progress_callback(step_index, steps)
            return callback_kwargs

        try:
            image = self.pipe(
                prompt=prompt,
                negative_prompt=neg_prompt,
                num_inference_steps=steps,
                guidance_scale=cfg,
                width=width,
                height=height,
                generator=generator,
                callback_on_step_end=step_callback
            ).images[0] # type: ignore
            
            duration = time.time() - start_time
            
            return {
                "success": True,
                "image": image,
                "seed": actual_seed,
                "duration": round(duration, 2)
            }
        except Exception as e:
            return {
                "success": False,
                "error": str(e)
            }