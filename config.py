# -*- coding: utf-8 -*-
"""
全局配置文件
用于统一管理模型路径、默认参数及系统常量。
"""
import os

# --- 路径配置 ---
BASE_DIR = os.path.dirname(os.path.abspath(__file__))

# 基础模型路径 (请确保该路径下包含完整的 diffusers 模型文件)
MODEL_PATH = os.path.join(BASE_DIR, "Z-Image-Turbo")

# 新增：LoRA 目录
LORA_DIR = os.path.join(BASE_DIR, "loras")

# 输出与数据库路径
OUTPUT_DIR = os.path.join(BASE_DIR, "outputs")
DB_PATH = os.path.join(BASE_DIR, "database", "history.db")

# 确保目录存在
os.makedirs(OUTPUT_DIR, exist_ok=True)
os.makedirs(os.path.dirname(DB_PATH), exist_ok=True)
os.makedirs(LORA_DIR, exist_ok=True)

# --- 默认生成参数 ---
DEFAULT_PROMPT = ""
DEFAULT_NEGATIVE_PROMPT = ""

# 尺寸与步数
DEFAULT_STEPS = 8      # Turbo 模型推荐步数
DEFAULT_CFG = 0.0      # Turbo 模型推荐 CFG 为 0
DEFAULT_WIDTH = 1024
DEFAULT_HEIGHT = 1024
DEFAULT_SEED = -1      # -1 代表随机种子

# LoRA 默认设置
DEFAULT_LORA_SCALE = 1.3
DEFAULT_LORA_ENABLE = False

# --- 系统配置 ---
# 代理设置 (解决本地连接问题)
os.environ['no_proxy'] = 'localhost,127.0.0.1,0.0.0.0'
