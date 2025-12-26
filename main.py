# -*- coding: utf-8 -*-
"""
Z-Image Studio 全栈入口
同时负责 API 服务和 静态页面托管。
"""
import uvicorn
from fastapi import FastAPI, HTTPException, WebSocket, WebSocketDisconnect
from fastapi.staticfiles import StaticFiles
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from typing import List
import os
import uuid
import asyncio

from core.engine import ZImageEngine
from core.lora_manager import LoRAMerger
from database.db_manager import DatabaseManager
import config

# --- 1. 初始化 ---
app = FastAPI(title="Z-Image Studio")

# 允许跨域 (保留作为保险)
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

engine = ZImageEngine()
db = DatabaseManager()

# --- 2. 数据模型 ---
class LoraConfig(BaseModel):
    name: str
    scale: float = 1.0

class GenerateRequest(BaseModel):
    prompt: str
    negative_prompt: str = ""
    steps: int = 8
    cfg: float = 0.0
    width: int = 1024
    height: int = 1024
    seed: int = -1
    seed_mode: str = "fixed"
    lora_enabled: bool = False
    loras: List[LoraConfig] = []

# --- 3. API 接口 (先定义 API，优先级最高) ---

@app.on_event("startup")
async def startup_event():
    print("🌟 系统启动中，正在加载模型...")
    engine.load_model()
    # 默认不加载 LoRA
    engine.update_lora(False, [])

@app.get("/api/status")
def get_status():
    return {
        "loaded": engine.is_loaded(),
        "device": engine.device,
        "dtype": str(engine.dtype),
        "lora_enabled": engine.current_lora_applied
    }

@app.get("/api/loras")
def get_loras():
    return LoRAMerger.scan_loras(config.LORA_DIR)

@app.post("/api/generate")
def generate_image(req: GenerateRequest):
    if not engine.is_loaded():
        raise HTTPException(status_code=503, detail="模型未加载")

    # update_lora 内部会自动检查配置是否变更，无需在此重复判断
    lora_configs = [l.dict() for l in req.loras]
    engine.update_lora(req.lora_enabled, lora_configs)
    
    result = engine.generate(
        prompt=req.prompt,
        neg_prompt=req.negative_prompt,
        steps=req.steps,
        cfg=req.cfg,
        width=req.width,
        height=req.height,
        seed=req.seed,
        seed_mode=req.seed_mode
    )
    
    if not result["success"]:
        raise HTTPException(status_code=500, detail=result["error"])
        
    filename = f"{uuid.uuid4().hex}.png"
    save_path = os.path.join(config.OUTPUT_DIR, filename)
    result["image"].save(save_path, format="PNG")
    
    record = {
        "filename": filename,
        "prompt": req.prompt,
        "negative_prompt": req.negative_prompt,
        "steps": req.steps,
        "cfg": req.cfg,
        "seed": result["seed"],
        "width": req.width,
        "height": req.height,
        "lora_enabled": req.lora_enabled,
        "loras": lora_configs, # 记录 LoRA 详情
        "device": engine.device,
        "duration": result["duration"]
    }
    new_id = db.add_record(record)
    
    return {
        "id": new_id,
        "url": f"/outputs/{filename}",
        "seed": result["seed"],
        "duration": result["duration"],
        "meta": record
    }

@app.get("/api/history")
def get_history(limit: int = 20, offset: int = 0):
    records = db.get_history(limit, offset)
    for r in records:
        r["url"] = f"/outputs/{r['filename']}"
    return records

@app.delete("/api/history/{record_id}")
def delete_history(record_id: int):
    success = db.delete_record(record_id)
    if not success:
        raise HTTPException(status_code=404, detail="记录不存在")
    return {"status": "deleted"}

@app.websocket("/ws/generate")
async def websocket_generate(websocket: WebSocket):
    await websocket.accept()
    try:
        data = await websocket.receive_json()
        req = GenerateRequest(**data)

        if not engine.is_loaded():
            await websocket.send_json({"type": "error", "message": "模型未加载"})
            return

        loop = asyncio.get_event_loop()

        # 处理 LoRA 配置
        lora_configs = [l.dict() for l in req.loras]
        await loop.run_in_executor(
            None, lambda: engine.update_lora(req.lora_enabled, lora_configs)
        )

        def progress_callback(step, total):
            asyncio.run_coroutine_threadsafe(
                websocket.send_json({"type": "progress", "step": step + 1, "total": total}),
                loop
            )

        result = await loop.run_in_executor(
            None,
            lambda: engine.generate(
                prompt=req.prompt,
                neg_prompt=req.negative_prompt,
                steps=req.steps,
                cfg=req.cfg,
                width=req.width,
                height=req.height,
                seed=req.seed,
                seed_mode=req.seed_mode,
                progress_callback=progress_callback
            )
        )

        if not result["success"]:
            await websocket.send_json({"type": "error", "message": result["error"]})
        else:
            filename = f"{uuid.uuid4().hex}.png"
            save_path = os.path.join(config.OUTPUT_DIR, filename)
            await loop.run_in_executor(None, lambda: result["image"].save(save_path, format="PNG"))

            record = {
                "filename": filename,
                "prompt": req.prompt,
                "negative_prompt": req.negative_prompt,
                "steps": req.steps,
                "cfg": req.cfg,
                "seed": result["seed"],
                "width": req.width,
                "height": req.height,
                "lora_enabled": req.lora_enabled,
                "loras": lora_configs,
                "device": engine.device,
                "duration": result["duration"]
            }

            new_id = await loop.run_in_executor(None, lambda: db.add_record(record))

            await websocket.send_json({
                "type": "complete",
                "result": {
                    "id": new_id,
                    "url": f"/outputs/{filename}",
                    "seed": result["seed"],
                    "duration": result["duration"],
                    "meta": record
                }
            })

    except WebSocketDisconnect:
        print("Client disconnected")
    except Exception as e:
        try:
             await websocket.send_json({"type": "error", "message": str(e)})
        except:
             pass
    finally:
        try:
            await websocket.close()
        except:
            pass

# --- 4. 静态文件托管 (最后定义，作为兜底) ---

# 挂载 outputs 目录，用于访问生成的图片
app.mount("/outputs", StaticFiles(directory=config.OUTPUT_DIR), name="outputs")

# [关键修改] 挂载 web 目录到根路径 '/'，实现“打开网址即由后端提供页面”
# 注意：html=True 表示访问 / 会自动寻找 index.html
app.mount("/", StaticFiles(directory="web", html=True), name="web")

if __name__ == "__main__":
    print("🚀 Z-Image Studio 全栈版已启动!")
    print("👉 请访问: http://127.0.0.1:8888")
    uvicorn.run("main:app", host="0.0.0.0", port=8888, reload=True)