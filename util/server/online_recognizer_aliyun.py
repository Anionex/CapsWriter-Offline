# coding: utf-8
"""
阿里云 DashScope 在线语音识别器（Qwen3-ASR Realtime WebSocket）

实现与 FunASREngine / sherpa-onnx 相同的接口（create_stream / decode_stream），
内部通过 WebSocket 协议调用 Qwen3-ASR 实时模型。
"""

import asyncio
import base64
import json
import logging
import os
import uuid
from dataclasses import dataclass, field
from typing import List, Optional

import numpy as np
import websockets

from config_server import AliyunASRConfig

logger = logging.getLogger(__name__)

WS_URL = 'wss://dashscope.aliyuncs.com/api-ws/v1/realtime'
AUDIO_CHUNK_SIZE = 8000  # 每次发送的 int16 采样数（约 0.5s @16kHz）


@dataclass
class RecognitionResult:
    text: str = ""
    timestamps: List[float] = field(default_factory=list)
    tokens: List[str] = field(default_factory=list)


@dataclass
class RecognitionStream:
    sample_rate: int = 16000
    audio_data: Optional[np.ndarray] = None
    _result: Optional[RecognitionResult] = field(default=None, init=False, repr=False)

    def accept_waveform(self, sample_rate: int, audio: np.ndarray):
        self.sample_rate = sample_rate
        self.audio_data = audio.astype(np.float32)

    @property
    def result(self) -> RecognitionResult:
        if self._result is None:
            self._result = RecognitionResult()
        return self._result

    def set_result(self, text: str, timestamps: List[float] = None, tokens: List[str] = None):
        self._result = RecognitionResult(text=text, timestamps=timestamps or [], tokens=tokens or [])


class AliyunRecognizer:
    """阿里云 DashScope 在线语音识别器（Qwen3-ASR Realtime WebSocket），接口兼容 sherpa-onnx / FunASREngine"""

    def __init__(self):
        self.sample_rate = 16000
        cfg = AliyunASRConfig
        self._model = cfg.model
        self._language = cfg.language
        self._enable_itn = cfg.enable_itn
        self._context = cfg.context
        self._api_key = cfg.api_key or os.environ.get('DASHSCOPE_API_KEY', '')
        if not self._api_key:
            logger.warning("未设置 DashScope API Key，请在 config_server.py 或环境变量中配置")

    def create_stream(self, **kwargs):
        return RecognitionStream(sample_rate=self.sample_rate)

    def _build_ws_url(self):
        return f"{WS_URL}?model={self._model}"

    def _make_event(self, event_type: str, **kwargs):
        """构造 WebSocket 事件消息"""
        msg = {"event_id": uuid.uuid4().hex, "type": event_type}
        msg.update(kwargs)
        return json.dumps(msg, ensure_ascii=False)

    def decode_stream(self, stream: RecognitionStream, context=None, is_final=True, **kwargs):
        """通过 WebSocket 实时接口调用 Qwen3-ASR 识别音频（仅处理最终片段）"""
        if not is_final:
            stream.set_result("")
            return

        audio = stream.audio_data
        if audio is None:
            stream.set_result("")
            return

        # float32 → int16 PCM
        pcm_int16 = np.clip(audio * 32768, -32768, 32767).astype(np.int16)
        pcm_bytes = pcm_int16.tobytes()

        # 合并语境
        ctx_parts = [self._context, context or ""]
        ctx = " ".join(p for p in ctx_parts if p)

        final_text = ""
        try:
            final_text = self._ws_recognize(pcm_bytes, ctx)
        except Exception as e:
            logger.error(f"Qwen3-ASR WebSocket 识别失败: {e}")

        # 生成字符级 tokens 和均匀时间戳
        chars = list(final_text.replace(' ', ''))
        duration = len(audio) / self.sample_rate
        if chars and duration > 0:
            time_per_char = duration / len(chars)
            timestamps = [i * time_per_char for i in range(len(chars))]
        else:
            timestamps = []

        stream.set_result(final_text, timestamps, chars)

    def _ws_recognize(self, pcm_bytes: bytes, context: str) -> str:
        """通过 WebSocket 完成一次完整的识别流程"""
        return asyncio.run(self._ws_recognize_async(pcm_bytes, context))

    async def _ws_recognize_async(self, pcm_bytes: bytes, context: str) -> str:
        headers = {
            "Authorization": f"bearer {self._api_key}",
            "X-DashScope-DataInspection": "enable",
        }
        async with websockets.connect(self._build_ws_url(), additional_headers=headers) as ws:
            return await self._ws_session(ws, pcm_bytes, context)

    async def _ws_session(self, ws, pcm_bytes: bytes, context: str) -> str:
        """WebSocket 会话：配置 → 发送音频 → 收集结果"""
        # 1. 等待 session.created
        await self._recv_until(ws, 'session.created')

        # 2. 发送 session.update（手动模式，不用 VAD）
        session_cfg = {
            "session": {
                "modalities": ["audio", "text"],
                "input_audio_format": "pcm",
                "input_audio_sample_rate": self.sample_rate,
                "turn_detection": None,
                "input_audio_transcription": {
                    "language": self._language,
                    "enable_itn": self._enable_itn,
                },
            }
        }
        if context:
            session_cfg["session"]["input_audio_transcription"]["context"] = context
        await ws.send(self._make_event("session.update", **session_cfg))
        await self._recv_until(ws, 'session.updated')

        # 3. 分块发送音频
        offset = 0
        chunk_bytes = AUDIO_CHUNK_SIZE * 2  # int16 = 2 bytes per sample
        while offset < len(pcm_bytes):
            chunk = pcm_bytes[offset:offset + chunk_bytes]
            b64 = base64.b64encode(chunk).decode('ascii')
            await ws.send(self._make_event("input_audio_buffer.append", audio=b64))
            offset += chunk_bytes

        # 4. commit + finish
        await ws.send(self._make_event("input_audio_buffer.commit"))
        await ws.send(self._make_event("session.finish"))

        # 5. 收集识别结果
        final_text = ""
        while True:
            data = json.loads(await ws.recv())
            evt = data.get("type", "")

            if evt == "conversation.item.input_audio_transcription.completed":
                final_text = data.get("transcript", "")
            elif evt == "session.finished":
                break
            elif evt == "error":
                logger.error(f"Qwen3-ASR 服务端错误: {data}")
                break

        return final_text

    async def _recv_until(self, ws, target_type: str):
        """接收消息直到收到指定类型的事件"""
        while True:
            data = json.loads(await ws.recv())
            evt = data.get("type", "")
            if evt == target_type:
                return data
            if evt == "error":
                raise RuntimeError(f"Qwen3-ASR 服务端错误: {data}")
