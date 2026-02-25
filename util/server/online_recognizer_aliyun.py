# coding: utf-8
"""
阿里云 DashScope 在线语音识别器（Qwen3-ASR Realtime WebSocket）

实现与 FunASREngine / sherpa-onnx 相同的接口（create_stream / decode_stream），
内部通过 WebSocket 协议调用 Qwen3-ASR 实时模型。

流式策略：
- 用户开始说话时立即建立 WebSocket 连接并配置 session
- 每个音频片段到达时实时发送，API 同步处理
- 松开按键时发送 commit+finish，此时大部分音频已处理完毕
- 最终结果几乎立即返回，消除 ~2s 固定等待
"""

import asyncio
import base64
import concurrent.futures
import json
import logging
import os
import threading
import uuid
from dataclasses import dataclass, field
from typing import Dict, List, Optional

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

@dataclass
class _SessionState:
    audio_queue: asyncio.Queue
    result_future: concurrent.futures.Future
    task: asyncio.Task


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

        self._sessions: Dict[str, _SessionState] = {}

        # 持久事件循环，运行在独立线程，避免每次 asyncio.run() 的开销
        self._loop = asyncio.new_event_loop()
        self._loop_thread = threading.Thread(target=self._loop.run_forever, daemon=True)
        self._loop_thread.start()

    def create_stream(self, **kwargs):
        return RecognitionStream(sample_rate=self.sample_rate)

    def _build_ws_url(self):
        return f"{WS_URL}?model={self._model}"

    def _make_event(self, event_type: str, **kwargs):
        msg = {"event_id": uuid.uuid4().hex, "type": event_type}
        msg.update(kwargs)
        return json.dumps(msg, ensure_ascii=False)

    # ------------------------------------------------------------------
    # 主入口
    # ------------------------------------------------------------------

    def decode_stream(self, stream: RecognitionStream, context=None,
                      is_final=True, socket_id=None, **kwargs):
        audio = stream.audio_data
        if audio is None:
            stream.set_result("")
            return

        pcm_int16 = np.clip(audio * 32768, -32768, 32767).astype(np.int16)
        pcm_bytes = pcm_int16.tobytes()

        ctx_parts = [self._context, context or ""]
        ctx = " ".join(p for p in ctx_parts if p)

        if socket_id is None:
            # 无 socket_id 时退化为批量模式（兼容旧调用）
            if not is_final:
                stream.set_result("")
                return
            final_text = self._run(self._ws_recognize_async(pcm_bytes, ctx))
            self._fill_result(stream, final_text, audio)
            return

        # ---- 流式模式 ----
        if socket_id not in self._sessions:
            self._open_session(socket_id, ctx)  # 同步等待 session 对象创建完成

        session = self._sessions[socket_id]  # 此时一定存在

        # 直接放入队列，_run_session 会在 WebSocket 就绪后读取（无需等待建连）
        asyncio.run_coroutine_threadsafe(
            session.audio_queue.put(pcm_bytes), self._loop
        ).result()

        if not is_final:
            stream.set_result("")
            return

        # 最终片段：发送结束信号，等待结果
        asyncio.run_coroutine_threadsafe(
            session.audio_queue.put(None), self._loop
        ).result()

        try:
            final_text = session.result_future.result(timeout=30)
        except Exception as e:
            logger.error(f"Qwen3-ASR 识别失败: {e}")
            final_text = ""

        self._fill_result(stream, final_text, audio)

    # ------------------------------------------------------------------
    # Session 管理
    # ------------------------------------------------------------------

    def _open_session(self, socket_id: str, context: str):
        """同步等待 session 对象创建完成（WebSocket 建连在后台异步进行）"""
        asyncio.run_coroutine_threadsafe(
            self._create_session(socket_id, context),
            self._loop
        ).result()

    async def _create_session(self, socket_id: str, context: str):
        audio_queue: asyncio.Queue = asyncio.Queue()
        result_future: concurrent.futures.Future = concurrent.futures.Future()
        task = self._loop.create_task(
            self._run_session(socket_id, audio_queue, result_future, context)
        )
        self._sessions[socket_id] = _SessionState(
            audio_queue=audio_queue,
            result_future=result_future,
            task=task,
        )

    async def _run_session(self, socket_id: str, audio_queue: asyncio.Queue,
                           result_future: concurrent.futures.Future, context: str):
        headers = {
            "Authorization": f"bearer {self._api_key}",
            "X-DashScope-DataInspection": "enable",
        }
        try:
            async with websockets.connect(self._build_ws_url(), additional_headers=headers) as ws:
                await self._recv_until(ws, 'session.created')

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

                # 持续接收音频块并发送
                chunk_bytes = AUDIO_CHUNK_SIZE * 2
                while True:
                    pcm = await audio_queue.get()
                    if pcm is None:  # 结束信号
                        break
                    offset = 0
                    while offset < len(pcm):
                        chunk = pcm[offset:offset + chunk_bytes]
                        b64 = base64.b64encode(chunk).decode('ascii')
                        await ws.send(self._make_event("input_audio_buffer.append", audio=b64))
                        offset += chunk_bytes

                await ws.send(self._make_event("input_audio_buffer.commit"))
                await ws.send(self._make_event("session.finish"))

                # 收集最终结果
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

                result_future.set_result(final_text)

        except Exception as e:
            logger.error(f"Qwen3-ASR session 异常: {e}")
            if not result_future.done():
                result_future.set_result("")
        finally:
            self._sessions.pop(socket_id, None)

    # ------------------------------------------------------------------
    # 工具方法
    # ------------------------------------------------------------------

    def _run(self, coro):
        """在持久事件循环上同步运行协程"""
        return asyncio.run_coroutine_threadsafe(coro, self._loop).result(timeout=30)

    async def _ws_recognize_async(self, pcm_bytes: bytes, context: str) -> str:
        """批量模式：一次性发送所有音频"""
        headers = {
            "Authorization": f"bearer {self._api_key}",
            "X-DashScope-DataInspection": "enable",
        }
        async with websockets.connect(self._build_ws_url(), additional_headers=headers) as ws:
            await self._recv_until(ws, 'session.created')

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

            offset = 0
            chunk_bytes = AUDIO_CHUNK_SIZE * 2
            while offset < len(pcm_bytes):
                chunk = pcm_bytes[offset:offset + chunk_bytes]
                b64 = base64.b64encode(chunk).decode('ascii')
                await ws.send(self._make_event("input_audio_buffer.append", audio=b64))
                offset += chunk_bytes

            await ws.send(self._make_event("input_audio_buffer.commit"))
            await ws.send(self._make_event("session.finish"))

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

    def _fill_result(self, stream: RecognitionStream, final_text: str, audio: np.ndarray):
        chars = list(final_text.replace(' ', ''))
        duration = len(audio) / self.sample_rate
        if chars and duration > 0:
            time_per_char = duration / len(chars)
            timestamps = [i * time_per_char for i in range(len(chars))]
        else:
            timestamps = []
        stream.set_result(final_text, timestamps, chars)

    async def _recv_until(self, ws, target_type: str):
        while True:
            data = json.loads(await ws.recv())
            evt = data.get("type", "")
            if evt == target_type:
                return data
            if evt == "error":
                raise RuntimeError(f"Qwen3-ASR 服务端错误: {data}")
