# coding: utf-8
"""
阿里云 DashScope 在线语音识别器

实现与 FunASREngine / sherpa-onnx 相同的接口（create_stream / decode_stream），
内部调用 DashScope SDK 进行语音识别。
"""

import logging
import os
import threading
from dataclasses import dataclass, field
from typing import List, Optional

import numpy as np

from config_server import AliyunASRConfig

logger = logging.getLogger(__name__)


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
    """阿里云 DashScope 在线语音识别器，接口兼容 sherpa-onnx / FunASREngine"""

    def __init__(self):
        self.sample_rate = 16000
        cfg = AliyunASRConfig
        self._model = cfg.model

        # 设置 API Key
        import dashscope
        if cfg.api_key:
            dashscope.api_key = cfg.api_key
        elif 'DASHSCOPE_API_KEY' not in os.environ:
            logger.warning("未设置 DashScope API Key，请在 config_server.py 或环境变量中配置")

    def create_stream(self, **kwargs):
        return RecognitionStream(sample_rate=self.sample_rate)

    def decode_stream(self, stream: RecognitionStream, context=None, **kwargs):
        """调用 DashScope API 识别音频"""
        audio = stream.audio_data
        if audio is None:
            stream.set_result("")
            return

        from dashscope.audio.asr import Recognition, RecognitionCallback, RecognitionResult as DashResult

        # float32 → int16 PCM bytes
        pcm = (audio * 32767).astype(np.int16).tobytes()

        # 用于同步等待结果
        collected_texts = []
        done_event = threading.Event()
        error_msg = [None]

        class _Callback(RecognitionCallback):
            def on_complete(self):
                done_event.set()

            def on_error(self, message):
                error_msg[0] = str(message.message) if hasattr(message, 'message') else str(message)
                logger.error(f"DashScope ASR 错误: {error_msg[0]}")
                done_event.set()

            def on_event(self, result: DashResult):
                sentence = result.get_sentence()
                if sentence and 'text' in sentence:
                    if DashResult.is_sentence_end(sentence):
                        collected_texts.append(sentence['text'])

        callback = _Callback()
        recognition = Recognition(
            model=self._model,
            format='pcm',
            sample_rate=self.sample_rate,
            semantic_punctuation_enabled=False,
            callback=callback,
        )

        try:
            recognition.start()

            # 分块发送音频 (3200 bytes = 100ms of 16kHz 16-bit mono)
            chunk_size = 3200
            for i in range(0, len(pcm), chunk_size):
                recognition.send_audio_frame(pcm[i:i + chunk_size])

            recognition.stop()
            done_event.wait(timeout=30)
        except Exception as e:
            logger.error(f"DashScope ASR 调用失败: {e}", exc_info=True)

        final_text = ''.join(collected_texts)

        # 生成字符级 tokens 和均匀时间戳
        chars = list(final_text.replace(' ', ''))
        duration = len(audio) / self.sample_rate
        if chars and duration > 0:
            time_per_char = duration / len(chars)
            timestamps = [i * time_per_char for i in range(len(chars))]
        else:
            timestamps = []

        stream.set_result(final_text, timestamps, chars)
