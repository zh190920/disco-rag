"""
异步LLM客户端 - 高性能版

核心优化：
1. asyncio + httpx 异步HTTP（替代同步openai库）
2. 连接池复用（减少TCP握手开销）
3. 信号量并发控制（防止API限流）
4. 自动重试 + 指数退避
5. LRU缓存（相同Prompt不重复调用）
6. 批量并发调度器
7. 请求级超时控制
"""

import asyncio
import hashlib
import json
import logging
import time
from dataclasses import dataclass, field
from functools import lru_cache
from typing import Optional, List, Dict, Any, Callable

logger = logging.getLogger(__name__)


@dataclass
class AsyncLLMConfig:
    """异步LLM配置"""
    api_base: str = "https://open.bigmodel.cn/api/paas/v4"
    api_key: str = ""
    model_name: str = "glm-4-flash"
    temperature: float = 0.1
    max_tokens: int = 4096
    timeout: float = 120.0

    # 并发控制
    max_concurrency: int = 10        # 最大并发请求数
    requests_per_minute: int = 60    # 每分钟请求限制（0=不限）

    # 重试策略
    max_retries: int = 3             # 最大重试次数
    retry_base_delay: float = 1.0    # 重试基础延迟(秒)
    retry_max_delay: float = 30.0    # 重试最大延迟(秒)

    # 缓存
    enable_cache: bool = True        # 是否启用内存缓存
    cache_max_size: int = 512        # 缓存最大条目数


class AsyncLLMClient:
    """
    高性能异步LLM客户端

    性能优化点：
    ┌──────────────────────────────────────────────┐
    │ 1. httpx AsyncClient 连接池复用               │
    │    → 每次请求无需新建TCP连接                   │
    │    → 减少 ~50ms/请求 的连接开销               │
    │                                              │
    │ 2. asyncio.Semaphore 并发控制                  │
    │    → 同时发起 N 个请求（默认10个）             │
    │    → 自动排队，不会压垮API                    │
    │                                              │
    │ 3. Token Bucket 限流                          │
    │    → 精确控制每分钟请求数                     │
    │    → 平滑发送，避免突发                       │
    │                                              │
    │ 4. 指数退避重试                               │
    │    → 429/5xx 自动重试                         │
    │    → 2^n * base_delay，避免雪崩               │
    │                                              │
    │ 5. Prompt哈希缓存                             │
    │    → 相同Prompt直接返回缓存结果               │
    │    → 测试/调试时大幅减少重复调用               │
    │                                              │
    │ 6. 批量并发调度                               │
    │    → gather_with_limit 控制并发上限            │
    │    → 任务完成即释放信号量                     │
    └──────────────────────────────────────────────┘
    """

    def __init__(self, config: Optional[AsyncLLMConfig] = None):
        self.config = config or AsyncLLMConfig()
        self._client = None  # 延迟初始化
        self._semaphore = asyncio.Semaphore(self.config.max_concurrency)
        self._cache: Dict[str, Any] = {}
        self._cache_order: List[str] = []  # LRU顺序
        self._token_bucket = _TokenBucket(self.config.requests_per_minute)
        self._stats = {
            "total_requests": 0,
            "cache_hits": 0,
            "retries": 0,
            "total_time": 0.0,
        }
        self._initialized = False

    async def _ensure_client(self):
        """延迟初始化httpx异步客户端"""
        if self._initialized:
            return
        try:
            import httpx
            self._client = httpx.AsyncClient(
                base_url=self.config.api_base,
                headers={
                    "Authorization": f"Bearer {self.config.api_key}",
                    "Content-Type": "application/json",
                },
                timeout=httpx.Timeout(self.config.timeout, connect=10.0),
                limits=httpx.Limits(
                    max_connections=self.config.max_concurrency + 5,
                    max_keepalive_connections=self.config.max_concurrency,
                    keepalive_expiry=60,
                ),
            )
            self._initialized = True
            logger.info(
                f"异步LLM客户端初始化: {self.config.api_base} / {self.config.model_name} "
                f"(并发上限={self.config.max_concurrency})"
            )
        except ImportError:
            raise ImportError("请安装httpx: pip install httpx")

    async def close(self):
        """关闭客户端连接池"""
        if self._client:
            await self._client.aclose()
            self._client = None
            self._initialized = False

    async def __aenter__(self):
        await self._ensure_client()
        return self

    async def __aexit__(self, exc_type, exc_val, exc_tb):
        await self.close()

    def _cache_key(self, system_prompt: str, user_prompt: str, temperature: float) -> str:
        """生成缓存Key"""
        raw = f"{system_prompt}||{user_prompt}||{temperature}"
        return hashlib.md5(raw.encode()).hexdigest()

    def _get_cache(self, key: str) -> Optional[str]:
        """读取缓存"""
        if not self.config.enable_cache:
            return None
        if key in self._cache:
            # 更新LRU顺序
            self._cache_order.remove(key)
            self._cache_order.append(key)
            self._stats["cache_hits"] += 1
            return self._cache[key]
        return None

    def _set_cache(self, key: str, value: str):
        """写入缓存"""
        if not self.config.enable_cache:
            return
        self._cache[key] = value
        self._cache_order.append(key)
        # LRU淘汰
        while len(self._cache) > self.config.cache_max_size:
            oldest = self._cache_order.pop(0)
            self._cache.pop(oldest, None)

    async def chat(
        self,
        system_prompt: str,
        user_prompt: str,
        temperature: Optional[float] = None,
    ) -> str:
        """
        异步聊天请求

        执行流程：
        1. 检查缓存 → 命中则直接返回
        2. 获取信号量 → 控制并发上限
        3. 获取Token → 限流控制
        4. 发送HTTP请求 → 异步非阻塞
        5. 失败重试 → 指数退避
        6. 写入缓存 → 供后续复用
        """
        await self._ensure_client()

        temp = temperature if temperature is not None else self.config.temperature
        cache_key = self._cache_key(system_prompt, user_prompt, temp)

        # 1. 检查缓存
        cached = self._get_cache(cache_key)
        if cached is not None:
            logger.debug("缓存命中")
            return cached

        # 2-4. 并发控制 + 限流 + 请求
        async with self._semaphore:
            await self._token_bucket.acquire()
            result = await self._request_with_retry(system_prompt, user_prompt, temp)

        # 5. 写入缓存
        self._set_cache(cache_key, result)
        return result

    async def chat_json(
        self,
        system_prompt: str,
        user_prompt: str,
        temperature: Optional[float] = None,
    ) -> dict:
        """异步聊天请求，返回解析后的JSON"""
        response_text = await self.chat(system_prompt, user_prompt, temperature)
        return self._parse_json_response(response_text)

    async def batch_chat(
        self,
        tasks: List[Dict[str, Any]],
        temperature: Optional[float] = None,
    ) -> List[str]:
        """
        批量并发聊天

        Args:
            tasks: [{"system_prompt": ..., "user_prompt": ...}, ...]

        Returns:
            与tasks等长的结果列表

        示例：
            results = await client.batch_chat([
                {"system_prompt": "你是RST分析专家", "user_prompt": "分析段落1..."},
                {"system_prompt": "你是RST分析专家", "user_prompt": "分析段落2..."},
                {"system_prompt": "你是RST分析专家", "user_prompt": "分析段落3..."},
            ])
            # 3个请求并发发出，而非串行等待
        """
        coros = [
            self.chat(t["system_prompt"], t["user_prompt"], temperature)
            for t in tasks
        ]
        return await asyncio.gather(*coros)

    async def batch_chat_json(
        self,
        tasks: List[Dict[str, Any]],
        temperature: Optional[float] = None,
    ) -> List[dict]:
        """批量并发聊天，返回JSON列表"""
        coros = [
            self.chat_json(t["system_prompt"], t["user_prompt"], temperature)
            for t in tasks
        ]
        return await asyncio.gather(*coros)

    async def _request_with_retry(
        self, system_prompt: str, user_prompt: str, temperature: float
    ) -> str:
        """带重试的HTTP请求"""
        payload = {
            "model": self.config.model_name,
            "messages": [
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_prompt},
            ],
            "temperature": temperature,
            "max_tokens": self.config.max_tokens,
        }

        last_error = None
        for attempt in range(self.config.max_retries):
            t0 = time.monotonic()
            try:
                response = await self._client.post("/chat/completions", json=payload)
                elapsed = time.monotonic() - t0
                self._stats["total_requests"] += 1
                self._stats["total_time"] += elapsed

                if response.status_code == 200:
                    data = response.json()
                    result = data["choices"][0]["message"]["content"]
                    logger.debug(f"LLM响应: {elapsed:.2f}s, {len(result)}字符")
                    return result

                elif response.status_code == 429:
                    # 限流，等待更长时间
                    retry_after = float(response.headers.get("retry-after", "5"))
                    logger.warning(f"API限流，等待{retry_after}s后重试...")
                    await asyncio.sleep(retry_after)
                    self._stats["retries"] += 1
                    continue

                elif response.status_code >= 500:
                    # 服务端错误，指数退避重试
                    delay = min(
                        self.config.retry_base_delay * (2 ** attempt),
                        self.config.retry_max_delay,
                    )
                    logger.warning(
                        f"服务端错误 {response.status_code}，{delay:.1f}s后重试 "
                        f"({attempt+1}/{self.config.max_retries})"
                    )
                    await asyncio.sleep(delay)
                    self._stats["retries"] += 1
                    continue

                else:
                    # 其他错误，不重试
                    raise RuntimeError(
                        f"API错误: {response.status_code} - {response.text[:200]}"
                    )

            except (ConnectionError, TimeoutError, asyncio.TimeoutError) as e:
                last_error = e
                delay = min(
                    self.config.retry_base_delay * (2 ** attempt),
                    self.config.retry_max_delay,
                )
                logger.warning(f"连接错误，{delay:.1f}s后重试: {e}")
                await asyncio.sleep(delay)
                self._stats["retries"] += 1
                continue

        raise RuntimeError(f"请求失败，已重试{self.config.max_retries}次: {last_error}")

    @staticmethod
    def _parse_json_response(text: str) -> dict:
        """从LLM回复中提取JSON"""
        text = text.strip()
        if text.startswith("```json"):
            text = text[7:]
        elif text.startswith("```"):
            text = text[3:]
        if text.endswith("```"):
            text = text[:-3]
        text = text.strip()

        try:
            return json.loads(text)
        except json.JSONDecodeError:
            start = text.find("{")
            end = text.rfind("}") + 1
            if start >= 0 and end > start:
                try:
                    return json.loads(text[start:end])
                except json.JSONDecodeError:
                    pass
            start = text.find("[")
            end = text.rfind("]") + 1
            if start >= 0 and end > start:
                try:
                    return {"data": json.loads(text[start:end])}
                except json.JSONDecodeError:
                    pass
            logger.warning(f"无法解析JSON: {text[:200]}...")
            return {"raw_response": text}

    def get_stats(self) -> dict:
        """获取客户端统计信息"""
        stats = dict(self._stats)
        if stats["total_requests"] > 0:
            stats["avg_time"] = stats["total_time"] / stats["total_requests"]
        else:
            stats["avg_time"] = 0
        stats["cache_size"] = len(self._cache)
        stats["cache_hit_rate"] = (
            stats["cache_hits"] / (stats["total_requests"] + stats["cache_hits"])
            if (stats["total_requests"] + stats["cache_hits"]) > 0
            else 0
        )
        return stats


class _TokenBucket:
    """
    Token Bucket 限流器

    原理：
    ┌─────────────────────────────────┐
    │  桶容量 = requests_per_minute   │
    │  每秒补充 = rpm / 60 个token    │
    │                                 │
    │  请求前取1个token:              │
    │  - 桶里有 → 直接通过            │
    │  - 桶空   → 等到补充再通过      │
    └─────────────────────────────────┘
    """

    def __init__(self, rpm: int = 0):
        self.rpm = rpm
        if rpm > 0:
            self.capacity = rpm
            self.tokens = float(rpm)
            self.refill_rate = rpm / 60.0  # 每秒补充数
            self._last_refill = time.monotonic()
            self._lock = asyncio.Lock()

    async def acquire(self):
        """获取一个token，如果桶空则等待"""
        if self.rpm <= 0:
            return

        async with self._lock:
            now = time.monotonic()
            elapsed = now - self._last_refill
            self.tokens = min(self.capacity, self.tokens + elapsed * self.refill_rate)
            self._last_refill = now

            if self.tokens >= 1.0:
                self.tokens -= 1.0
                return

            # 需要等待
            wait_time = (1.0 - self.tokens) / self.refill_rate
            await asyncio.sleep(wait_time)
            self.tokens = 0
            self._last_refill = time.monotonic()
