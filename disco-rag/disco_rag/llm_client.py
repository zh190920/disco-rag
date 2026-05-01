"""
LLM客户端 - 统一封装大模型调用接口

支持：
- OpenAI兼容API（包括本地部署的vLLM、Ollama等）
- 可扩展对接其他LLM服务
"""

import json
import logging
from typing import Optional
from dataclasses import dataclass, field

logger = logging.getLogger(__name__)


@dataclass
class LLMConfig:
    """LLM配置"""
    api_base: str = "https://open.bigmodel.cn/api/paas/v4"
    api_key: str = ""
    model_name: str = "glm-4-flash"
    temperature: float = 0.1
    max_tokens: int = 4096
    timeout: int = 120

    # 备选配置（如果主模型不可用）
    fallback_api_base: Optional[str] = None
    fallback_api_key: Optional[str] = None
    fallback_model_name: Optional[str] = None


class LLMClient:
    """
    统一的LLM调用客户端
    
    支持任何OpenAI兼容的API接口，包括：
    - OpenAI官方API
    - 智谱AI (BigModel)
    - 本地vLLM / Ollama
    - 其他兼容服务
    """

    def __init__(self, config: Optional[LLMConfig] = None):
        self.config = config or LLMConfig()
        self._client = None
        self._init_client()

    def _init_client(self):
        """初始化OpenAI兼容客户端"""
        try:
            from openai import OpenAI
            self._client = OpenAI(
                api_key=self.config.api_key or "dummy-key",
                base_url=self.config.api_base,
                timeout=self.config.timeout,
            )
            logger.info(f"LLM客户端初始化成功: {self.config.api_base} / {self.config.model_name}")
        except ImportError:
            logger.warning("openai包未安装，请运行: pip install openai")
            self._client = None

    def chat(self, system_prompt: str, user_prompt: str, temperature: Optional[float] = None) -> str:
        """
        发送聊天请求并获取回复
        
        Args:
            system_prompt: 系统提示词
            user_prompt: 用户输入
            temperature: 生成温度（可选，覆盖默认值）
        
        Returns:
            模型的文本回复
        """
        if self._client is None:
            raise RuntimeError("LLM客户端未初始化，请检查openai包是否安装")

        messages = [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_prompt},
        ]

        temp = temperature if temperature is not None else self.config.temperature

        try:
            response = self._client.chat.completions.create(
                model=self.config.model_name,
                messages=messages,
                temperature=temp,
                max_tokens=self.config.max_tokens,
            )
            result = response.choices[0].message.content
            logger.debug(f"LLM响应长度: {len(result)} 字符")
            return result
        except Exception as e:
            logger.error(f"LLM调用失败: {e}")
            # 尝试备选配置
            if self.config.fallback_api_base:
                logger.info("尝试使用备选LLM配置...")
                return self._fallback_chat(messages, temp)
            raise

    def _fallback_chat(self, messages: list, temperature: float) -> str:
        """使用备选配置进行调用"""
        from openai import OpenAI
        fallback_client = OpenAI(
            api_key=self.config.fallback_api_key or "dummy-key",
            base_url=self.config.fallback_api_base,
            timeout=self.config.timeout,
        )
        response = fallback_client.chat.completions.create(
            model=self.config.fallback_model_name,
            messages=messages,
            temperature=temperature,
            max_tokens=self.config.max_tokens,
        )
        return response.choices[0].message.content

    def chat_json(self, system_prompt: str, user_prompt: str, temperature: Optional[float] = None) -> dict:
        """
        发送聊天请求并解析JSON回复
        
        Args:
            system_prompt: 系统提示词
            user_prompt: 用户输入
            temperature: 生成温度
        
        Returns:
            解析后的字典
        """
        response_text = self.chat(system_prompt, user_prompt, temperature)
        return self._parse_json_response(response_text)

    @staticmethod
    def _parse_json_response(text: str) -> dict:
        """从LLM回复中提取JSON"""
        # 尝试直接解析
        text = text.strip()
        
        # 去除markdown代码块包裹
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
            # 尝试提取JSON部分
            start = text.find("{")
            end = text.rfind("}") + 1
            if start >= 0 and end > start:
                try:
                    return json.loads(text[start:end])
                except json.JSONDecodeError:
                    pass
            
            # 尝试找数组
            start = text.find("[")
            end = text.rfind("]") + 1
            if start >= 0 and end > start:
                try:
                    return {"data": json.loads(text[start:end])}
                except json.JSONDecodeError:
                    pass

            logger.warning(f"无法解析JSON回复: {text[:200]}...")
            return {"raw_response": text}
