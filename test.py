import asyncio
from disco_rag.async_core import AsyncDiscoRAGPipeline, AsyncLLMConfig

async def main():
    config = AsyncLLMConfig(api_key="your-key", max_concurrency=10)
    
    async with AsyncDiscoRAGPipeline() as pipeline:
        # 单次提问
        answer = await pipeline.ask("维生素D能预防流感吗？", paragraphs=[...])
        
        # 批量并发
        # results = await pipeline.batch_ask(
        #     questions=["问题1", "问题2", "问题3"],
        #     paragraphs_list=[[...], [...], [...]],
        # )

asyncio.run(main())