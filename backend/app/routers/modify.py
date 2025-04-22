from fastapi import APIRouter, Request, HTTPException
from dotenv import load_dotenv
from anthropic._exceptions import RateLimitError
from app.prompts import SYSTEM_MODIFY_PROMPT
from pydantic import BaseModel
import logging
import os
from app.services.deepseek_chat_service import DeepseekChatService
import re

# 配置日志
log_dir = "/app/logs"
if not os.path.exists(log_dir):
    os.makedirs(log_dir)

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler(os.path.join(log_dir, "mermaid_modification.log")),
        logging.StreamHandler()  # 同时输出到控制台
    ]
)
logger = logging.getLogger(__name__)

load_dotenv()

router = APIRouter(prefix="/modify", tags=["DeepSeek"])

# Initialize services
deepseek_service = DeepseekChatService()


class ModifyRequest(BaseModel):
    instructions: str
    current_diagram: str
    repo: str
    username: str
    explanation: str


@router.post("")
async def modify(request: Request, body: ModifyRequest):
    try:
        # Check instructions length
        if not body.instructions or not body.current_diagram:
            return {"error": "Instructions and/or current diagram are required"}
        elif (
            len(body.instructions) > 1000 or len(body.current_diagram) > 100000
        ):  # just being safe
            return {"error": "Instructions exceed maximum length of 1000 characters"}

        if body.repo in [
            "fastapi",
            "streamlit",
            "flask",
            "api-analytics",
            "monkeytype",
        ]:
            return {"error": "Example repos cannot be modified"}

        # 记录修改请求
        logger.info(f"Modification requested for repo: {body.repo}")
        logger.info(f"Instructions: {body.instructions}")
        logger.info("\n============ Original Mermaid Code ============")
        logger.info(body.current_diagram)
        logger.info("=============================================\n")

        # 使用 DeepSeek 模型修改图表
        modified_mermaid_code = await deepseek_service.call_chat_api(
            system_prompt=SYSTEM_MODIFY_PROMPT,
            data={
                "instructions": body.instructions,
                "explanation": body.explanation,
                "diagram": body.current_diagram,
            },
            api_key=None,  # 使用默认的 API key
            reasoning_effort="low",  # 对于修改任务，low 级别就足够了
        )

        # Check for BAD_INSTRUCTIONS response
        if "BAD_INSTRUCTIONS" in modified_mermaid_code:
            logger.warning("Invalid or unclear instructions provided")
            return {"error": "Invalid or unclear instructions provided"}

        # 使用正则表达式提取 ```mermaid 和 ``` 之间的内容
        mermaid_pattern = r"```mermaid\n(.*?)```"
        match = re.search(mermaid_pattern, modified_mermaid_code, re.DOTALL)
        
        if match:
            modified_mermaid_code = match.group(1).strip()
        else:
            # 如果没有找到标准格式，记录原始内容并尝试清理
            logger.warning("No standard mermaid code block found, attempting to clean up the response")
            logger.warning("Original content:")
            logger.warning(modified_mermaid_code)
            # 移除所有 ```mermaid 和 ``` 标记作为后备方案
            modified_mermaid_code = modified_mermaid_code.replace("```mermaid", "").replace("```", "").strip()

        # 记录修改后的代码
        logger.info("\n============ Modified Mermaid Code ============")
        logger.info(modified_mermaid_code)
        logger.info("=============================================\n")

        return {"diagram": modified_mermaid_code}
    except RateLimitError as e:
        logger.error(f"Rate limit error: {str(e)}")
        raise HTTPException(
            status_code=429,
            detail="Service is currently experiencing high demand. Please try again in a few minutes.",
        )
    except Exception as e:
        logger.error(f"Error in modify endpoint: {str(e)}")
        return {"error": str(e)}