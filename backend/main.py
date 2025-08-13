from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from hybrid_sqlcoder import HybridSQLCoder 
from db_utils import execute_sql, get_detail_link_by_question
import logging
import re
from pydantic import BaseModel
from db_utils import get_connection
from db_utils import execute_sql, get_detail_link_by_question, fetch_one
...
sqlcoder = HybridSQLCoder(db_conn=get_connection())
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

app = FastAPI()
logger.info("Initializing LLM with LangChain or Gemini...")

logger.info("Server initialization complete.")

from fastapi.middleware.cors import CORSMiddleware

class ClearCacheRequest(BaseModel):
    question: str

class ManualSqlRequest(BaseModel):
    sql: str

app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:3000"],  # Hoặc ["http://localhost:3000"] nếu muốn an toàn hơn
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

class QueryRequest(BaseModel):
    question: str
    is_independent: bool  # ✅ người dùng tick câu hỏi độc lập
    force_no_cache: bool = False  # ✅ optional, mặc định False

@app.post("/query")
async def process_query(query: QueryRequest):
    logger.info(f"Processing query: {query.question}")
    try:
        # ✅ Reset memory nếu là câu hỏi độc lập
        if query.is_independent:
            sqlcoder.reset_context()

        # Bước 1: LLM sinh câu lệnh SQL và thực thi luôn (chỉ lưu cache nếu execute thành công)
        logger.info("Generating and executing SQL...")

        sql_raw = sqlcoder.generate_sql(
            question=query.question,
            is_follow_up=not query.is_independent,  # ✅ follow-up khi không tick hỏi mới
            force_no_cache=query.force_no_cache,
            reset=query.is_independent              # ✅ chỉ reset khi câu hỏi mới
        )
        logger.info(f"Generated SQL:\n{sql_raw}")

        # Bước 2: Tìm tất cả các block SQL
        sql_blocks = re.findall(r"```sql(.*?)```", sql_raw, re.DOTALL | re.IGNORECASE)
        if not sql_blocks:
            raise ValueError("Không tìm thấy câu lệnh SQL nào trong phản hồi LLM.")

        results = []
        answers = []

        for idx, sql_block in enumerate(sql_blocks, start=1):
            sql_clean = sql_block.strip()
            logger.info(f"Executing SQL #{idx}: {sql_clean}")

            result = sqlcoder.generate_and_execute_sql(
                question=query.question,
                is_follow_up=not query.is_independent,
                execute_fn=lambda _: execute_sql(sql_clean),
                force_no_cache=query.force_no_cache,
                reset=query.is_independent
            )
            logger.info(f"Type of result: {type(result)}")

            filter_link = get_detail_link_by_question(query.question)
            formatted_answer = sqlcoder.format_result_for_user(result, filter_link)

            # --- Gọi Groq để viết lại câu trả lời ---
            try:
                groq_answer = sqlcoder.response(query.question, result, filter_link)

                answers.append(groq_answer)
            except Exception as e:
                logger.error(f"Groq API error: {e}")
                answers.append(formatted_answer)

        final_answer = "\n\n".join(answers)

        # ✅ Lưu log hội thoại vào memory
        if sqlcoder.engine == "openai":
            sqlcoder.memory.chat_memory.add_user_message(query.question)
            sqlcoder.memory.chat_memory.add_ai_message(final_answer)
        elif sqlcoder.engine == "gemini":
            sqlcoder.memory.append((query.question, final_answer))

        # Bước 3: Trả về kết quả từng câu
        return {
            "question": query.question,
            "answer": final_answer
        }

    except Exception as e:
        logger.error(f"Error processing query: {str(e)}")
        raise HTTPException(status_code=500, detail={
            "error": str(e),
            "source": "query_processing"
        })

@app.post("/clear_cache")
async def clear_cache_for_question(request: ClearCacheRequest):
    logger.info(f"Yêu cầu xóa cache cho câu hỏi: {request.question}")
    sqlcoder.clear_specific_cache(request.question)
    return {"message": f"Đã gửi yêu cầu xóa cache cho câu hỏi: '{request.question}'"}

@app.post("/execute_manual")
async def execute_manual_sql(request: ManualSqlRequest):
    logger.info(f"Yêu cầu thực thi SQL thủ công: {request.sql}")
    try:
        results = execute_sql(request.sql)
        return {"sql": request.sql, "result": results}
    except Exception as e:
        logger.error(f"Lỗi khi thực thi SQL thủ công: {e}")
        return {"sql": request.sql, "error": str(e)}