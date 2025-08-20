from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from hybrid_sqlcoder import HybridSQLCoder
import logging
import re
from pydantic import BaseModel
from db_utils import get_connection
from db_utils import execute_sql
# Khởi tạo coder
sqlcoder = HybridSQLCoder(db_conn=get_connection())
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

app = FastAPI()
logger.info("Server initialization complete.")

from fastapi.middleware.cors import CORSMiddleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:3000", "*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

class QueryRequest(BaseModel):
    question: str
    is_independent: bool  # người dùng tick câu hỏi độc lập
    force_no_cache: bool = False

class ClearCacheRequest(BaseModel):
    question: str

class ManualSqlRequest(BaseModel):
    sql: str

@app.post("/query")
async def process_query(query: QueryRequest):
    logger.info(f"Processing query: {query.question}")
    try:
        final_answer = sqlcoder.response(
            question=query.question,
            execute_fn=lambda sql: execute_sql(sql),
            is_independent=query.is_independent,
            force_no_cache=query.force_no_cache,
        )

        # Lưu memory
        if sqlcoder.engine == "openai":
            sqlcoder.memory.chat_memory.add_user_message(query.question)
            sqlcoder.memory.chat_memory.add_ai_message(final_answer)
        else:
            sqlcoder.memory.append((query.question, final_answer))

        return {"question": query.question, "answer": final_answer}

    except Exception as e:   # 👈 thêm khối except
        logger.error(f"Error processing query: {str(e)}")
        raise HTTPException(status_code=500, detail={
            "error": str(e),
            "source": "query_processing"
        })

@app.post("/clear_cache")
async def clear_cache_for_question(request: ClearCacheRequest):
    logger.info(f"Yêu cầu xóa cache cho: {request.question}")
    sqlcoder.clear_specific_cache(request.question)
    return {"message": f"Đã xóa cache cho câu: '{request.question}'"}

@app.post("/execute_manual")
async def execute_manual_sql(request: ManualSqlRequest):
    logger.info(f"Thực thi SQL thủ công: {request.sql}")
    try:
        results = execute_sql(request.sql)
        return {"sql": request.sql, "result": results}
    except Exception as e:
        logger.error(f"Lỗi khi thực thi SQL thủ công: {e}")
        return {"sql": request.sql, "error": str(e)}
