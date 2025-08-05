import oracledb
import os
from dotenv import load_dotenv
import logging
import re
from typing import List, Dict, Union

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

load_dotenv()

def get_connection():
    try:
        conn = oracledb.connect(
            user=os.getenv("DB_USER"),
            password=os.getenv("DB_PASSWORD"),
            dsn=os.getenv("DB_DSN")
        )
        logger.info("✅ Database connection established.")
        return conn
    except oracledb.DatabaseError as e:
        error, = e.args
        logger.error(f"❌ Database connection error: {error.code} - {error.message}")
        raise Exception(f"Lỗi kết nối CSDL: {error.code} - {error.message}")

def extract_sql_code(raw_sql: str) -> List[str]:
    matches = re.findall(r"```sql(.*?)```", raw_sql, re.DOTALL | re.IGNORECASE)
    sqls = []
    if matches:
        for match in matches:
            sql = match.strip()
            if sql.endswith(";"):
                sql = sql[:-1].strip()
            sqls.append(sql)
    else:
        if raw_sql.endswith(";"):
            raw_sql = raw_sql[:-1].strip()
        sqls.append(raw_sql)
    return sqls

def execute_sql(sql_input: str) -> Union[Dict, List[Dict]]:
    sql_statements = extract_sql_code(sql_input)
    all_results = []

    for sql_clean in sql_statements:
        logger.info(f"🔎 Executing SQL:\n{sql_clean}")
        try:
            with get_connection() as conn:
                with conn.cursor() as cur:
                    cur.execute(sql_clean)
                    if cur.description:
                        columns = [desc[0] for desc in cur.description]
                        raw_rows = cur.fetchall()
                        rows = []
                        for row in raw_rows:
                            new_row = []
                            for val in row:
                                if isinstance(val, oracledb.LOB):
                                    new_row.append(val.read())
                                else:
                                    new_row.append(val)
                            rows.append(tuple(new_row))
                        result = {"columns": columns, "rows": rows}
                        logger.info(f"✅ SQL execution successful with {len(rows)} rows.")
                        all_results.append(result)
                    else:
                        message = "✅ Thực thi thành công, không có dữ liệu trả về."
                        logger.info(message)
                        all_results.append({"message": message})
                conn.commit()
        except oracledb.DatabaseError as e:
            error, = e.args
            logger.error(f"❌ SQL error: {error.code} - {error.message}")
            # ✅ Raise exception để generate_and_execute_sql không cache SQL lỗi
            raise Exception(f"Lỗi CSDL: {error.code} - {error.message}")
        except Exception as e:
            logger.error(f"❌ Unexpected error: {str(e)}")
            raise Exception(f"Lỗi không xác định: {str(e)}")

    # Xử lý kết quả trả về
    if len(all_results) == 1:
        return all_results[0]
    elif len(all_results) > 1:
        count_result = all_results[0]
        detail_result = all_results[1]
        combined_result = {}
        if "rows" in count_result and count_result["rows"]:
            combined_result["count"] = count_result["rows"][0][0]
        if "rows" in detail_result:
            combined_result["details"] = detail_result
        return combined_result
    else:
        return {"message": "⚠️ Không có câu lệnh SQL nào được thực thi."}

def get_filter_link_by_keywords(keywords: list):
    conn = get_connection()
    cursor = conn.cursor()

    # Build single keyword string with wildcards for LIKE
    keyword_str = "%" + " ".join(keywords).lower() + "%"

    sql = """
        SELECT f.ID, f.FILTER
        FROM FB_FILTER f
        JOIN PAKH p ON p.ID = f.ID
        WHERE LOWER(p.NOI_DUNG_PHAN_ANH) LIKE :keyword
    """

    cursor.execute(sql, keyword=keyword_str)
    rows = cursor.fetchall()

    result = []
    for row in rows:
        filter_id, filter_name = row
        link = f"http://14.160.91.174:8180/smartw/feedback/list.htm?filter_id={filter_id}"
        result.append({"filter_id": filter_id, "link": link})

    cursor.close()
    conn.close()
    return result
