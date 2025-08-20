import oracledb
import os
import re
from dotenv import load_dotenv
import logging
from typing import List, Dict, Union, Optional
import json

from difflib import get_close_matches


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
    import re
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

                        # Đọc LOB nếu có
                        rows = [
                            tuple(val.read() if isinstance(val, oracledb.LOB) else val for val in row)
                            for row in raw_rows
                        ]

                        # ✅ trả về toàn bộ dữ liệu
                        result = {"columns": columns, "rows": rows, "total_rows": len(rows)}

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
            raise Exception(f"Lỗi CSDL: {error.code} - {error.message}")
        except Exception as e:
            logger.error(f"❌ Unexpected error: {str(e)}")
            raise Exception(f"Lỗi không xác định: {str(e)}")

    # Nếu nhiều câu lệnh SQL, giữ nguyên kết quả từng câu lệnh
    if len(all_results) == 1:
        return all_results[0]
    return all_results


def detect_dept_column(user_input: str) -> Optional[str]:
    """
    Nhận diện xem user nói 'phòng' hay 'tổ' để chọn đúng cột.
    """
    text = user_input.lower()
    if "phòng" in text:
        return "PHONG_BAN"
    elif "tổ" in text:
        return "TO_NHOM"
    return None
def resolve_dept_code(user_input: str, db_conn) -> Optional[tuple]:
    """
    Từ user_input tìm DEPT_CODE tương ứng trong M_DEPARTMENT.
    Trả về (dept_code, dept_name).
    """
    cursor = db_conn.cursor()
    cursor.execute("SELECT DEPT_CODE, DEPT_NAME FROM M_DEPARTMENT")
    rows = cursor.fetchall()

    dept_map = {normalize_canhan(name): (code, name) for code, name in rows}
    norm_input = normalize_canhan(user_input)

    matches = get_close_matches(norm_input, dept_map.keys(), n=1, cutoff=0.7)
    if matches:
        return dept_map[matches[0]]
    return None
def build_dept_condition(user_input: str) -> Optional[tuple]:
    """
    Trả về (sql_condition, dept_name) nếu tìm được phòng/tổ từ input.
    sql_condition: ví dụ "PAKH_TO_NHOM.TO_NHOM = 'DVT_DN_TVT_QN'"
    dept_name: tên hiển thị của phòng/tổ
    """
    column = detect_dept_column(user_input)
    if not column:
        return None

    try:
        with get_connection() as conn:
            match = resolve_dept_code(user_input, conn)
            if match:
                dept_code, dept_name = match
                if column == "PHONG_BAN":
                    sql_condition = f"PAKH_PHONG_BAN.PHONG_BAN = '{dept_code}'"
                else:
                    sql_condition = f"PAKH_TO_NHOM.TO_NHOM = '{dept_code}'"
                return sql_condition, dept_name
    except Exception as e:
        logger.error(f"Lỗi trong build_dept_condition: {e}")
        return None

def fetch_one(sql: str, params: list) -> dict:
    try:
        with get_connection() as conn:
            with conn.cursor() as cur:
                cur.execute(sql, params)
                row = cur.fetchone()
                if row and cur.description:
                    columns = [desc[0] for desc in cur.description]
                    return dict(zip(columns, row))
                return None
    except Exception as e:
        logger.error(f"Lỗi trong fetch_one: {e}")
        return None

CACHE_JSON_FILE = "locality_cache.json"

def build_locality_cache_json() -> dict:
    sql = "SELECT FULL_NAME, PROVINCE, DISTRICT FROM FB_LOCALITY"
    
    try:
        result = execute_sql(sql)
        rows = result.get("rows", [])
        columns = result.get("columns", [])
        
        cache = {}
        for row in rows:
            row_dict = dict(zip(columns, row))
            full_name = row_dict.get("FULL_NAME")
            province_code = row_dict.get("PROVINCE")
            district_code = row_dict.get("DISTRICT")
            
            if full_name:
                # Tách huyện/quận
                district_name = None
                if " H." in full_name:
                    district_name = full_name.split(" H.")[1]
                    if " T." in district_name:
                        district_name = district_name.split(" T.")[0]
                    elif " TP." in district_name:
                        district_name = district_name.split(" TP.")[0]
                    district_name = district_name.strip()
                elif " Q." in full_name:
                    district_name = full_name.split(" Q.")[1]
                    if " T." in district_name:
                        district_name = district_name.split(" T.")[0]
                    elif " TP." in district_name:
                        district_name = district_name.split(" TP.")[0]
                    district_name = district_name.strip()

                # Tách tỉnh/thành phố
                province_name = None
                if " T." in full_name:
                    province_name = full_name.split(" T.")[1].strip()
                elif " TP." in full_name:
                    province_name = full_name.split(" TP.")[1].strip()
                
                if district_name and province_name:
                    cache[district_name] = {
                        "district_code": district_code,
                        "province_code": province_code,
                        "district_name": district_name,
                        "province_name": province_name
                    }
                    # Thêm mapping tỉnh → tỉnh_code, chỉ thêm một lần
                if province_name and province_name:
                    cache[province_name] = {
                        "district_code": None,
                        "province_code": province_code,
                        "district_name": None,
                        "province_name": province_name
                    }
        
        with open(CACHE_JSON_FILE, "w", encoding="utf-8") as f:
            json.dump(cache, f, ensure_ascii=False, indent=2)
        
        logger.info(f"✅ Cache FB_LOCALITY đã lưu vào {CACHE_JSON_FILE} với {len(cache)} bản ghi.")
        return cache
    except Exception as e:
        logger.error(f"❌ Lỗi khi build cache JSON: {e}")
        return {}


def load_locality_cache_json() -> dict:
    """
    Load cache JSON nếu tồn tại, nếu chưa có sẽ tạo mới.
    """
    if os.path.exists(CACHE_JSON_FILE):
        with open(CACHE_JSON_FILE, "r", encoding="utf-8") as f:
            cache = json.load(f)
        logger.info(f"✅ Cache FB_LOCALITY đã được load từ {CACHE_JSON_FILE}.")
        return cache
    else:
        logger.info("⚠️ Cache JSON chưa tồn tại, tạo mới từ DB...")
        return build_locality_cache_json()

def build_dept_cache_json() -> list:
    """
    Lấy toàn bộ DEPT_CODE, DEPT_NAME từ M_DEPARTMENT
    và lưu ra file JSON dạng list[dict].
    """
    sql = "SELECT DEPT_CODE, DEPT_NAME FROM M_DEPARTMENT"

    try:
        result = execute_sql(sql)
        rows = result.get("rows", [])
        columns = result.get("columns", [])
        
        departments = []
        for row in rows:
            row_dict = dict(zip(columns, row))
            dept_code = row_dict.get("DEPT_CODE")
            dept_name = row_dict.get("DEPT_NAME")
            if dept_code and dept_name:
                departments.append({
                    "dept_code": dept_code,
                    "dept_name": dept_name
                })

        with open(DEPT_CACHE_JSON_FILE, "w", encoding="utf-8") as f:
            json.dump(departments, f, ensure_ascii=False, indent=2)

        logger.info(f"✅ Cache M_DEPARTMENT đã lưu vào {DEPT_CACHE_JSON_FILE} với {len(departments)} bản ghi.")
        return departments

    except Exception as e:
        logger.error(f"❌ Lỗi khi build dept cache JSON: {e}")
        return []



DEPT_CACHE_JSON_FILE = "departments.json"
DEPT_CACHE = []  # cache toàn cục

def load_dept_cache_json() -> list:
    """
    Load cache departments từ JSON, nếu chưa có thì tạo mới từ DB.
    """
    global DEPT_CACHE
    if DEPT_CACHE:  # nếu đã load rồi thì không cần đọc lại
        return DEPT_CACHE

    if os.path.exists(DEPT_CACHE_JSON_FILE):
        with open(DEPT_CACHE_JSON_FILE, "r", encoding="utf-8") as f:
            DEPT_CACHE = json.load(f)
        logger.info(f"✅ Cache M_DEPARTMENT đã được load từ {DEPT_CACHE_JSON_FILE}.")
    else:
        logger.info("⚠️ Chưa có cache departments.json, tạo mới từ DB...")
        DEPT_CACHE = build_dept_cache_json()

    return DEPT_CACHE

