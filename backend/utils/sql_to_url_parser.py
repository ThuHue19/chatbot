# utils/sql_to_url_parser.py
import re
import json
import os
import logging
from datetime import datetime

logger = logging.getLogger(__name__)

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
URL_MAPPING_FILE = os.path.join(BASE_DIR, "url_mapping.json")

URL_CONFIG = {}
try:
    with open(URL_MAPPING_FILE, "r", encoding="utf-8") as f:
        URL_CONFIG = json.load(f)
except FileNotFoundError:
    logger.error(f"Error: {URL_MAPPING_FILE} not found. Please create it.")
except json.JSONDecodeError:
    logger.error(f"Error: Could not decode JSON from {URL_MAPPING_FILE}. Check file syntax.")

def get_mapped_sql_value(param_name: str, sql_value: str) -> str:
    """Ánh xạ giá trị SQL sang giá trị URL nếu có định nghĩa."""
    mapping_table = URL_CONFIG.get("sql_value_to_url_value_mapping", {}).get(param_name, {})
    return mapping_table.get(sql_value, sql_value)

def extract_url_params_from_sql(sql_query: str) -> dict:
    """
    Trích xuất các điều kiện lọc từ câu truy vấn SQL và ánh xạ sang tham số URL.
    Hỗ trợ các điều kiện WHERE cơ bản (AND, =, LIKE, IN, >=, <=).
    """
    params = {}
    sql_upper = sql_query.upper()
    
    # Tìm kiếm mệnh đề WHERE
    where_match = re.search(r"WHERE\s+(.*)", sql_upper, re.DOTALL)
    if not where_match:
        return params

    where_clause = where_match.group(1).strip()
    
    # Tách các điều kiện bằng AND (đơn giản, có thể cần phức tạp hơn cho OR)
    conditions = re.split(r'\s+AND\s+', where_clause)

    sql_param_mapping = URL_CONFIG.get("sql_to_url_param_mapping", {})

    for condition in conditions:
        condition = condition.strip()
        
        # Regex cho các loại điều kiện: COLUMN = 'VALUE', COLUMN LIKE '%VALUE%', COLUMN IN ('VAL1', 'VAL2'), COLUMN >= DATE, COLUMN <= DATE
        match_eq = re.match(r"(\w+\.\w+)\s*=\s*['\"]?([^'\"]+)['\"]?", condition)
        match_like = re.match(r"(\w+\.\w+)\s+LIKE\s+['\"]?%?([^'%]+)%?['\"]?", condition)
        match_in = re.match(r"(\w+\.\w+)\s+IN\s+\(([^)]+)\)", condition)
        # Regex cho TO_DATE('DD/MM/YYYY')
        match_date_gte = re.match(r"(\w+\.\w+)\s*>=\s*TO_DATE\('(\d{2}/\d{2}/\d{4})','DD/MM/YYYY'\)", condition)
        match_date_lte = re.match(r"(\w+\.\w+)\s*<=\s*TO_DATE\('(\d{2}/\d{2}/\d{4})','DD/MM/YYYY'\)", condition)
        
        sql_full_col_name = None
        sql_value = None

        if match_eq:
            sql_full_col_name = match_eq.group(1)
            sql_value = match_eq.group(2)
        elif match_like:
            sql_full_col_name = match_like.group(1)
            sql_value = match_like.group(2)
        elif match_in:
            sql_full_col_name = match_in.group(1)
            sql_value = match_in.group(2).replace("'", "").replace(" ", "").split(",")
            sql_value = "%2C".join(sql_value) # Mã hóa cho URL
        elif match_date_gte:
            sql_full_col_name = match_date_gte.group(1)
            sql_value = match_date_gte.group(2)
            # Dành riêng cho ngày tháng: ánh xạ THOI_GIAN_GHI_NHAN thành startDate
            # Cần ánh xạ cụ thể trong sql_to_url_param_mapping
            
        elif match_date_lte:
            sql_full_col_name = match_date_lte.group(1)
            sql_value = match_date_lte.group(2)
            # Dành riêng cho ngày tháng: ánh xạ THOI_GIAN_GHI_NHAN thành endDate
        
        if sql_full_col_name and sql_value is not None:
            table_name, col_name = sql_full_col_name.split('.')
            
            if table_name in sql_param_mapping:
                col_map = sql_param_mapping[table_name]

                if col_name in col_map:
                    # Xử lý trường hợp ngày tháng đặc biệt
                    if isinstance(col_map[col_name], dict) and col_name == "THOI_GIAN_GHI_NHAN":
                        if match_date_gte:
                            url_param_name = col_map[col_name].get("start")
                        elif match_date_lte:
                            url_param_name = col_map[col_name].get("end")
                        else:
                            url_param_name = None # Fallback
                    else:
                        url_param_name = col_map[col_name]
                    
                    if url_param_name:
                        # Ánh xạ giá trị SQL sang giá trị URL nếu cần
                        mapped_value = get_mapped_sql_value(url_param_name, sql_value)
                        params[url_param_name] = mapped_value
                else:
                    logger.debug(f"SQL column '{col_name}' from table '{table_name}' not found in sql_to_url_param_mapping.")
            else:
                logger.debug(f"Table '{table_name}' not found in sql_to_url_param_mapping.")

    # Áp dụng các quy tắc phức tạp từ URL_CONFIG (ví dụ: cho tab "quá hạn")
    # Các quy tắc này vẫn dựa trên logic tổng thể, có thể không trực tiếp từ SQL
    # nhưng bổ sung các tham số mặc định cho URL.
    
    # Kiểm tra và áp dụng quy tắc nếu có tab_from_status được trích xuất
    if params.get("tab_from_status"):
        inferred_tab = params["tab_from_status"]
        
        # Tìm trong các quy tắc phức tạp
        for rule in URL_CONFIG.get("complex_rules", []):
            if inferred_tab in rule.get("trigger_keywords", []): # Giả định trigger_keywords có thể chứa giá trị tab
                for param_key, param_value in rule.get("set_params", {}).items():
                    if param_key == "tab": # Đảm bảo tab chính thức được đặt
                        params[param_key] = param_value
                    elif param_key not in params: # Chỉ thêm nếu chưa có
                        params[param_key] = param_value
                break
        
        # Nếu không có quy tắc phức tạp nào phù hợp, gán tab từ status
        if "tab" not in params:
            params["tab"] = inferred_tab
            
    # Xóa tham số trung gian
    params.pop("tab_from_status", None)

    return params