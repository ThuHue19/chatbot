import os
import json
from rapidfuzz import fuzz, process

SCHEMA_DIR = "schema"
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
KEYWORDS_FILE = os.path.join(BASE_DIR, "table_keywords.json")

with open(KEYWORDS_FILE, "r", encoding="utf-8") as f:
    TABLE_KEYWORDS = json.load(f)

def load_schema(table_name):
    txt_path = os.path.join(SCHEMA_DIR, f"{table_name}.txt")
    json_path = os.path.join(SCHEMA_DIR, f"{table_name}.json")

    if os.path.exists(json_path):
        with open(json_path, "r", encoding="utf-8") as f:
            data = json.load(f)

        lines = [f"# TABLE: {data.get('table', table_name)}"]

        if "select_instructions" in data:
            lines.append("\nSELECT_INSTRUCTIONS:")
            lines.append(f"{data['select_instructions']}")

        if "columns" in data:
            lines.append("COLUMNS:")
            for col, info in data["columns"].items():
                col_type = info.get("type", "UNKNOWN")
                desc = info.get("description", "")
                lines.append(f"- {col} ({col_type}): {desc}")

        if "aggregate_columns" in data:
            lines.append("\nAGGREGATE_COLUMNS:")
            for col, desc in data["aggregate_columns"].items():
                lines.append(f"- {col} (NUMBER): {desc}")

        if "note" in data:
            lines.append("\nNOTE:")
            lines.append(f"- {data['note']}")

        # ✅ Thêm dữ liệu mẫu nếu có
        if "sample_data" in data and isinstance(data["sample_data"], list):
            lines.append("\nSAMPLE DATA (Top 10 rows):")
            for i, row in enumerate(data["sample_data"], 1):
                # Định dạng dữ liệu mẫu thành bảng dễ đọc
                sample_str = " | ".join(f"{k}: {v}" for k, v in row.items())
                lines.append(f"{i}. {sample_str}")

        return "\n".join(lines)

    elif os.path.exists(txt_path):
        with open(txt_path, "r", encoding="utf-8") as f:
            return f"# TABLE: {table_name}\n" + f.read().strip()

    return ""

def extract_column_names(columns_data):
    """Hỗ trợ cả 2 dạng columns: dict hoặc list"""
    if isinstance(columns_data, dict):
        return list(columns_data.keys())
    elif isinstance(columns_data, list):
        return [col["name"] for col in columns_data if "name" in col]
    else:
        return []

def load_all_schemas() -> dict:
    """
    Đọc toàn bộ schema trong folder schema/
    Trả về dict {table_name: [list_columns]}
    """
    schema_dict = {}
    for file in os.listdir(SCHEMA_DIR):
        if file.endswith(".json"):
            table_name = file.replace(".json", "")
            with open(os.path.join(SCHEMA_DIR, file), "r", encoding="utf-8") as f:
                data = json.load(f)
                columns_data = data.get("columns", [])
                schema_dict[table_name] = extract_column_names(columns_data)
    return schema_dict


def extract_relevant_tables(question: str):
    question = question.lower()
    candidate_tables = set()
    
    for table, data in TABLE_KEYWORDS.items():
        table_kw = data.get("keywords", [])
        col_kw_map = data.get("columns", {})

        # Check table-level keywords
        if any(kw.lower() in question for kw in table_kw):
            candidate_tables.add(table)
            continue

        # Check column-level keywords
        for col, col_data in col_kw_map.items():
            if any(kw.lower() in question for kw in col_data.get("keywords", [])):
                candidate_tables.add(table)
                break

    # Nếu không tìm thấy match rõ ràng → trả rỗng
    if not candidate_tables:
        return []

    final_tables = set()

    is_statistical_query = any(kw in question for kw in ["bao nhiêu", "tổng", "số lượng", "thống kê", "tỷ lệ", "đếm", "thời gian xử lý", "trung bình"])
    is_detail_listing_query = any(kw in question for kw in ["liệt kê", "danh sách", "chi tiết", "các phản ánh đó"]) # Thêm từ khóa cho liệt kê
    is_person_or_org_related = any(kw in question for kw in ["cá nhân", "người xử lý", "tổ nhóm", "phòng ban", "trung tâm"])

    # Logic ưu tiên cho các câu hỏi thống kê liên quan đến cá nhân/tổ chức/phòng ban/trung tâm
    if is_statistical_query and is_person_or_org_related:
        if "cá nhân" in question or "người xử lý" in question:
            final_tables.add("PAKH_SLA_CA_NHAN")
        elif "tổ nhóm" in question:
            # If "tổ nhóm" is explicitly mentioned, strongly prefer PAKH_SLA_TO_NHOM
            final_tables.add("PAKH_SLA_TO_NHOM")
        elif "phòng ban" in question and not "tổ nhóm" in question:
            # If "phòng ban" is mentioned but not "tổ nhóm", prefer PAKH_SLA_PHONG_BAN
            final_tables.add("PAKH_SLA_PHONG_BAN")        
        elif "trung tâm" in question:
            final_tables.add("PAKH_SLA_TRUNG_TAM")

        # Nếu một bảng SLA đã được xác định cho thống kê
        if final_tables:
            # Thêm các bảng không phải SLA mà vẫn cần thiết cho thông tin bổ sung (ví dụ: FB_GROUP, FB_TYPE_NOC)
            for table in candidate_tables:
                if not table.startswith("PAKH_SLA_") and table != "PAKH": # Đảm bảo không thêm PAKH nếu có SLA cho thống kê
                    final_tables.add(table)
            
            # QUAN TRỌNG: Nếu câu hỏi cũng yêu cầu LIỆT KÊ CHI TIẾT, hãy thêm PAKH và PAKH_CA_NHAN
            if is_detail_listing_query:
                final_tables.add("PAKH")
                final_tables.add("PAKH_CA_NHAN")

            return list(final_tables)
            
    # Nếu không phải là một truy vấn thống kê mạnh mẽ về cá nhân/tổ chức (hoặc không tìm thấy SLA phù hợp)
    # HOẶC chỉ là một truy vấn liệt kê chi tiết (không có thống kê SLA)
    # thì trả về tất cả các bảng ứng cử viên, bao gồm PAKH nếu có.
    return list(candidate_tables)

def load_schema_for_question(question) -> str:
    if isinstance(question, list):
        question = " ".join(question)
    question = question.lower()

    matched_tables = extract_relevant_tables(question)
    if not matched_tables:
        return ""
    schemas = [load_schema(tbl) for tbl in matched_tables if tbl]
    return "\n\n".join(schemas)

