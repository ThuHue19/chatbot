import os
import re
import logging
from dotenv import load_dotenv
from utils.sql_planner import SQLPlanner
from utils.column_mapper import extract_tables_and_columns, generate_column_mapping_hint
from utils.schema_loader import TABLE_KEYWORDS, extract_relevant_tables
from utils.relation_loader import load_relations
from typing import List, Dict
import json
import unidecode
from db_utils import get_connection

conn = get_connection()


# --- Setup môi trường & logger ---
load_dotenv()
logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO)

LLM_ENGINE = os.getenv("LLM_ENGINE", "openai").lower()
FORM_DETAIL_PARAMS = [
    "type", "trungTam", "phongBan", "toNhom", "caNhan", "loaiPa", "loaiThueBao",
    "nguyenNhan", "caTruc", "fo_bo", "tinh", "ctdv", "day", "today", "week",
    "month", "quarter", "year", "kpiName", "quanHuyen", "level", "nhomNguyenNhan"
]
LIST_PARAMS = [
    "tab", "id", "year", "tinhThanhPho", "quanHuyen", "is_ticket", "phong", "to", "caTruc",
    "dept_xuly", "dept", "ttTicket", "nhomPa", "loaiPa", "fo_bo", "loaiThueBao", "is_dongpa", "ctdv"
]
def normalize(text):
    return unidecode.unidecode(text).replace(' ', '_').replace('-', '_').lower()

def extract_form_detail_params_from_sql(sql: str, context: dict = None) -> Dict[str, str]:
    default_values = {k: "" for k in FORM_DETAIL_PARAMS}
    default_values.update({
        "type": "year",
        "today": "undefined",
        "week": "undefined",
        "month": "undefined",
        "quarter": "undefined",
    })
    params = default_values.copy()
    regex_map = {
        "trungTam": r"TRUNG_TAM\s*=\s*'([^']+)'",
        "phongBan": r"PHONG_BAN\s*=\s*'([^']+)'",
        "toNhom": r"TO_NHOM\s*=\s*'([^']+)'",
        "caNhan": r"CA_NHAN\s*=\s*'([^']+)'",
        "loaiPa": r"LOAI_PA\s*=\s*'([^']+)'",
        "loaiThueBao": r"LOAI_THUE_BAO\s*=\s*'([^']+)'",
        "nguyenNhan": r"NGUYEN_NHAN\s*=\s*'([^']+)'",
        "caTruc": r"CA_TRUC\s*=\s*'([^']+)'",
        "fo_bo": r"FO_BO\s*=\s*'([^']+)'",
        "tinh": r"TINH\s*=\s*'([^']+)'",
        "ctdv": r"CTDV\s*=\s*'([^']+)'",
        "day": r"DAY\s*=\s*'([^']+)'",
        "week": r"WEEK\s*=\s*'([^']+)'",
        "month": r"MONTH\s*=\s*'([^']+)'",
        "quarter": r"QUARTER\s*=\s*'([^']+)'",
        "year": r"YEAR\s*=\s*'([^']+)'",
        "kpiName": r"KPI_NAME\s*=\s*'([^']+)'",
        "quanHuyen": r"QUAN_HUYEN\s*=\s*'([^']+)'",
        "level": r"LEVEL\s*=\s*'([^']+)'",
        "nhomNguyenNhan": r"NHOM_NGUYEN_NHAN\s*=\s*'([^']+)'",
    }
    for key, regex in regex_map.items():
        match = re.search(regex, sql, re.IGNORECASE)
        if match:
            params[key] = match.group(1)
    if context:
        for key in params:
            if key in context and context[key]:
                params[key] = context[key]
    if not params["year"]:
        match = re.search(r"\b(20[2-3][0-9])\b", sql)
        if match:
            params["year"] = match.group(1)
        elif context and "year" in context and context["year"]:
            params["year"] = context["year"]
    # Gợi ý tự động cho type, level, kpiName nếu có từ khóa đặc biệt trong SQL
    if not params["kpiName"]:
        trang_thai_map = {
            "TU_CHOI": "soPaTuChoi",
            "DONG": "soPaDong",
            "DANG_XU_LY": "soPaDangXlDh",
            "DA_XU_LY": "soPaDaXlyDh",
            "HOAN_THANH": "soPaHoanThanh",
        }
        match = re.search(r"TRANG_THAI\s*=\s*'([^']+)'", sql, re.IGNORECASE)
        if match:
            trang_thai = match.group(1).upper()
            if trang_thai in trang_thai_map:
                params["kpiName"] = trang_thai_map[trang_thai]
    if not params["level"]:
        if "PAKH_CA_NHAN" in sql.upper():
            params["level"] = "ca_nhan"
        elif "PAKH_TO_NHOM" in sql.upper():
            params["level"] = "to_nhom"
        elif "PAKH_PHONG_BAN" in sql.upper():
            params["level"] = "phong_ban"
        elif "PAKH_TRUNG_TAM" in sql.upper():
            params["level"] = "trung_tam"
    return params

def extract_list_params_from_sql(sql: str, context: dict = None) -> Dict[str, str]:
    default_values = {k: "" for k in LIST_PARAMS}
    default_values.update({
        "tab": "year",
        "is_ticket": "Y%2CN",
        "is_dongpa": "DANG_XU_LY%2CDA_XU_LY",
    })
    params = default_values.copy()
    regex_map = {
        "tab": r"tab\s*=\s*'([^']+)'",
        "id": r"id\s*=\s*'([^']+)'",
        "year": r"year\s*=\s*'([^']+)'",
        "tinhThanhPho": r"TINH_THANH_PHO\s*=\s*'([^']+)'",
        "quanHuyen": r"QUAN_HUYEN\s*=\s*'([^']+)'",
        "is_ticket": r"is_ticket\s*=\s*'([^']+)'",
        "phong": r"PHONG\s*=\s*'([^']+)'",
        "to": r"TO_NHOM\s*=\s*'([^']+)'",
        "caTruc": r"CA_TRUC\s*=\s*'([^']+)'",
        "dept_xuly": r"DEPT_XULY\s*=\s*'([^']+)'",
        "dept": r"DEPT\s*=\s*'([^']+)'",
        "ttTicket": r"TT_TICKET\s*=\s*'([^']+)'",
        "nhomPa": r"NHOM_PA\s*=\s*'([^']+)'",
        "loaiPa": r"LOAI_PA\s*=\s*'([^']+)'",
        "fo_bo": r"FO_BO\s*=\s*'([^']+)'",
        "loaiThueBao": r"LOAI_THUE_BAO\s*=\s*'([^']+)'",
        "is_dongpa": r"is_dongpa\s*=\s*'([^']+)'",
        "ctdv": r"CTDV\s*=\s*'([^']+)'"
    }
    for key, regex in regex_map.items():
        match = re.search(regex, sql, re.IGNORECASE)
        if match:
            params[key] = match.group(1)
    if context:
        for key in params:
            # LUÔN ƯU TIÊN context nếu context có giá trị
            if key in context and context[key]:
                params[key] = context[key]
    return params

def build_filter_url(base: str, params: Dict[str, str]) -> str:
    query = "&".join(f"{k}={v}" for k, v in params.items())
    return f"{base}?{query}"

# --- Lớp Cache cho Schema ---
class SchemaCache:
    def __init__(self):
        self.cache = {}

    def get_schema(self, table_name):
        if table_name not in self.cache:
            from utils.schema_loader import load_schema
            self.cache[table_name] = load_schema(table_name)
        return self.cache[table_name]

    def clear(self):
        self.cache.clear()
        logger.info("Đã xóa schema cache.")

# --- Lớp chính HybridSQLCoder ---
class HybridSQLCoder:
    def __init__(self, db_conn=None):
        self.engine = LLM_ENGINE
        self.sql_cache = {}
        self.schema_cache = SchemaCache()
        self.relations, self.concat_mappings = load_relations()
        self.last_question = None
        self.last_sql = None
        self.last_result = None
        self.query_mode = None
        self.last_user_subject = None
        self.last_user_subjects = set()
        self.last_context = {}  # Tổng hợp mọi filter context
        self.db_conn = db_conn
        BASE_DIR = os.path.dirname(os.path.abspath(__file__))
        EXAMPLES_FILE = os.path.join(BASE_DIR, "utils", "sql_examples.json")
        self.sql_examples = self._load_sql_examples(EXAMPLES_FILE)

        if self.engine == "openai":
            self._init_openai()
        else:
            self._init_gemini()

    def _init_openai(self):
        from langchain_openai import ChatOpenAI
        from langchain.memory import ConversationSummaryMemory

        self.model_name = os.getenv("OPENAI_MODEL", "gpt-4")
        self.llm = ChatOpenAI(model=self.model_name, temperature=0)
        self.memory = ConversationSummaryMemory(
            llm=self.llm, memory_key="chat_history", return_messages=True
        )

    def _init_gemini(self):
        import google.generativeai as genai

        api_key = os.getenv("GEMINI_API_KEY")
        genai.configure(api_key=api_key)
        self.model = genai.GenerativeModel("models/gemini-2.5-flash", generation_config={"temperature": 0})
        self.memory = []  # list of (user_message, ai_message)

    def _fallback_to_gemini(self):
        logger.warning("OpenAI gặp lỗi, chuyển sang Gemini.")
        self._init_gemini()
        self.engine = "gemini"

    # --- Helper ---
    def _get_last_n_user_questions(self, n=5):
        if self.engine == "openai" and hasattr(self.memory, "chat_memory"):
            return [
                msg.content for msg in reversed(self.memory.chat_memory.messages)
                if msg.type == "human"
            ][:n][::-1]
        elif self.engine == "gemini":
            return [user for user, _ in self.memory[-n:]][::-1]
        return []

    def _update_context_from_sql(self, sql: str):
        # Dùng regex để bóc các filter phổ biến
        context = {}
        patterns = {
            "phongBan": r"PHONG_BAN\s*=\s*'([^']+)'",
            "toNhom": r"TO_NHOM\s*=\s*'([^']+)'",
            "trungTam": r"TRUNG_TAM\s*=\s*'([^']+)'",
            "caNhan": r"CA_NHAN\s*=\s*'([^']+)'",
            "tinhThanhPho": r"TINH_THANH_PHO\s*=\s*'([^']+)'",
            "quanHuyen": r"QUAN_HUYEN\s*=\s*'([^']+)'",
            "nhomPa": r"NHOM_PA\s*=\s*'([^']+)'",
            "loaiPa": r"LOAI_PA\s*=\s*'([^']+)'",
            "maTinh": r"MA_TINH\s*=\s*'([^']+)'",
            "maHuyen": r"MA_HUYEN\s*=\s*'([^']+)'",
            "level": r"LEVEL\s*=\s*'([^']+)'",
            "nhomNguyenNhan": r"NHOM_NGUYEN_NHAN\s*=\s*'([^']+)'",
        }
        for key, pat in patterns.items():
            m = re.search(pat, sql, re.IGNORECASE)
            if m:
                context[key] = m.group(1)
        if context:
            self.last_context.update(context)
    # File: hybrid_sqlcoder.py

    def get_ma_dia_chi_fuzzy(self, dia_chi_text: str) -> dict:
        if not self.db_conn:
            logger.warning("[WARN] DB connection is None. Không thể truy vấn mã địa lý.")
            return {}
        
        # Chuẩn hóa địa chỉ: chuyển lowercase, loại bỏ ký tự đặc biệt
        text = unidecode.unidecode(dia_chi_text).lower()
        text = re.sub(r"[^\w\s]", " ", text)
        keywords = [kw.strip() for kw in text.split() if kw.strip()]
        if not keywords:
            return {}

        # Tạo điều kiện WHERE động
        conditions = []
        params = []
        for i, kw in enumerate(keywords):
            # Thay thế unaccent(LOWER(full_name)) bằng LOWER(full_name) để tránh lỗi ORA-00904
            conditions.append(f"LOWER(full_name) LIKE :{i+1}")
            params.append(f"%{kw}%")
        
        where_clause = " AND ".join(conditions)
        query = f"""
            SELECT ma_tinh AS tinhThanhPho, ma_huyen AS quanHuyen 
            FROM FB_LOCALITY 
            WHERE {where_clause}
            ORDER BY LENGTH(full_name) DESC
            FETCH FIRST 1 ROWS ONLY
        """

        try:
            with self.db_conn.cursor() as cursor:
                cursor.execute(query, params)
                row = cursor.fetchone()
                if row:
                    result = {
                        "tinhThanhPho": row[0],
                        "quanHuyen": row[1]
                    }
                    logger.info(f"[DEBUG] Mapping địa chỉ '{dia_chi_text}' -> {result}")
                    return result
        except Exception as e:
            logger.error(f"Lỗi truy vấn mã địa lý: {e}")
        return {}

    def _extract_dia_chi_from_question(self, question: str) -> str:
        # Mở rộng pattern bắt địa chỉ
        patterns = [
            r"(?:tại|ở|địa chỉ|địa bàn|tại khu vực|tại)\s*([^,.;?]+)",
            r"(\b(?:quận|huyện|thị xã|tp\.?|tỉnh)\s+[\w\s]+)",
            r"([\w\s]+(?:,\s*[\w\s]+){1,2})"
        ]
        
        for pattern in patterns:
            match = re.search(pattern, question, re.IGNORECASE)
            if match:
                return match.group(1).strip()
        
        # Fallback: lấy cụm cuối cùng nếu không khớp pattern
        return " ".join(question.split()[-3:])
    def _detect_new_filter(self, question: str) -> bool:
        # Nếu câu hỏi có từ khóa về phòng ban, tổ nhóm, trung tâm, địa bàn, nhóm, loại...
        keywords = [
            "phòng", "phòng ban", "tổ", "tổ nhóm", "trung tâm", "địa bàn", "tỉnh", "huyện",
            "nhóm phản ánh", "loại phản ánh", "cá nhân", "số thuê bao", "mã tỉnh", "mã huyện"
        ]
        q = question.lower()
        return any(kw in q for kw in keywords)

    def _resolve_follow_up_question(self, question: str, is_follow_up: bool = False) -> str:
        """"Dùng LLM để diễn đạt lại câu hỏi tiếp nối thành câu đầy đủ, có thể hiểu độc lập"""
        if not is_follow_up:
            return question
        last_questions = self._get_last_n_user_questions(n=1)
        prompt  = f"""
Bạn là trợ lý ngôn ngữ.
Dưới đây là  câu hỏi trước của người dùng:
{os.linesep.join(f"{i+1}. {q}" for i, q in enumerate(last_questions))}

Câu hỏi hiện tại:
{question}

Hãy viết lại câu hỏi hiện tại thành một câu hỏi đầy đủ, rõ nghĩa, có thể hiểu độc lập. 
Nếu có từ như “đó”, “này”, “người đó”,... thì hãy thay thế bằng thông tin ngữ cảnh từ các câu hỏi trước.
Chỉ trả lại câu hỏi mới, không thêm giải thích.
"""
        resolved = self._invoke_model(prompt)
        return resolved.strip() if resolved else question

    def _format_memory_context(self):
        history = []
        if self.engine == "openai" and hasattr(self.memory, "chat_memory"):
            for m in self.memory.chat_memory.messages[-10:]:
                prefix = "Người dùng" if m.type == "human" else "Trợ lý"
                history.append(f"{prefix}: {m.content}")
        elif self.engine == "gemini":
            for user_msg, ai_msg in self.memory[-5:]:
                history.extend([f"Người dùng: {user_msg}", f"Trợ lý: {ai_msg}"])
        return os.linesep.join(history)

    def _is_statistical_list(self, question: str) -> bool:
        q = question.lower()
        return any(kw in q for kw in [
            "theo từng nhóm", "theo từng loại", "thống kê", "group by", "từng nhóm", "từng loại", "mỗi nhóm", "mỗi loại"
        ])

    def _format_relations_for_prompt(self):
        return os.linesep.join(
            f"- {table}.{col} → {ref['ref_table']}.{ref['ref_column']}"
            for table, cols in self.relations.items()
            for col, ref in cols.items()
        )

    def _generate_sum_hint(self) -> str:
        hints = [
            "- Đối với các bảng SLA (PAKH_SLA_*):",
            "   - Luôn dùng SUM() với các cột tổng hợp như SO_PA_NHAN, SO_PA_DA_XL, SO_PA_QH.",
            "   - Ánh xạ từ khóa trong câu hỏi sang cột tổng hợp:",
            "     - 'quá hạn' → SUM(SO_PA_QH) (Số phản ánh quá hạn)",
            "     - 'đã xử lý đúng hạn' → SUM(SO_PA_DA_XL_DH)",
            "     - 'đang xử lý đúng hạn' → SUM(SO_PA_DANG_XL_DH)",
            "     - 'từ chối' hoặc 'bị từ chối' → SUM(SO_PA_TU_CHOI)",
            "     - 'tổng thời gian xử lý' → SUM(TONG_TG_XL) / 60 (đổi sang giờ)",
            "     - 'thời gian trung bình xử lý' →  ROUND(SUM(TONG_TG_XL) / NULLIF(SUM(SO_PA_DA_XL), 0), 2) AS TB_PHUT, TRUNC(SUM(TONG_TG_XL) / NULLIF(SUM(SO_PA_DA_XL), 0) / 60) AS GIO, MOD(ROUND(SUM(TONG_TG_XL) / NULLIF(SUM(SO_PA_DA_XL), 0)), 60) AS PHUT",
            "   - TUYỆT ĐỐI KHÔNG JOIN bảng SLA trực tiếp với PAKH hoặc PAKH_CA_NHAN cho các truy vấn thống kê.",
            "- Công thức tính tỷ lệ xử lý đúng hạn: (SUM(SO_PA_DA_XL_DH) + SUM(SO_PA_DANG_XL_DH)) / NULLIF(SUM(SO_PA_NHAN), 0) * 100.",
            "- Dùng bảng SLA phù hợp với ngữ cảnh cá nhân, tổ nhóm, phòng ban, trung tâm",
            "- Lấy tên cá nhân trong cột CA_NHAN"
        ]
        return os.linesep.join(hints)

    def _extract_last_user_subject_from_result(self, result: dict):
        target_result = result.get("details") or result
        if not target_result or "columns" not in target_result or "rows" not in target_result:
            return
        try:
            for colname in ["CA_NHAN", "SO_THUE_BAO", "ID"]:
                if colname in target_result["columns"]:
                    idx = target_result["columns"].index(colname)
                    for row in target_result["rows"]:
                        value = str(row[idx])
                        if value:
                            self.last_user_subject = value
                            self.last_user_subjects.add(value)
                            logger.info(f"✅ Cập nhật last_user_subject: {value}")
        except Exception as e:
            logger.warning(f"Lỗi khi trích last_user_subject từ kết quả: {e}")

    def _load_sql_examples(self, file_path: str) -> List[Dict[str, str]]:
        if os.path.exists(file_path):
            with open(file_path, "r", encoding="utf-8") as f:
                return json.load(f)
        logger.warning(f"File ví dụ SQL không tìm thấy: {file_path}")
        return []

    def _select_relevant_examples(self, question: str, num_examples: int = 3) -> str:
        question_lower = question.lower()
        question_words = set(question_lower.split())
        scored_examples = []

        for example in self.sql_examples:
            example_question_lower = example["question"].lower()
            example_sql_upper = example["sql"].upper()

            score = 0

            common_keywords = question_words.intersection(set(example_question_lower.split()))
            score += len(common_keywords) * 2

            if ("mỗi" in question_lower or "từng" in question_lower) and \
               ("mỗi" in example_question_lower or "từng" in example_question_lower):
                score += 5

            for table, info in TABLE_KEYWORDS.items():
                if any(kw.lower() in question_lower for kw in info.get("keywords", [])) and \
                   table.upper() in example_sql_upper:
                    score += 4

            if "mỗi tổ" in question_lower and \
               "TO_NHOM" in example_sql_upper and \
               "GROUP BY" in example_sql_upper and \
               "PAKH_SLA_TO_NHOM" in example_sql_upper and \
               "JOIN FB_GROUP" not in example_sql_upper:
                score += 20

            if score > 0:
                scored_examples.append((score, example))

        scored_examples.sort(key=lambda x: x[0], reverse=True)

        relevant_examples = []
        added_questions = set()

        for score, example in scored_examples:
            if example["question"] not in added_questions:
                relevant_examples.append(example)
                added_questions.add(example["question"])
            if len(relevant_examples) >= num_examples:
                break

        if len(relevant_examples) < num_examples:
            for ex in self.sql_examples:
                if ex["question"] not in added_questions:
                    relevant_examples.append(ex)
                    added_questions.add(ex["question"])
                if len(relevant_examples) >= num_examples:
                    break

        example_str = ""
        if relevant_examples:
            example_str += os.linesep + "---EXAMPLES START---" + os.linesep
            for ex in relevant_examples:
                example_str += f"Câu hỏi: {ex['question']}" + os.linesep + "SQL:" + os.linesep + f"```sql{os.linesep}{ex['sql']}{os.linesep}```" + os.linesep + os.linesep
            example_str += "---EXAMPLES END---" + os.linesep + os.linesep
        return example_str

    # --- Sinh SQL ---
    def generate_sql(self, question: str, is_follow_up: bool = False, previous_error: str = None, retries: int = 2, force_no_cache: bool = False):
        key = question.strip().lower()
        self.last_question = question
        # Nếu có ngữ cảnh người dùng, thêm vào cache key để phân biệt
        if self.last_user_subjects:
            key += "__" + "__".join(sorted(sub.lower() for sub in self.last_user_subjects))

        # Sử dụng cache nếu có và không force_no_cache
        if not force_no_cache and key in self.sql_cache:
            logger.info(f"[CACHE] Sử dụng SQL đã cache cho câu hỏi: {question}")
            return self.sql_cache[key]

        for attempt in range(retries):
            try:
                # Nếu là câu hỏi tiếp nối, bổ sung context filter vào prompt nếu không có filter mới
                context_for_prompt = ""
                if is_follow_up and self.last_context and not self._detect_new_filter(question):
                    context_for_prompt = (
                        "\n# Ngữ cảnh hội thoại trước đó:\n"
                        + "\n".join(f"- {k}: {v}" for k, v in self.last_context.items())
                        + "\n"
                    )

                current_question_for_resolution = question
                if is_follow_up and self.last_user_subjects:
                    joined_subjects = ", ".join(self.last_user_subjects)
                    current_question_for_resolution = f"{question.strip()} (liên quan đến các cá nhân: {joined_subjects})"
                    logger.info(f"Đã thêm ngữ cảnh cá nhân vào câu hỏi tiếp nối: {current_question_for_resolution}")

                question_resolved = self._resolve_follow_up_question(current_question_for_resolution, is_follow_up)

                planner = SQLPlanner(self._invoke_model)
                plan_result = planner.plan(question_resolved)
                relevant_tables = plan_result.get("tables", [])
                relevant_examples_str = self._select_relevant_examples(question)

                if not relevant_tables:
                    logger.warning("Planner không xác định được bảng → fallback extract_relevant_tables")
                    relevant_tables = extract_relevant_tables(question_resolved)

                if any(kw in question_resolved.lower() for kw in ["liệt kê", "các phản ánh"]):
                    if "PAKH_CA_NHAN" not in relevant_tables:
                        relevant_tables.append("PAKH_CA_NHAN")
                    if "PAKH" not in relevant_tables and "PAKH_NOI_DUNG_PHAN_ANH" not in relevant_tables:
                        relevant_tables.append("PAKH")

                if "FB_LOCALITY" not in relevant_tables:
                    relevant_tables.append("FB_LOCALITY")

                schema_text = os.linesep + os.linesep.join(
                    self.schema_cache.get_schema(tbl) for tbl in relevant_tables if tbl
                )
                if not schema_text:
                    raise ValueError("Không tìm thấy schema phù hợp.")

                column_mapping_hint = generate_column_mapping_hint(question_resolved)

                prompt_text = f"""
{context_for_prompt}
Bạn là trợ lý sinh truy vấn SQL cho CSDL Oracle.
# Mục tiêu
Sinh truy vấn SQL chính xác. Ưu tiên trả lời nhanh chóng, đơn giản.

# Lịch sử
{self._format_memory_context()}

{relevant_examples_str}

# Ánh xạ từ từ khóa → bảng.cột
{column_mapping_hint}
        
# SCHEMA
{schema_text}

# Hướng dẫn
{self._generate_sum_hint()}
- Chấp nhận các hình thức viết tắt, ví dụ p/a tương ứng với phản ánh.
- Với số thuê bao người dùng, trong cơ sở dữ liệu đang có định dạng không có số 0 ở đầu, khi người dùng nhập câu hỏi BẮT BUỘC chấp nhận cả kiểu nhập CÓ SỐ 0 và KHÔNG có số 0.
- với những câu hỏi liệt kê phản ánh, thông tin phản ánh BẮT BUỘC SELECT SO_THUE_BAO, NOI_DUNG_PHAN_ANH để chỉ cần lấy thông tin từ cột này trong bảng PAKH
- ƯU TIÊN xem mức độ tương thích với từ khóa trong table_keywords.json để truy vấn và trả ra đúng cột được hỏi, không trả lời thiếu hay thừa thông tin, nếu người dùng hỏi về nội dung SIM, gói cước, mạng yếu,... thì thông tin sẽ được lưu trong bảng `PAKH_NOI_DUNG_PHAN_ANH`, KHÔNG PHẢI bảng `PAKH`.
- Với những câu hỏi về số lượng bao nhiêu BẮT BUỘC dùng các bảng PAKH_SLA_* vì các bảng này đã chứa các số liệu tổng hợp, KHÔNG CẦN VÀ KHÔNG ĐƯỢC PHÉP JOIN với bảng PAKH hoặc PAKH_CA_NHAN.
- Khi người dùng tiếp tục hỏi “liệt kê các phản ánh đó” → phải chuyển sang JOIN PAKH_CA_NHAN và PAKH qua ID để lấy đầy đủ thông tin từ bảng PAKH, và bắt buộc trả ra danh sách thông tin phản ánh từ bảng PAKH. 
- Nếu câu hỏi tiếp nối chỉ là "liệt kê" hoặc "liệt kê đi", hãy liệt kê tất cả các phản ánh của cá nhân đã hỏi trước đó KHÔNG tự động thêm điều kiện trạng thái bị từ chối/trạng thái khác nếu câu hỏi không nêu rõ.
- Với các trường mã như `LOAI_PHAN_ANH`, `FB_GROUP`, `HIEN_TUONG`, `NGUYEN_NHAN`, `DON_VI_NHAN` cần JOIN bảng tương ứng (FB_TYPE, FB_GROUP, FB_HIEN_TUONG, FB_REASON, FB_DEPARTMENT) để trả ra tên (`NAME`) thay vì trả ra mã.
- LƯU Ý: cột `TRANG_THAI` chỉ tồn tại trong các bảng `PAKH_CA_NHAN`, `PAKH_PHONG_BAN`, `PAKH_TO_NHOM`, `PAKH_TRUNG_TAM` với giá trị hợp lệ là các mã viết hoa không dấu: 'TU_CHOI' (từ chối), 'HOAN_THANH' (hoàn thành), 'DONG' (đóng),'DANG_XU_LY' (đang xử lý), 'DA_XU_LY' (đã xử lý) → Nếu người dùng viết tiếng Việt như 'Từ chối', hãy ánh xạ sang 'TU_CHOI'.
- TUYỆT ĐỐI KHÔNG được truy vấn trực tiếp các cột địa chỉ như TINH_THANH_PHO, QUAN_HUYEN, PHUONG_XA bằng LIKE. Phải luôn JOIN với bảng FB_LOCALITY để lấy FULL_NAME. Với các cột `PAKH.TINH_THANH_PHO`, `PAKH_QUAN_HUYEN`, `PAKH_PHUONG_XA`, phải nối lại và JOIN với bảng FB_LOCALITY thông qua quan hệ như trong RELATIONS, và khi hỏi thì thay vì trả về 3 cột mã trong PAKH, hãy trả ra FULL_NAME trong FB_LOCALITY và phải join đủ 3 cột.
- Ưu tiên dùng bảng `PAKH_NOI_DUNG_PHAN_ANH` nếu cần truy vấn các trường bán cấu trúc đã chuẩn hóa. Khi hỏi khu vực bị lỗi của phản ánh, ưu tiên truy vấn cột KHU_VUC_BI_LOI của bảng PAKH_NOI_DUNG_PHAN_ANH, ngoài ra có thể trả về tên địa danh đầy đủ truy vấn từ các cột trong PAKH nhưng đã liên kết với FB_LOCALITY để lấy tên thay vì mã khu vực. Hỏi gì trả lời đó, đừng đưa ra thừa thông tin.
 {previous_error if previous_error else ''}
- Nếu có lỗi trước đó, sửa lỗi và tạo lại câu SQL chính xác.
- Nếu chỉ hỏi "bao nhiêu" mà không đề cập "liệt kê", **chỉ cần trả về kết quả tính**.
- **QUAN TRỌNG VỚI CÂU HỎI TIẾP NỐI:** Nếu câu hỏi hiện tại là câu hỏi tiếp nối và có các cá nhân đã được xác định trước đó ({", ".join(self.last_user_subjects) if self.last_user_subjects else 'không có'}), hãy ưu tiên trả lời cho các cá nhân đó, trừ khi câu hỏi hiện tại rõ ràng chỉ định cá nhân khác hoặc yêu cầu thống kê chung. Ví dụ, nếu câu trước hỏi về 'khainx' và 'duc.vole', câu sau hỏi 'số lượng phản ánh của mỗi cá nhân đó' thì phải hiểu là 'khainx' và 'duc.vole'.

# RELATIONS
{self._format_relations_for_prompt()}

# Quan trọng
- TUYỆT ĐỐI KHÔNG BỊA RA tên bảng hoặc tên cột KHÔNG có trong SCHEMA.
- Luôn ưu tiên dùng đúng SELECT gợi ý trong SCHEMA nếu có.
- Nếu không đủ thông tin để trả lời, hãy thông báo rõ thay vì đoán.
- TUYỆT ĐỐI KHÔNG JOIN quá nhiều bảng khi câu hỏi không yêu cầu nhiều thông tin như vậy

Câu hỏi:
{question_resolved}

SQL:
"""
                sql_raw = self._invoke_model(prompt_text)
                self._update_context_from_sql(sql_raw)
                match = re.findall(r"CA_NHAN\s*=\s*'([^']+)'", sql_raw)
                if match:
                    for subject in match:
                        self.last_user_subjects.add(subject)
                        logger.info(f"✅ Đã cập nhật last_user_subjects từ SQL: {subject}")

                return sql_raw

            except Exception as e:
                logger.error(f"Lỗi sinh SQL (lần thử {attempt + 1}): {e}")
                if attempt < retries - 1:
                    previous_error = str(e)
                    continue
                raise
    def generate_and_execute_sql(self, question: str, is_follow_up: bool = False, previous_error: str = None, retries: int = 2, force_no_cache: bool = False, execute_fn=None):
        sql_raw = self.generate_sql(question, is_follow_up, previous_error, retries, force_no_cache)
        self.last_sql = sql_raw

        if execute_fn:
            try:
                result = execute_fn(sql_raw)
                key = question.strip().lower()
                if self.last_user_subjects:
                    key += "__" + "__".join(sorted(sub.lower() for sub in self.last_user_subjects))

                self.sql_cache[key] = sql_raw
                logger.info(f"[CACHE] Đã lưu SQL vào cache sau khi execute thành công cho câu hỏi: {question}")

                return result
            except Exception as e:
                logger.error(f"Lỗi khi execute SQL: {e}")
                return {"error": str(e)}
        else:
            logger.warning("Chưa truyền hàm execute_fn vào generate_and_execute_sql.")
            return None

    def _invoke_model(self, prompt_text: str, retries: int = 2) -> str:
        attempt = 0
        while attempt < retries:
            try:
                if self.engine == "openai":
                    from langchain.prompts import PromptTemplate
                    from langchain.chains import LLMChain

                    sql_prompt = PromptTemplate(input_variables=["input"], template="{input}")
                    sql_chain = LLMChain(llm=self.llm, prompt=sql_prompt)
                    return sql_chain.run(prompt_text).strip()

                elif self.engine == "gemini":
                    return self.model.generate_content(prompt_text).text.strip()

            except Exception as e:
                error_msg = str(e).lower()
                if "rate limit" in error_msg:
                    attempt += 1
                    logger.warning(f"OpenAI rate-limit, retrying attempt {attempt}/{retries}...")
                    if attempt >= retries:
                        logger.warning("OpenAI vẫn rate-limit sau retry, fallback Gemini.")
                        self._fallback_to_gemini()
                        return self._invoke_model(prompt_text)
                else:
                    raise

    def _postprocess_sql(self, sql_raw: str, schema_text: str) -> str:
        return sql_raw.strip()

    def _extract_year_from_question(self):
        if self.last_question:
            match = re.search(r"\b(20[2-3][0-9])\b", self.last_question)
            if match:
                return match.group(1)
        return None

    def _extract_nhom_nguyen_nhan_from_question(self):
        if self.last_question:
            match = re.search(r"nhóm nguyên nhân\s*(\d+)", self.last_question, re.IGNORECASE)
            if match:
                return match.group(1)
        return ""

    # --- Trả kết quả luôn ---
    def format_result_for_user(self, result: dict, filter_link: str = None) -> str:
        target = result.get("details") or result if "rows" in result else None
        if not target or not target.get("rows"):
            if "error" in result:
                return f"❌ Lỗi SQL: {result['error']}"
            elif "message" in result:
                return f"✅ {result['message']}"
            return "⚠️ Không có dữ liệu."

        columns = target["columns"]
        rows = target["rows"]
        q = (self.last_question or "").lower()

        is_list_query = any(kw in q for kw in [
            "liệt kê", "danh sách", "list", "show", "các pa", "các phản ánh", "xem chi tiết", "chi tiết các phản ánh"
        ])
        is_statistical = any(kw in q for kw in [
            "theo từng nhóm", "theo từng loại", "thống kê", "group by", "từng nhóm", "từng loại", "mỗi nhóm", "mỗi loại"
        ])

        # ✅ Xử lý thống kê nhóm
        nhom_keys = [
            "nhom_pa", "ten_nhom_pa", "nhomphananh", "tennhomphananh",
            "nhom_nguyen_nhan", "ten_nhom_nguyen_nhan"
        ]
        nhom_col = next((col for col in columns if col.lower() in nhom_keys), None)
        if (
            any(kw in q for kw in [
                "theo từng nhóm phản ánh", "mỗi nhóm phản ánh", "group by nhóm phản ánh",
                "theo từng nhóm nguyên nhân", "mỗi nhóm nguyên nhân", "group by nhóm nguyên nhân"
            ])
            or nhom_col
        ):
            if nhom_col:
                nhom_idx = columns.index(nhom_col)
                lines = []
                for row in rows:
                    nhom_val = row[nhom_idx]
                    context_nhom = {
                        "year": self._extract_year_from_question() or "2024",
                        "nhomPa": nhom_val if "nhom" in nhom_col.lower() else "",
                        "nhomNguyenNhan": nhom_val if "nguyen_nhan" in nhom_col.lower() else "",
                    }
                    params = extract_list_params_from_sql(self.last_sql, context_nhom)
                    link = build_filter_url("http://14.160.91.174:8180/smartw/feedback/list.htm", params)
                    lines.append(f"{nhom_col}: {nhom_val}\n🔗 [Xem chi tiết]({link})")
                return os.linesep.join(lines)

        # ✅ Thống kê tổng hợp (không có link)
        if is_statistical:
            return os.linesep.join(
                "- " + ", ".join(f"{col}: {val}" for col, val in zip(columns, row))
                for row in rows
            )

        # ✅ Liệt kê chi tiết
        if is_list_query:
            row_limit = 1
            lines = [
                "- " + ", ".join(f"{col}: {val}" for col, val in zip(columns, row))
                for row in rows[:row_limit]
            ]

            if not filter_link:
                context_common = {"year": self._extract_year_from_question() or "2024"}
                has_xuly = any(keyword in self.last_sql.lower() for keyword in [
                    'ca_nhan', 'phong_ban', 'to_nhom', 'trung_tam'
                ])
                has_diachi = (
                    any(re.search(rf"\b{key}\b", self.last_sql, re.IGNORECASE)
                        for key in ["TINH_THANH_PHO", "QUAN_HUYEN", "MA_TINH", "MA_HUYEN"]) or
                    "FB_LOCALITY" in self.last_sql.upper()
                )

                if has_xuly:
                    context_form = {
                        **context_common,
                        "caNhan": list(self.last_user_subjects)[0] if self.last_user_subjects else "",
                    }
                    params = extract_form_detail_params_from_sql(self.last_sql, context_form)
                    base_url = "http://14.160.91.174:8180/smartw/feedback/form/detail.htm"
                else:
                    base_url = "http://14.160.91.174:8180/smartw/feedback/list.htm"
                    ma_diachi = {}

                    # Ưu tiên lấy mã từ kết quả truy vấn
                    ma_col_mapping = {
                        "tinhThanhPho": ["ma_tinh", "province", "tinh_thanh_pho"],
                        "quanHuyen": ["ma_huyen", "district", "quan_huyen"],
                    }

                    for key, aliases in ma_col_mapping.items():
                        for alias in aliases:
                            if alias in map(str.lower, columns):
                                idx = next(i for i, col in enumerate(columns) if col.lower() == alias)
                                ma_diachi[key] = rows[0][idx]
                                logger.info(f"[DEBUG] Dùng trực tiếp mã {key} = {ma_diachi[key]} từ kết quả truy vấn")
                                break

                    full_name_idx = next(
                        (i for i, col in enumerate(columns) if col.upper() in ["FULL_NAME", "DIA_DIEM_PHAN_ANH"]),
                        None
                    )

                    # Truy vấn địa chỉ từ full_name nếu cần
                    if not ma_diachi.get("tinhThanhPho") or not ma_diachi.get("quanHuyen"):
                        if full_name_idx is not None:
                            dia_chi = rows[0][full_name_idx]
                            logger.info(f"[DEBUG] Địa chỉ trong kết quả (full_name): {dia_chi}")
                            try:
                                query = """
                                    SELECT province AS tinhThanhPho, district AS quanHuyen
                                    FROM FB_LOCALITY
                                    WHERE full_name = :1
                                    FETCH FIRST 1 ROWS ONLY
                                """
                                with self.db_conn.cursor() as cursor:
                                    cursor.execute(query, [dia_chi])
                                    result_fine = cursor.fetchone()
                                    if result_fine:
                                        col_names = [desc[0].lower() for desc in cursor.description]
                                        for idx, col in enumerate(col_names):
                                            if col in ["tinhthanhpho", "quanhuyen"]:
                                                ma_diachi[col] = result_fine[idx]
                                        logger.info(f"[DEBUG] Mapping FULL_NAME → mã địa lý: {ma_diachi}")
                            except Exception as e:
                                logger.error(f"[ERROR] Truy vấn mã địa lý từ FULL_NAME lỗi: {e}")
                        else:
                            logger.warning("[DEBUG] Không tìm thấy cột địa chỉ phù hợp trong kết quả.")

                    # Fallback: từ câu hỏi
                    if not ma_diachi.get("tinhThanhPho") or not ma_diachi.get("quanHuyen"):
                        dia_chi_cau_hoi = self._extract_dia_chi_from_question(self.last_question or "")
                        logger.info(f"[DEBUG] Địa chỉ từ câu hỏi: {dia_chi_cau_hoi}")
                        fuzzy_result = self.get_ma_dia_chi_fuzzy(dia_chi_cau_hoi)
                        if fuzzy_result:
                            logger.info(f"[DEBUG] Mapping địa chỉ từ câu hỏi → mã địa lý: {fuzzy_result}")
                            ma_diachi.update(fuzzy_result)

                    if ma_diachi:
                        context_common.update(ma_diachi)
                        # 🔁 Normalize key trước khi truyền
                    if "tinhthanhpho" in context_common:
                        context_common["tinhThanhPho"] = context_common.pop("tinhthanhpho")
                    if "quanhuyen" in context_common:
                        context_common["quanHuyen"] = context_common.pop("quanhuyen")


                    params = extract_list_params_from_sql(self.last_sql, context_common)
                    logger.info(f"[DEBUG] context_common: {context_common}")
                    logger.info(f"[DEBUG] params for build_filter_url: {params}")
                    filter_link = build_filter_url(base_url, params)

            if filter_link:
                lines.append(f"\n🔗 [Xem toàn bộ danh sách tại đây]({filter_link})")
            return os.linesep.join(lines)


 
    def clear_memory(self):
        if self.engine == "openai":
            self.memory.clear()
        else:
            self.memory = []
        logger.info("Đã xóa bộ nhớ hội thoại.")

    def clear_specific_cache(self, question: str):
        key = question.strip().lower()
        if self.last_user_subjects:
            key += "__" + "__".join(sorted(sub.lower() for sub in self.last_user_subjects))

        if key in self.sql_cache:
            del self.sql_cache[key]
            logger.info(f"Đã xóa SQL cache cho câu hỏi: {question}")

    def clear_cache(self):
        self.sql_cache.clear()
        self.schema_cache.clear()
        logger.info("Đã xóa SQL cache và schema cache.")

    def clear_all(self):
        self.clear_cache()
        self.clear_memory()
        self.last_context = {}
        logger.info("Đã xóa toàn bộ bộ nhớ và cache.")

    def format_result_context(self, result: dict) -> str:
        if "rows" in result and result["rows"]:
            header = ", ".join(result["columns"])
            content = os.linesep.join(
                "- " + ", ".join(f"{col}: {val}" for col, val in zip(result["columns"], row))
                for row in result["rows"][:5]
            )
            return f"{header}{os.linesep}{content}"
        elif "error" in result:
            return f"Lỗi: {result['error']}"
        return "Không có dữ liệu."