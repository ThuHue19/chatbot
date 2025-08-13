import os
import re
import logging
import json
import unidecode
from typing import List, Dict
from dotenv import load_dotenv
from groq import Groq

from utils.sql_planner import SQLPlanner
from utils.column_mapper import extract_tables_and_columns, generate_column_mapping_hint
from utils.schema_loader import TABLE_KEYWORDS, extract_relevant_tables
from utils.relation_loader import load_relations
from db_utils import get_connection

# --- Setup môi trường & logger ---
load_dotenv()
logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO)

LLM_ENGINE = os.getenv("LLM_ENGINE", "openai").lower()
conn = get_connection()

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
def _clean_sql_for_param_extraction(sql: str) -> str:
    """Loại bỏ LOWER(), UPPER(), TRIM()... quanh tên cột để regex bắt dễ hơn"""
    sql_clean = re.sub(r"\bLOWER\s*\(\s*([A-Z_]+)\s*\)", r"\1", sql, flags=re.IGNORECASE)
    sql_clean = re.sub(r"\bUPPER\s*\(\s*([A-Z_]+)\s*\)", r"\1", sql_clean, flags=re.IGNORECASE)
    sql_clean = re.sub(r"\bTRIM\s*\(\s*([A-Z_]+)\s*\)", r"\1", sql_clean, flags=re.IGNORECASE)
    return sql_clean

def extract_form_detail_params_from_sql(sql: str, context: dict = None) -> Dict[str, str]:
    sql = _clean_sql_for_param_extraction(sql)
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
        "caNhan": r"(?:\b\w+\.)?CA_NHAN\s*=\s*'([^']+)'",
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
    if not params["level"]:
        if "PAKH_CA_NHAN" in sql.upper():
            params["level"] = "ca_nhan"
        elif "PAKH_TO_NHOM" in sql.upper():
            params["level"] = "to_nhom"
        elif "PAKH_PHONG_BAN" in sql.upper():
            params["level"] = "phong_ban"
        elif "PAKH_TRUNG_TAM" in sql.upper():
            params["level"] = "trung_tam"
    if not params["year"]:
        match = re.search(r"\b(20[2-3][0-9])\b", sql)
        if match:
            params["year"] = match.group(1)
        elif context and "year" in context and context["year"]:
            params["year"] = context["year"]
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
    
    return params

def extract_list_params_from_sql(sql: str, context: dict = None) -> Dict[str, str]:
    sql = _clean_sql_for_param_extraction(sql)
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
        self.sql_cache = {}
        self.context = {}
        self.context_filters = None
        self.groq_client = None
        self._init_groq()

        self.relations, self.concat_mappings = load_relations()
        self.context = {
            "filters": {},
            "subjects": set(),
            "last_question": None,
            "last_sql": None,
            "last_result": None
        }
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

    def _init_groq(self):
        api_key = os.getenv("GROQ_API_KEY")
        if not api_key:
            raise ValueError("Thiếu GROQ_API_KEY trong file .env")
        self.groq_client = Groq(api_key=api_key)
        
    def _fallback_to_gemini(self):
        logger.warning("OpenAI gặp lỗi, chuyển sang Gemini.")
        self._init_gemini()
        self.engine = "gemini"
    def _extract_where_conditions(self, sql: str) -> str:
        """
        Trích xuất phần điều kiện WHERE từ câu SQL.
        Trả về chuỗi điều kiện (không bao gồm chữ WHERE).
        """
        # Bỏ ; ở cuối để tránh match sai
        sql = sql.strip().rstrip(";")
        
        # Tìm WHERE ... (đến hết hoặc trước GROUP/ORDER)
        match = re.search(r"\bWHERE\b\s+(.*?)(?:\bGROUP\b|\bORDER\b|$)", sql, re.IGNORECASE | re.DOTALL)
        if match:
            return match.group(1).strip()
        return ""
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
    def reset_context(self):
        self.context = {
            "filters": {},
            "subjects": set(),
            "last_question": None,
            "last_sql": None,
            "last_result": None
        }
        logger.info("[CONTEXT] Đã reset toàn bộ context.")
    def update_context(self, sql=None, result=None, question=None):
        """Cập nhật context từ SQL, kết quả truy vấn và câu hỏi."""
        if question is not None:
            self.context["last_question"] = question
        if sql is not None:
            self.context["last_sql"] = sql
        if result is not None:
            self.context["last_result"] = result

    def get_ma_dia_chi_fuzzy(self, dia_chi_text: str) -> dict:
        if not self.db_conn:
            logger.warning("[WARN] DB connection is None. Không thể truy vấn mã địa lý.")
            return {}
        
        import unidecode

        # Bỏ dấu, lowercase, loại bỏ ký tự đặc biệt
        text = unidecode.unidecode(dia_chi_text).lower()
        text = re.sub(r"[^\w\s]", " ", text)
        keywords = [kw.strip() for kw in text.split() if kw.strip()]
        if not keywords:
            return {}

        # Tạo điều kiện WHERE bằng LIKE
        conditions = []
        params = []
        for i, kw in enumerate(keywords):
            conditions.append(f"LOWER(full_name) LIKE :{i+1}")
            params.append(f"%{kw}%")
        
        where_clause = " AND ".join(conditions)
        query = f"""
    SELECT province AS tinhThanhPho, district AS quanHuyen 
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
        if not question:
            return ""

        import unidecode
        q_norm = unidecode.unidecode(question).lower()

        # Chuẩn hóa các dạng viết tắt
        q_norm = re.sub(r"\bq\.?\s", "quan ", q_norm)  # Q Hoàng Mai, Q. Hoàng Mai → quan Hoàng Mai
        q_norm = re.sub(r"\bh\.?\s", "huyen ", q_norm) # H Hóc Môn, H. Hóc Môn → huyen Hóc Môn
        q_norm = re.sub(r"\btp\.?\s", "thanh pho ", q_norm) # TP, TP. → thanh pho
        q_norm = re.sub(r"\btp\b", "thanh pho", q_norm)

        # Regex nhận diện đầy đủ các loại địa chỉ
        patterns = [
            r"(?:tai|o|thuoc|dia ban|khu vuc|tinh|thanh pho|quan|huyen|xa|phuong)\s+([^\.,;?\n]+)",
            r"(quan|huyen|phuong|xa|thi xa|tp|thanh pho)\s+\w+(?:\s+\w+)?"
        ]

        for pattern in patterns:
            match = re.search(pattern, q_norm, re.IGNORECASE)
            if match:
                return match.group(0).strip()

        # fallback: lấy 3 từ cuối
        return " ".join(q_norm.strip().split()[-3:])

    def _detect_new_filter(self, question: str) -> bool:
        # Nếu câu hỏi có từ khóa về phòng ban, tổ nhóm, trung tâm, địa bàn, nhóm, loại...
        keywords = [
            "phòng", "phòng ban", "tổ", "tổ nhóm", "trung tâm", "địa bàn", "tỉnh", "huyện",
            "nhóm phản ánh", "loại phản ánh", "cá nhân", "số thuê bao", "mã tỉnh", "mã huyện"
        ]
        q = question.lower()
        return any(kw in q for kw in keywords)

    def _resolve_follow_up_question(self, question: str, is_follow_up: bool = False) -> str:
        if not is_follow_up:
            return question
        last_q = self.context["last_question"] or ""
        if not last_q:
            return question
        prompt = f"""
Bạn là trợ lý ngôn ngữ.
Câu hỏi trước: {last_q}
Câu hỏi hiện tại: {question}
Hãy viết lại câu hỏi hiện tại thành câu hỏi đầy đủ, rõ nghĩa.
"""
        resolved = self._invoke_model(prompt)
        return resolved.strip() if resolved else question

    def _detect_new_filter(self, question: str) -> bool:
        keywords = [
            "phòng", "phòng ban", "tổ", "tổ nhóm", "trung tâm", "địa bàn", "tỉnh", "huyện",
            "nhóm phản ánh", "loại phản ánh", "cá nhân", "số thuê bao", "mã tỉnh", "mã huyện"
        ]
        return any(kw in question.lower() for kw in keywords)

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
            score = len(question_words.intersection(set(example["question"].lower().split())))
            if score > 0:
                scored_examples.append((score, example))
        scored_examples.sort(key=lambda x: x[0], reverse=True)
        relevant_examples = [ex for _, ex in scored_examples[:num_examples]]
        example_str = ""
        if relevant_examples:
            example_str += "\n---EXAMPLES START---\n"
            for ex in relevant_examples:
                example_str += f"Câu hỏi: {ex['question']}\nSQL:\n```sql\n{ex['sql']}\n```\n\n"
            example_str += "---EXAMPLES END---\n\n"
        return example_str

    # --- Sinh SQL ---
    def generate_sql(self, question: str, is_follow_up: bool = False, previous_error: str = None,
                 retries: int = 2, force_no_cache: bool = False, reset: bool = False):
    # Reset nếu là câu hỏi mới
        if reset and not is_follow_up:  # <-- chỉ reset khi không phải tiếp tục
            self.reset_context()

        # Lưu câu hỏi ngay
        self.update_context(question=question)

        key = question.strip().lower()
        if self.context.get("subjects"):
            key += "__" + "__".join(sorted(sub.lower() for sub in self.context["subjects"]))

        # Cache
        if not force_no_cache and key in self.sql_cache:
            logger.info(f"[CACHE] Sử dụng SQL cache cho: {question}")
            return self.sql_cache[key]


        for attempt in range(retries):
            try:
                context_for_prompt = ""
                if is_follow_up and all(self.context.get(k) for k in ["last_question", "last_sql", "last_result"]):
                    filter_str = ""
                    if self.context["filters"]:
                        filter_str = "\nCác bộ lọc đã lưu từ trước: " + json.dumps(self.context["filters"], ensure_ascii=False)
                    context_for_prompt = f"""
                    # Ngữ cảnh hội thoại trước đó:
                    Câu hỏi: {self.context['last_question']}
                    SQL: {self.context['last_sql']}
                    Kết quả: {json.dumps(self.context['last_result'], ensure_ascii=False)}
                    {filter_str}
                    """
                question_resolved = self._resolve_follow_up_question(question, is_follow_up)
                planner = SQLPlanner(self._invoke_model)
                plan_result = planner.plan(question_resolved)
                relevant_tables = plan_result.get("tables", []) or extract_relevant_tables(question_resolved)

                relevant_examples_str = self._select_relevant_examples(question)
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

{relevant_examples_str}

# Ánh xạ từ từ khóa → bảng.cột
{column_mapping_hint}
        
# SCHEMA
{schema_text}

# Hướng dẫn
{self._generate_sum_hint()}
- Chấp nhận các hình thức viết tắt, ví dụ KHCN - Khách hàng cá nhân, KHDN - Khách hàng doanh nghiệp, pa - phản ánh, HCM - Thành phố Hồ Chí Minh.
- Với số thuê bao người dùng, trong cơ sở dữ liệu đang có định dạng không có số 0 ở đầu, khi người dùng nhập câu hỏi BẮT BUỘC chấp nhận cả kiểu nhập CÓ SỐ 0 và KHÔNG có số 0.
- Với những câu liệt kê, chỉ trả về cột PAKH.SO_THUE_BAO và PAKH.NOI_DUNG_PHAN_ANH, KHÔNG TRẢ CÁC CỘT KHÁC.
- Nếu câu hỏi có yếu tố địa chỉ, luôn SELECT thêm cột mã địa lý (TINH_THANH_PHO, QUAN_HUYEN) hoặc PROVINCE, DISTRICT từ bảng địa chỉ (như FB_LOCALITY) để phục vụ mapping.
- ƯU TIÊN xem mức độ tương thích với từ khóa trong table_keywords.json để truy vấn và trả ra đúng cột được hỏi, không trả lời thiếu hay thừa thông tin, nếu người dùng hỏi về nội dung SIM, gói cước, mạng yếu,... thì thông tin sẽ được lưu trong bảng `PAKH_NOI_DUNG_PHAN_ANH`, KHÔNG PHẢI bảng `PAKH`.
- Với những câu hỏi về số lượng bao nhiêu BẮT BUỘC dùng các bảng PAKH_SLA_* vì các bảng này đã chứa các số liệu tổng hợp, KHÔNG CẦN VÀ KHÔNG ĐƯỢC PHÉP JOIN với bảng PAKH hoặc PAKH_CA_NHAN.
- Khi người dùng tiếp tục hỏi “liệt kê các phản ánh đó” → phải chuyển sang JOIN PAKH_CA_NHAN và PAKH qua ID để lấy đầy đủ thông tin từ bảng PAKH, và bắt buộc trả ra danh sách thông tin phản ánh từ bảng PAKH. 
- Với các trường mã như `LOAI_PHAN_ANH`, `FB_GROUP`, `HIEN_TUONG`, `NGUYEN_NHAN`, `DON_VI_NHAN` cần JOIN bảng tương ứng (FB_TYPE, FB_GROUP, FB_HIEN_TUONG, FB_REASON, FB_DEPARTMENT) để trả ra tên (`NAME`) thay vì trả ra mã.
- LƯU Ý: cột `TRANG_THAI` chỉ tồn tại trong các bảng `PAKH_CA_NHAN`, `PAKH_PHONG_BAN`, `PAKH_TO_NHOM`, `PAKH_TRUNG_TAM` với giá trị hợp lệ là các mã viết hoa không dấu: 'TU_CHOI' (từ chối), 'HOAN_THANH' (hoàn thành), 'DONG' (đóng),'DANG_XU_LY' (đang xử lý), 'DA_XU_LY' (đã xử lý) → Nếu người dùng viết tiếng Việt như 'Từ chối', hãy ánh xạ sang 'TU_CHOI'.
- TUYỆT ĐỐI KHÔNG được truy vấn trực tiếp các cột địa chỉ như TINH_THANH_PHO, QUAN_HUYEN, PHUONG_XA bằng LIKE. Phải luôn JOIN với bảng FB_LOCALITY để lấy FULL_NAME. Với các cột `PAKH.TINH_THANH_PHO`, `PAKH_QUAN_HUYEN`, `PAKH_PHUONG_XA`, phải nối lại và JOIN với bảng FB_LOCALITY thông qua quan hệ như trong RELATIONS, và khi hỏi thì thay vì trả về 3 cột mã trong PAKH, hãy trả ra FULL_NAME trong FB_LOCALITY và phải join đủ 3 cột.
- Ưu tiên dùng bảng `PAKH_NOI_DUNG_PHAN_ANH` nếu cần truy vấn các trường bán cấu trúc đã chuẩn hóa. Khi hỏi khu vực bị lỗi của phản ánh, ưu tiên truy vấn cột KHU_VUC_BI_LOI của bảng PAKH_NOI_DUNG_PHAN_ANH, ngoài ra có thể trả về tên địa danh đầy đủ truy vấn từ các cột trong PAKH nhưng đã liên kết với FB_LOCALITY để lấy tên thay vì mã khu vực. Hỏi gì trả lời đó, đừng đưa ra thừa thông tin.
 {previous_error if previous_error else ''}
- Nếu có lỗi trước đó, sửa lỗi và tạo lại câu SQL chính xác.
- Nếu chỉ hỏi "bao nhiêu" mà không đề cập "liệt kê", **chỉ cần trả về kết quả tính**.
- Nếu câu hỏi tiếp nối, ưu tiên context: {self.context["filters"]}

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
                self.update_context(sql=sql_raw)
                
                return sql_raw

            except Exception as e:
                logger.error(f"Lỗi sinh SQL (lần thử {attempt + 1}): {e}")
                if attempt < retries - 1:
                    previous_error = str(e)
                    continue
                raise
    def generate_and_execute_sql(self, question: str, is_follow_up: bool = False, previous_error: str = None,
                             retries: int = 2, force_no_cache: bool = False, execute_fn=None,  reset: bool = False):
        sql_raw = self.generate_sql(question, is_follow_up, previous_error, retries, force_no_cache)

        if execute_fn:
            try:
                result = execute_fn(sql_raw)

                # Lưu kết quả vào context
                self.update_context(result=result)
                # Nếu chạy thành công và có dữ liệu thì lưu điều kiện WHERE
                if result and isinstance(result, dict):
                    extracted_filter = self._extract_where_conditions(sql_raw)
                    if extracted_filter:
                        self.context_filters = extracted_filter
                        logger.info(f"[CONTEXT] Lưu filter: {self.context_filters}")

                        # 🔹 Parse các tham số chi tiết để lưu vào context["filters"]
                        form_params = extract_form_detail_params_from_sql(sql_raw, self.context.get("filters", {}))
                        # Bỏ các param rỗng
                        form_params = {k: v for k, v in form_params.items() if v}
                        self.context["filters"].update(form_params)
                        logger.info(f"[CONTEXT] Cập nhật filters: {self.context['filters']}")

                # Cache SQL
                key = question.strip().lower()
                if self.context.get("subjects"):
                    key += "__" + "__".join(sorted(sub.lower() for sub in self.context["subjects"]))
                self.sql_cache[key] = sql_raw

                return result
            except Exception as e:
                logger.error(f"Lỗi execute SQL: {e}")
                return {"error": str(e)}
        else:
            logger.warning("Chưa truyền execute_fn vào generate_and_execute_sql.")
            return None


    def _invoke_model(self, prompt_text: str, retries: int = 1) -> str:
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
        if self.context["last_question"]:
            match = re.search(r"\b(20[2-3][0-9])\b", self.context["last_question"], re.IGNORECASE)
            if match:
                return match.group(1)
        return None

    def format_result_for_user(self, result: dict, filter_link: str = None) -> str:
        logger.info(f"[DEBUG] last_question: {self.context.get('last_question')}")
        logger.info(f"[DEBUG] last_sql: {self.context.get('last_sql')}")
        logger.info(f"[DEBUG] last_result: {self.context.get('last_result')}")
        logger.info(f"[DEBUG] filters: {self.context.get('filters')}")

        def _safe_join(items):
            return os.linesep.join("" if item is None else str(item) for item in items)

        # --- Fix toán tử ưu tiên ---
        target = result.get("details") or (result if "rows" in result else None)

        # --- Debug để xem target thực tế là gì ---
        logger.info(f"[DEBUG] target type: {type(target)}, target: {target}")

        if not target or not target.get("rows"):
            if "error" in result:
                return f"❌ Lỗi SQL: {result['error']}"
            elif "message" in result:
                return f"✅ {result['message']}"
            return "⚠️ Không có dữ liệu."

        columns = target["columns"]
        rows = target["rows"]

        logger.info(f"[DEBUG] columns: {columns}")
        logger.info(f"[DEBUG] rows: {rows}")

        q = (self.context["last_question"] or "").lower()

        is_statistical, is_info_query, is_list_query = self.detect_query_type(self.context["last_question"], target)

        # --- Xử lý thống kê ---
        if is_statistical:
            logger.info("[DEBUG] Đang xử lý dạng thống kê")
            lines = []
            if len(columns) == 1 and len(rows) == 1:
                val = rows[0][0]
                logger.info(f"[DEBUG] Giá trị thống kê đơn: {val}")
                return str(val if val is not None else 0)

            for row in rows:
                safe_parts = []
                for col, val in zip(columns, row):
                    col_str = str(col) if col is not None else ""
                    val_str = str(val) if val is not None else "0"
                    safe_parts.append(f"{col_str}: {val_str}")
                lines.append(", ".join(safe_parts) if safe_parts else "")

            logger.info(f"[DEBUG] lines trước khi join: {lines}")
            return _safe_join(lines)

        # --- Truy vấn thông tin ---
        if is_info_query:
            # Nếu chỉ có 1 cột → trả danh sách giá trị
            if len(columns) == 1:
                values = [str(row[0]) if row[0] is not None else "N/A" for row in rows]
                return _safe_join(values)

            # Nhiều cột → trả dạng "col: value"
            lines = []
            for row in rows:
                formatted_row = ", ".join(
                    f"{col}: {str(val) if val is not None else 'N/A'}"
                    for col, val in zip(columns, row)
                )
                lines.append(formatted_row)
            return _safe_join(lines)


        # --- Liệt kê chi tiết ---
        if is_list_query:
            row_limit = 1
            lines = []
            for row in rows[:row_limit]:
                formatted_row = "- " + ", ".join(
                    f"{col}: {str(val) if val is not None else 'N/A'}"
                    for col, val in zip(columns, row)
                )
                lines.append(formatted_row)

            if not filter_link and len(rows) > 2:
                    context_common = {"year": self._extract_year_from_question() or "2024"}
                    has_xuly = any(keyword in self.context['last_sql'].lower() for keyword in [
                        'ca_nhan', 'phong_ban', 'to_nhom', 'trung_tam'
                    ])
                    has_diachi = (
                        any(re.search(rf"\b{key}\b", self.context['last_sql'], re.IGNORECASE)
                            for key in ["TINH_THANH_PHO", "QUAN_HUYEN", "MA_TINH", "MA_HUYEN"]) or
                        "FB_LOCALITY" in self.context['last_sql'].upper()
                    )

                    if has_xuly:
                        context_form = context_common.copy()
                        # KHÔNG set cứng caNhan từ câu hỏi nữa, để extract_form_detail_params_from_sql lo
                        params = extract_form_detail_params_from_sql(self.context['last_sql'], context_form)            
                        base_url = "http://14.160.91.174:8180/smartw/feedback/form/detail.htm"
                        filter_link = build_filter_url(base_url, params)
                    else:
                        base_url = "http://14.160.91.174:8180/smartw/feedback/list.htm"
                        ma_diachi = {}

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

                        if not ma_diachi.get("tinhThanhPho") or not ma_diachi.get("quanHuyen"):
                            dia_chi_cau_hoi = self._extract_dia_chi_from_question(self.context["last_question"] or "")
                            logger.info(f"[DEBUG] Địa chỉ từ câu hỏi: {dia_chi_cau_hoi}")
                            fuzzy_result = self.get_ma_dia_chi_fuzzy(dia_chi_cau_hoi)
                            if fuzzy_result:
                                logger.info(f"[DEBUG] Mapping địa chỉ từ câu hỏi → mã địa lý: {fuzzy_result}")
                                ma_diachi.update(fuzzy_result)

                        if ma_diachi:
                            context_common.update(ma_diachi)

                        if "tinhthanhpho" in context_common:
                            context_common["tinhThanhPho"] = context_common.pop("tinhthanhpho")
                        if "quanhuyen" in context_common:
                            context_common["quanHuyen"] = context_common.pop("quanhuyen")

                        params = extract_list_params_from_sql(self.context['last_sql'], context_common)
                        logger.info(f"[DEBUG] context_common: {context_common}")
                        logger.info(f"[DEBUG] params for build_filter_url: {params}")
                        filter_link = build_filter_url(base_url, params) 

            if filter_link:
                lines.append(f"\n🔗 [Xem toàn bộ danh sách tại đây]({filter_link})")
            return _safe_join(lines)

        return "⚠️ Không xác định được loại câu hỏi."
    def detect_query_type(self, question: str, result: dict):
        """Xác định loại câu hỏi: thống kê, thông tin, liệt kê"""
        q = (question or "").lower()

        is_list_query = any(kw in q for kw in [
            "liệt kê", "danh sách", "list", "show", "các pa", 
            "các phản ánh", "xem chi tiết", "chi tiết các phản ánh", "pa ở"
        ])
        is_statistical = any(kw in q for kw in [
            "theo từng nhóm", "theo từng loại", "thống kê", 
            "từng nhóm", "từng loại", "mỗi nhóm", "mỗi loại",
            "bao nhiêu", "tổng số", "tổng cộng", "số lượng", 
            "bao nhiêu phản ánh", "bn phản ánh", "bn pa", "bao nhiêu pa"
        ]) or "group by" in q or "từng nhóm" in q

        # Xác định có dữ liệu hay không
        rows = result.get("rows") or result.get("details", {}).get("rows")
        is_info_query = not is_list_query and not is_statistical and bool(rows)

        return is_statistical, is_info_query, is_list_query
    def response(self, question: str, result: dict, filter_link: str = None) -> str:
        self.context["last_question"] = question
        formatted_answer = self.format_result_for_user(result, filter_link)

        is_statistical, is_info_query, is_list_query = self.detect_query_type(
            question, result
        )

        if getattr(self, "groq_client", None) \
            and formatted_answer \
            and "Không có dữ liệu" not in formatted_answer \
            and (is_statistical or is_info_query):

            try:
                prompt = f"""
Bạn là trợ lý trả lời câu hỏi dựa trên kết quả truy vấn.

Câu hỏi: {question}
Kết quả truy vấn: {formatted_answer}

HƯỚNG DẪN NGHIÊM NGẶT:
- Chỉ trả lời dựa trên dữ liệu trong "Kết quả truy vấn".
- Phản hồi phải giữ nguyên toàn bộ giá trị trong {formatted_answer}, không được bỏ bớt hoặc suy luận thêm.
- Nếu câu hỏi yêu cầu thống kê hoặc tổng hợp, chỉ việc diễn đạt lại nhưng vẫn phải nêu rõ số liệu chính xác từ {formatted_answer}.
- Nếu không chắc chắn, hãy trả nguyên {formatted_answer} mà không thay đổi.
"""

                resp = self.groq_client.chat.completions.create(
                    model="llama-3.1-8b-instant",
                    messages=[
                        {"role": "system", "content": "Bạn là trợ lý trả lời câu hỏi dựa trên kết quả truy vấn SQL."},
                        {"role": "user", "content": prompt}
                    ]
                )
                return resp.choices[0].message.content.strip()
            except Exception as e:
                import logging
                logging.error(f"[Groq ERROR] {e}")
                return formatted_answer

        return formatted_answer


    def clear_memory(self):
        if self.engine == "openai":
            self.memory.clear()
        else:
            self.memory = []
        logger.info("Đã xóa bộ nhớ hội thoại.")
    def clear_cache(self):
        self.sql_cache.clear()
        self.schema_cache.clear()
        logger.info("Đã xóa SQL cache và schema cache.")

    def clear_all(self):
        self.clear_cache()
        self.clear_memory()
        self.last_context = {}
        logger.info("Đã xóa toàn bộ bộ nhớ và cache.")

    