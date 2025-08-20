import os
import re
import logging
import json
import unidecode
from typing import List, Dict, Optional
from dotenv import load_dotenv
from difflib import get_close_matches
from rapidfuzz import process, fuzz

from utils.sql_planner import SQLPlanner
from utils.column_mapper import generate_column_mapping_hint
from utils.schema_loader import TABLE_KEYWORDS, extract_relevant_tables
from utils.relation_loader import load_relations
from db_utils import get_connection, execute_sql, load_locality_cache_json, load_dept_cache_json


# --- Setup môi trường & logger ---
load_dotenv()
logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO)
LOCALITY_CACHE = load_locality_cache_json()
DEPT_CACHE_FILE = load_dept_cache_json()

logger.info(f"✅ Đã load {len(LOCALITY_CACHE)} bản ghi địa chỉ từ cache.")
logger.info(f"✅ Đã load {len(DEPT_CACHE_FILE)} bản ghi tổ nhóm, phòng ban từ cache.")

LLM_ENGINE = os.getenv("LLM_ENGINE", "openai").lower()
conn = get_connection()
def get_ma_dia_chi_from_cache(dia_chi_text: str) -> dict:
    """
    Map địa chỉ user nhập sang mã tỉnh/quận dựa trên LOCALITY_CACHE.
    """
    if not dia_chi_text:
        return {}

    # Chuẩn hóa và tách các từ khóa
    tokens = [t.strip() for t in re.split(r"[,\n]", dia_chi_text) if t.strip()]
    for token in tokens:
        if token in LOCALITY_CACHE:
            item = LOCALITY_CACHE[token]
            return {
                "tinhThanhPho": item.get("province_code", ""),
                "quanHuyen": item.get("district_code", "")
            }
    # Nếu không tìm thấy
    return {}

dept_keywords = ["tổ", "nhóm", "phòng", "ban", "tổ nhóm", "phòng ban","đài"]
canhan_keywords = ["cá nhân", "nhân viên", "người xử lý", "anh", "cô"]

def summarize_rows(columns, rows, max_rows=20):
    """
    Nếu rows nhiều hơn max_rows thì trả về thống kê/tóm tắt,
    ngược lại trả về full rows.
    """
    if len(rows) <= max_rows:
        return {"columns": columns, "rows": rows}

    # Tóm tắt: đếm số bản ghi, lấy vài ví dụ
    summary = {
        "total_records": len(rows),
        "sample_rows": [dict(zip(columns, r)) for r in rows[:3]]
    }

    # Nếu có cột CA_NHAN / TO_NHOM → gom nhóm luôn
    if "CA_NHAN" in columns:
        idx = columns.index("CA_NHAN")
        counter = {}
        for r in rows:
            key = r[idx]
            counter[key] = counter.get(key, 0) + 1
        summary["count_by_canhan"] = counter

    if "TO_NHOM" in columns:
        idx = columns.index("TO_NHOM")
        counter = {}
        for r in rows:
            key = r[idx]
            counter[key] = counter.get(key, 0) + 1
        summary["count_by_tonhom"] = counter

    return {"summary": summary}

# Hàm kiểm tra xuất hiện chính xác
def contains_keyword(text: str, keywords: list) -> bool:
    text_lower = text.lower()
    for kw in keywords:
        # \b để match nguyên từ, không nhận các từ chứa
        if re.search(rf"\b{re.escape(kw)}\b", text_lower):
            return True
    return False
FORM_DETAIL_PARAMS = [
    "type", "trungTam", "phongBan", "toNhom", "caNhan", "loaiPa", "loaiThueBao",
    "nguyenNhan", "caTruc", "fo_bo", "tinh", "ctdv", "day", "today", "week",
    "month", "quarter", "year", "kpiName", "quanHuyen", "level", "nhomNguyenNhan"
]
LIST_PARAMS = [
    "tab", "id", "year", "tinhThanhPho", "quanHuyen", "is_ticket", "phong", "to", "caTruc",
    "dept_xuly", "dept", "ttTicket", "nhomPa", "loaiPa", "fo_bo", "loaiThueBao", "is_dongpa", "ctdv"
]
def normalize_canhan(value: str) -> str:
    """Chuẩn hóa tên để so sánh fuzzy (bỏ dấu, bỏ ký tự đặc biệt, viết liền)"""
    if not value:
        return ""
    # Bỏ dấu, chuyển thường
    value = unidecode.unidecode(str(value)).lower()
    # Xóa ký tự không phải chữ/số
    value = re.sub(r'[^a-z0-9]', '', value)
    return value

def fuzzy_match_canhan(user_input: str, db_values: List[str], threshold: float = 70) -> Optional[str]:
    """
    Fuzzy match tên cá nhân từ user_input với danh sách ca_nhan (db_values).
    Trả về chuỗi gốc nếu tìm được match.
    threshold: % độ tương đồng tối thiểu (0-100)
    """
    if not user_input or not db_values:
        return None

    norm_input = normalize_canhan(user_input)

    # Map normalized -> original
    norm_map = {normalize_canhan(v): v for v in db_values}

    # Tính score từng ứng viên
    candidates = []
    for norm_val, orig in norm_map.items():
        score = fuzz.partial_ratio(norm_input, norm_val)
        candidates.append((orig, score))

    # Lấy match tốt nhất
    best_match = max(candidates, key=lambda x: x[1], default=(None, 0))
    if best_match[1] >= threshold:
        return best_match[0]
    return None
def resolve_canhan(user_input: str) -> Optional[str]:
    try:
        with get_connection() as conn:
            with conn.cursor() as cursor:
                cursor.execute("SELECT DISTINCT ca_nhan FROM PAKH_SLA_CA_NHAN")
                ca_nhans = [row[0] for row in cursor.fetchall()]
                match = fuzzy_match_canhan(user_input, ca_nhans)
                
                if match:
                    logger.info(f"👤 Fuzzy match cá nhân: '{user_input}' → '{match}'")
                return match
    except Exception as e:
        logger.error(f"Lỗi trong resolve_canhan: {e}")
        return None
import unicodedata
def normalize_text(text: str) -> str:
    text = text.lower()
    text = unicodedata.normalize("NFD", text)  # bỏ dấu
    text = re.sub(r"[\u0300-\u036f]", "", text)
    text = re.sub(r"[^a-z0-9\s]", " ", text)
    return re.sub(r"\s+", " ", text).strip()
def load_dept_cache_json_separated():
    all_depts = load_dept_cache_json()  # list dict
    to_nhom_list = [d for d in all_depts if "TVT" in d["dept_name"] or "TVT" in d["dept_code"].lower()]
    phong_ban_list = [d for d in all_depts if d not in to_nhom_list]
    return to_nhom_list, phong_ban_list

def fuzzy_match_department(user_input: str, kind: str = "all", cutoff: float = 0.7):
    """
    kind = "to_nhom", "phong_ban", hoặc "all"
    """
    to_nhom_list, phong_ban_list = load_dept_cache_json_separated()

    if kind == "to_nhom":
        departments = to_nhom_list
    elif kind == "phong_ban":
        departments = phong_ban_list
    else:
        departments = to_nhom_list + phong_ban_list

    if not departments:
        return None, None, 0

    norm_input = normalize_text(user_input)
    dept_map = {normalize_text(d["dept_name"]): (d["dept_code"], d["dept_name"]) for d in departments}

    best_match, score, _ = process.extractOne(norm_input, dept_map.keys(), scorer=fuzz.WRatio)
    if score >= cutoff * 100:
        dept_code, dept_name = dept_map[best_match]
        return dept_code, dept_name, score
    return None, None, 0

def load_known_places(db_conn):
    cursor = db_conn.cursor()
    cursor.execute("SELECT full_name FROM FB_LOCALITY")
    rows = cursor.fetchall()
    return {normalize_text(r[0]) for r in rows if r[0]}
    
def _clean_sql_for_param_extraction(sql: str) -> str:
    """Loại bỏ LOWER(), UPPER(), TRIM()... quanh tên cột để regex bắt dễ hơn"""
    sql = str(sql or "")
    sql_clean = re.sub(r"\bLOWER\s*\(\s*([A-Z_]+)\s*\)", r"\1", sql, flags=re.IGNORECASE)
    sql_clean = re.sub(r"\bUPPER\s*\(\s*([A-Z_]+)\s*\)", r"\1", sql_clean, flags=re.IGNORECASE)
    sql_clean = re.sub(r"\bTRIM\s*\(\s*([A-Z_]+)\s*\)", r"\1", sql_clean, flags=re.IGNORECASE)
    return sql_clean

def extract_form_detail_params_from_sql(sql: str, context: dict = None, get_ma_dia_chi_fuzzy=None) -> dict:
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

    # --- Extract bằng regex bình thường ---
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

    # --- Override bằng context nếu có ---
    if context:
        for key in params:
            if key in context and context[key]:
                params[key] = context[key]

    # --- Xác định level ---
    sql_upper = sql.upper()
    if not params["level"]:
        if "PAKH_CA_NHAN" in sql_upper:
            params["level"] = "ca_nhan"
        elif "PAKH_TO_NHOM" in sql_upper:
            params["level"] = "to_nhom"
        elif "PAKH_PHONG_BAN" in sql_upper:
            params["level"] = "phong_ban"
        elif "PAKH_TRUNG_TAM" in sql_upper:
            params["level"] = "trung_tam"

    # --- Xác định năm ---
    if not params["year"]:
        match = re.search(r"\b(20[2-3][0-9])\b", sql)
        if match:
            params["year"] = match.group(1)
        elif context and "year" in context and context["year"]:
            params["year"] = context["year"]

    # --- Xác định KPI ---
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

    # --- Xác định caTruc ---
    if not params["caTruc"]:
        ca_truc_map = {"SANG": "SANG", "CHIEU": "CHIEU", "TOI": "TOI"}
        match = re.search(r"CA_TRUC\s*=\s*'([^']+)'", sql, re.IGNORECASE)
        if match:
            ca_truc = match.group(1).upper()
            if ca_truc in ca_truc_map:
                params["caTruc"] = ca_truc_map[ca_truc]

    # --- Loại thuê bao ---
    if not params["loaiThueBao"]:
        loai_thue_bao_map = {"KHDN": "KHDN", "KHCN": "KHCN", "VIP": "VIP"}
        match = re.search(r"LOAI_THUE_BAO\s*=\s*'([^']+)'", sql, re.IGNORECASE)
        if match:
            loai_thue_bao = match.group(1).upper()
            if loai_thue_bao in loai_thue_bao_map:
                params["loaiThueBao"] = loai_thue_bao_map[loai_thue_bao]

    return params


def extract_list_params_from_sql(sql: str, context: dict = None, get_ma_dia_chi_fuzzy=None) -> dict:
    sql = _clean_sql_for_param_extraction(sql)
    default_values = {k: "" for k in LIST_PARAMS}
    default_values.update({
        "tab": "year",
        "is_ticket": "Y%2CN",
        "is_dongpa": "DANG_XU_LY%2CDA_XU_LY",
    })
    params = default_values.copy()

    # --- Extract bằng regex ---
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

    
    # --- Override bằng context nếu có ---
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
        self.db_conn = db_conn
        self.engine = LLM_ENGINE
        self.sql_cache = {}
        self.schema_cache = SchemaCache()
        self.context = {}
        self.context_filters = None
        self.known_places = load_known_places(self.db_conn) if self.db_conn else set()
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

    
    def _fallback_to_gemini(self):
        logger.warning("OpenAI gặp lỗi, chuyển sang Gemini.")
        self._init_gemini()
        self.engine = "gemini"
    def _extract_where_conditions(self, sql: str) -> str:
        """Trích xuất phần điều kiện WHERE từ câu SQL (không gồm chữ WHERE)."""
        sql = str(sql or "").strip().rstrip(";")
        match = re.search(r"\bWHERE\b\s+(.*?)(?:\bGROUP\b|\bORDER\b|$)", sql, re.IGNORECASE | re.DOTALL)
        if match:
            return match.group(1).strip()
        return ""
    # --- Helper ---
    def _get_last_n_user_questions(self, n=5):
        if self.engine == "openai" and hasattr(self.memory, "chat_memory"):
            return [
                msg.content for msg in reversed(self.memory.chat_memory.messages)
                if getattr(msg, "type", "") == "human"
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
    def encode(self, text: str) -> str:
        return str(normalize_text(text) or "")
    def get_ma_dia_chi_fuzzy(self, dia_chi_text: str) -> dict:
        """Map địa chỉ user nhập sang mã tỉnh/quận trong DB."""
        if not self.db_conn or not dia_chi_text:
            return {}

        norm_text = normalize_text(dia_chi_text)
        # tách theo dấu phẩy hoặc khoảng trắng
        parts = [p.strip() for p in re.split(r"[,\n]", dia_chi_text) if p.strip()]

        # Load danh sách FULL_NAME từ DB (có thể cache để tối ưu)
        cursor = self.db_conn.cursor()
        cursor.execute("SELECT province, district, full_name FROM FB_LOCALITY")
        rows = cursor.fetchall()
        cursor.close()

        best_match = {"tinhThanhPho": None, "quanHuyen": None}
        max_score = 0

        for province, district, full_name in rows:
            norm_full = normalize_text(full_name)
            # score = số từ trùng khớp giữa input và full_name
            input_tokens = set(norm_text.split())
            full_tokens = set(norm_full.split())
            score = len(input_tokens & full_tokens) / max(len(full_tokens), 1)
            if score > max_score:
                max_score = score
                best_match["tinhThanhPho"] = province
                best_match["quanHuyen"] = district

        if max_score < 0.3:  # nếu quá ít token trùng, coi như không tìm thấy
            return {}

        return best_match



    from difflib import get_close_matches

    @staticmethod
    def fuzzy_match_diachi(user_input: str, db_fullnames: List[str], cutoff=0.8):
        """Map địa chỉ user nhập sang FULL_NAME trong DB (static method tránh tự bơm self)."""
        norm_input = normalize_text(user_input)
        norm_map = {normalize_text(v): v for v in db_fullnames}
        matches = get_close_matches(norm_input, norm_map.keys(), n=1, cutoff=cutoff)
        if matches:
            return norm_map[matches[0]]
        return None

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
        question_lower = str(question or "").lower()
        question_words = set(question_lower.split())
        scored_examples = []
        for example in self.sql_examples or []:
            score = len(question_words.intersection(set(str(example.get("question", "")).lower().split())))
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
        # Reset nếu là câu hỏi mới độc lập
        if reset and not is_follow_up:
            self.reset_context()

        # Lưu câu hỏi ngay
        self.update_context(question=question)

        key = str(question or "").strip().lower()
        if self.context.get("subjects"):
            key += "__" + "__".join(sorted(str(sub).lower() for sub in self.context["subjects"]))

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
Câu hỏi mới: {question}
# Hướng dẫn 
- Nếu người dùng dùng từ "đó", "tiếp", "liệt kê đi", "thêm chi tiết", hãy coi đó là tiếp nối của câu trước và phải lọc dựa trên kết quả/SQL trước.
- Nếu người dùng nhập câu hỏi mới đầy đủ (có đủ chủ thể, bộ lọc),  hãy coi là câu độc lập.

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
- Chấp nhận các hình thức viết tắt, ví dụ KHCN - Khách hàng cá nhân, KHDN - Khách hàng doanh nghiệp, pa hay p/a - phản ánh, HCM - Thành phố Hồ Chí Minh.
- Với số thuê bao người dùng, trong cơ sở dữ liệu đang có định dạng không có số 0 ở đầu, khi người dùng nhập câu hỏi BẮT BUỘC chấp nhận cả kiểu nhập CÓ SỐ 0 và KHÔNG có số 0.
- Với những câu liệt kê, chỉ trả về cột PAKH.SO_THUE_BAO và PAKH.NOI_DUNG_PHAN_ANH, KHÔNG TRẢ CÁC CỘT KHÁC.
- TUYỆT ĐỐI KHÔNG được truy vấn trực tiếp các cột địa chỉ như TINH_THANH_PHO, QUAN_HUYEN, PHUONG_XA bằng LIKE. Phải luôn JOIN với bảng FB_LOCALITY để lấy FULL_NAME  like (và join đủ 3 cột khi trả ra tên).
- Với những câu hỏi về số lượng bao nhiêu BẮT BUỘC dùng các bảng PAKH_SLA_* vì các bảng này đã chứa các số liệu tổng hợp, KHÔNG CẦN VÀ KHÔNG ĐƯỢC PHÉP JOIN với bảng PAKH hoặc PAKH_CA_NHAN.
- Khi người dùng tiếp tục hỏi “liệt kê các phản ánh đó” → phải chuyển sang JOIN PAKH_CA_NHAN và PAKH qua ID để lấy đầy đủ thông tin từ bảng PAKH, và bắt buộc trả ra danh sách thông tin phản ánh từ bảng PAKH. 
- Với các trường mã như `LOAI_PHAN_ANH`, `FB_GROUP`, `HIEN_TUONG`, `NGUYEN_NHAN`, `DON_VI_NHAN` cần JOIN bảng tương ứng (FB_TYPE, FB_GROUP, FB_HIEN_TUONG, FB_REASON, FB_DEPARTMENT) để trả ra tên (`NAME`) thay vì trả ra mã.
- LƯU Ý: cột `TRANG_THAI` chỉ tồn tại trong các bảng `PAKH_CA_NHAN`, `PAKH_PHONG_BAN`, `PAKH_TO_NHOM`, `PAKH_TRUNG_TAM` với giá trị hợp lệ là: 'TU_CHOI','HOAN_THANH','DONG','DANG_XU_LY','DA_XU_LY'.
{previous_error if previous_error else ''}
- Nếu có lỗi trước đó, sửa lỗi và tạo lại câu SQL chính xác.
- Nếu chỉ hỏi "bao nhiêu" mà không đề cập "liệt kê", **chỉ cần trả về kết quả tính**.
- Nếu câu hỏi tiếp nối, ưu tiên context: {self.context["filters"]}

# RELATIONS
{self._format_relations_for_prompt()}

# Quan trọng
- TUYỆT ĐỐI KHÔNG BỊA TÊN BẢNG, TÊN CỘT KHÔNG CÓ TRONG SCHEMA
- Ưu tiên dùng đúng SELECT gợi ý trong SCHEMA nếu có.
- Nếu không đủ thông tin để trả lời, hãy thông báo rõ thay vì đoán.
- Hạn chế JOIN không cần thiết.

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

    def _invoke_model(self, prompt_text: str, retries: int = 1) -> str:
        attempt = 0
        while attempt < retries:
            try:
                if self.engine == "openai":
                    from langchain.prompts import PromptTemplate
                    from langchain.chains import LLMChain

                    sql_prompt = PromptTemplate(input_variables=["input"], template="{input}")
                    sql_chain = LLMChain(llm=self.llm, prompt=sql_prompt)
                    return str(sql_chain.run(prompt_text)).strip()

                elif self.engine == "gemini":
                    return str(self.model.generate_content(prompt_text).text).strip()

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
        sql = str(sql_raw or "").strip()
        if "fb_locality" in sql.lower():
            # ép select thêm province, district nếu chưa có
            if not re.search(r"\bprovince\b", sql, re.IGNORECASE):
                sql = sql.replace("SELECT", "SELECT l.PROVINCE, l.DISTRICT,", 1)
        return sql

    def _extract_year_from_question(self):
        if self.context.get("last_question"):
            match = re.search(r"\b(20[2-3][0-9])\b", str(self.context["last_question"]), re.IGNORECASE)
            if match:
                return match.group(1)
        return None
    import re


    def response(self, question: str, execute_fn=None, is_independent: bool = False, force_no_cache: bool = False):
        if is_independent:
            self.reset_context()

        if execute_fn is None:
            execute_fn = self.execute_sql

        # --- Chỉ fuzzy match cá nhân nếu câu hỏi có từ khóa liên quan ---
        dept_keywords_to_nhom = ["tổ", "nhóm"]
        dept_keywords_phong_ban = ["phòng ban", "ban"]

        dept_code, dept_name, score = None, None, 0

        # Kiểm tra tổ nhóm
        if contains_keyword(question, dept_keywords_to_nhom):
            dept_code, dept_name, score = fuzzy_match_department(question, kind="to_nhom")

        # Kiểm tra phòng ban
        elif contains_keyword(question, dept_keywords_phong_ban):
            dept_code, dept_name, score = fuzzy_match_department(question, kind="phong_ban")

        if dept_code:
            logger.info(f"🔎 Fuzzy match: '{question}' → {dept_name} ({dept_code}), score={score}")
            question += f" (Mã phòng ban tương ứng: {dept_code})"

        canhan_code = None
        if contains_keyword(question, canhan_keywords):
            canhan_code = resolve_canhan(question)
            if canhan_code:
                question += f" (CA_NHAN: {canhan_code})"

        try:
            # --- Sinh SQL ---
            sql_raw = self.generate_sql(
                question, is_follow_up=not is_independent, force_no_cache=force_no_cache
            )
            sql_clean = re.sub(r"```sql|```", "", sql_raw, flags=re.IGNORECASE).strip()

            # --- Thực thi SQL ---
            try:
                result = execute_fn(sql_clean)
            except Exception as e:
                # Nếu SQL lỗi hoàn toàn → fallback
                prompt = f"Câu hỏi: {question}\nTrả lời ngắn gọn, đúng ngôn ngữ câu hỏi:"
                return self._invoke_model(prompt)

            # --- Xử lý kết quả ---
            target = result.get("details") or (result if "rows" in result else None)
            if not target or not target.get("rows"):
                return "Xin phép được thông tin tới Anh/Chị: hiện tại không tìm thấy dữ liệu phù hợp."

            columns, rows_full = target["columns"], target["rows"]
            is_statistical, is_info_query, is_list_query = self.detect_query_type(question, target)

            formatted_result = {}
            filter_link = None

            if is_statistical:
                try:
                    # Nếu nhiều dòng → convert thành dict {col0: col1, ...}
                    if len(rows_full) > 1 and len(columns) >= 2:
                        formatted_result = {
                            "statistical": True,
                            "data": {row[0]: row[1] for row in rows_full}
                        }
                    else:
                        val = rows_full[0][0] if rows_full else 0
                        if isinstance(val, (int, float)):
                            formatted_result = {"total": val}
                        else:
                            try:
                                formatted_result = {"total": float(val)}
                            except Exception:
                                formatted_result = {"statistical": True, "value": str(val)}
                except Exception:
                    formatted_result = {"statistical": True, "message": "⚠️ Không có dữ liệu thống kê."}


            elif is_list_query:
                # lấy preview vài dòng + đếm tổng
                row_preview = rows_full[:3]
                sql_upper = sql_clean.upper()
                has_handler = any(keyword.upper() in sql_upper for keyword in ['CA_NHAN', 'PHONG_BAN', 'TO_NHOM', 'TRUNG_TAM'])
                year_in_question = self._extract_year_from_question()

                # Xác định năm mặc định
                if has_handler:
                    default_year = year_in_question or "2024"
                else:
                    default_year = "2024"
                    if "FB_DATE" in columns:
                        fb_date_value = rows_full[0][columns.index("FB_DATE")]
                        if fb_date_value:
                            default_year = str(fb_date_value)[:4]

                context_common = {"year": default_year}

                try:
                    if has_handler:
                        params = extract_form_detail_params_from_sql(
                            sql_clean, context_common, get_ma_dia_chi_fuzzy=get_ma_dia_chi_from_cache
                        )
                        base_url = "http://14.160.91.174:8180/smartw/feedback/form/detail.htm"
                    else:
                        params = extract_list_params_from_sql(
                            sql_clean, context_common, get_ma_dia_chi_fuzzy=get_ma_dia_chi_from_cache
                        )
                        base_url = "http://14.160.91.174:8180/smartw/feedback/list.htm"

                    filter_link = build_filter_url(base_url, params)
                    logger.info(f"[LINK] Link build thành công: {filter_link}")
                except Exception as e:
                    logger.error(f"[ERROR] Xây dựng link thất bại: {e}")
                    filter_link = None

                formatted_result = summarize_rows(columns, rows_full)
                formatted_result["count"] = len(rows_full)
                if filter_link:
                        formatted_result["link"] = f"[truy cập tại đây]({filter_link})"


            else:
                # Trường hợp khác → show 1 record đầu tiên
                formatted_result = {"columns": columns, "rows": rows_full[:1], "count": len(rows_full)}

            # --- Luôn sinh phản hồi tự nhiên ---
            prompt = f"""
    Câu hỏi: {question}
    Kết quả truy vấn (JSON):
    {json.dumps(formatted_result, ensure_ascii=False)}

    Yêu cầu:
    - Trả lời cùng NGÔN NGỮ với câu hỏi.
    - Trả lời lịch sự, có xưng hô phù hợp.
    - KHI ĐÃ CÓ DỮ LIỆU TRUY VẤN ĐƯỢC THÌ PHẢI TRẢ LỜI BÁM SÁT VÀO DỮ LIỆU, TUYỆT ĐỐI KHÔNG TỰ BỊA ĐẶT THÊM
    - Nếu có 'count', hãy nêu rõ số lượng bản ghi.
    - Nếu có 'link', cuối câu thêm: "🔗 {formatted_result.get('link')}".
    """
            answer = self._invoke_model(prompt)
            self.update_context(sql=sql_clean, result=result, question=question)
            return answer

        except Exception as e:
            # fallback cuối cùng
            prompt = f"""
    Người dùng hỏi: {question}

    Yêu cầu:
    - Trả lời đúng ngôn ngữ câu hỏi.
    - Trả lời lịch sự, có xưng hô phù hợp.
    - BÁM SÁT dữ liệu (nếu có).
    """
            return self._invoke_model(prompt)



    def detect_query_type(self, question: str, result: dict):
        q = str(question or "").lower()
        rows = result.get("rows") or result.get("details", {}).get("rows")

        # Nếu kết quả chỉ có 1 cột và tên cột chứa "COUNT" → chắc chắn là thống kê
        if rows and len(rows[0]) == 1 and "count" in (result.get("columns", [""])[0].lower()):
            return True, False, False  # statistical

        is_list_query = any(kw in q for kw in [
            "liệt kê", "danh sách", "list", "show", "các pa",
            "các phản ánh", "xem chi tiết", "chi tiết các phản ánh", "pa ở","lk"
        ])
        is_statistical = any(kw in q for kw in [
            "theo từng nhóm", "theo từng loại", "thống kê",
            "từng nhóm", "từng loại", "mỗi nhóm", "mỗi loại",
            "bao nhiêu", "tổng số", "tổng cộng", "số lượng",
            "bao nhiêu phản ánh", "bn phản ánh", "bn pa", "bao nhiêu pa"
        ]) or "group by" in q

        is_info_query = not is_list_query and not is_statistical and bool(rows)
        return is_statistical, is_info_query, is_list_query


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