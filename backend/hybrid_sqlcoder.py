
import os
import re
import logging
from dotenv import load_dotenv
from utils.sql_planner import SQLPlanner
from utils.column_mapper import extract_tables_and_columns, generate_column_mapping_hint # Import now
from utils.schema_loader import TABLE_KEYWORDS, extract_relevant_tables
from utils.relation_loader import load_relations
from typing import List, Dict
import json

# --- Setup môi trường & logger ---
load_dotenv()
logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO)

LLM_ENGINE = os.getenv("LLM_ENGINE", "openai").lower()
def extract_filter_params_from_sql(sql: str) -> Dict[str, str]:
    params = {}

    if "TRANG_THAI" in sql:
        match = re.search(r"TRANG_THAI\\s*=\\s*'([^']+)'", sql)
        if match:
            params["is_dongpa"] = match.group(1)

    if "LOAI_THUE_BAO" in sql:
        match = re.search(r"LOAI_THUE_BAO\\s*=\\s*'([^']+)'", sql)
        if match:
            params["loaiThueBao"] = match.group(1)

    if "IS_TICKET" in sql:
        match = re.search(r"IS_TICKET\\s+IN\\s*\\(([^)]+)\)", sql)
        if match:
            values = match.group(1).replace("'", "").replace(" ", "")
            params["is_ticket"] = values

    if "NHOM_PA" in sql:
        match = re.search(r"NHOM_PA\\s*=\\s*'([^']+)'", sql)
        if match:
            params["nhomPa"] = match.group(1)

    if "SO_PA_QH" in sql or "QUA_HAN" in sql.upper():
        params["tab"] = "quaHan"
    elif "DONG" in sql.upper():
        params["tab"] = "dong"
    elif "NAM" in sql.upper():
        params["tab"] = "year"
    elif "THANG" in sql.upper():
        params["tab"] = "month"
    elif "QUY" in sql.upper():
        params["tab"] = "quarter"
    elif "NGAY" in sql.upper():
        params["tab"] = "day"
    else:
        params["tab"] = "xuLy"
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
    def __init__(self):
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
        BASE_DIR = os.path.dirname(os.path.abspath(__file__))
        # Correct path for examples file, typically relative to the project root or server dir
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
            "     - 'thời gian trung bình xử lý' → AVG(TONG_TG_XL) / 60 (đổi sang giờ)",
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
                            self.last_user_subject = value  # ✅ cập nhật cá nhân gần nhất
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
            example_sql_upper = example["sql"].upper() # Chuyển SQL sang chữ hoa để so sánh từ khóa bảng

            score = 0

            # 1. Tăng điểm cho từ khóa chung giữa câu hỏi người dùng và câu hỏi mẫu
            common_keywords = question_words.intersection(set(example_question_lower.split()))
            score += len(common_keywords) * 2 # Nhân 2 để tăng trọng số cho sự trùng lặp từ khóa

            # 2. Tăng điểm cho ý định GROUP BY (ví dụ: "mỗi", "từng")
            if ("mỗi" in question_lower or "từng" in question_lower) and \
               ("mỗi" in example_question_lower or "từng" in example_question_lower):
                score += 5 # Điểm cộng cao cho ý định GROUP BY

            # 3. Tăng điểm nếu ví dụ SQL sử dụng bảng phù hợp với từ khóa trong câu hỏi
            # Dynamically check against TABLE_KEYWORDS for relevance
            for table, info in TABLE_KEYWORDS.items():
                if any(kw.lower() in question_lower for kw in info.get("keywords", [])) and \
                   table.upper() in example_sql_upper:
                    score += 4 # Score for relevant table keywords and table in SQL

            # 4. Tăng điểm đặc biệt cho trường hợp "mỗi tổ" không JOIN FB_GROUP
            # Đây là trường hợp bạn gặp lỗi, cần ưu tiên cực cao
            if "mỗi tổ" in question_lower and \
               "TO_NHOM" in example_sql_upper and \
               "GROUP BY" in example_sql_upper and \
               "PAKH_SLA_TO_NHOM" in example_sql_upper and \
               "JOIN FB_GROUP" not in example_sql_upper:
                score += 20 # Ưu tiên cực cao cho ví dụ giải quyết vấn đề cụ thể này

            # Chỉ thêm vào danh sách nếu có điểm dương
            if score > 0:
                scored_examples.append((score, example))

        # Sắp xếp các ví dụ theo điểm số giảm dần
        scored_examples.sort(key=lambda x: x[0], reverse=True)

        # Chọn ra số lượng ví dụ mong muốn
        relevant_examples = []
        added_questions = set() # Dùng để tránh thêm các câu hỏi trùng lặp

        for score, example in scored_examples:
            if example["question"] not in added_questions:
                relevant_examples.append(example)
                added_questions.add(example["question"])
            if len(relevant_examples) >= num_examples:
                break
        
        # Nếu không đủ ví dụ liên quan, thêm một số ví dụ tổng quát từ đầu danh sách (nếu có)
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
        # Nếu có ngữ cảnh người dùng, thêm vào cache key để phân biệt
        if self.last_user_subjects:
            key += "__" + "__".join(sorted(sub.lower() for sub in self.last_user_subjects))

        # Sử dụng cache nếu có và không force_no_cache
        if not force_no_cache and key in self.sql_cache:
            logger.info(f"[CACHE] Sử dụng SQL đã cache cho câu hỏi: {question}")
            return self.sql_cache[key]

        for attempt in range(retries):
            try:
                # ✅ SỬA ĐỔI ĐIỂM 1: Đảm bảo câu hỏi được giải quyết ngữ cảnh TỪ ĐẦU
                # và nếu là follow-up, thêm ngữ cảnh từ last_user_subjects
                current_question_for_resolution = question
                if is_follow_up and self.last_user_subjects:
                    # Tạo một chuỗi chứa tất cả các cá nhân đã biết để đưa vào ngữ cảnh
                    joined_subjects = ", ".join(self.last_user_subjects)
                    # Thêm ngữ cảnh vào câu hỏi hiện tại cho LLM xử lý
                    current_question_for_resolution = f"{question.strip()} (liên quan đến các cá nhân: {joined_subjects})"
                    logger.info(f"Đã thêm ngữ cảnh cá nhân vào câu hỏi tiếp nối: {current_question_for_resolution}")

                question_resolved = self._resolve_follow_up_question(current_question_for_resolution, is_follow_up)
                
                # ✅ SỬA ĐỔI ĐIỂM 2: Cải thiện cách xác định bảng liên quan
                # Bắt đầu với các bảng được plan bởi SQLPlanner
                planner = SQLPlanner(self._invoke_model)
                plan_result = planner.plan(question_resolved) # Dùng question_resolved ở đây
                relevant_tables = plan_result.get("tables", [])
                relevant_examples_str = self._select_relevant_examples(question)

                # Nếu planner không nhận diện được -> fallback rule-based
                if not relevant_tables:
                    logger.warning("Planner không xác định được bảng → fallback extract_relevant_tables")
                    # Dùng question_resolved để tìm bảng
                    relevant_tables = extract_relevant_tables(question_resolved)
                
                # Ép thêm nếu câu hỏi có từ khóa liệt kê + tên người
                if any(kw in question_resolved.lower() for kw in ["liệt kê", "các phản ánh"]):
                    if "PAKH_CA_NHAN" not in relevant_tables:
                        relevant_tables.append("PAKH_CA_NHAN")
                    if "PAKH" not in relevant_tables and "PAKH_NOI_DUNG_PHAN_ANH" not in relevant_tables:
                        relevant_tables.append("PAKH")

                # Thêm FB_LOCALITY nếu chưa có (như logic cũ)
                if "FB_LOCALITY" not in relevant_tables:
                    relevant_tables.append("FB_LOCALITY")

                schema_text = os.linesep + os.linesep.join(
                    self.schema_cache.get_schema(tbl) for tbl in relevant_tables if tbl
                )
                if not schema_text:
                    raise ValueError("Không tìm thấy schema phù hợp.")
                
                # ✅ SỬA ĐỔI ĐIỂM 3: Điều chỉnh prompt để LLM hiểu rõ hơn về ngữ cảnh cá nhân
                # và hướng dẫn rõ ràng hơn về cách xử lý câu hỏi tiếp nối
                column_mapping_hint = generate_column_mapping_hint(question_resolved) # Dùng question_resolved và import từ utils.column_mapper

                prompt_text = f"""
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
        """
        Sinh SQL và thực thi luôn. Chỉ lưu vào cache khi execute thành công.
        - execute_fn: hàm thực thi sql, ví dụ db.execute(sql)
        """
        sql_raw = self.generate_sql(question, is_follow_up, previous_error, retries, force_no_cache)

        if execute_fn:
            try:
                # Thực thi SQL
                result = execute_fn(sql_raw)
                # ✅ Nếu thành công, lưu cache
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

    # Removed the redundant _generate_column_mapping_hint function from here.
    # It is now imported from utils.column_mapper.

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

    # --- Trả kết quả luôn ---
    def format_result_for_user(self, result: dict, filter_link: str = None) -> str:
        target = result.get("details") or result if "rows" in result else None

        if target and target["rows"]:
            columns = target["columns"]
            rows = target["rows"]

            # Tạo filter link nếu chưa có
            if filter_link is None and self.last_sql:
                params = extract_filter_params_from_sql(self.last_sql)
                filter_link = build_filter_url("http://14.160.91.174:8180/smartw/feedback/list.htm", params)

            row_limit = 3 if filter_link else len(rows)
            if len(columns) == 1 and len(rows) == 1:
                return str(rows[0][0])

            if len(columns) == 1:
                return os.linesep.join(str(row[0]) for row in rows)

            lines = [
                "- " + ", ".join(f"{col}: {val}" for col, val in zip(columns, row))
                for row in rows[:row_limit]
            ]

            if filter_link:
                lines.append(f"{os.linesep}🔗 [Xem toàn bộ danh sách tại đây]{filter_link}")

            return os.linesep.join(lines)

        elif "error" in result:
            return f"❌ Lỗi SQL: {result['error']}"
        elif "message" in result:
            return f"✅ {result['message']}"
        return "⚠️ Không có dữ liệu."

# --- Quản lý bộ nhớ & cache ---
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