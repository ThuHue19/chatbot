# --- File: server/utils/sql_planner.py ---
import logging
import json

logger = logging.getLogger(__name__)

class SQLPlanner:
    def __init__(self, llm):
        self.llm = llm

    def plan(self, question: str) -> dict:
        prompt = f"""
Bạn là một trợ lý SQL Planner.

Dựa vào câu hỏi dưới đây, hãy:
1. Xác định "intent" của câu hỏi: một trong các giá trị sau:
   - "thống kê" nếu câu hỏi về tổng số, số lượng, tỷ lệ...
   - "liệt kê" nếu câu hỏi yêu cầu danh sách, phản ánh, chi tiết
   - "so sánh" nếu câu hỏi yêu cầu so sánh nhiều hơn 1 đối tượng
   - "khác" nếu không xác định được

2. Đưa ra danh sách tên bảng liên quan cần truy vấn (dưới dạng mảng). Một số bảng có thể bao gồm: PAKH, PAKH_CA_NHAN, PAKH_SLA_CA_NHAN, PAKH_SLA_PHONG_BAN, FB_TYPE_NOC, FB_GROUP, FB_LOCALITY...

Trả về kết quả dưới dạng JSON như ví dụ:
{{
  "intent": "thống kê",
  "tables": ["PAKH_SLA_CA_NHAN", "FB_REASON"]
}}

Chỉ trả về JSON, không giải thích thêm.

Câu hỏi:
{question}
"""
        try:
            response = self.llm(prompt)
            response = response.strip()
            if response.startswith("```json"):
                response = response[7:].strip()
            if response.endswith("```"):
                response = response[:-3].strip()
            result = json.loads(response)
            if "intent" not in result or "tables" not in result:
                raise ValueError("Thiếu trường 'intent' hoặc 'tables'")
            return result
        except Exception as e:
            logger.warning(f"\u26a0\ufe0f Lỗi khi phân tích kế hoạch truy vấn: {e}")
            return {"intent": "unknown", "tables": []}