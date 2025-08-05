# server/utils/relation_loader.py

import logging
import os

def load_relations(filepath="./utils/relations.txt"):
    relations = {}
    concat_mappings = []

    with open(filepath, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line or line.startswith("#") or line.startswith("TYPE:"):
                continue

            if "→" not in line:
                logging.warning(f"Bỏ qua dòng không hợp lệ: {line}")
                continue

            try:
                left, right = [s.strip() for s in line.split("→")]

                # Xử lý CONCAT_MAPPING
                if "+" in left:
                    # Ví dụ: PAKH.TINH_THANH_PHO + '' + PAKH.QUAN_HUYEN + '' + PAKH.PHUONG_XA
                    parts = [part.strip() for part in left.split("+") if "." in part]
                    from_fields = [p for p in parts if "." in p]

                    to_table, to_col = [s.strip() for s in right.split(".")]

                    concat_mappings.append({
                        "type": "concat_mapping",
                        "from": from_fields,
                        "to": {"table": to_table, "column": to_col}
                    })
                    continue

                # Xử lý ánh xạ bình thường
                if "." not in left or "." not in right:
                    logging.warning(f"Bỏ qua dòng không hợp lệ (thiếu dấu '.'): {line}")
                    continue

                left_table, left_col = left.split(".")
                right_table, right_col = right.split(".")

                relations.setdefault(left_table, {})[left_col] = {
                    "ref_table": right_table,
                    "ref_column": right_col
                }

            except Exception as e:
                logging.error(f"Lỗi khi phân tích dòng: {line} → {e}")
                continue

    return relations, concat_mappings
