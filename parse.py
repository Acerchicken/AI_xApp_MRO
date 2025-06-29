# pip install pandas
import pandas as pd
import re
from datetime import datetime
import os

def parse_log_file(filepath: str) -> pd.DataFrame:
    # Pattern để trích xuất dữ liệu từ mỗi dòng log
    # .*? là regex non-greedy, dùng để bỏ qua đoạn text không quan trọng nằm giữa các trường.
    # Tất cả (\d+) là để bắt số nguyên dương trong chuỗi log.
    log_pattern = re.compile(
        r"\[(\d{2}-\d{2}:\d{2}:\d{2}:\d{2}\.\d{3})\].*?"
        r"\[UE ID: (\d+)\].*?AMF_UE_ID: (\d+), RSRP_SOURCE: (\d+), RSRP_TARGET: (\d+), "
        r"RSRQ_SOURCE: (\d+), RSRQ_TARGET: (\d+), LOAD_SOURCE: (\d+), "
        r"PCI_SOURCE: (\d+), CELL_ID_SOURCE: (\d+), PCI_TARGET: (\d+), CELL_ID_TARGET: (\d+), "
        r"HO_FAIL_COUNT: (\d+), HO_MARGIN: (\d+), TIME_TO_TRIGGER: (\d+), HOF: (\d+)"
    )

    records = []

    with open(filepath, "r") as file:
        for line in file:
            match = log_pattern.search(line)
            if match:
                timestamp_str = match.group(1)
                # Chuyển đổi về datetime (Date-Month:Hour:Minute:Second:MilliSecond
                timestamp = datetime.strptime(timestamp_str, "%d-%m:%H:%M:%S.%f")  

                record = {
                    "Timestamp": timestamp,
                    "UE_ID": int(match.group(2)),
                    "AMF_UE_ID": int(match.group(3)),
                    "RSRP_SOURCE": int(match.group(4)),
                    "RSRP_TARGET": int(match.group(5)),
                    "RSRQ_SOURCE": int(match.group(6)),
                    "RSRQ_TARGET": int(match.group(7)),
                    "LOAD_SOURCE": int(match.group(8)),
                    "PCI_SOURCE": int(match.group(9)),
                    "CELL_ID_SOURCE": int(match.group(10)),
                    "PCI_TARGET": int(match.group(11)),
                    "CELL_ID_TARGET": int(match.group(12)),
                    "HO_FAIL_COUNT": int(match.group(13)),
                    "HO_MARGIN": int(match.group(14)),  # A3_offset
                    "TIME_TO_TRIGGER": int(match.group(15)) * 10,  # đổi sang ms
                    "HOF": int(match.group(16)),  # 0: success, 1: fail
                }
                #print(record)
                records.append(record)

    return pd.DataFrame(records)
