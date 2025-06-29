import pandas as pd

def detect_ping_pong(df: pd.DataFrame, time_threshold: float = 2.0) -> pd.DataFrame:
    df["PingPong"] = False
    last_seen = {}

    for i in range(len(df)):
        row = df.iloc[i]
        ue_id = row["UE_ID"]
        curr_src = row["CELL_ID_SOURCE"]
        curr_tgt = row["CELL_ID_TARGET"]
        curr_time = row["Timestamp"]

        # Kiểm tra nếu đã từng HO đến curr_src trước đó
        if ue_id in last_seen:
            prev_src, prev_tgt, prev_time = last_seen[ue_id]
            if (
                prev_src == curr_tgt and
                prev_tgt == curr_src and
                (curr_time - prev_time).total_seconds() <= time_threshold
            ):
                df.at[i, "PingPong"] = True

        # Cập nhật trạng thái HO gần nhất
        last_seen[ue_id] = (curr_src, curr_tgt, curr_time)

    return df