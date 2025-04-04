import chess
import chess.engine
import pandas as pd

from src.explain_chess import extract_features, parse_connect6_fen

ENGINE_PATH = "engine/lc0.exe"


def process_fen_file(csv_path):
    df = pd.read_csv(csv_path)

    features_list = []

    for fen in df['fen'].head(5000):
        board = chess.Board(fen)
        if board.is_checkmate():
            continue
        features = extract_features(fen)
        if features is not None:
            features['advantage'] = get_best_move(fen)
            features_list.append(features)

    features_df = pd.DataFrame(features_list)
    features_df.to_csv("data/chess_result_test.csv", index=True)

    return features_df

def process_connect6_fen_file(csv_path, output_path="data/connect6_result.csv"):
    df = pd.read_csv(csv_path)
    board_count = 0

    with open(output_path, mode='w', newline='') as f:
        # Ghi header một lần
        first_row = True

        for fen in df['fen']:
            if fen.startswith("="):
                board_count += 1
                continue
            features_df = pd.DataFrame([parse_connect6_fen(fen)])  # Chuyển dict thành DataFrame
            print(features_df)
            # Ghi từng dòng vào file
            features_df.to_csv(f, mode='a', header=first_row, index=True)
            first_row = False  # Sau lần đầu tiên, không ghi lại header

    print(f"Processed {board_count} boards.")
    return output_path

def get_best_move(fen):
    board = chess.Board(fen)

    # Kiểm tra nếu ván cờ đã kết thúc
    if board.is_game_over():
        result = board.result()
        return {
            'advantage': 0 if result == '1/2-1/2' else (1 if result == '1-0' else -1)
        }

    with chess.engine.SimpleEngine.popen_uci(ENGINE_PATH) as engine:
        # Phân tích nước đi tốt nhất với thời gian giới hạn 2 giây
        analysis = engine.analyse(board, chess.engine.Limit(time=5))

        # Trích xuất điểm số đánh giá lợi thế
        score = analysis['score'].relative.score(mate_score=10000)
        advantage = 1 if score > 0 else 0

        print("--- Phân tích bàn cờ ---")
        print(f"Trạng thái FEN: {fen}")
        print(f"Lợi thế: {'Trắng' if advantage == 1 else 'Đen'}")
        print(f"Điểm đánh giá: {score} (số dương lợi thế trắng, âm lợi thế đen)")
        print("------------------------")

        return advantage


