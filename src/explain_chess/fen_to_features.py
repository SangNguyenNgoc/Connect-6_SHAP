import chess
import chess.engine


def get_material_score(board):
    piece_values = {
        chess.PAWN: 1,
        chess.KNIGHT: 3,
        chess.BISHOP: 3,
        chess.ROOK: 5,
        chess.QUEEN: 9,
    }
    score_white = sum(piece_values.get(piece.piece_type, 0) for piece in board.piece_map().values() if piece.color)
    score_black = sum(piece_values.get(piece.piece_type, 0) for piece in board.piece_map().values() if not piece.color)
    return score_white, score_black, score_white - score_black


def get_castling_rights(board):
    return int(board.has_kingside_castling_rights(chess.WHITE)), int(
        board.has_queenside_castling_rights(chess.WHITE)), int(board.has_kingside_castling_rights(chess.BLACK)), int(
        board.has_queenside_castling_rights(chess.BLACK))


def get_piece_counts(board):
    return len(board.pieces(chess.PAWN, chess.WHITE)), len(board.pieces(chess.PAWN, chess.BLACK)), len(
        board.pieces(chess.ROOK, chess.WHITE)) - len(board.pieces(chess.ROOK, chess.BLACK)), len(
        board.pieces(chess.QUEEN, chess.WHITE)) - len(board.pieces(chess.QUEEN, chess.BLACK))


def get_attack_info(board):
    white_attackers = sum(board.is_attacked_by(chess.WHITE, square) for square in board.pieces(chess.KING, chess.BLACK))
    black_attackers = sum(board.is_attacked_by(chess.BLACK, square) for square in board.pieces(chess.KING, chess.WHITE))
    return white_attackers, black_attackers


def extract_features(fen):
    board = chess.Board(fen)

    material_white, material_black, material_diff = get_material_score(board)
    w_castle_ks, w_castle_qs, b_castle_ks, b_castle_qs = get_castling_rights(board)
    pawns_white, pawns_black, rook_diff, queen_diff = get_piece_counts(board)
    w_attackers, b_attackers = get_attack_info(board)

    features = {
        "material_score_white": material_white,  # Điểm vật chất của trắng
        "material_score_black": material_black,  # Điểm vật chất của đen
        "material_difference": material_diff,  # Chênh lệch điểm vật chất (trắng - đen)
        "white_castle_kingside": w_castle_ks,  # Trắng có quyền nhập thành cánh vua (1: có, 0: không)
        "white_castle_queenside": w_castle_qs,  # Trắng có quyền nhập thành cánh hậu (1: có, 0: không)
        "black_castle_kingside": b_castle_ks,  # Đen có quyền nhập thành cánh vua (1: có, 0: không)
        "black_castle_queenside": b_castle_qs,  # Đen có quyền nhập thành cánh hậu (1: có, 0: không)
        "pawns_white": pawns_white,  # Số quân tốt của trắng
        "pawns_black": pawns_black,  # Số quân tốt của đen
        "rook_difference": rook_diff,  # Chênh lệch số quân xe (trắng - đen)
        "queen_difference": queen_diff,  # Chênh lệch số quân hậu (trắng - đen)
        "king_attackers_white": w_attackers,  # Số quân trắng tấn công vua đen
        "king_attackers_black": b_attackers,  # Số quân đen tấn công vua trắng
        "legal_moves_white": len(list(board.legal_moves)) if board.turn else 0,  # Số nước đi hợp lệ của trắng
        "legal_moves_black": len(list(board.legal_moves)) if not board.turn else 0,  # Số nước đi hợp lệ của đen
        "check_moves_white": int(board.is_check() and board.turn),  # Trắng đang chiếu vua
        "check_moves_black": int(board.is_check() and not board.turn),  # Đen đang chiếu vua
        "advantage": 0
    }

    return features


import numpy as np
import re


def parse_connect6_fen(fen):
    board_state, turn, move_count, a , last_move, b, next_move = fen.split()

    # Kích thước bàn cờ
    rows = board_state.split('/')
    height = 10
    width = 10
    # Lượt đi tiếp theo, số lượt đã đi
    next_turn = 'black' if turn == '[b]' else 'white'
    move_count = int(move_count)

    # Nước đi cuối cùng là của ai và ở đâu
    last_move_player = 'black' if move_count % 2 == 1 else 'white'
    last_move_position = last_move if last_move != '-' else None

    # Chuyển bàn cờ sang ma trận
    board = np.zeros((10, 10), dtype=int)
    piece_map = {'w': 2, 'b': 1}

    for i, row in enumerate(rows):
        col_idx = 0
        for char in row:
            if char.isdigit():
                col_idx += int(char)
            elif char in piece_map:
                board[i, col_idx] = piece_map[char]
                col_idx += 1

    # Đếm số lượng đường thẳng (ngang, dọc, chéo) có độ dài 2, 3, 4, 5
    def count_lines(board, player):
        directions = [(1, 0), (0, 1), (1, 1), (1, -1)]  # dọc, ngang, chéo xuống, chéo lên
        line_counts = {2: 0, 3: 0, 4: 0, 5: 0}
        seen = set()  # Dùng để tránh đếm trùng các đoạn con

        height, width = board.shape

        for r in range(height):
            for c in range(width):
                if board[r, c] != player:
                    continue

                for dr, dc in directions:
                    length = 0
                    positions = []  # Lưu lại vị trí để kiểm tra trùng lặp

                    nr, nc = r, c
                    while 0 <= nr < height and 0 <= nc < width and board[nr, nc] == player:
                        length += 1
                        positions.append((nr, nc))  # Lưu vị trí của đường
                        nr += dr
                        nc += dc

                    if length >= 2 and tuple(positions) not in seen:
                        seen.add(tuple(positions))  # Đánh dấu đường đã đếm
                        if length >= 5:
                            line_counts[5] += 1  # Đếm đường 5 và bỏ qua đường con
                        elif length == 4 and line_counts[5] == 0:
                            line_counts[4] += 1
                        elif length == 3 and line_counts[5] == 0 and line_counts[4] == 0:
                            line_counts[3] += 1
                        elif length == 2 and line_counts[5] == 0 and line_counts[4] == 0 and line_counts[3] == 0:
                            line_counts[2] += 1

        return line_counts

    opponent = 2 if next_turn == 'black' else 1
    line_counts_competitor = count_lines(board, opponent)
    line_counts_player = count_lines(board, 1 if opponent == 2 else 1)


    # return {
    #     "next_turn": 1 if next_turn == 'black' else 2,
    #     "move_count": move_count,
    #     "last_move_player": 1 if last_move_player == 'black' else 2,
    #     "last_move_position": last_move_position[1:],
    #     "line_counts_competitor_2": 0 if line_counts_competitor[2] == 0 else 1,
    #     "line_counts_competitor_3": 0 if line_counts_competitor[3] == 0 else 1,
    #     "line_counts_competitor_4": 0 if line_counts_competitor[4] == 0 else 1,
    #     "line_counts_competitor_5": 0 if line_counts_competitor[5] == 0 else 1,
    #     "line_counts_player_2": 0 if line_counts_player[2] == 0 else 1,
    #     "line_counts_player_3": 0 if line_counts_player[3] == 0 else 1,
    #     "line_counts_player_4": 0 if line_counts_player[4] == 0 else 1,
    #     "line_counts_player_5": 0 if line_counts_player[5] == 0 else 1,
    #     "next_move": check_move(board, int(next_move[1:]), 1 if next_turn == 'black' else 2),
    # }

    return {
        "next_turn": 0 if next_turn == 'black' else 1,
        "move_count": move_count,
        "last_move_player": 0 if last_move_player == 'black' else 1,
        "last_move_position": last_move_position[1:],
        "line_counts_competitor_2": 0 if line_counts_competitor[2] == 0 else 1,
        "line_counts_competitor_3": 0 if line_counts_competitor[3] == 0 else 1,
        "line_counts_competitor_4": 0 if line_counts_competitor[4] == 0 else 1,
        "line_counts_competitor_5": 0 if line_counts_competitor[5] == 0 else 1,
        "line_counts_player_2": 0 if line_counts_player[2] == 0 else 1,
        "line_counts_player_3": 0 if line_counts_player[3] == 0 else 1,
        "line_counts_player_4": 0 if line_counts_player[4] == 0 else 1,
        "line_counts_player_5": 0 if line_counts_player[5] == 0 else 1,
        "next_move": check_move(board, int(next_move[1:]), 1 if next_turn == 'black' else 2),
    }


def check_move(board, move, player):
    """
    Kiểm tra nước đi có tạo chuỗi 4 hoặc chặn chuỗi 4 của đối thủ hay không.

    Parameters:
    - board: Mảng 2 chiều đại diện cho bàn cờ (0: trống, 1: người chơi 1, 2: người chơi 2)
    - move: Số từ 0 đến 99 đại diện cho vị trí nước đi mới
    - player: Người chơi thực hiện nước đi (1 hoặc 2)

    Returns:
    - 1: Nếu nước đi tạo chuỗi 4 liên tiếp
    - 2: Nếu nước đi chặn chuỗi 4 của đối thủ
    - 3: Nếu nước đi thỏa cả hai điều kiện trên
    - 0: Nếu không thỏa điều kiện nào
    """
    row, col = divmod(move, 10)  # Chuyển đổi số thành tọa độ hàng và cột
    directions = [(1, 0), (0, 1), (1, 1), (1, -1)]  # Dọc, ngang, chéo chính, chéo phụ
    opponent = 3 - player  # Lấy đối thủ (1 -> 2, 2 -> 1)
    board[row][col] = player  # Tạm thời đặt nước đi lên bàn cờ

    creates_4 = False
    blocks_4 = False

    for dr, dc in directions:
        for check_player in [player, opponent]:  # Kiểm tra cả 2 người chơi
            count = 1
            for d in [-1, 1]:  # Kiểm tra 2 hướng
                r, c = row + dr * d, col + dc * d
                while 0 <= r < len(board) and 0 <= c < len(board[0]) and board[r][c] == check_player:
                    count += 1
                    r += dr * d
                    c += dc * d

            if count >= 4:
                if check_player == player:
                    creates_4 = True
                else:
                    blocks_4 = True

    board[row][col] = 0  # Hoàn nguyên nước đi

    if creates_4 and blocks_4:
        return 1
    elif creates_4:
        return 1
    elif blocks_4:
        return 2
    return 0

# Ví dụ sử dụng
# fen = "w9/5w3b/5b4/5bw3/4wb4/w3bb2w1/3w1bw3/10/10/10 [b] 15 - w0 - b75"
# features = parse_connect6_fen(fen)
# print(features)