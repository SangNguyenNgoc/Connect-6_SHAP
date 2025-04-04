import pandas as pd
import shap
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import confusion_matrix, classification_report, accuracy_score, precision_score, recall_score, f1_score, ConfusionMatrixDisplay
from imblearn.under_sampling import RandomUnderSampler, ClusterCentroids


def split_features_and_labels(csv_path):
    # Đọc dữ liệu từ file CSV
    df = pd.read_csv(csv_path)

    # Tách X (đặc trưng) và y (nhãn)
    X = df.drop(columns=['next_move']).drop(columns=['Unnamed: 0'])
    y = pd.DataFrame(df, columns=['next_move'])
    return X, y


def save_shap_explanation(model, X_test, y_test, explainer, shap_values,
                          output_path="chess_result/shap_explanation.csv"):
    # Chuyển SHAP values thành DataFrame
    shap_df = pd.DataFrame(shap_values.values, columns=X_test.columns)

    # Thêm cột base value và tổng giá trị SHAP
    shap_df['base_value'] = shap_values.base_values
    shap_df['shap_sum'] = shap_df.sum(axis=1)

    # Thêm nhãn dự đoán và nhãn thực tế (nếu có)
    shap_df['predicted_class'] = model.predict(X_test)
    shap_df['actual_class'] = y_test

    # Thêm giá trị kỳ vọng từ explainer
    shap_df['expected_value'] = shap_df['predicted_class'].apply(lambda c: explainer.expected_value[c])

    # Xuất DataFrame ra file CSV
    shap_df.to_csv(output_path, index=False)

    print(f"SHAP explanation saved to: {output_path}")

def explain():
    X_raw, y_raw = split_features_and_labels("data/connect6_result.csv")
    X, y = RandomUnderSampler().fit_resample(X_raw, y_raw)

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    y_train = y_train.values.ravel()
    y_test = y_test.values.ravel()

    # Huấn luyện mô hình proxy
    model = RandomForestClassifier(n_estimators=200)
    model.fit(X_train, y_train)

    # Áp dụng SHAP để giải thích quyết định
    explainer = shap.TreeExplainer(model, X_test)
    shap_values = explainer(X_test)

    print("Shape of SHAP values:", shap_values.shape)
    print("Shape of X_test:", X_test.shape)

    if len(shap_values.shape) > 2:
        shap_values = shap_values.mean(axis=2)  # Hoặc chọn slice phù hợp

    # Vẽ biểu đồ SHAP
    shap.summary_plot(shap_values, X_test, max_display=23)
    shap.summary_plot(shap_values, X_test, max_display=23, plot_type="bar")


    # Xuất kết quả ra file
    # shap_df = pd.DataFrame(shap_values.values, columns=X.columns)
    # shap_df.to_csv("chess_result/shap_explanation.csv", index=False)
    # print(shap_values.values.shape)
    save_shap_explanation(model, X_test, y_test, explainer, shap_values)

    # Dự đoán trên tập kiểm tra
    y_pred = model.predict(X_test)

    # Tính ma trận lỗi
    cm = confusion_matrix(y_test, y_pred)

    # In ra ma trận lỗi
    print("Confusion Matrix:")
    print(cm)

    print(cm.shape)  # Kích thước ma trận nhầm lẫn
    print(model.classes_)

    # Vẽ trực quan ma trận lỗi
    disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=model.classes_)
    disp.plot(cmap="Blues")

    # Tính các chỉ số đánh giá
    accuracy = accuracy_score(y_test, y_pred)
    precision = precision_score(y_test, y_pred, average='weighted')  # Trọng số theo số lượng mẫu
    recall = recall_score(y_test, y_pred, average='weighted')
    f1 = f1_score(y_test, y_pred, average='weighted')

    # In ra kết quả
    print(f"\nAccuracy: {accuracy * 100:.2f}%")
    print(f"Precision: {precision * 100:.2f}%")
    print(f"Recall: {recall * 100:.2f}%")
    print(f"F1-score: {f1 * 100:.2f}%")

    # In báo cáo chi tiết từng lớp
    print("\nClassification Report:")
    print(classification_report(y_test, y_pred))