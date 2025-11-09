import numpy as np
import tensorflow as tf
from fastapi import FastAPI, WebSocket, WebSocketDisconnect
from fastapi.responses import HTMLResponse

app = FastAPI()

# --- 1. モデルのロード ---
# アプリケーションの起動時に一度だけモデルをロードする
try:
    model = tf.keras.models.load_model('har_model.h5')
    # モデルが期待する入力形状 (例: (1, 50, 3) -> 50サンプル, 3軸)
    # これはあなたのモデルに合わせて変更する必要があります
    WINDOW_SIZE = 50 
    NUM_AXES = 3
    LABELS = ["stay", "walk", "jog"] # あなたが学習させた順番に
except Exception as e:
    print(f"モデル(har_model.h5)のロードに失敗しました: {e}")
    model = None

# --- 2. リアルタイム推論のためのヘルパークラス ---
# WebSocket接続ごとに、このクラスのインスタンスを作成します。
# これにより、ユーザーごとにデータを独立して溜めることができます。
class RealTimePredictor:
    def __init__(self):
        self.data_buffer = [] # センサーデータを溜めるバッファ

    def predict_action(self, sensor_data: dict):
        # {x: ..., y: ..., z: ...} の形式でデータが来ると仮定
        self.data_buffer.append([sensor_data['x'], sensor_data['y'], sensor_data['z']])

        # バッファがモデルの要求するウィンドウサイズに達したら推論
        if len(self.data_buffer) < WINDOW_SIZE:
            return None # まだデータが足りない

        # データをNumPy配列に変換し、モデルの入力形状に合わせる
        # (50, 3) -> (1, 50, 3)
        input_data = np.array(self.data_buffer[-WINDOW_SIZE:]) # 末尾から50件取得
        input_data = np.expand_dims(input_data, axis=0)

        # !!! 注意 !!!
        # モデル学習時と同じ前処理（正規化など）がここで必要になる場合があります
        # input_data = (input_data - mean) / std ...など

        # 推論実行
        prediction = model.predict(input_data)
        predicted_index = np.argmax(prediction[0])
        action = LABELS[predicted_index]

        # バッファを少しずらす（例: 10件分ずらして、次の推論に備える）
        # これをしないと毎回同じデータで推論してしまう
        self.data_buffer = self.data_buffer[10:] # スライディングウィンドウ

        return action

# --- 3. WebSocket エンドポイント ---
@app.websocket("/ws")
async def websocket_endpoint(websocket: WebSocket):
    await websocket.accept()
    
    if model is None:
        await websocket.send_text("Error: Model is not loaded on server.")
        await websocket.close()
        return

    predictor = RealTimePredictor() # ユーザー専用の予測器を作成

    try:
        while True:
            # クライアント (JS) からJSON形式でデータを受け取る
            data = await websocket.receive_json() 
            # data は {x: 0.1, y: 0.2, z: 9.8} のようになっていると想定

            # 推論を実行
            action = predictor.predict_action(data)

            # 推論結果が出た場合のみクライアントに送信
            if action:
                await websocket.send_text(action)

    except WebSocketDisconnect:
        print("Client disconnected")

# --- 4. フロントエンド (HTML) を提供するエンドポイント ---
# (Renderではフロントとバックを分けることもできますが、簡単のため一緒にします)
@app.get("/")
async def get():
    # index.html を読み込んで返す
    with open("index.html", "r", encoding="utf-8") as f:
        html_content = f.read()
    return HTMLResponse(html_content)