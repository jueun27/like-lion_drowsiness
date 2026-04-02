from fastapi import FastAPI, WebSocket
from fastapi.websockets import WebSocketDisconnect
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse
from contextlib import asynccontextmanager
import asyncio
import json
import sys
import os

sys.path.append(os.path.dirname(__file__))
from pipeline import capture_and_crop, build_payload, update_feature_buffer

connected_clients = []

async def feature_loop():
    """10fps로 feature 추출 + 버퍼 쌓기"""
    while True:
        await asyncio.sleep(0.1)
        try:
            tiles = capture_and_crop()
            for student_id, tile in tiles.items():
                update_feature_buffer(student_id, tile)
        except Exception as e:
            print(f"feature 에러: {e}")

async def broadcast_loop():
    """1초마다 상태 판정 + 이미지 전송"""
    while True:
        await asyncio.sleep(1)
        if not connected_clients:
            continue
        try:
            tiles = capture_and_crop()
            for student_id, tile in tiles.items():
                payload = build_payload(student_id, tile)
                for client in connected_clients:
                    await client.send_text(json.dumps(payload))
        except Exception as e:
            print(f"전송 에러: {e}")

@asynccontextmanager
async def lifespan(app):
    asyncio.create_task(feature_loop())
    asyncio.create_task(broadcast_loop())
    yield

app = FastAPI(lifespan=lifespan)

@app.get("/new_dashboard.html")
def dashboard_page():
    return FileResponse("../front/new_dashboard.html")

@app.get("/sample_detail_updated.html")
def detail_page():
    return FileResponse("../front/sample_detail_updated.html")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

@app.websocket("/ws")
async def websocket_endpoint(websocket: WebSocket):
    await websocket.accept()
    connected_clients.append(websocket)
    print(f"클라이언트 연결됨. 현재 연결 수: {len(connected_clients)}")
    try:
        while True:
            await websocket.receive_text()
    except WebSocketDisconnect:
        connected_clients.remove(websocket)
        print("클라이언트 연결 해제됨")

@app.get("/")
def root():
    return {"status": "running"}

@app.get("/test-send")
async def test_send():
    dummy = {
        "student_id": "tile_01",
        "state": "drowsy_sign",
        "score": 0.85,
        "event_label": "eyes_closed_long",
        "flags": {"fsm_triggered": True},
        "features": {"ear": 0.15, "pitch": -12.5}
    }
    await send_state(dummy)
    return {"status": "sent"}

async def send_state(data: dict):
    for client in connected_clients:
        await client.send_text(json.dumps(data))

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)