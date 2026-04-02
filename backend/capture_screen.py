import mss
import cv2
import numpy as np
import time
import os
import win32gui
import win32ui
import win32con
from ctypes import windll

_captured_window_logged = False

ZOOM_KEYWORDS = ["Zoom 회의", "Zoom Meeting"]

def find_zoom_window():
    result = []
    def callback(hwnd, _):
        title = win32gui.GetWindowText(hwnd)
        rect = win32gui.GetWindowRect(hwnd)
        w = rect[2] - rect[0]
        h = rect[3] - rect[1]
        if w > 400 and h > 300:
            for kw in ZOOM_KEYWORDS:
                if kw in title:
                    result.append((hwnd, title, w, h))
                    print(f"후보 창: {title} | {w}x{h}")  # 추가
                    break
    win32gui.EnumWindows(callback, None)
    
    if not result:
        return None, None
    
    result.sort(key=lambda x: x[2] * x[3], reverse=True)
    print(f"선택된 창: {result[0][1]} | {result[0][2]}x{result[0][3]}")  # 추가
    return result[0][0], result[0][1]

def capture_zoom():
    """Zoom 창 자동 찾아서 백그라운드 캡처"""
    global _captured_window_logged
    
    hwnd, title = find_zoom_window()
    if not hwnd:
        print("Zoom 창을 찾을 수 없어요!")
        return None
    
    if not _captured_window_logged:
        print(f"캡처 대상: {title}")
        _captured_window_logged = True
    
    left, top, right, bottom = win32gui.GetWindowRect(hwnd)
    w = right - left
    h = bottom - top
    
    hwndDC = win32gui.GetWindowDC(hwnd)
    mfcDC = win32ui.CreateDCFromHandle(hwndDC)
    saveDC = mfcDC.CreateCompatibleDC()
    
    saveBitMap = win32ui.CreateBitmap()
    saveBitMap.CreateCompatibleBitmap(mfcDC, w, h)
    saveDC.SelectObject(saveBitMap)
    
    windll.user32.PrintWindow(hwnd, saveDC.GetSafeHdc(), 3)
    
    bmpinfo = saveBitMap.GetInfo()
    bmpstr = saveBitMap.GetBitmapBits(True)
    
    frame = np.frombuffer(bmpstr, dtype=np.uint8).reshape(bmpinfo['bmHeight'], bmpinfo['bmWidth'], 4)
    frame = cv2.cvtColor(frame, cv2.COLOR_BGRA2BGR)

    # h 대신 frame 실제 높이 사용
    # fh = frame.shape[0]
    # top_cut = 90
    # bottom_cut = 90
    # frame = frame[top_cut:fh-bottom_cut, :]

    # print(f"창 크기: {w}x{h}")
    # print(f"비트맵 크기: {bmpinfo['bmWidth']}x{bmpinfo['bmHeight']}")
    
    win32gui.DeleteObject(saveBitMap.GetHandle())
    saveDC.DeleteDC()
    mfcDC.DeleteDC()
    win32gui.ReleaseDC(hwnd, hwndDC)
    
    return frame

def capture_window(window_title="Zoom 회의"):
    """특정 창만 캡처 - 제목이 window_title로 시작하는 창"""
    global _captured_window_logged
    
    result = []
    def callback(hwnd, _):
        title = win32gui.GetWindowText(hwnd)
        if title.startswith(window_title):
            result.append((hwnd, title))
    win32gui.EnumWindows(callback, None)
    
    if not result:
        print(f"창을 찾을 수 없어요: {window_title}")
        return None
    
    hwnd = result[0][0]
    
    if not _captured_window_logged:
        print(f"캡처 대상 창: {result[0][1]}")
        _captured_window_logged = True
    
    left, top, right, bottom = win32gui.GetWindowRect(hwnd)
    w = right - left
    h = bottom - top
    
    hwndDC = win32gui.GetWindowDC(hwnd)
    mfcDC = win32ui.CreateDCFromHandle(hwndDC)
    saveDC = mfcDC.CreateCompatibleDC()
    
    saveBitMap = win32ui.CreateBitmap()
    saveBitMap.CreateCompatibleBitmap(mfcDC, w, h)
    saveDC.SelectObject(saveBitMap)
    
    windll.user32.PrintWindow(hwnd, saveDC.GetSafeHdc(), 3)
    
    bmpinfo = saveBitMap.GetInfo()
    bmpstr = saveBitMap.GetBitmapBits(True)
    
    frame = np.frombuffer(bmpstr, dtype=np.uint8).reshape(bmpinfo['bmHeight'], bmpinfo['bmWidth'], 4)
    frame = cv2.cvtColor(frame, cv2.COLOR_BGRA2BGR)
    frame = cv2.resize(frame, (1920, 1080))
    
    win32gui.DeleteObject(saveBitMap.GetHandle())
    saveDC.DeleteDC()
    mfcDC.DeleteDC()
    win32gui.ReleaseDC(hwnd, hwndDC)
    
    return frame

class ScreenCapture:
    def __init__(self, fps=10):
        self.fps = fps
        self.interval = 1.0 / fps

    def capture_frame(self):
        with mss.mss() as sct:
            monitor = {
                "top": 30,
                "left": 0,
                "width": 1920,
                "height": 990
            }
            screenshot = sct.grab(monitor)
            frame = np.array(screenshot)
            frame = cv2.cvtColor(frame, cv2.COLOR_BGRA2BGR)
            return frame

    def start_capture(self, save_dir=None, save_interval=10):
        print(f"캡처 시작 - {self.fps}fps")
        if save_dir:
            os.makedirs(save_dir, exist_ok=True)
        
        frame_count = 0
        cv2.namedWindow("capture", cv2.WINDOW_NORMAL)
        
        while True:
            start = time.time()
            frame = self.capture_frame()
            
            if frame is not None:
                cv2.imshow("capture", frame)
                
            if save_dir and frame_count % save_interval == 0:
                timestamp = int(time.time() * 1000)
                filename = f"{save_dir}/frame_{timestamp}.jpg"
                cv2.imwrite(filename, frame)
                print(f"저장: {filename}")
            
            frame_count += 1
            
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break
            
            elapsed = time.time() - start
            time.sleep(max(0, self.interval - elapsed))
        
        cv2.destroyAllWindows()

if __name__ == "__main__":
    cap = ScreenCapture(fps=10)
    cap.start_capture(save_dir="../data/video_capture", save_interval=10)