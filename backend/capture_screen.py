import mss
import cv2
import numpy as np
import time
import os
import win32gui
import win32ui
import win32con
from ctypes import windll

_captured_window_logged = True

def capture_window(window_title="Zoom 회의"):
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
    
    # 처음 한 번만 출력
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
        """현재 화면을 캡처"""
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
            # frame = cv2.resize(frame, (1920, 1080))
            return frame

    def start_capture(self, save_dir=None, save_interval=10):
        """실시간 캡처 시작 (테스트용)"""
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