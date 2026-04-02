import cv2
import numpy as np
import time
import mss
from crop_grid import GridCrop

def main():
    cropper = GridCrop()
    print("캡처 + crop 시작 - q 눌러서 종료")
    
    cv2.namedWindow("tile_0", cv2.WINDOW_NORMAL)
    cv2.namedWindow("tile_1", cv2.WINDOW_NORMAL)

    with mss.mss() as sct:
        monitor = sct.monitors[1]
        while True:
            screenshot = sct.grab(monitor)
            frame = np.array(screenshot)
            frame = cv2.cvtColor(frame, cv2.COLOR_BGRA2BGR)
            
            tiles = cropper.crop(frame)
            
            cv2.imshow("tile_0", tiles[0])
            cv2.imshow("tile_1", tiles[1])
            
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break

    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()