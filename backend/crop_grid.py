import cv2
import numpy as np
import os

class GridCrop:
    def __init__(self, rows=4, cols=4, screen_w=1920, screen_h=1080):
        self.rows = rows
        self.cols = cols
        self.screen_w = screen_w
        self.screen_h = screen_h
        self.tile_w = screen_w // cols   # 480
        self.tile_h = screen_h // rows   # 270

    def crop(self, frame):
        tiles = {}
        student_id = 1
        for row in range(self.rows):
            for col in range(self.cols):
                x1 = col * self.tile_w
                y1 = row * self.tile_h
                x2 = x1 + self.tile_w
                y2 = y1 + self.tile_h
                tile = frame[y1:y2, x1:x2]
                tiles[student_id] = tile
                student_id += 1
        return tiles

    def save_tiles(self, tiles, save_dir, filename):
        os.makedirs(save_dir, exist_ok=True)
        name, ext = os.path.splitext(filename)
        for student_id, tile in tiles.items():
            filepath = os.path.join(save_dir, f"{name}_student{student_id}{ext}")
            cv2.imwrite(filepath, tile)

if __name__ == "__main__":
    frame = cv2.imread(r"C:\Users\User\Desktop\drowsiness\data\video_capture\test.jpg")
    
    cropper = GridCrop()
    tiles = cropper.crop(frame)
    
    print(f"타일 개수: {len(tiles)}")
    for sid, tile in tiles.items():
        print(f"학생 {sid}: {tile.shape}")
    
    cropper.save_tiles(tiles, "../data/crops", "test.jpg")
    print("저장 완료!")
    
    for sid, tile in tiles.items():
        cv2.imshow(f"tile_{sid}", tile)
    cv2.waitKey(0)
    cv2.destroyAllWindows()