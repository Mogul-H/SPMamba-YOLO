import warnings
warnings.filterwarnings('ignore')
from ultralytics import YOLO

if __name__ == '__main__':
    model = YOLO('C:/Users/Mogul/Desktop/yolov8/runs/train/exp63/weights/best.pt')
    model.val(data=r"C:\Users\Mogul\Desktop\DUO\data.yaml",
              split='val',
              imgsz=640,
              batch=16,
              # rect=False,
              # save_json=True, # 这个保存coco精度指标的开关
              project='runs/val',
              name='exp',
              )