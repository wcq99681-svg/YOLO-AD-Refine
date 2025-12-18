# train11.py (示例)
import warnings
warnings.filterwarnings('ignore')
from ultralytics import YOLO
import torch
# 确保导入 Detect_Aphid_Light 类
from ultralytics.nn.modules.head import Detect_Aphid_Light # <<<--- 再次确认这个导入路径！

if __name__ == '__main__':
    # 1. 加载模型
    model = YOLO(r'z-yaml\yolo11-newfpn copy 4.yaml', task='detect')

    # 2. <<<--- 手动设置 Stride --->>>
    try:
        detection_head = model.model.model[-1]
        if isinstance(detection_head, Detect_Aphid_Light):
            correct_strides = torch.tensor([8., 16., 32.])
            print(f"Manually setting stride for {type(detection_head).__name__}: {correct_strides.numpy().tolist()}")
            # 使用 copy_() 来更新 buffer
            detection_head.stride.copy_(correct_strides.to(model.device))
            detection_head.anchors_calculated = False # Reset flag
            print("Running bias_init() after setting strides...")
            detection_head.bias_init()
        else:
            print(f"Warning: Last module is type {type(detection_head).__name__}, not Detect_Aphid_Light. Stride not set manually.")
    except Exception as e:
        print(f"Error accessing or setting stride on the detection head: {e}")
        # ... (错误处理)

    # 3. 开始训练
    model.train(data=r'F:\work\paper\dataset\apid.v2i.yolov8\data.yaml',
                # seed=42,
                cache=False, 
                imgsz=640,
                pretrained = False,
                epochs=220,
                batch=16,
                close_mosaic=0, 
                workers=4,# device='0',x
                optimizer='SGD', # using SGD
                # patience=0, # close earlystop
                # resume=True, # 断点续训,YOLO初始化时选择last.pt
                # amp=False, # close amp
                # fraction=0.2,
                project='runs/train',
                name='exp',
                )

