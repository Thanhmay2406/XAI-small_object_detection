from ultralytics import YOLO

model = YOLO('/kaggle/working/XAI-small_object_detection/configs/yolov8s-p2.yaml')

'''
python src/yolov8_p2.py \
  --data your-data.yaml \
  --model config/yolov8s-p2.yaml \
  --epochs 100 \
  --imgsz 640 \
  --batch 16 \
  --device 0 \
  --workers 8 \
  --patience 20 \
  --project runs/train \
  --name yolov8_p2_run \
  --seed 42
'''

model.train(
        data='/kaggle/input/datasets/thanhmay2406/dataset-for-research/drill_bit/data.yaml',
        epochs=1,
        imgsz=640,
        batch=16,
        device=0,
        workers=8,
        patience=20,
        name='baseline_yolov8_p2_train',
        seed=42,
        )

model.val(
        data='/kaggle/input/datasets/thanhmay2406/dataset-for-research/drill_bit/data.yaml',
        split='val',
        name='baseline_eval_val',
        )

model.val(
        data='/kaggle/input/datasets/thanhmay2406/dataset-for-research/drill_bit/data.yaml',
        split='test',
        name='baseline_eval_test',
        )

