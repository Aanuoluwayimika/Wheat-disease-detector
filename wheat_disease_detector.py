
import os
import json
import shutil
from ultralytics import YOLO

# Dataset configuration
yaml_content = """
path: /content/wheat_disease/wheat_disease_combined
train: train/images
val: valid/images
test: test/images
nc: 9
names:
- Healthy
- Leaf_Rust
- Yellow_Rust
- Powdery_Mildew
- Smut
- Stem_Rust
- barley_yellow_dwarf
- Septoria
- CrownAndRootRot
"""

with open('/content/wheat_disease/wheat_disease_combined/data.yaml', 'w') as f:
    f.write(yaml_content)

# Load and train YOLOv8 model
model = YOLO('yolov8n.pt')

model.train(
    data='/content/wheat_disease/wheat_disease_combined/data.yaml',
    epochs=10,
    imgsz=640,
    batch=16,
    name='wheat_disease_detector',
    patience=5
)

# Run detection on test images
import matplotlib.pyplot as plt

test_path = '/content/wheat_disease/wheat_disease_combined/test/images'
test_images = os.listdir(test_path)[:6]

fig, axes = plt.subplots(2, 3, figsize=(16, 10))
axes = axes.flatten()

for idx, img_name in enumerate(test_images):
    img_path = os.path.join(test_path, img_name)
    results = model.predict(source=img_path, conf=0.25, save=False, verbose=False)
    result_img = results[0].plot()
    axes[idx].imshow(result_img[:, :, ::-1])
    axes[idx].axis('off')
    axes[idx].set_title(f"Detections: {len(results[0].boxes)}", fontsize=10)

plt.suptitle('YOLOv8 Wheat Disease Detection - Test Images', fontsize=14)
plt.tight_layout()
plt.savefig('detection_results.png', dpi=100, bbox_inches='tight')
plt.show()
