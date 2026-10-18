import os, random, shutil

random.seed(42)

images_dir = "data/dataset/train/images"
labels_dir = "data/dataset/train/labels"

val_images_dir = "data/dataset/valid/images"
val_labels_dir = "data/dataset/valid/labels"
os.makedirs(val_images_dir, exist_ok=True)
os.makedirs(val_labels_dir, exist_ok=True)

all_images = [f for f in os.listdir(images_dir) if f.lower().endswith((".jpg", ".jpeg", ".png"))]
random.shuffle(all_images)

val_count = int(len(all_images) * 0.15)
val_images = all_images[:val_count]

for img_name in val_images:
    label_name = os.path.splitext(img_name)[0] + ".txt"
    shutil.move(os.path.join(images_dir, img_name), os.path.join(val_images_dir, img_name))
    label_path = os.path.join(labels_dir, label_name)
    if os.path.exists(label_path):
        shutil.move(label_path, os.path.join(val_labels_dir, label_name))

print(f"Moved {len(val_images)} images (and matching labels) to validation set.")