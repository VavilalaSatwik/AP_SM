from ultralytics import YOLO

# Provide the path to the model file
model_path = "traced_model.pt"

# Initialize YOLO model with the model file path
yolo_model = YOLO(model_path)

# Perform inference
results = yolo_model("forest.mp4", show=True, save=True)
print(results)







