import sys
import time
import json
import math
from pathlib import Path
import numpy as np
import cv2
from PIL import Image
from ultralytics import YOLO
import torch
import torch.nn as nn
from torchvision import models, transforms
from numba import cuda

MODEL_SAVE_PATH_CLASSIFIER = 'C:/PersonID/models/person_classifier_model.pth'
CLASSES_SAVE_PATH = 'C:/PersonID/models/person_reid_classes.json'
GALLERY_BASE_PATH = 'C:/PersonID/gallery_embeddings/'
YOLO_MODEL_PATH = 'C:/PersonID/scripts/yolov8n.pt'
TORCH_DEVICE = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")#safety net for hardware use (looks for GPU on device if not attempts to utilize CPU)
SIMILARITY_THRESHOLD = 0.55

#numba cuda kernel implementation
@cuda.jit #lets cosine sim function run on the GPU
def cosine_similarity_kernel(query_embedding, gallery_embeddings, out_similarities):
    idx = cuda.grid(1) #each thread will handle on gallery embedding
    if idx < gallery_embeddings.shape[0]:
        embedding_dim = query_embedding.shape[0]
        dot, mag_query, mag_gallery = 0.0, 0.0, 0.0
        #cosine sim formula & dot product computations
        for i in range(embedding_dim):
            q_val = query_embedding[i]
            g_val = gallery_embeddings[idx, i]
            dot += q_val * g_val
            mag_query += q_val * q_val #query vector
            mag_gallery += g_val * g_val #our vector
        mag_query = math.sqrt(mag_query)
        mag_gallery = math.sqrt(mag_gallery)
        if mag_query > 0.0 and mag_gallery > 0.0:
            out_similarities[idx] = dot / (mag_query * mag_gallery)
        else:
            out_similarities[idx] = 0.0

#loads resnet18. prepping & cropping human images into vectors for cosim
def setup_embedding_model(model_path, num_classes_resnet, device):
    resnet_model = models.resnet18(weights=None) #clean resnet model
    num_ftrs = resnet_model.fc.in_features
    resnet_model.fc = nn.Linear(num_ftrs, num_classes_resnet) #matching # of classes used during training (2 for ben and pryce)
    resnet_model.load_state_dict(torch.load(model_path, map_location=device))
    resnet_model.eval()
    #process of removing layers and converting images to vectors
    feature_extractor = nn.Sequential(*list(resnet_model.children())[:-1]);
    feature_extractor = feature_extractor.to(device);
    feature_extractor.eval()
    return feature_extractor, num_ftrs #length of embedding vector(512)

#person identification via cosine similarity
def get_embedding_from_cv_crop(cv_bgr_crop, model, transform, device_to_use): #passes BGR image of detected humans through feature extractor model
    if cv_bgr_crop is None or cv_bgr_crop.size == 0: #check for valid image input
        return None
    rgb_pil_image = Image.fromarray(cv2.cvtColor(cv_bgr_crop, cv2.COLOR_BGR2RGB)) #BGR to RGB conversion
    img_transformed = transform(rgb_pil_image)
    img_batch = img_transformed.unsqueeze(0).to(device_to_use)
    #forward passing
    with torch.no_grad(): #passes preprocessed image through model to get embedding
        embedding = model(img_batch)
    embedding = torch.flatten(embedding, 1) #flattens into 1D vector
    return embedding.cpu().numpy() #convert to numpy

#main logic that uses cosine sim on GPU to find out which person a new embedding belongs to
def identify_person(current_embedding_np, threshold): #numpy vector-will out person name and similarity score
    if current_embedding_np is None or GALLERY_EMBEDDINGS_NP_ARRAY is None:
        return "no gallery", 0.0 #safety case if embedding or gallery is missing
    num_gallery = GALLERY_EMBEDDINGS_NP_ARRAY.shape[0]  #get number of people in gallery
    query_gpu = cuda.to_device(current_embedding_np.astype(np.float32).flatten()) #convert query embedding to float23 array
    gallery_gpu = cuda.to_device(GALLERY_EMBEDDINGS_NP_ARRAY) #send stored embeddings to GPU
    out_similarities_gpu = cuda.device_array(num_gallery, dtype=np.float32)
    #cuda kernel
    threads_per_block = 256
    blocks_per_grid = math.ceil(num_gallery / threads_per_block)
    cosine_similarity_kernel[blocks_per_grid, threads_per_block](query_gpu, gallery_gpu, out_similarities_gpu) #cuda kernel to compute cosin sim for all entries in parallel
    similarity_scores = out_similarities_gpu.copy_to_host() #get results
    if similarity_scores.size == 0: #error check
        return "Scoring Error", 0.0
    #finding best match with highest cosine sim score
    max_similarity = np.max(similarity_scores)
    best_match_idx = np.argmax(similarity_scores)
    #comparing against threshold
    if max_similarity >= threshold:
        return GALLERY_PERSON_NAMES[best_match_idx], max_similarity
    else:
        return "person unknown", max_similarity
#loads all models and ensures system is ready to detect and identify humans
try:
    with open(CLASSES_SAVE_PATH, 'r') as f: #load number of classes for resnet
        NUM_CLASSES_RESNET = len(json.load(f)) #load stored json files for pryce and ben
    yolo_model = YOLO(YOLO_MODEL_PATH)  #load yolov8 model
    PERSON_CLASS_INDEX = next((idx for idx, name in yolo_model.names.items() if name == 'person'), -1) #scan yolo calss for "person"
    #load resnet embedding model
    EMBEDDING_MODEL, EMBEDDING_DIM = setup_embedding_model(MODEL_SAVE_PATH_CLASSIFIER, NUM_CLASSES_RESNET, TORCH_DEVICE) #resnet18 feature extractor
    #prepare gallery of known detected people
    GALLERY_EMBEDDINGS = {}
    TARGET_PERSON_NAMES = ["Pryce", "Kenny", "Ben"]
    Path(GALLERY_BASE_PATH).mkdir(parents=True, exist_ok=True)
    #load embeddings from disk
    for person_name in TARGET_PERSON_NAMES:
        emb_path = Path(GALLERY_BASE_PATH) / f"{person_name.lower()}_gallery_embedding.npy"
        if emb_path.exists():
            GALLERY_EMBEDDINGS[person_name] = np.load(emb_path)
    GALLERY_PERSON_NAMES = list(GALLERY_EMBEDDINGS.keys())
    if GALLERY_EMBEDDINGS: #save list of names that were uploaded
        GALLERY_EMBEDDINGS_NP_ARRAY = np.vstack(list(GALLERY_EMBEDDINGS.values())).astype(np.float32)
    else:
        GALLERY_EMBEDDINGS_NP_ARRAY = None
    #inference transforms
    INFERENCE_TRANSFORM = transforms.Compose([
        transforms.Resize(256),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ])
  #safety net for errors
except Exception as e:
    print(f"ERROR. {e}")
    sys.exit()

#starts webcam, uses yolov8 and resnet18 and cosine sim
def main():
    #start webcam
    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        print("error. webcam not working")
        return
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640); cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480) #capture resolution
    print("press 'q' to quit.")
    #reading frame by frame
    while True:
        ret, frame_bgr = cap.read()
        if not ret: break
        frame_display = frame_bgr.copy() #captures frame
        results = yolo_model(frame_bgr, verbose=False, conf=0.5) #runs yolov8 on frame with 0.4 confidence level
        #process each detected box
        if results and results[0].boxes:
            for box in results[0].boxes.cpu().numpy():
                if int(box.cls[0]) == PERSON_CLASS_INDEX: #filter for person class
                    x_min, y_min, x_max, y_max = map(int, box.xyxy[0])  #get bounding box coordinates and confidence level
                    conf = box.conf[0]
                    #crop, embed, and identify embeddings
                    if x_min < x_max and y_min < y_max:
                        cropped_person = frame_bgr[y_min:y_max, x_min:x_max]
                        current_embedding = get_embedding_from_cv_crop(cropped_person, EMBEDDING_MODEL, INFERENCE_TRANSFORM, TORCH_DEVICE)
                        id_name, id_score = identify_person(current_embedding, SIMILARITY_THRESHOLD)
                        #displaying labels and bounding box
                        identified_label_display = f"{id_name} ({id_score*100:.0f}%)"
                        display_color = (0, 255, 0) if id_name not in ["unknown person", "analyzing..."] else (0, 165, 255)
                        yolo_text = f"person {conf*100:.0f}%"
                        #puts yolo confidence level and identify label
                        cv2.rectangle(frame_display, (x_min, y_min), (x_max, y_max), display_color, 2)
                        cv2.putText(frame_display, yolo_text, (x_min, y_min - 25 if y_min > 25 else y_min + 15), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0), 2)
                        cv2.putText(frame_display, identified_label_display, (x_min, y_min - 5 if y_min > 5 else y_min + 35), cv2.FONT_HERSHEY_SIMPLEX, 0.5, display_color, 2)
        cv2.imshow("Webcam", frame_display) #displays frame and exit feature
        #if shutting down, webcam and opencv turn off
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
    cap.release()
    cv2.destroyAllWindows()

# This is here to ensure that unexpected errors while running do not freeze the webcam and ensure it closes properly
# without going into task manager.
if __name__ == "__main__":
    try:
        main()
    except Exception as e:
        print(f"unexpected error occurred {e}")
    finally:
        cv2.destroyAllWindows()
