import torch
import cv2
import easyocr
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from google.colab import files

def extract_text_features(text):
    return [
        len(text),                         
        sum(c.isdigit() for c in text),    
        sum(c.isalpha() for c in text),      
        1 if "ID" in text.upper() else 0,    
    ]

X_train = [extract_text_features("ID-8829"), extract_text_features("CAUTION"), extract_text_features("12345")]
y_train = [1, 2, 1]

rf_model = RandomForestClassifier(n_estimators=100)
rf_model.fit(X_train, y_train)
print(" Random Forest trained on industrial patterns")

reader = easyocr.Reader(['en'], gpu=torch.cuda.is_available())

def hybrid_ocr_pipeline():
    print("Upload your industrial box image:")
    uploaded = files.upload()

    for filename in uploaded.keys():
        img = cv2.imread(filename)
     
        raw_results = reader.readtext(img, paragraph=True)

        print(f"\n--- Results for {filename} ---")
        for (bbox, text) in raw_results:
           
            features = extract_text_features(text)
            prediction = rf_model.predict([features])[0]

            label_map = {0: "Other", 1: "SERIAL NUMBER", 2: "HAZARD LABEL"}
            final_type = label_map.get(prediction)

            print(f"DL Extracted: '{text}' | RF Classified as: {final_type}")

            (tl, tr, br, bl) = bbox
            cv2.rectangle(img, (int(tl[0]), int(tl[1])), (int(br[0]), int(br[1])), (255, 0, 0), 2)
            cv2.putText(img, f"{final_type}: {text}", (int(tl[0]), int(tl[1])-10),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)

        from google.colab.patches import cv2_imshow
        cv2_imshow(img)

hybrid_ocr_pipeline()
