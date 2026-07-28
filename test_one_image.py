import sys
sys.path.insert(0, "src")
import prediction as pred

image_path = "C:/Users/pc/Desktop/BackupDocs/GitHub/crop_disease_indetifier/data/raw/Healthy/0a966c0b-85d0-41ea-82d2-3121f9325460___R.S_HL 8180 copy 2.jpg"
result = pred.predict_single(image_path)
print(result)