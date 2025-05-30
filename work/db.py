from pymongo import MongoClient

client = MongoClient('mongodb://localhost:27017/')
db = client['face_detection_db']

detections_col = db['detections']