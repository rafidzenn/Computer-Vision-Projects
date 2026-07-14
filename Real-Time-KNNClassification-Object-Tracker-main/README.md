# Real-Time-KNNClassification-Object-Tracker
Object detection and tracking using KNN Classification plus Kalman Filter Tracking. In the frame processing, MOG2 algorithm learns bg model over time to detect moving foreground objects. We classify detections using trained KNN model, use kalman filters for smooth tracking and remove stale trackers i.e, those objects that are no longer visible. 
