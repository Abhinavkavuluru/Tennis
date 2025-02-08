from ultralytics import YOLO
import cv2
import numpy as np

class DotLine:
    def __init__(self, model_path, input_video, output_video, max_trail=50):
        self.model = YOLO(model_path)
        self.video_path = input_video
        self.output_video_path = output_video
        self.max_trail = max_trail

        # Open video capture
        self.cap = cv2.VideoCapture(self.video_path)
        if not self.cap.isOpened():
            raise ValueError(f"Error: Could not open video {self.video_path}")

        # Get video properties
        self.fps = int(self.cap.get(cv2.CAP_PROP_FPS))
        self.width = int(self.cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        self.height = int(self.cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

        # Define the codec and create VideoWriter
        fourcc = cv2.VideoWriter_fourcc(*"mp4v")
        self.out = cv2.VideoWriter(self.output_video_path, fourcc, self.fps, (self.width, self.height))

        # Create a blank canvas to store persistent dots and lines
        self.trail_canvas = np.zeros((self.height, self.width, 3), dtype=np.uint8)

        # List to store detected ball positions
        self.trajectory_points = []

    def process_video(self):
        while self.cap.isOpened():
            ret, frame = self.cap.read()
            if not ret:
                break

            frame_with_trail = self.detect_and_track(frame)
            self.out.write(frame_with_trail)

        self.release_resources()

    def detect_and_track(self, frame):
        results = self.model.predict(frame, conf=0.5, verbose=False)

        for result in results:
            for box in result.boxes:
                class_id = int(box.cls[0])
                class_name = self.model.model.names[class_id]
                x1, y1, x2, y2 = map(int, box.xyxy[0])
                center_x = (x1 + x2) // 2
                center_y = (y1 + y2) // 2

                if class_name.lower() == "tennis ball":
                    if 0 <= center_x < self.width and 0 <= center_y < self.height:
                        self.trajectory_points.append((center_x, center_y))
                        if len(self.trajectory_points) > self.max_trail:
                            self.trajectory_points.pop(0)

                        for i in range(1, len(self.trajectory_points)):
                            cv2.line(self.trail_canvas, self.trajectory_points[i - 1], self.trajectory_points[i], (0, 255, 0), 2)

                        cv2.circle(self.trail_canvas, (center_x, center_y), 5, (0, 0, 255), -1)

        return cv2.addWeighted(frame, 0.8, self.trail_canvas, 0.5, 0)

    def release_resources(self):
        self.cap.release()
        self.out.release()
        cv2.destroyAllWindows()
        print(f"dotline video saved to: {self.output_video_path}")
