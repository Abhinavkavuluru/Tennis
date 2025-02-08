import os
import cv2
import pandas as pd

class ImagePlotter:
    def __init__(self, output_csv, image_path, output_image_path):
        self.output_csv = output_csv
        self.image_path = image_path
        self.output_image_path = output_image_path

    def plot_coordinates_on_image(self):
        # Ensure output directory exists
        output_dir = os.path.dirname(self.output_image_path)
        if not os.path.exists(output_dir):
            os.makedirs(output_dir)

        # Load the image
        image = cv2.imread(self.image_path)

        # Load the CSV file with the new coordinates
        data = pd.read_csv(self.output_csv)

        # Iterate through the data and plot points on the image
        for _, row in data.iterrows():
            x, y = int(row['x']), int(row['y'])
            cv2.circle(image, (x, y), radius=4, color=(0, 0, 255), thickness=-1)  # Red dot

        # Save the resulting image
        cv2.imwrite(self.output_image_path, image)

        # ✅ Confirm file creation
        if os.path.exists(self.output_image_path):
            print(f"✅ Image with plotted coordinates saved to: {self.output_image_path}")
        else:
            print(f"❌ Error: Failed to save {self.output_image_path}")
