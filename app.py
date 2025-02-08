import streamlit as st
import os
import time
import tempfile
import cv2
import gdown  # For downloading models from Google Drive


# Set up Streamlit page configuration
st.set_page_config(page_title="Tennis Analysis App", layout="wide")

# 📌 Google Drive File IDs (Replace with actual IDs)
GDRIVE_FILES = {
    "yolo5_last.pt": "1YegZe9_HXEVuXEA-dbjn70DbBv0vxbFR",
    "keypoints_model.pth": "1NFIfBkD9OCSMIN8Q-s8z8Pf-YkkS7Byb",
    "ball_tracker.pkl": "1tjM6IVFVf-q5-fcWryC3H1ytlfNbcNeR"
}

# 📂 Directory to store models
MODEL_DIR = "models"
os.makedirs(MODEL_DIR, exist_ok=True)

# 📥 Function to download files from Google Drive
def download_file(file_name, file_id):
    file_path = os.path.join(MODEL_DIR, file_name)
    if not os.path.exists(file_path):  # Download only if missing
        with st.spinner(f"Downloading {file_name} from Google Drive..."):
            gdown.download(f"https://drive.google.com/uc?id={file_id}", file_path, quiet=False)
    return file_path  # Return the local file path

# ✅ Download required models from Google Drive
MODEL_PATH = download_file("yolo5_last.pt", GDRIVE_FILES["yolo5_last.pt"])
STUB_PATH = download_file("ball_tracker.pkl", GDRIVE_FILES["ball_tracker.pkl"])

# 🎯 Sidebar with Instructions
st.sidebar.title("📋 How to Use")
st.sidebar.markdown(
    """
    - **Upload a Tennis Video** 🎾 (MP4, AVI, MOV)  
    - **Preview the Video** after uploading  
    - **View Ball Tracking Data** 📊  
    - **Check the Heatmap** 🔥  
    - **Download the Heatmap**  
    """
)

# 🏆 Streamlit App Title
st.title("🎾 Tennis Match Analysis App")
st.write("Upload a tennis match video to process, track ball movements, and generate a heatmap.")

# Ensure output directory exists
OUTPUT_DIR = "output"
os.makedirs(OUTPUT_DIR, exist_ok=True)

# Initialize session state
if "processed_video" not in st.session_state:
    st.session_state.processed_video = None
if "heatmap_image" not in st.session_state:
    st.session_state.heatmap_image = None
if "output_image" not in st.session_state:
    st.session_state.output_image = None
if "processing_done" not in st.session_state:
    st.session_state.processing_done = False

# 📂 Upload video file
uploaded_file = st.file_uploader("📂 Upload a Tennis Match Video", type=["mp4", "avi", "mov", "mkv"])

if uploaded_file:
    # Create a temporary directory
    temp_dir = tempfile.mkdtemp()

    input_video_path = os.path.join(temp_dir, uploaded_file.name)
    output_video_path = os.path.join(temp_dir, "processed_video.mp4")

    heatmap_image = os.path.join(OUTPUT_DIR, "heatmap.jpg")
    output_image = os.path.join(OUTPUT_DIR, "court_plot.jpg")
    ball_hits_csv = os.path.join(OUTPUT_DIR, "ball_hits_coordinates.csv")
    transformed_csv = os.path.join(OUTPUT_DIR, "transformed_ball_hits_coordinates.csv")

    # Save uploaded file
    with open(input_video_path, "wb") as f:
        f.write(uploaded_file.read())

    # 🎥 Display uploaded video
    st.subheader("🎥 Uploaded Video")
    st.video(input_video_path)

    # ⚡ Processing button
    if st.button("⚡ Process Video & Generate Heatmap"):
        st.write("⏳ Processing video, please wait...")

        # Step 1: Process the video
        with st.spinner("🔄 Processing video..."):
            tracker = DotLine(MODEL_PATH, input_video_path, output_video_path)
            tracker.process_video()

        # Step 2: Track ball hits and generate coordinates
        with st.spinner("📌 Tracking ball hits..."):
            hits = BallTracker(MODEL_PATH, input_video_path, STUB_PATH, ball_hits_csv)
            hits.process_ball_hits()

        # Step 3: Generate heatmap
        with st.spinner("🌡️ Generating heatmap..."):
            heatmap = TennisHeatmap(transformed_csv, heatmap_image)
            heatmap.generate_heatmap()

        # Step 4: Plot ball hits on the court
        with st.spinner("📍 Plotting ball hits on the court..."):
            plotter = ImagePlotter(transformed_csv, input_video_path, output_image)
            plotter.plot_coordinates_on_image()

        time.sleep(2)

        # ✅ Verify files
        if os.path.exists(output_video_path) and os.path.getsize(output_video_path) > 0:
            st.session_state.processed_video = output_video_path
            st.success("✅ Video processing complete!")
        else:
            st.error("❌ Processed video is missing.")

        if os.path.exists(heatmap_image) and os.path.getsize(heatmap_image) > 0:
            st.session_state.heatmap_image = heatmap_image
            st.success("✅ Heatmap generated successfully!")
        else:
            st.error("❌ Heatmap missing!")

        if os.path.exists(output_image) and os.path.getsize(output_image) > 0:
            st.session_state.output_image = output_image
            st.success("✅ Court plot generated successfully!")
        else:
            st.error("❌ Court plot missing!")

        st.session_state.processing_done = True

# 🎬 Display outputs only if processing is done
if st.session_state.processing_done:
    st.subheader("🎬 Processed Video")

    if st.session_state.processed_video:
        st.video(st.session_state.processed_video)

    # 📊 Display Heatmap
    st.subheader("📊 Heatmap of Ball Hits")
    if st.session_state.heatmap_image:
        st.image(st.session_state.heatmap_image, use_column_width=True)
    else:
        st.error("⚠️ Heatmap image not found.")

    # 📌 Display Ball Hits on Court
    st.subheader("📌 Ball Hits on the Court")
    if st.session_state.output_image:
        st.image(st.session_state.output_image, use_column_width=True)
    else:
        st.error("⚠️ Court plot image not found.")

    # 📥 Download buttons
    st.write("📥 Download Processed Files:")
    if st.session_state.heatmap_image:
        with open(st.session_state.heatmap_image, "rb") as file:
            st.download_button("⬇ Download Heatmap", data=file, file_name="heatmap.jpg")
    if st.session_state.output_image:
        with open(st.session_state.output_image, "rb") as file:
            st.download_button("⬇ Download Court Plot", data=file, file_name="court_plot.jpg")
