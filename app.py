import cv2
import tempfile
import streamlit as st

st.title("UCLan SkyView 🚀")
uploaded_file = st.file_uploader(
    "Choose a rocket launch video...", type=["mp4", "mov", "avi"]
)

if uploaded_file is not None:
  st.video(uploaded_file)

  if st.button("Run Flight Analysis"):
    with st.spinner("Processing trajectory and tracking frames..."):

      # 1. Save uploaded video to a temporary file path OpenCV can read
      tfile = tempfile.NamedTemporaryFile(delete=False, suffix=".mp4")
      tfile.write(uploaded_file.read())

      # 2. Open the video with OpenCV
      cap = cv2.VideoCapture(tfile.name)
      width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
      height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
      fps = cap.get(cv2.CAP_PROP_FPS)

      # Setup output video writer (MP4 format)
      output_path = tempfile.NamedTemporaryFile(delete=False, suffix=".mp4").name
      fourcc = cv2.VideoWriter_fourcc(*"mp4v")
      out = cv2.VideoWriter(output_path, fourcc, fps, (width, height))

      # 3. Frame-by-frame loop
      frame_count = 0
      while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
          break

        frame_count += 1

        # --- THIS IS WHERE THE "AI" / PROCESSING GOES ---
        # For a quick cool effect without heavy AI training:
        # Draw a simulated targeting reticle or text overlay on every frame
        cv2.putText(
            frame,
            f"UCLan SkyView | Frame: {frame_count}",
            (50, 50),
            cv2.FONT_HERSHEY_SIMPLEX,
            1,
            (0, 255, 0),
            2,
        )
        cv2.circle(
            frame, (width // 2, height // 2), 40, (0, 0, 255), 2
        )  # Target box in center

        # Write processed frame to output video
        out.write(frame)

      cap.release()
      out.release()

    st.success("Analysis Complete!")

    # 4. Show output video to user
    st.video(output_path)

    with open(output_path, "rb") as f:
      st.download_button(
          "Download Processed Video",
          f,
          file_name="uclan_skyview_tracked.mp4",
          mime="video/mp4",
      )