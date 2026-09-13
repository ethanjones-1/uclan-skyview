import tempfile
import cv2
import numpy as np
import streamlit as st

st.title("UCLan SkyView 🚀")
st.write("Aerospace Runway & Roadway Surface Tracker — UCLan Aerospace Society")

uploaded_file = st.file_uploader(
    "Choose a rocket launch or test video...", type=["mp4", "mov", "avi"]
)

if uploaded_file is not None:
  st.video(uploaded_file)

  if st.button("Run Runway/Road Analysis"):
    with st.spinner("Processing video frames and isolating surface track..."):

      tfile = tempfile.NamedTemporaryFile(delete=False, suffix=".mp4")
      tfile.write(uploaded_file.read())

      cap = cv2.VideoCapture(tfile.name)
      width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
      height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
      fps = cap.get(cv2.CAP_PROP_FPS)
      if fps <= 0:
        fps = 30

      output_path = tempfile.NamedTemporaryFile(delete=False, suffix=".mp4").name
      fourcc = cv2.VideoWriter_fourcc(*"mp4v")
      out = cv2.VideoWriter(output_path, fourcc, fps, (width, height))

      frame_count = 0

      while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
          break

        frame_count += 1

        # Convert frame to grayscale and isolate lower half asphalt/concrete surfaces (Runway/Road region)
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        roi = gray[int(height * 0.4) : height, 0:width]  # Lower field of view

        # Edge and contour filtering for paved strips
        blur = cv2.GaussianBlur(roi, (5, 5), 0)
        _, thresh = cv2.threshold(blur, 60, 255, cv2.THRESH_BINARY_INV)
        contours, _ = cv2.findContours(
            thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE
        )

        if contours:
          # Find the largest structural strip representing the runway/road surface
          largest_c = max(contours, key=cv2.contourArea)
          if cv2.contourArea(largest_c) > (width * height * 0.05):
            x, y, w, h = cv2.boundingRect(largest_c)
            y += int(
                height * 0.4
            )  # Offset back to full frame coordinate scale

            # Draw clean tracking box labeled exclusively as RUNWAY/ROAD
            cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 165, 255), 3)
            cv2.putText(
                frame,
                "RUNWAY / ROAD LOCKED",
                (x, max(y - 10, 30)),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.6,
                (0, 165, 255),
                2,
            )

        # HUD Telemetry Overlay
        cv2.putText(
            frame,
            "UCLan SkyView | SURFACE TRACKER ACTIVE",
            (30, 40),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.7,
            (0, 255, 255),
            2,
        )
        cv2.putText(
            frame,
            f"FRAME: {frame_count}",
            (30, 80),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.6,
            (255, 255, 255),
            2,
        )

        out.write(frame)

      cap.release()
      out.release()

    st.success("Analysis Complete!")
    st.video(output_path)

    with open(output_path, "rb") as f:
      st.download_button(
          label="Download Tracked Video",
          data=f,
          file_name="uclan_skyview_runway.mp4",
          mime="video/mp4",
      )