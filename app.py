import tempfile
import cv2
import numpy as np
import streamlit as st

st.title("UCLan SkyView 🚀")
st.write("Dynamic Terrain & Infrastructure Scanner — UCLan Aerospace Society")

uploaded_file = st.file_uploader(
    "Choose a flight test or launch video...", type=["mp4", "mov", "avi"]
)

if uploaded_file is not None:
  st.video(uploaded_file)

  if st.button("Run Dynamic Frame Analysis"):
    with st.spinner("Processing video frames efficiently..."):

      tfile = tempfile.NamedTemporaryFile(delete=False, suffix=".mp4")
      tfile.write(uploaded_file.read())

      cap = cv2.VideoCapture(tfile.name)
      width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
      height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
      fps = cap.get(cv2.CAP_PROP_FPS)
      if fps <= 0:
        fps = 30

      # Scale down resolution slightly to prevent cloud memory overflow
      proc_width = int(width * 0.75)
      proc_height = int(height * 0.75)

      output_path = tempfile.NamedTemporaryFile(delete=False, suffix=".mp4").name
      fourcc = cv2.VideoWriter_fourcc(*"mp4v")
      out = cv2.VideoWriter(
          output_path, fourcc, fps, (proc_width, proc_height)
      )

      frame_count = 0

      while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
          break

        frame_count += 1

        # Resize frame to save memory
        frame = cv2.resize(frame, (proc_width, proc_height))

        # Skip every other frame during heavy contour math to stay well under RAM limits
        if frame_count % 4 != 0:
          out.write(frame)
          continue

        hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

        # 1. OCEAN DETECTION (HSV Blue Range Masking)
        lower_blue = np.array([90, 50, 50])
        upper_blue = np.array([130, 255, 255])
        blue_mask = cv2.inRange(hsv, lower_blue, upper_blue)
        blue_contours, _ = cv2.findContours(
            blue_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE
        )

        for c in blue_contours:
          if cv2.contourArea(c) > (proc_width * proc_height * 0.03):
            x, y, w, h = cv2.boundingRect(c)
            cv2.rectangle(frame, (x, y), (x + w, y + h), (255, 100, 0), 2)
            cv2.putText(
                frame,
                "OCEAN DETECTED",
                (x, max(y - 10, 20)),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.5,
                (255, 100, 0),
                2,
            )
            break

        # 2. RUNWAY / TARMAC DETECTION
        lower_region = gray[int(proc_height * 0.5) : proc_height, 0:proc_width]
        blurred = cv2.GaussianBlur(lower_region, (5, 5), 0)
        edges = cv2.Canny(blurred, 50, 150)
        runway_contours, _ = cv2.findContours(
            edges, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE
        )

        for c in runway_contours:
          if cv2.contourArea(c) > (proc_width * proc_height * 0.02):
            x, y, w, h = cv2.boundingRect(c)
            if w > (proc_width * 0.3):
              y += int(proc_height * 0.5)
              cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 165, 255), 2)
              cv2.putText(
                  frame,
                  "RUNWAY / ROAD",
                  (x, max(y - 10, 20)),
                  cv2.FONT_HERSHEY_SIMPLEX,
                  0.5,
                  (0, 165, 255),
                  2,
              )
              break

        # 3. BUILDINGS & LAUNCH SITE DETECTION
        _, thresh = cv2.threshold(gray, 100, 255, cv2.THRESH_BINARY)
        struct_contours, _ = cv2.findContours(
            thresh, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE
        )

        for c in struct_contours:
          area = cv2.contourArea(c)
          if (proc_width * proc_height * 0.005) < area < (
              proc_width * proc_height * 0.1
          ):
            approx = cv2.approxPolyDP(
                c, 0.04 * cv2.arcLength(c, True), True
            )
            if len(approx) >= 4:
              x, y, w, h = cv2.boundingRect(c)
              if int(proc_height * 0.3) < y < int(proc_height * 0.8):
                cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 2)
                cv2.putText(
                    frame,
                    "BUILDING / LAUNCH SITE",
                    (x, max(y - 10, 20)),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.5,
                    (0, 255, 0),
                    2,
                )
                break

        # Telemetry HUD
        cv2.putText(
            frame,
            f"UCLan SkyView | FRAME {frame_count}",
            (30, 40),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.6,
            (0, 255, 255),
            2,
        )

        out.write(frame)

      cap.release()
      out.release()

    st.success("Analysis Complete!")
    st.video(output_path)

    with open(output_path, "rb") as f:
      st.download_button(
          label="Download Scanned Video",
          data=f,
          file_name="uclan_skyview_scanned.mp4",
          mime="video/mp4",
      )