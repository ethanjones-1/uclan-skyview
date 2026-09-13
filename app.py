import tempfile
import cv2
import streamlit as st
from ultralytics import YOLOWorld

st.title("UCLan SkyView 🚀")
st.write("Custom Open-Vocabulary Aerospace Asset Tracker — UCLan Aerospace Society")


# Load the open-vocabulary YOLO-World model
@st.cache_resource
def load_model():
  model = YOLOWorld("yolov8s-world.pt")
  # Define the exact custom text classes you want to detect
  model.set_classes(["building", "runway", "ocean", "launch site"])
  return model


model = load_model()

uploaded_file = st.file_uploader(
    "Choose a rocket launch or test video...", type=["mp4", "mov", "avi"]
)

if uploaded_file is not None:
  st.video(uploaded_file)

  if st.button("Run Custom AI Analysis"):
    with st.spinner(
        "Scanning frames for buildings, runways, ocean, and launch sites..."
    ):

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

        # Frame-skipping optimization to keep processing smooth on cloud servers
        if frame_count % 2 != 0:
          out.write(frame)
          continue

        # Run open-vocabulary AI detection on the frame
        results = model(frame, verbose=False)

        for r in results:
          boxes = r.boxes
          for box in boxes:
            x1, y1, x2, y2 = map(int, box.xyxy[0])
            conf = float(box.conf[0])
            cls = int(box.cls[0])
            class_name = model.names[cls].upper()

            # Filter with a reasonable confidence threshold
            if conf > 0.25:
              cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 128), 2)
              cv2.putText(
                  frame,
                  f"{class_name} ({conf:.2f})",
                  (x1, max(y1 - 10, 20)),
                  cv2.FONT_HERSHEY_SIMPLEX,
                  0.6,
                  (0, 255, 128),
                  2,
              )

        # Telemetry HUD Overlay
        cv2.putText(
            frame,
            "UCLan SkyView | OPEN-VOCABULARY ACTIVE",
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
          label="Download Custom Tracked Video",
          data=f,
          file_name="uclan_skyview_custom_tracked.mp4",
          mime="video/mp4",
      )