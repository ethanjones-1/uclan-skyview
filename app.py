import tempfile
import cv2
import streamlit as st
from ultralytics import YOLO

st.title("UCLan SkyView 🚀")
st.write(
    "Aerospace Launch Asset Tracker (People & Vehicles) — UCLan Aerospace"
    " Society"
)


@st.cache_resource
def load_model():
  return YOLO("yolov8n.pt")


model = load_model()

uploaded_file = st.file_uploader(
    "Choose a rocket launch or test video...", type=["mp4", "mov", "avi"]
)

if uploaded_file is not None:
  st.video(uploaded_file)

  if st.button("Run Asset Tracking"):
    with st.spinner("AI scanning for launch crew and vehicles..."):

      tfile = tempfile.NamedTemporaryFile(delete=False, suffix=".mp4")
      tfile.write(uploaded_file.read())

      cap = cv2.VideoCapture(tfile.name)
      width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
      height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
      fps = cap.get(cv2.CAP_PROP_FPS)

      output_path = tempfile.NamedTemporaryFile(delete=False, suffix=".mp4").name
      fourcc = cv2.VideoWriter_fourcc(*"mp4v")
      out = cv2.VideoWriter(output_path, fourcc, fps, (width, height))

      frame_count = 0

      # Classes we care about for a launch site: Person (0), Vehicle classes (Car, Truck, Bus, Motorcycle, Bicycle)
      target_classes = [0, 1, 2, 3, 5, 7]

      while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
          break

        frame_count += 1

        results = model(frame, verbose=False)

        for r in results:
          boxes = r.boxes
          for box in boxes:
            x1, y1, x2, y2 = map(int, box.xyxy[0])
            conf = float(box.conf[0])
            cls = int(box.cls[0])

            # Filter for our specific target classes and confidence > 45%
            if cls in target_classes and conf > 0.45:
              class_name = model.names[cls].upper()
              if class_name in ["CAR", "TRUCK", "BUS", "MOTORCYCLE", "BICYCLE"]:
                label_text = f"VEHICLE ({conf:.2f})"
                box_color = (0, 165, 255)  # Orange for vehicles
              elif class_name == "PERSON":
                label_text = f"CREW / PERSON ({conf:.2f})"
                box_color = (0, 255, 0)  # Green for people
              else:
                label_text = f"{class_name} ({conf:.2f})"
                box_color = (255, 0, 0)

              cv2.rectangle(frame, (x1, y1), (x2, y2), box_color, 2)
              cv2.putText(
                  frame,
                  label_text,
                  (x1, max(y1 - 10, 20)),
                  cv2.FONT_HERSHEY_SIMPLEX,
                  0.5,
                  box_color,
                  2,
              )

        # HUD Telemetry Overlay
        cv2.putText(
            frame,
            "UCLan SkyView | SITE MONITOR ACTIVE",
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

    st.success("Tracking Analysis Complete!")
    st.video(output_path)

    with open(output_path, "rb") as f:
      st.download_button(
          label="Download Tracked Video",
          data=f,
          file_name="uclan_skyview_tracked.mp4",
          mime="video/mp4",
      )