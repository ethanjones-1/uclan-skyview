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
            fgbg = cv2.createBackgroundSubtractorMOG2()

            while cap.isOpened():
                ret, frame = cap.read()
                if not ret:
                    break

                frame_count += 1

                # Detect motion (finds the rocket/smoke against the background sky)
                fgmask = fgbg.apply(frame)

                # Find contours of moving parts
                contours, _ = cv2.findContours(
                    fgmask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE
                )

                for c in contours:
                    # Filter out tiny movements (noise) and focus on large moving objects (the rocket)
                    if cv2.contourArea(c) > 500:
                        (x, y, w, h) = cv2.boundingRect(c)
                        # Draw a tracking box around the moving rocket/launch
                        cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 2)
                        cv2.putText(
                            frame,
                            "TARGET LOCKED",
                            (x, y - 10),
                            cv2.FONT_HERSHEY_SIMPLEX,
                            0.5,
                            (0, 255, 0),
                            2,
                        )

                # Add professional flight HUD overlay text
                cv2.putText(
                    frame,
                    "UCLan SkyView | TELEMETRY ACTIVE",
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

        # 4. Show output video to user
        st.video(output_path)

        with open(output_path, "rb") as f:
            st.download_button(
                label="Download Processed Video",
                data=f,
                file_name="uclan_skyview_tracked.mp4",
                mime="video/mp4",
            )