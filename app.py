import time
import streamlit as st

st.title("UCLan SkyView 🚀")
st.write(
    "Welcome to the UCLan Aerospace Society video analysis tool. Upload a"
    " launch video below to test the pipeline."
)

# File uploader widget
uploaded_file = st.file_uploader(
    "Choose a rocket launch video...", type=["mp4", "mov", "avi"]
)

if uploaded_file is not None:
  st.subheader("Original Launch Footage")
  st.video(uploaded_file)

  # Generate button simulation
  if st.button("Run Simulation Analysis"):
    with st.spinner("Processing flight frames..."):
      progress_bar = st.progress(0)
      for percent_complete in range(100):
        time.sleep(0.02)
        progress_bar.progress(percent_complete + 1)

    st.success("Analysis complete!")
    st.subheader("Processed Output (Mock)")
    st.video(uploaded_file)

    # Added download button for the output video
    st.download_button(
        label="Download Processed Video",
        data=uploaded_file,
        file_name="uclan_skyview_processed.mp4",
        mime="video/mp4",
    )