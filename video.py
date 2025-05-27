import streamlit as st
import cv2
import tempfile
import base64
import openai
from PIL import Image
from io import BytesIO
import os

# Set your OpenAI API key
openai.api_key = os.getenv("OPENAI_API_KEY")  # Or set directly for testing

def extract_frames(video_path, frame_rate=1):
    cap = cv2.VideoCapture(video_path)
    frames = []
    count = 0
    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break
        if count % int(cap.get(cv2.CAP_PROP_FPS)) == 0:
            img = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            frames.append(Image.fromarray(img))
        count += 1
    cap.release()
    return frames

def image_to_base64(image: Image.Image) -> str:
    buffered = BytesIO()
    image.save(buffered, format="JPEG")
    img_str = base64.b64encode(buffered.getvalue()).decode("utf-8")
    return f"data:image/jpeg;base64,{img_str}"

def analyze_damage_with_openai(image_b64):
    response = openai.ChatCompletion.create(
        model="gpt-4-vision-preview",
        messages=[
            {
                "role": "user",
                "content": [
                    {"type": "text", "text": "List all visible damages on the car in this image."},
                    {"type": "image_url", "image_url": image_b64}
                ]
            }
        ],
        max_tokens=500
    )
    return response['choices'][0]['message']['content']

# Streamlit app
st.title("Car Damage Detection from Video (Insurance Claim Support)")

uploaded_file = st.file_uploader("Upload Car Video", type=["mp4", "mov", "avi"])

if uploaded_file:
    with tempfile.NamedTemporaryFile(delete=False) as temp_file:
        temp_file.write(uploaded_file.read())
        temp_video_path = temp_file.name

    st.info("Extracting frames from video...")
    frames = extract_frames(temp_video_path)
    st.success(f"Extracted {len(frames)} frame(s). Analyzing...")

    damage_reports = []
    for i, frame in enumerate(frames):
        st.image(frame, caption=f"Frame {i+1}", use_column_width=True)
        image_b64 = image_to_base64(frame)
        with st.spinner(f"Analyzing frame {i+1} with OpenAI Vision..."):
            try:
                result = analyze_damage_with_openai(image_b64)
                damage_reports.append(result)
                st.text_area(f"Detected Damage (Frame {i+1})", result, height=150)
            except Exception as e:
                st.error(f"Error analyzing frame {i+1}: {e}")

    if damage_reports:
        st.subheader("Final Consolidated Damage Report")
        combined = "\n".join(damage_reports)
        with st.spinner("Summarizing damages..."):
            summary = openai.ChatCompletion.create(
                model="gpt-4",
                messages=[
                    {"role": "user", "content": f"Summarize the following observations into a final list of car damages:\n{combined}"}
                ]
            )
            final_report = summary['choices'][0]['message']['content']
            st.success("Summary Complete")
            st.text_area("Final Damage Summary", final_report, height=200)

    os.remove(temp_video_path)
