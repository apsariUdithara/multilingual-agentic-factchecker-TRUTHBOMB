import streamlit as st
import json
import os
from src.pipeline import run_fact_checker
from PIL import Image

# Page Configuration
st.set_page_config(
    page_title="Sri Lankan Social Media Fact-Checking Hub",
    page_icon="🔍",
    layout="centered"
)

st.title("🔍 Sri Lankan Social Media Fact-Checking Hub")
st.markdown("""
Verify claims from Facebook, X (Twitter), WhatsApp, and more. 
Our system uses **Gemini 1.5 Flash** for general news and a **Fine-tuned Phi-3.5** for Sri Lankan Traffic Laws.
""")

# Input Section
st.subheader("Submit a Claim")
input_type = st.radio("Choose input type:", ["Text", "Image/Screenshot"])

user_input = None
if input_type == "Text":
    user_input = st.text_area("Paste the social media post content here:", placeholder="e.g., Aluth traffic fines announced today...")
else:
    uploaded_file = st.file_uploader("Upload a screenshot of the post:", type=["png", "jpg", "jpeg"])
    if uploaded_file is not None:
        # Save temporary file for the pipeline
        temp_dir = "temp_uploads"
        if not os.path.exists(temp_dir):
            os.makedirs(temp_dir)
        temp_path = os.path.join(temp_dir, uploaded_file.name)
        with open(temp_path, "wb") as f:
            f.write(uploaded_file.getbuffer())
        user_input = temp_path
        st.image(Image.open(uploaded_file), caption="Uploaded Screenshot", use_container_width=True)

# Verification Logic
if st.button("Verify Claim", type="primary"):
    if not user_input:
        st.warning("Please provide some input first!")
    else:
        try:
            with st.status("Fact-checking in progress...") as status:
                st.write("🤖 Triage Agent: Analyzing input and detecting language...")
                # Note: For real-time updates from LangGraph, we'd use streaming, 
                # but for this MVP we'll use step-by-step messaging in the status box.
                
                # Execute Pipeline
                result = run_fact_checker(user_input)
                
                status.update(label="Fact-check Complete!", state="complete", expanded=False)

            # Extract Data
            final_verdict = result.get("final_verdict", {})
            verdict = final_verdict.get("verdict", "Unverified")
            confidence = final_verdict.get("confidence_score", "0%")
            rationale = final_verdict.get("rationale", "No rationale provided.")
            summaries = final_verdict.get("summaries", {})
            
            # Display Results
            st.divider()
            st.header("Verification Result")
            
            # Metrics
            col1, col2 = st.columns(2)
            with col1:
                # Color coding for verdict
                color = "green" if verdict == "True" else "red" if verdict == "False" else "orange"
                st.markdown(f"### Verdict: :{color}[{verdict}]")
            with col2:
                st.metric("Confidence Score", confidence)

            # Rationale
            st.subheader("Rationale")
            st.write(rationale)

            # Multilingual Summaries
            st.subheader("Summaries")
            tab1, tab2, tab3 = st.tabs(["English", "සිංහල", "தமிழ்"])
            
            with tab1:
                st.write(summaries.get("english", "N/A"))
            with tab2:
                st.write(summaries.get("sinhala", "N/A"))
            with tab3:
                st.write(summaries.get("tamil", "N/A"))

            # Source Info
            with st.expander("Technical Evidence Details"):
                st.json(result.get("agent_outputs", []))

        except Exception as e:
            st.error(f"An error occurred during verification: {e}")

st.divider()
st.caption("Disclaimer: This tool provides AI-generated fact-checking based on available data. Always consult official gazettes for legal matters.")
