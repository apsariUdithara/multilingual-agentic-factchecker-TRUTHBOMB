import streamlit as st

st.title("Sri Lankan Social Media Fact-Checker")

claim = st.text_area("Enter a claim to verify:")
if st.button("Check"):
    st.write("Processing claim...")
    # TODO: Invoke LangGraph pipeline
