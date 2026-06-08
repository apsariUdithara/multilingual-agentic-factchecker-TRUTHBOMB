import requests
import streamlit as st

API_URL = "http://127.0.0.1:8000/api/v1/verify"

st.set_page_config(
    page_title="Traffic Telemetry Fact-Checker",
    page_icon="TT",
    layout="wide",
)

st.markdown(
    """
    <style>
        .stApp {
            background: radial-gradient(circle at 15% 15%, #eef4ff 0%, #f7fafc 45%, #edf6f1 100%);
        }
        .result-card {
            border: 1px solid #d7dde5;
            border-radius: 12px;
            padding: 16px;
            background: #ffffff;
        }
    </style>
    """,
    unsafe_allow_html=True,
)

st.title("Traffic Telemetry Fact-Checker")
st.caption("Paste a Singlish or English social media claim and verify it against retrieved context.")

user_input = st.text_area(
    "Social media post",
    placeholder="e.g., Nawammahara railway gate close wela traffic jam ekak thiyenawa",
    height=180,
)

verify_clicked = st.button("Verify", type="primary", use_container_width=True)

if verify_clicked:
    if not user_input.strip():
        st.warning("Please enter a post before verification.")
    else:
        try:
            with st.spinner("Verifying claim..."):
                response = requests.post(
                    API_URL,
                    json={"text": user_input},
                    timeout=90,
                )

            if response.status_code != 200:
                detail = "Unknown server error"
                try:
                    detail = response.json().get("detail", detail)
                except ValueError:
                    detail = response.text or detail
                st.error(f"Verification failed ({response.status_code}): {detail}")
            else:
                data = response.json()
                verdict = str(data.get("verdict", "UNVERIFIED")).upper()

                if verdict == "TRUE":
                    verdict_color = "#18794e"
                elif verdict == "FALSE":
                    verdict_color = "#b42318"
                else:
                    verdict_color = "#667085"

                confidence = data.get("confidence_score", 0.0)
                try:
                    confidence_pct = f"{float(confidence) * 100:.1f}%"
                except (TypeError, ValueError):
                    confidence_pct = "N/A"

                st.markdown("<div class='result-card'>", unsafe_allow_html=True)
                st.subheader("Verification Result")
                st.write("**Extracted claim (English)**")
                st.write(data.get("extracted_claim_en", "N/A"))

                col1, col2 = st.columns(2)
                with col1:
                    st.markdown(
                        f"<h3 style='color:{verdict_color}; margin-bottom:0;'>Verdict: {verdict}</h3>",
                        unsafe_allow_html=True,
                    )
                with col2:
                    st.metric("Confidence", confidence_pct)

                st.write("**Reasoning**")
                st.write(data.get("reasoning", "No reasoning provided."))
                st.markdown("</div>", unsafe_allow_html=True)

                with st.expander("Retrieved Context Sources"):
                    context_rows = data.get("retrieved_context", [])
                    if not context_rows:
                        st.info("No context documents were returned.")
                    else:
                        for idx, row in enumerate(context_rows, start=1):
                            st.markdown(f"**Source {idx}**")
                            st.write(f"Source: {row.get('source', 'Unknown')}")
                            st.write(f"Timestamp: {row.get('timestamp', 'N/A')}")
                            st.write(f"Distance Score: {row.get('distance_score', 'N/A')}")
                            st.write(row.get("content", ""))
                            st.divider()

        except requests.exceptions.ConnectionError:
            st.error(
                "Cannot connect to FastAPI server at http://127.0.0.1:8000. "
                "Please start the backend and try again."
            )
        except requests.exceptions.Timeout:
            st.error("The verification request timed out. Please retry.")
        except requests.RequestException as exc:
            st.error(f"Network error during verification: {exc}")
        except ValueError:
            st.error("Received an invalid JSON response from the server.")
