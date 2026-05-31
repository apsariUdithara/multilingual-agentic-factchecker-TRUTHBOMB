import pytest
from unittest.mock import patch, MagicMock
from src.pipeline import run_fact_checker

# Mocking the heavy/external dependencies to allow tests to run in any environment
@patch("src.agents.triage.model.generate_content")
@patch("src.agents.traffic_expert.verify_traffic_law_claim")
@patch("src.agents.web_researcher.web_search")
@patch("src.agents.synthesis.model.generate_content")
def test_traffic_claim_routing(mock_synthesis, mock_web_search, mock_traffic_tool, mock_triage):
    """
    Verifies that a traffic-related claim routes to the traffic_expert.
    """
    # 1. Setup Mocks
    mock_triage.return_value.text = '{"language": "English", "extracted_claim": "A 16-year-old can register a car.", "routing_destination": "traffic_expert"}'
    mock_traffic_tool.invoke.return_value = "According to the Motor Traffic Act, the minimum age is 18."
    mock_synthesis.return_value.text = '{"verdict": "False", "confidence_score": "95%", "rationale": "Legal age is 18.", "summaries": {"english": "False.", "sinhala": "අසත්‍යයි.", "tamil": "பொய்."}}'
    
    claim = "A 16-year-old can register a car in Sri Lanka."
    result = run_fact_checker(claim)
    
    # 2. Assert Routing and Logic
    assert result["routing_destination"] == "traffic_expert"
    assert any(out["agent"] == "traffic_expert" for out in result["agent_outputs"])
    assert result["final_verdict"]["verdict"] == "False"
    assert result["final_verdict"]["summaries"]["sinhala"] == "අසත්‍යයි."

@patch("src.agents.triage.model.generate_content")
@patch("src.agents.traffic_expert.verify_traffic_law_claim")
@patch("src.agents.web_researcher.web_search")
@patch("src.agents.synthesis.model.generate_content")
def test_general_claim_routing(mock_synthesis, mock_web_search, mock_traffic_tool, mock_triage):
    """
    Verifies that a general news claim routes to the web_researcher.
    """
    # 1. Setup Mocks
    mock_triage.return_value.text = '{"language": "English", "extracted_claim": "New 10000 note.", "routing_destination": "web_researcher"}'
    mock_web_search.invoke.return_value = "CBSL denied rumors of a new 10,000 rupee note."
    mock_synthesis.return_value.text = '{"verdict": "False", "confidence_score": "90%", "rationale": "CBSL denied it.", "summaries": {"english": "False.", "sinhala": "අසත්‍යයි.", "tamil": "பொய்."}}'
    
    claim = "The Central Bank of Sri Lanka just announced a new 10,000 rupee note today."
    result = run_fact_checker(claim)
    
    # 2. Assert Routing and Logic
    assert result["routing_destination"] == "web_researcher"
    assert any(out["agent"] == "web_researcher" for out in result["agent_outputs"])
    assert result["final_verdict"]["verdict"] == "False"

if __name__ == "__main__":
    pytest.main([__file__])
