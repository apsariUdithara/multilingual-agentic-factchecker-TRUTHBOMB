import pytest
from unittest.mock import MagicMock, patch
from src.pipeline import run_fact_checker

@pytest.fixture
def mock_agents():
    with patch("src.agents.claim_extractor.model.generate_content") as mock_extractor, \
         patch("src.agents.local_researcher.verify_claim_against_local_db.invoke") as mock_local, \
         patch("src.agents.web_researcher.web_search.invoke") as mock_web, \
         patch("src.agents.synthesis.model.generate_content") as mock_synth:
        
        # Mock Extractor
        mock_extractor_response = MagicMock()
        mock_extractor_response.text = '{"language": "English", "extracted_claim": "The sky is blue", "routing_destination": "web_researcher"}'
        mock_extractor.return_value = mock_extractor_response
        
        # Mock Web Search
        mock_web.return_value = "Search results: The sky is confirmed to be blue."
        
        # Mock Synthesis
        mock_synth_response = MagicMock()
        mock_synth_response.text = '{"verdict": "True", "confidence_score": "100%", "rationale": "Clear evidence.", "summaries": {"english": "True", "sinhala": "සත්‍යයි", "tamil": "உண்மை"}}'
        mock_synth.return_value = mock_synth_response
        
        yield {
            "extractor": mock_extractor,
            "local": mock_local,
            "web": mock_web,
            "synth": mock_synth
        }

def test_full_pipeline_web_routing(mock_agents):
    input_text = "I think the sky is blue."
    result = run_fact_checker(input_text)
    
    assert result["extracted_claim"] == "The sky is blue"
    assert result["routing_destination"] == "web_researcher"
    assert result["final_verdict"]["verdict"] == "True"
    mock_agents["web"].assert_called_once()
    mock_agents["local"].assert_not_called()

def test_full_pipeline_local_routing(mock_agents):
    # Update mock to route locally
    mock_agents["extractor"].return_value.text = '{"language": "English", "extracted_claim": "Local news", "routing_destination": "local_researcher"}'
    mock_agents["local"].return_value = "Local evidence found."
    
    input_text = "Local news update."
    result = run_fact_checker(input_text)
    
    assert result["routing_destination"] == "local_researcher"
    mock_agents["local"].assert_called_once()
