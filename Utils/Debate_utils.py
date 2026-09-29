import re
import time
import sys
import json
from Utils.API_utils import call_gpt35Turbo_api,gpt4_vision_caption
from Agents.Agent2_InstrumentIdentification import Instrument_Recognition_Agent
from Agents.Agent1_ActionRecognition import Action_Recognition_Agent

# =============================================================================
# Knowledge Graph and Mappings
# =============================================================================

validation_rules = {
    "instrument-action": {
        "monopolar curved scissors": [
            "cutting",       # (e)
            "cauterization",
            "retraction",
            "tool manipulation"# (c)
        ],
        "forceps": [
            "grasping",         # (d)
            "cauterization",     # (c)
            "retraction",        # (a)
            "tool manipulation"  # (f)
        ],
        "needle driver": [
            "suturing",          # (b)
            "tool manipulation"  # (f)
        ],
        "clip applier": [
            "applying clip",     # (g)
            "tool manipulation"  # (f)
        ],
        "grasper": [  # note: key now matches MCQ option (g)
            "grasping",          # (d)
            "retraction",
            "cauterization",        # (a)
            "tool manipulation"  # (f)
        ],
        "stapler": [         # (e)
            "applying clip",     # (g)  # Staples can act as clips
            "tool manipulation"  # (f)
        ],
        "permanent cautery hook": [
            "cauterization",
            "tool manipulation",
            "retraction",
            "grasping",          
            "cutting"
        ]
    }
}

# Mapping dictionaries for converting Option letters to instrument/action names

instrument_map = {
    "A": "stapler",
    "B": "monopolar curved scissors",
    "C": "needle driver",
    "D": "forceps",
    "E": "permanent cautery hook",
    "F": "clip applier",
    "G": "grasper"
}

action_map = {
    "A": "retraction",
    "B": "suturing",
    "C": "cauterization",
    "D": "grasping",
    "E": "cutting",
    "F": "tool manipulation",
    "G": "applying clip"
}


def transform_action_to_instrument_question(action_question: str) -> str:
    
    instrument_question = action_question.replace(
        "What is the most likely ongoing action of the surgical instrument",
        "What is the most likely surgical instrument"
    )
    
    if "?" in instrument_question:
        parts = instrument_question.split("?")
        instrument_question_only = parts[0]  # everything before the first question mark
    else:
        instrument_question_only = instrument_question
    
    instrument_mcq = """(a) Stapler
    (b) Monopolar Curved Scissors
    (c) Needle Driver
    (d) Forceps
    (e) Permanent Cautery Hook
    (f) Clip Applier
    (g) Grasper
    """

    # Construct final question
    transformed = f"{instrument_question_only.strip()}?\n{instrument_mcq}"

    return transformed


def summarize_with_gpt(response: str, task: str) -> str:
    """
    Uses GPT-3.5 to summarize the agent response and extract either the instrument or action.

    :param response: The detailed response from the agent.
    :param task: Either "instrument" or "action" to specify what to extract.
    :return: The extracted instrument or action.
    """

    # Define mapping dictionaries
    instrument_map = {
        "A": "Stapler",
        "B": "Monopolar Curved Scissors",
        "C": "Needle Driver",
        "D": "Forceps",
        "E": "Permanent Cautery Hook",
        "F": "Clip Applier",
        "G": "Grasper"
    }

    action_map = {
        "A": "Retraction",
        "B": "Suturing",
        "C": "Cauterization",
        "D": "Grasping",
        "E": "Cutting",
        "F": "Tool Manipulation",
        "G": "Applying Clip"
    }

    # Select the appropriate mapping
    mapping = instrument_map if task == "instrument" else action_map
    mapping_text = "\n".join([f"{key} → {value}" for key, value in mapping.items()])

    prompt = f"""
    Please analyze the following response from a surgical AI agent and extract only the final {task} prediction.
    If the answer is unclear, return "unknown". If the response specifies an option (e.g., "Option (D)"), use the mapping below:

    {task.capitalize()} Mapping:
    {mapping_text}

    Response:
    {response}

    Return only the extracted {task} name as the output, without any extra text.
    """

    try:
        print(f"Extracting {task} with GPT-3.5...")
        print("Prompt:")
        print(prompt)
        extracted_result = call_gpt35Turbo_api(prompt).strip()
        print(f"Extracted {task}: {extracted_result}")
    except Exception as e:
        print(f"Error with GPT-3.5 extraction: {e}")
        extracted_result = "unknown"

    return extracted_result

def parse_instrument_response(agent_response: str) -> str:
    """
    Uses GPT-3.5 to extract the instrument from the agent response.
    """
    instrument_name = summarize_with_gpt(agent_response, "instrument")
    print("parse_instrument_response_instrument_name: ", instrument_name)
    return instrument_name

def parse_action_response(agent_response: str) -> str:
    """
    Uses GPT-3.5 to extract the action from the agent response.
    """
    action_name = summarize_with_gpt(agent_response, "action")
    print("parse_action_response_action_name: ", action_name)
    return action_name

def instrument_action_consistency_check(instrument_name, action_name) -> bool:
    """
    Checks if the predicted action is valid for the predicted instrument using the knowledge graph.
    This function ensures that the action identified by the action recognition agent is
    compatible with the instrument identified by the instrument identification agent.
    """
    instrument_key = instrument_name.lower().strip()
    print("instrument_action_consistency_check_instrument_key: ", instrument_key)
    action_key = action_name.lower().strip()
    print("instrument_action_consistency_check_action_key: ", action_key)
    valid_actions = validation_rules["instrument-action"].get(instrument_key, [])
    print(valid_actions)
    print(action_key in valid_actions)
    return action_key in valid_actions
    

def gpt_evaluate_metric(metric_name, instrument_agent, action_agent, rubric, max_retries=3) -> int:
    """
    Asks GPT-3.5 to evaluate the given response_text based on a provided rubric
    for the metric 'metric_name' and return an integer rating between 1 and 5.
    """
    prompt = f"""
    You are an expert evaluator of a multi-agent collaboration in an agentic system called Surg-CoT, which is a Chain-of-Thought embedded knowledge-based surgical agent which provide chain-of-thought reasoning for surgical image analysis. 
    Surg-CoT is designed to be trustworthy, accurate, and explainable in high-stakes medical contexts. 
    The ultimate goal of the system is to correctly identify the surgical action.
    
    To enhance the accuracy of action recognition, two agents collaborate:
      1. The action recognition agent provides an initial prediction of the surgical action.
      2. The instrument identification agent supplements this prediction by accurately identifying the surgical instrument,
         thereby reinforcing the action prediction.
    
    Evaluate the following agent response based on the metric "{metric_name}".
    {rubric}

    Response from the instrument_identification_agent:
    {instrument_agent}

    Response from the action_recognition_agent:
    {action_agent}

    Please provide only an integer rating between 1 (Very Poor) and 5 (Excellent) as your output.
    """
    for attempt in range(1, max_retries + 1):
        try:
            print(f"GPT rating attempt {attempt} for {metric_name}...")
            rating_str = call_gpt35Turbo_api(prompt).strip()
            print(f"Rating extracted: {rating_str}")
            match = re.search(r'\b([1-5])\b', rating_str)
            print(match)
            if match:
                return int(match.group(1))  # Extract valid rating
                print(f"Rating extracted: {rating_str}")
            else:
                print(f"Attempt {attempt}: Failed to extract valid rating. Retrying...")
                time.sleep(1)  # Short delay before retrying
        except Exception as e:
            print(f"Error in attempt {attempt} evaluating {metric_name}: {e}")
            time.sleep(1)  # Prevent immediate retries on errors

    print(f"All {max_retries} attempts failed. Defaulting to rating 3 (average).")
    return 3  # Default rating on repeated failure


def evaluate_consensus(instrument_name, action_name, 
                         instrument_answer, action_answer, question) -> dict:
    """
    Computes evaluation metrics:
      1. Knowledge graph consistency.
      2. Chain-of-Thought (CoT) Coherence (using GPT-generated evaluation).
      3. Answer Relevance (a new metric, also GPT-evaluated).
    """
    # 1) Check knowledge graph consistency
    kg_consistency = instrument_action_consistency_check(instrument_name, action_name)
    
    # Define rubrics for the metrics.
    coherence_rubric = (
        "Definition: Is the combined instrument–action prediction logically consistent and free from contradictions?\n"
        "Considerations:\n"
        " - Does the instrument logically enable the predicted action?\n"
        " - Are the two final answers free of contradictions?\n"
        " - Does the chain-of-thought or reasoning align the instrument with the action?\n"
        "Rating Guide:\n"
        " 1 = Very Poor: Completely contradictory or implausible.\n"
        " 2 = Below Average: Partially aligned, noticeable logical gaps.\n"
        " 3 = Average: Not contradictory, but minimal coherence.\n"
        " 4 = Good: Largely consistent, minor gaps.\n"
        " 5 = Excellent: Perfectly consistent and logically robust."
    )
    
    Collaborative_Synergy_rubric = (
        "Definition: Definition: How well do the InstrumentIdentification_Agent and ActionRecognition_Agent reinforce each other’s predictions?\n"
        "Considerations:\n"
        " - Do the two agents build on each other’s reasoning?\n"
        " - Does having both agents lead to a stronger final conclusion than a single agent?\n"
        " - Is there clear information sharing or cross-referencing?\n"
        "Rating Guide:\n"
        " 1 = Very Poor: No evidence of collaboration; responses conflict or are uncoordinated.\n"
        " 2 = Below Average: Minimal synergy; answers are weakly connected.\n"
        " 3 = Average: Reasonable collaboration; some mutual reinforcement.\n"
        " 4 = Good: Agents support each other effectively.\n"
        " 5 = Excellent: Collaboration significantly elevates clarity and correctness"
    )
    # metric_name, instrument_agent, action_agent, rubric
    Coherence_rating = gpt_evaluate_metric("Coherence of both answers", instrument_answer, action_answer, coherence_rubric)
    Collaborative_Synergy_rating = gpt_evaluate_metric("Collaborative Synergy of both answers", instrument_answer, action_answer, Collaborative_Synergy_rubric)
    
    metrics = {
        "kg_consistency": kg_consistency,
        "Coherence": Coherence_rating,
        "Collaborative_Synergy": Collaborative_Synergy_rating
    }
    print("Evaluation Metrics: ", metrics)
    return metrics

def select_best_action_output(candidates: list) -> dict:
    """
    Given a list of candidate refinement outputs (each a dict with an action answer),
    use GPT-3.5 to select the candidate with the highest confidence.
    """
    candidate_texts = ""
    for idx, candidate in enumerate(candidates, 1):
        candidate_texts += f"Candidate {idx}:\n"
        candidate_texts += f"- Parsed Instrument Name: {candidate['parsed_instrument_name']}\n"
        candidate_texts += f"- **Raw Action Recognition Agent Output:**\n{candidate['action_answer']}\n"
        candidate_texts += f"- Evaluation Metrics: {candidate['metrics']}\n\n"

    # Ask GPT-3.5 to return the candidate number (as an integer) that exhibits the highest confidence.
    prompt = f"""
    You are an expert surgical AI evaluator. Your task is to analyze and select the best reasoning 
    from multiple candidate responses generated by an Action Recognition Agent in a robotic surgery context.

    The system consists of two collaborating agents:
    - **Instrument Identification Agent:** Identifies the surgical instrument.
    - **Action Recognition Agent:** Predicts the surgical action using chain-of-thought reasoning.

    Each candidate response includes:
    - The **parsed instrument name** (validated by the Instrument Identification Agent).
    - The **full raw response** of the Action Recognition Agent.
    - The evaluation metrics, including knowledge graph consistency and coherence.

    **Task:**
    Evaluate the candidates based on the following criteria:
    1. **Chain-of-thought coherence:** Does the reasoning flow logically from observation to conclusion?
    2. **Confidence and clarity:** Is the response well-structured, unambiguous, and supported by logical steps?
    3. **Instrument-action alignment:** Does the predicted action align with the identified instrument?
    4. **Overall reliability:** Which candidate provides the strongest evidence-based conclusion?

    **Candidates:**
    {candidate_texts}

    Provide your decision by stating ONLY the best candidate number (as an integer).
    """
    print("=====================================================================================================")
    print("######################################################################################################")
    print("GPT candidate selection prompt:", prompt)
    print("######################################################################################################")
    print("=====================================================================================================")
    candidate_number_response = call_gpt35Turbo_api(prompt).strip()
    
    print("GPT candidate selection response:", candidate_number_response)
    match = re.search(r'\b([1-9]\d*)\b', candidate_number_response)
    if match:
        candidate_number = int(match.group(1))
        if 1 <= candidate_number <= len(candidates):
            print(f"Selected candidate #{candidate_number}")
            return candidates[candidate_number - 1]
    # Fallback: return the first candidate if parsing fails.
    print("Falling back to the first candidate.")
    return candidates[0]

def save_candidates_to_file(candidates, filename="candidates_log.json"):
    """
    Save all candidate refinement outputs to a JSON file for later reference.
    """
    try:
        with open(filename, "w") as f:
            json.dump(candidates, f, indent=4)
        print(f"[Logger] Successfully saved {len(candidates)} candidates to {filename}")
    except Exception as e:
        print(f"[Logger] Error saving candidates to file: {e}")
