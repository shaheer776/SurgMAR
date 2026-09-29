import re
import time
import sys
import json
from Utils.API_utils import call_gpt35Turbo_api,gpt4_vision_caption
from Agents.Agent2_InstrumentIdentification import Instrument_Recognition_Agent
from Agents.Agent1_ActionRecognition import Action_Recognition_Agent
from Utils.Debate_utils import (
    transform_action_to_instrument_question,
    parse_instrument_response,
    parse_action_response,
    instrument_action_consistency_check,
    evaluate_consensus,
    select_best_action_output,
    save_candidates_to_file,
    Instrument_Recognition_Agent,
    Action_Recognition_Agent
)

def multi_agent_debate(question, image_path):
    """
    Orchestrates the multi-agent collaboration to ultimately recognize the surgical action.
    
    Overview:
      - The primary goal is to identify the surgical action correctly.
      - The action recognition agent attempts to predict the surgical action based on the input question and image.
      - The instrument identification agent rephrases the question to focus on identifying the surgical instrument.
      - By combining the outputs, we verify consistency with a knowledge graph mapping instruments to valid actions.
      - If the predictions are consistent, the action is accepted. Otherwise, fallback logic is applied.
      
    This collaborative design reinforces the action recognition by ensuring that the instrument identity is correct,
    thereby leading to a more accurate and trustworthy final action description.
    """

    print("\n[Debate_Agent] Received question:\n", question)
    print("=====================================================================================================")

     # 1) Transform the action question into an instrument question
    instrument_question = transform_action_to_instrument_question(question)
    print("\n[Debate_Agent] Transformed instrument question:\n", instrument_question)
    print("=====================================================================================================")

    # 2) Get the instrument guess from the instrument identification agent
    instrument_answer = Instrument_Recognition_Agent(instrument_question, image_path)
    print("\n[Debate_Agent] Instrument_Recognition_Agent response:")
    print(instrument_answer)
    print("=====================================================================================================")
    print("=====================================================================================================")

    # 3) Get the action guess from the action recognition agent
    action_answer = Action_Recognition_Agent(question, image_path)
    print("\n[Debate_Agent] Action_Recognition_Agent response:")
    print(action_answer)
    print("=====================================================================================================")
    print("=====================================================================================================")

    # 4) Parse the final answers from both
    instrument_name = parse_instrument_response(instrument_answer)
    print("\n[Debate_Agent] Parsed instrument name:", instrument_name)
    action_name = parse_action_response(action_answer)
    print("[Debate_Agent] Parsed action name:", action_name)
    print("=====================================================================================================")

    # 5) Evaluate with our chosen metrics
    metrics = evaluate_consensus(instrument_name, action_name, instrument_answer, action_answer, question)

    # 6) Extract individual metric values
    kg_consistency = metrics["kg_consistency"]
    cot_coherence = metrics["Coherence"]
    collab_synergy = metrics["Collaborative_Synergy"]  # This is now collaborative synergy
    print("=====================================================================================================")

    # List to store all candidate refinement outputs
    refined_candidates = []

    # Initial candidate (before reruns)
    initial_candidate = {
        "instrument_answer": instrument_answer,
        "parsed_instrument_name": instrument_name,
        "action_answer": action_answer,
        "parsed_action_name": action_name,
        "metrics": metrics
    }
    refined_candidates.append(initial_candidate)

    # 6) If coherence, collaboration, or consistency is poor, refine action recognition
    if kg_consistency and cot_coherence > 3 and collab_synergy > 3:
        print("\n[Moderator] Initial responses are acceptable. No refinement needed.")
        final_instrument = instrument_answer
        final_action = action_answer

    else:
        print("=====================================================================================================")
        print("\n[Moderator] Detected inconsistency or weak collaboration. Triggering action refinement...")

        max_refinements = 3
        # Perform up to max_refinements reruns
        for i in range(max_refinements):
            print(f"\n[Moderator] Refinement iteration {i+1} ...")
            # Rerun Instrument Identification Agent (Optional, if we want updated reasoning)
            refined_instrument_answer = Instrument_Recognition_Agent(instrument_question, image_path)
            refined_instrument_name = parse_instrument_response(refined_instrument_answer)
            print("=====================================================================================================")
            print("\n[Debate_Agent] Rerun: Instrument_Recognition_Agent Response:\n", refined_instrument_answer)
            print("[Debate_Agent] Rerun: Parsed Instrument Name:", refined_instrument_name)

            # Rerun Action Recognition Agent with instrument information explicitly fed in
            refined_action_prompt_as_question_input = f"""
            The Instrument Identification Agent has identified the instrument in question to be: {refined_instrument_name}.
            Validate and confirm if you agree that instrument in question is {refined_instrument_name}. 
            If you agree with the Instrument Identification Agent and the identity of the instrument in question, determine the most appropriate ongoing surgical action using the Action Recognition Chain of Thought Process.

            {question}
            """
            # Run ActionRecognition_Agent again with guided input
            refined_action_answer = Action_Recognition_Agent(refined_action_prompt_as_question_input, image_path)
            refined_action_name = parse_action_response(refined_action_answer)
            print("=====================================================================================================")
            print("\n[Debate_Agent] Rerun: Action_Recognition_Agent Response:\n", refined_action_answer)
            print("[Debate_Agent] Rerun: Parsed Action Name:", refined_action_name)

            candidate = {
                "instrument_answer": refined_instrument_answer,
                "parsed_instrument_name": refined_instrument_name,
                "action_answer": refined_action_answer,
                "parsed_action_name": refined_action_name
            }
            refined_candidates.append(candidate)
            # Re-evaluate the new candidate with our metrics
            new_metrics = evaluate_consensus(refined_instrument_name, refined_action_name, refined_instrument_answer, refined_action_answer, question)
            candidate["metrics"] = new_metrics

            if new_metrics["kg_consistency"] and new_metrics["Coherence"] > 3 and new_metrics["Collaborative_Synergy"] > 3:
                print("[Moderator] This refinement meets our quality thresholds. Exiting refinement loop early.")
                break  # Accept this candidate and exit the loop
            else:
                print("[Moderator] This refinement did not meet the thresholds. Continuing to next iteration if available...")

        # Save all refined candidates to a file
        save_candidates_to_file(refined_candidates)

        # After up to 3 refinements, use GPT-3.5 to select the candidate with the highest confidence.
        selected_candidate = select_best_action_output(refined_candidates)
        final_instrument = selected_candidate["instrument_answer"]
        final_action = selected_candidate["action_answer"]

    # 7) Return the final collaboration result
    final_output = {
        "instrument_agent_answer": final_instrument,
        "action_agent_answer": final_action,
        "metrics": metrics 
    }
    
    print("\n[Debate_Agent] Final Output:")
    print(final_output)

    return final_output
