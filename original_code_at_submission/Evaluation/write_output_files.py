import os
import logging
from itertools import product
from relations import extract_in_file3
from lexnet import get_word_from_id_mapping

# Configure logger
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Flag to switch between manual and auto modes
MODE = os.getenv("MODE", "auto")  # Can be "manual" or "auto"
MIXED_ROUTE = False  # Option for mixed route in auto mode

# File paths
SYNONYM_MAPPING_FILE = "synonym_mapping.txt"

def create_directory_safely(directory_path):
    """Create a directory if it doesn't already exist."""
    try:
        os.makedirs(directory_path, exist_ok=True)
        logger.info(f"Directory '{directory_path}' created or already exists.")
    except OSError as e:
        logger.error(f"Error creating directory '{directory_path}': {e}")

def load_synonym_mapping():
    """Load synonym mapping from file."""
    try:
        with open(SYNONYM_MAPPING_FILE) as f:
            return eval(f.read())
    except Exception as e:
        logger.error(f"Error reading synonym mapping: {e}")
        return {}

def auto_output(response, output, tajp, word_map, syn_map):
    """Process output in auto mode."""
    try:
        extract_in_file3(response, output, tajp, word_map, syn_map)
        logger.info(f"{tajp} --> done")
    except Exception as e:
        logger.error(f"Error processing auto output for {tajp}: {e}")

def manual_mode(base_dir, options):
    """Run in manual mode over combinations of options."""
    for combination in product(*options):
        history, simple, english, k, feedback_ness = combination

        extra_dir = "history/" if history else "no_history/"
        extra_dir += "simple/" if simple else "complex/"
        extra_dir += "English/" if english else "OtherLanguage/"
        extra_dir += f"{k}shot/"
        extra_dir += "feedback" if feedback_ness else "no_feedback"

        if not history and feedback_ness:
            continue

        main_dir = os.path.join(base_dir, extra_dir)

        for tajp in ["type1", "type2", "type3"]:
            response = f"{main_dir}/{tajp}_model_response.txt"
            output = f"{main_dir}/outputs/{tajp}"

            create_directory_safely(output)

            word_map = get_word_from_id_mapping()
            syn_map = load_synonym_mapping()

            auto_output(response, output, tajp, word_map, syn_map)

def auto_mode():
    """Run in auto mode, automatically processing model responses."""
    tajps = ["type1", "type2", "type3"] if not MIXED_ROUTE else ["mixed"]
    for tajp in tajps:
        if MIXED_ROUTE:
            sample = "samples/mixed_sample_dict.txt"
            response = "/path/to/mixed_model_response.txt"
            output = "/path/to/mixed_output"
        else:
            sample = f"samples/sample_{tajp}.txt"
            response = f"/path/to/{tajp}_model_response.txt"
            output = f"/path/to/{tajp}_output"

        word_map = get_word_from_id_mapping()
        syn_map = load_synonym_mapping()

        auto_output(response, output, tajp, word_map, syn_map)

def main():
    """Main function that switches between manual and auto mode."""
    if MODE == "manual":
        base_dir = "/path/to/manual/mode/base/directory"

        options = [
            (False, True),  # history_options
            (False, True),  # simple_options
            (False, True),  # english_options
            (5, 7),         # k_shots_options
            (False, True),  # feedback_options
        ]

        logger.info("Running in manual mode.")
        manual_mode(base_dir, options)

    elif MODE == "auto":
        logger.info("Running in auto mode.")
        auto_mode()

    else:
        logger.error(f"Invalid mode '{MODE}' specified. Please set to 'manual' or 'auto'.")

if __name__ == "__main__":
    main()
