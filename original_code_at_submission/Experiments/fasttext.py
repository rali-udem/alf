import fasttext
from numpy import linalg
import os
import logging
from dotenv import load_dotenv
from tqdm import tqdm

# Load environment variables (for model path and other settings)
load_dotenv()

# Configure logger
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Load the model
MODEL_PATH = os.getenv("MODEL_PATH", "/path/to/your/fasttext_fr300_model.ftz")
logger.info(f"Loading model from {MODEL_PATH}...")
try:
    model = fasttext.load_model(MODEL_PATH)
    logger.info("Model loaded successfully")
except Exception as e:
    logger.error(f"Error loading model: {e}")
    raise

def get_analogy(word_a, word_b, word_c):
    """Returns the analogy for the given words using FastText model."""
    try:
        result = model.get_analogies(word_b, word_a, word_c, k=1)
        analogous_word = result[0][1]
    except Exception as e:
        logger.error(f"Error in analogy computation: {e}")
        analogous_word = "Unknown"
    
    logger.info(f"'{word_a}' is to '{word_b}' as '{word_c}' is to '{analogous_word}'")
    return analogous_word

def neighbourhood_test(word_a, word_b, word_c, true, predicted):
    """Checks if the predicted word is within an acceptable distance of the true word."""
    try:
        v_a = model.get_word_vector(word_a)
        v_b = model.get_word_vector(word_b)
        v_c = model.get_word_vector(word_c)
        v_predicted = model.get_word_vector(predicted)
        v_true = model.get_word_vector(true)
    except Exception as e:
        logger.error(f"Error in neighbourhood test: {e}")
        return False

    # Calculate distances and compare
    ballpark1 = linalg.norm(v_b - v_true) / 2
    ballpark2 = linalg.norm(v_c - v_true) / 2
    ballpark = min(ballpark1, ballpark2)

    distance = linalg.norm(v_predicted - v_true)

    if distance <= ballpark:
        return True
    return False

def neighbourhood_test_multi(word_a, word_b, word_c, trues, predicted):
    """Performs multiple neighbourhood tests for a list of true words."""
    for true in trues:
        try:
            v_a = model.get_word_vector(word_a)
            v_b = model.get_word_vector(word_b)
            v_c = model.get_word_vector(word_c)
            v_predicted = model.get_word_vector(predicted.lower())
            v_true = model.get_word_vector(true.lower())
        except Exception as e:
            logger.error(f"Error in multi neighbourhood test: {e}")
            return False

        # Calculate distances and compare
        ballpark1 = linalg.norm(v_b - v_true) / 2
        ballpark2 = linalg.norm(v_c - v_true) / 2
        ballpark = min(ballpark1, ballpark2)

        distance = linalg.norm(v_predicted - v_true)

        if distance <= ballpark:
            return True

    return False

def gather_analogies_from_data(input_file, output_file):
    """Gather analogies from a file, generate predictions, and save results."""
    try:
        with open(input_file, "r") as f:
            tajp_dict = eval(f.read())
    except Exception as e:
        logger.error(f"Error reading input file {input_file}: {e}")
        return

    preds = {}
    for i, lf in tqdm(enumerate(tajp_dict)):
        preds[lf] = []
        for j, analogy in enumerate(tajp_dict[lf][1]):
            word_a = analogy[0][0]
            word_b = analogy[0][1]
            word_c = analogy[1][0]

            predicted = get_analogy(word_a, word_b, word_c)
            preds[lf].append(predicted)
        
        logger.info(f"Processed {i + 1}/{len(tajp_dict)} analogy sets")

    try:
        with open(output_file, 'w') as f:
            f.write(str(preds))
        logger.info(f"Predictions saved to {output_file}")
    except Exception as e:
        logger.error(f"Error saving output file {output_file}: {e}")

def main():
    """Main entry point for processing."""
    input_file = os.getenv("INPUT_FILE", "indata/regrouped_synta1_sample.txt")
    output_file = os.getenv("OUTPUT_FILE", "outdata/synta1_preds.txt")
    
    logger.info(f"Starting analogy gathering with input: {input_file} and output: {output_file}")
    gather_analogies_from_data(input_file, output_file)
    logger.info("Finished processing analogies")

if __name__ == "__main__":
    main()
