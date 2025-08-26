import os
import glob
import re
import numpy as np
import matplotlib.pyplot as plt
import string
import logging
from dotenv import load_dotenv

# Load environment variables (if any)
load_dotenv()

# Configure logger for better output management
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Constants
BASE_DIR = os.getenv("BASE_DIR", "/path/to/your/base")  # Set base directory as environment variable if necessary
NUM_BOOTSTRAP_SAMPLES = 1000

# Regular expression patterns
analogy_pattern = re.compile(r'IN: (.*?): (.*?) :: (.*?): (.*?) \|\|')
prediction_pattern = re.compile(r'«(.*?)»')
options_pattern = re.compile(r'\[(.*?)\]')

def calculate_correctness(first_term, second_term, third_term, fourth_term, prediction, options):
    """Calculates correctness based on various criteria."""
    prediction = operator(prediction)

    # Match by containment
    for option in options:
        if option.lower() in prediction.lower():
            return True
    return False

def plot_averages(avg1, avg2, avg3, avg4, avg5):
    """Plot the averages of the correctness metrics."""
    x = np.arange(8)
    width = 0.15  # Width of each bar
    positions = [x - 2 * width, x - width, x, x + width, x + 2 * width]

    plt.figure(figsize=(12, 7))

    plt.bar(positions[0], avg1, width=width, color='red', label='exact match')
    plt.bar(positions[1], avg2, width=width, color='pink', label='contain match')
    plt.bar(positions[2], avg3, width=width, color='blue', label='exact match synonimique')
    plt.bar(positions[3], avg4, width=width, color='turquoise', label='contain match synonimique')
    plt.bar(positions[4], avg5, width=width, color='orange', label='voisinage plongement de mots')

    plt.title('Pourcentage des différentes configurations Llama3 selon 5 métriques (synta1)')
    plt.xlabel('Configuration')
    plt.ylabel('Pourcentage (%)')

    xticks = ["0shot\nno context\nno feedback", "0shot\nno context\nwith feedback", 
              "0shot\nwith context\nno feedback", "0shot\nwith context\nwith feedback", 
              "3shot\nno context\nno feedback", "3shot\nno context\nwith feedback", 
              "3shot\nwith context\nno feedback", "3shot\nwith context\nwith feedback"]

    plt.xticks(ticks=x, labels=xticks, rotation=0, ha='right')
    plt.ylim(0, 100)
    plt.yticks(np.arange(0, 101, 5))

    plt.legend()
    plt.grid(True, linestyle='--', alpha=0.7)
    plt.tight_layout()
    plt.show()

def bootstrap(data, num_samples=NUM_BOOTSTRAP_SAMPLES):
    """Perform bootstrapping to calculate the confidence intervals."""
    means = []
    n = len(data)
    for _ in range(num_samples):
        sample = np.random.choice(data, size=n, replace=True)
        means.append(np.mean(sample) * 100)
    return np.array(means)

def operator(text):
    """Helper function to clean the text."""
    def operator_punct(text):
        text = text.replace('_', ' ')
        if text and text[-1] in string.punctuation:
            text = text[:-1]
        return text

    if ':' in text:
        if '::' in text:
            last_part = text.split('::')[-1].strip()
            if ':' in last_part:
                return operator_punct(last_part.split(':')[-1].strip())
        else:
            return operator_punct(text.split(':')[-1].strip())

    return operator_punct(text)

def process_line(line):
    """Process a line and extract the components of the analogy."""
    analogy_match = analogy_pattern.search(line)
    prediction_match = prediction_pattern.search(line)
    options_match = options_pattern.search(line)

    if analogy_match and prediction_match and options_match:
        first_term = analogy_match.group(1)
        second_term = analogy_match.group(2)
        third_term = analogy_match.group(3)
        fourth_term = analogy_match.group(4)

        prediction = prediction_match.group(1)
        options = options_match.group(1).split(', ')
        options = [opt.strip("'") for opt in options]

        return first_term, second_term, third_term, fourth_term, prediction, options
    return None

def process_files_in_directory(base_dir, options, bootstrap_samples=NUM_BOOTSTRAP_SAMPLES):
    """Process all the files in a given directory and calculate correctness."""
    correctness_dict = {}
    bootstrap_dict = {}
    exam_dict = {}

    for option in options:
        base_name = option  # Adjust this line to extract the file name correctly
        exam_scores = []
        file_path = os.path.join(base_dir, option)

        if not os.path.isdir(file_path):
            continue

        file_list = glob.glob(os.path.join(file_path, '*'))
        for file in file_list:
            if os.path.isfile(file):
                with open(file, 'r') as f:
                    lines = f.readlines()

                total_lines = 0
                correct_lines = 0
                backup_line = ""

                for line in lines:
                    if line.strip() and not line.startswith('-----'):
                        result = process_line(line)
                        if result is None:
                            backup_line += ' ' + line.strip()
                            result = process_line(backup_line)
                            if result is None:
                                continue
                            else:
                                backup_line = ""

                        if result:
                            total_lines += 1
                            first_term, second_term, third_term, fourth_term, prediction, options = result
                            is_correct = calculate_correctness(first_term, second_term, third_term, fourth_term, prediction, options)
                            exam_scores.append(1 if is_correct else 0)

                if total_lines > 0:
                    correctness_percentage = (correct_lines / total_lines) * 100
                    correctness_dict[base_name] = correctness_percentage
                    exam_dict[base_name] = exam_scores

                    bootstrap_results = bootstrap(np.array(exam_scores), num_samples=bootstrap_samples)
                    bootstrap_mean = np.mean(bootstrap_results)
                    bottom_whisker = np.percentile(bootstrap_results, 2.5)
                    top_whisker = np.percentile(bootstrap_results, 97.5)

                    bootstrap_dict[base_name] = (bootstrap_mean, bottom_whisker, top_whisker)

    return correctness_dict, bootstrap_dict

def main():
    # Example usage
    options = [
        "para", "synta0", "synta1"  # Adjust these options as needed
    ]

    correctness_dict, bootstrap_dict = process_files_in_directory(BASE_DIR, options)
    
    # Perform further processing and plotting as needed
    logger.info("Correctness dict: %s", correctness_dict)
    logger.info("Bootstrap dict: %s", bootstrap_dict)

    # Reorder and plot the results
    avg1, avg2, avg3, avg4, avg5 = [], [], [], [], []  # Populate with actual values
    plot_averages(avg1, avg2, avg3, avg4, avg5)

if __name__ == "__main__":
    main()
