import os
import glob
import re
import numpy as np
from matplotlib import pyplot as plt
import string

def plot_averages(avg1, avg2, avg3, avg4, avg5):

    x = np.arange(8)  # Number of bars
    width = 0.15  # Width of each bar

    # Adjust the positions
    positions = [x - 2 * width, x - width, x, x + width, x + 2 * width]

    # Plotting
    plt.figure(figsize=(12, 7))

    plt.bar(positions[0], avg1, width=width, color='red', label='exact match')
    plt.bar(positions[1], avg2, width=width, color='pink', label='contain match')
    plt.bar(positions[2], avg3, width=width, color='blue', label='exact match synonym')
    plt.bar(positions[3], avg4, width=width, color='turquoise', label='contain match synonym')
    plt.bar(positions[4], avg5, width=width, color='orange', label='word embedding neighbourhood')

    # Adding titles and labels
    plt.title('Percentage of Different Configurations based on 5 Metrics')
    plt.xlabel('Configuration')
    plt.ylabel('Percentage (%)')

    xticks = ["0shot\nno context\nno feedback", "0shot\nno context\nwith feedback", 
              "0shot\nwith context\nno feedback", "0shot\nwith context\nwith feedback", 
              "3shot\nno context\nno feedback", "3shot\nno context\nwith feedback",
              "3shot\nwith context\nno feedback", "3shot\nwith context\nwith feedback"]

    plt.xticks(ticks=x, labels=xticks, rotation=0, ha='right')
    plt.ylim(0, 100)
    plt.yticks(np.arange(0, 101, 5))

    # Adding a legend
    plt.legend()

    # Show grid
    plt.grid(True, linestyle='--', alpha=0.7)

    # Show plot
    plt.tight_layout()
    plt.show()

def bootstrap(data, num_samples=1000):
    means = []
    n = len(data)
    for _ in range(num_samples):
        sample = np.random.choice(data, size=n, replace=True)
        means.append(np.mean(sample) * 100)
    return np.array(means)

def operator(text):
    def operator_punct(text):
        # Remove only the final punctuation if it exists
        if text and text[-1] in string.punctuation:
            text = text[:-1]
        return text

    # Check the format of the input text
    if ':' in text:
        if '::' in text:
            # For 'text1: text2 :: text3 : text4', keep only text4
            last_part = text.split('::')[-1].strip()
            if ':' in last_part:
                # Remove punctuation and return the text after the last colon
                return operator_punct(last_part.split(':')[-1].strip())
        else:
            # For 'text1 : text2', keep only text2
            return operator_punct(text.split(':')[-1].strip())

    # If neither format matches, return the cleaned text
    return operator_punct(text)

# Function to process each line and extract components
def process_line(line):
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
        options = [opt.strip("'") for opt in options]  # Remove extra single quotes

        return first_term, second_term, third_term, fourth_term, prediction, options
    else:
        return None

if __name__ == "__main__":

    print("Evaluating models...")

    tajps = ["config1", "config2", "config3"]

    # Placeholder directories for input data
    dirnames = [
                "path/to/model1/outputs",
                "path/to/model2/outputs",
                "path/to/model3/outputs",
                "path/to/model4/outputs",
                "path/to/model5/outputs",
                "path/to/model6/outputs",
                "path/to/model7/outputs",
                "path/to/model8/outputs"
    ]

    for i, dirname in enumerate(dirnames):

        # Regular expression patterns to match the necessary components
        analogy_pattern = re.compile(r'IN: (.*?): (.*?) :: (.*?): (.*?) \|\|')
        prediction_pattern = re.compile(r'«(.*?)»')
        options_pattern = re.compile(r'\[(.*?)\]')

        # Function to calculate correctness
        def calculate_correctness(first_term, second_term, third_term, fourth_term, prediction, options):
            prediction = operator(prediction)

            # Contain match synonymic
            for option in options:
                if option.lower() in prediction.lower():
                    return True
            return False

        # Get a list of all files in the directory
        file_list = glob.glob(os.path.join(dirname, '*'))

        # Dictionary to store correctness percentage for each file
        correctness_dict = {}

        # Collect all exam scores
        exam_dict = {}

        bootstrap_dict = {}

        # Iterate through each file in the directory
        for file_path in file_list:

            exam_scores = []

            # Check if the path is a file (not a directory)
            if os.path.isfile(file_path):
                with open(file_path, 'r') as file:
                    lines = file.readlines()

                total_lines = 0
                correct_lines = 0

                backup_line = ""
                for line in lines:
                    if line.strip() and not line.startswith('-----'):
                        result = process_line(line)

                        if result is None:          # i.e. could not parse well. The model's response spans several lines, we glue them back
                            backup_line += ' ' + line.strip()
                            result = process_line(backup_line)
                            if result is None:  # check if it is still bad, with this partial fix from the iteration
                                continue
                            else:               # we have successfully fixed the problem, and we can reset buffer
                                backup_line = ""

                        if result:
                            total_lines += 1
                            first_term, second_term, third_term, fourth_term, prediction, options = result
                            is_correct = calculate_correctness(first_term, second_term, third_term, fourth_term,
                                                               prediction, options)
                            if is_correct:
                                correct_lines += 1
                                exam_scores.append(1)
                            else:
                                exam_scores.append(0)

                # calculating the average
                if total_lines > 0:
                    correctness_percentage = (correct_lines / total_lines) * 100
                else:
                    correctness_percentage = 0.0

                base_name = os.path.basename(file_path).replace("_output.txt", "")
                correctness_dict[base_name] = correctness_percentage
                exam_dict[base_name] = exam_scores

                # Perform bootstrapping
                bootstrap_results = bootstrap(np.array(exam_scores), num_samples=1000)
                bootstrap_mean = np.mean(bootstrap_results)
                bottom_whisker = np.percentile(bootstrap_results, 2.5)
                top_whisker = np.percentile(bootstrap_results, 97.5)

                bootstrap_info = (bootstrap_mean, bottom_whisker, top_whisker)
                bootstrap_dict[base_name] = bootstrap_info

        # Reorder dictionary and plot
        order = ["Config1", "Config2", "Config3", "Config4", "Config5", "Config6", "Config7", "Config8"]
        reordered_dict = {key: correctness_dict[key] for key in order if key in correctness_dict}

        # Extract values for plotting
        file_names = list(reordered_dict.keys())
        correctness_vals = np.array(list(reordered_dict.values()))
        errors = np.array([top_whisker - bootstrap_mean for bootstrap_mean, bottom_whisker, top_whisker in list(bootstrap_dict.values())])

        fig, ax = plt.subplots()

        # Bar plot with error bars
        bars = ax.bar(file_names, correctness_vals, yerr=errors, capsize=5)
        ax.set_xlabel('Lexical Function')
        eval_type = "Exact Match"
        ax.set_ylabel(f"{eval_type} Evaluation")
        ax.set_title("Evaluation Scores per Lexical Function")

        ax.set_xticks(file_names)
        ax.set_xticklabels(file_names, rotation=90)
        ax.set_ylim(0, 100)
        ax.set_yticks(range(0, 101, 10))

        # add horizontal mean line
        average_correctness = np.mean(list(correctness_dict.values()))
        ax.axhline(y=average_correctness, color='r', linestyle='--', label=f'Average: {average_correctness:.2f}')

        ax.legend()
        plt.tight_layout()
        plt.show()

        print(reordered_dict)
