import csv
import random

def make_csv_file(eval_file_path, header_row):
    """
        Creates a new CSV file at the specified path with the given header row.

        Parameters:
        eval_file_path (str): The path where the CSV file will be created.
        header_row (list): A list containing the column headers for the CSV file.

        Returns:
        list: The header row that was written to the CSV file.
        """
    with open(eval_file_path, 'w', newline='') as evaluation_file:
        eval_csv_writer = csv.writer(evaluation_file, delimiter=';', quotechar='"', quoting=csv.QUOTE_MINIMAL)
        eval_csv_writer.writerow(header_row)
        return header_row

def write_metrics_to_csv(eval_file_path, header_row, result_metrics):
    """
        Writes the specified metric values to the CSV file at the given path.

        Parameters:
        eval_file_path (str): The path of the CSV file.
        header_row (list): A list containing the column headers for the CSV file.
        result_metrics (dict): A dictionary containing the metric values to be written to the CSV file.

        Returns:
        None
        """
    with open(eval_file_path, 'a', newline='') as evaluation_file:
        eval_csv_writer = csv.writer(evaluation_file, delimiter=';', quotechar='"',
                                     quoting=csv.QUOTE_MINIMAL)
        eval_csv_writer.writerow(make_csv_row(header_row, result_metrics))

def make_csv_row(header_row, result_metrics):
    """
        Creates a new CSV row based on the specified header row and result metrics.

        Parameters:
        header_row (list): A list containing the column headers for the CSV file.
        result_metrics (dict): A dictionary containing the metric values to be written to the CSV file.

        Returns:
        list: A list containing the metric values in the order specified by the header row.
        """
    row = []
    for field in header_row:
        row.append(result_metrics[field])
    return row

def random_permutation_with_groups(lst, group_size):
    random.seed(42)
    # Check if the list length is divisible by the group size
    if len(lst) % group_size != 0:
        raise ValueError("List length must be divisible by the group size")

    # Reshape the list into a list of lists with each sublist representing a group
    grouped_lst = [lst[i:i + group_size] for i in range(0, len(lst), group_size)]

    # Shuffle the sublists (groups) independently
    random.shuffle(grouped_lst)

    # Flatten the list of lists to get the final permutation
    result = [item for sublist in grouped_lst for item in sublist]

    return result