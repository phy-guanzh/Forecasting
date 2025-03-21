import csv


def convert_text_to_csv(input_filename, output_filename):
    # Read all lines from the input file
    with open(input_filename, 'r', encoding='utf-8') as infile:
        lines = infile.readlines()

    # Remove any leading/trailing whitespace and filter out blank lines
    cleaned_lines = [line.strip() for line in lines if line.strip() != '']

    # Assume the first three non-empty lines form the header
    header = cleaned_lines[:3]
    # The remaining lines are data; group them into rows of three values each
    data_lines = cleaned_lines[3:]
    rows = [data_lines[i:i + 3] for i in range(0, len(data_lines), 3)]

    # Write the header and rows to the CSV file
    with open(output_filename, 'w', newline='', encoding='utf-8') as csvfile:
        writer = csv.writer(csvfile)
        writer.writerow(header)
        for row in rows:
            if len(row) == 3:  # Only write complete rows
                writer.writerow(row)


if __name__ == '__main__':
    # Change these filenames as needed
    input_filename = 'test.csv'
    output_filename = 'output.csv'
    convert_text_to_csv(input_filename, output_filename)

