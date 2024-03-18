mapping = {"0": "true", "1": "unknown", "2": "false"}

input_file = "test_labels.txt"
output_file = "output.txt"

with open(input_file, "r") as f:
    lines = f.readlines()

with open(output_file, "w") as f:
    for line in lines:
        label = line.strip()
        if label in mapping:
            updated_label = mapping[label]
            f.write(updated_label + "\n")
        else:
            f.write("unknown\n")  # Default to "unknown" if label not found in mapping


