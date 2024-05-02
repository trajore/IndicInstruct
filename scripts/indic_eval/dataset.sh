#!/bin/bash
# Check if the number of arguments is correct
if [ "$#" -ne 4 ]; then
    echo "Usage: $0 <folder_path> <ip_address> <port> <path_to_test>"
    exit 1
fi

# Extract arguments
folder_path=$1
ip_address=$2
port=$3
path_to_test=$4

# Create a temporary directory to store the archive
temp_dir=$(mktemp -d)
wait $!
archive_name="data1.tar.gz"

# Create a tar archive of the folder
echo "Creating a tar archive of the folder..."
tar -czf "$temp_dir/$archive_name" -C "$folder_path" .
wait $!
# Script section for dataset transfer
echo "Transferring dataset folder..."
echo "Transferring dataset folder to model owner..."
send_file(){

# Use openssl to transfer the tar archive over TLSv1.3 and specify the certificate and key
python3 - <<END
import socket
import ssl

def send_file(file_path, server_address, server_port, ca):
    # Check if the file exists
    try:
        with open(file_path, 'rb') as file:
            file_data = file.read()
    except FileNotFoundError:
        print(f"File not found: {file_path}")
        return

    # Create a TCP/IP socket
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
        # Wrap the socket in TLS encryption
        context = ssl.create_default_context(ssl.Purpose.SERVER_AUTH)
        context.load_verify_locations(cafile=ca)
        context.verify_mode = ssl.CERT_REQUIRED 
        context.check_hostname = False

        with context.wrap_socket(s, server_side=False, server_hostname=server_address) as ssl_socket:
            try:
                # Connect to the server
                ssl_socket.connect((server_address, server_port))
                print(f"Connected to {server_address}:{server_port}")

                # Send the file data
                ssl_socket.sendall(file_data)
                print("File sent successfully.")
            except Exception as e:
                print(f"Error occurred while sending file: {e}")
# Usage example
send_file("$temp_dir/$archive_name", "$ip_address",$port, "ca.crt")
END
}
send_file &
wait $!
echo "Dataset folder transferred successfully."

rm -rf "$temp_dir/$archive_name"
wait $!
echo "Connecting to model owner for predicted labels..."
wait $!
# Function to receive files
receive_file() {
    # Create a TCP/IP socket
    python3 - <<END
import socket
import ssl

def receive_file(peer_ip, peer_port, file_name, server_cert, server_key):
    # Create a TCP/IP socket
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
        # Bind the socket to the server address and port
        s.bind((peer_ip, peer_port))
        # Listen for incoming connections
        s.listen(1)
        print(f"Waiting for connection on {peer_ip}:{peer_port}...")

        # Accept the connection
        client_socket, client_address = s.accept()
        print(f"Connected to {client_address}")

        # Wrap the client socket in TLS encryption
        context = ssl.create_default_context(ssl.Purpose.CLIENT_AUTH)
        context.load_cert_chain(certfile=server_cert, keyfile=server_key)
        context.load_verify_locations(cafile="ca.crt")
        context.verify_mode = ssl.CERT_NONE

        with context.wrap_socket(client_socket, server_side=True) as ssl_socket:
            # Receive the file data
            with open(file_name, 'wb') as file:
                while True:
                    data = ssl_socket.recv(1024)
                    if not data:
                        break
                    file.write(data)

            print("File received successfully.")


# Usage example
receive_file("$ip_address", $port, 'data.tar.gz', 'server.crt', 'server.key')
END
}

# Start receiving the dataset folder
receive_file &
wait $!

# Clean up temporary directory
rm -rf "$temp_dir"
wait $!
#unzip the file
tar -xvf data.tar.gz
echo "predicted labels folder received successfully."
wait $!
echo "Running model accuracy tests..."
python3.10 $path_to_test > output_file.log
wait $!
echo "Model accuracy tests completed successfully."

log_file="output_file.log"
model_id=11
model_name="Bert"
owner_name="user1"
dataset_id=7

# Generate Python script
python_script=$(cat << END
import re
import argparse
import datetime
import requests

def extract_metrics(log_file):
    # Regular expressions to extract metrics
    accuracy_pattern = r'Accuracy: (\d+\.\d+)'
    precision_pattern = r'Precision: (\d+\.\d+)'
    recall_pattern = r'Recall: (\d+\.\d+)'
    f1_score_pattern = r'F1 Score: (\d+\.\d+)'

    # Initialize metrics
    accuracy = None
    precision = None
    recall = None
    f1_score = None

    # Read log file and extract metrics
    with open('$log_file', 'r') as f:
        log_content = f.read()
        accuracy_match = re.search(accuracy_pattern, log_content)
        if accuracy_match:
            accuracy = float(accuracy_match.group(1))

        precision_match = re.search(precision_pattern, log_content)
        if precision_match:
            precision = float(precision_match.group(1))

        recall_match = re.search(recall_pattern, log_content)
        if recall_match:
            recall = float(recall_match.group(1))

        f1_score_match = re.search(f1_score_pattern, log_content)
        if f1_score_match:
            f1_score = float(f1_score_match.group(1))

    return accuracy, precision, recall, f1_score

def main():
    # Extract metrics
    accuracy, precision, recall, f1_score = extract_metrics('$log_file')

    # Display metrics
    model_id=$model_id
    model_name="$model_name"
    owner_name="$owner_name"
    dataset_id=$dataset_id
    print("Model ID:", model_id)
    print("Model Name:", model_name)
    print("Owner Name:", owner_name)
    print("Accuracy:", accuracy)
    print("Precision:", precision)
    print("Recall:", recall)
    print("F1 Score:", f1_score)
     # Send metrics to update_leaderboard view
    auth_token = 'zxajnply1gkbpznixmyelkanqhokx12w'
    csrf_token = 'hMNjZootaF9zJDni3MRJPI5bY6TmIaaK'
    payload = {
        'model_id': model_id,
        'dataset_id': dataset_id,
        'accuracy': accuracy,
        'precision': precision,
        'recall': recall,
        'f1_score': f1_score,
    }
    cookies = {
    'sessionid': auth_token,
    'csrftoken': csrf_token,
}

    response = requests.post('http://127.0.0.1:8000/leaderboard/update', data=payload, cookies=cookies)
    if response.status_code == 200:
        print("Leaderboard updated successfully")
    else:
        print("Error updating leaderboard:", response.text)

if __name__ == "__main__":
    main()
END
)

# Save Python script to a temporary file
python_script_file="/tmp/metrics_script.py"
echo "$python_script" > "$python_script_file"

# Execute Python script
python3 "$python_script_file" "$@"

