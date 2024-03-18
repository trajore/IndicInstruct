#!/bin/bash
            # Check if the number of arguments is correct
if [ "$#" -ne 4 ]; then
    echo "Usage: $0 <evaluation_file_path> <ip_address> <port> <folder_path>"
    exit 1
fi

# Extract arguments
evaluation_file_path=$1
ip_address=$2
port=$3
folder_path=$4

echo "Connecting to dataset owner..."
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

tar -xzf data.tar.gz
wait $!
echo "Running model evaluation..."
# Add your model evaluation command here
$evaluation_file_path
#wait $!
rm -rf data.tar.gz
wait $!
# Add your command to retrieve accuracy results here
# Create a temporary directory to store the archive
temp_dir=$(mktemp -d)
archive_name="data.tar.gz"

# Create a tar archive of the folder
echo "Creating a tar archive of the folder..."
tar -czf "$temp_dir/$archive_name" -C "$folder_path" .

# Script section for dataset transfer
echo "Transferring predicted label folder..."

# Use openssl to transfer the tar archive over TLSv1.3 and specify the certificate and key
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
echo "Predicted labels sent successfully."
echo "Results will be available on the leaderboard"
# Clean up temporary directory
rm -rf "$temp_dir"

