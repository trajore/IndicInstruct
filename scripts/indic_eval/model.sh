#!/bin/bash
                # Check if the number of arguments is correct
            Third_party_ip_address=127.0.0.1
            Third_party_port=7001
            verification_code=1234
            model_file_path=$1

            if [ "$#" -ne 2 ]; then
                echo "Usage: $0 <model_file_path> <verification code>"
                exit 1
            fi
            ip_address=$Third_party_ip_address
port=$Third_party_port
            
# Create a temporary directory to store the archive
temp_dir=$(mktemp -d)
wait $!
archive_name="model.tar.gz"

# Create a tar archive of the folder
echo "Creating a tar archive of the folder..."
tar -czf "$temp_dir/$archive_name" -C "$model_file_path" .
wait $!
# Script section for model transfer
echo "Transferring model folder..."
echo "Transferring model folder to model owner..."
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
echo "model folder transferred successfully."
            