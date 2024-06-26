from http.server import HTTPServer, BaseHTTPRequestHandler
import ssl
import os, subprocess,re
from urllib.parse import parse_qs

# Define the directory where files will be stored
UPLOAD_DIR = 'uploads'

class RequestHandler(BaseHTTPRequestHandler):
    def do_POST(self):
        # Get the length of the content
        content_length = int(self.headers['Content-Length'])

        # Read the raw HTTP POST data from the socket till all is received

        post_data = self.rfile.read(content_length)
        # Convert the raw data to a string
        post_data = post_data.decode('utf-8')
        boundary = re.findall(r'--(.*?)\r\n', post_data)[0]
        parts = post_data.split('--' + boundary)[1:-1]

        # Initialize an empty dictionary to store the parsed data
        parsed_data = {}

        # Parse each part and extract the field name and value
        for part in parts:
            # Extract field name
            field_match = re.search(r'Content-Disposition: form-data; name="(.*?)"', part)
            if field_match:
                field_name = field_match.group(1)
                
                # Extract field value
                value_match = re.search(r'\r\n\r\n(.*?)\r\n', part, re.DOTALL)
                if value_match:
                    field_value = value_match.group(1)
                    
                    # Add the field name and value to the parsed data dictionary
                    parsed_data[field_name] = field_value

        # Print the parsed data
        print(parsed_data)
        modelID = parsed_data['modelID']
        modelName = parsed_data['modelName']
        datasetID = parsed_data['datasetID']
        datasetName = parsed_data['datasetName']
        verification_code = parsed_data['verification_code']




        # Create the upload directory if it doesn't exist
        os.makedirs(UPLOAD_DIR, exist_ok=True)

        # Save the uploaded files
        ca_cert_file = parsed_data['ca_cert']
        server_cert_file = parsed_data['server_cert']
        server_key_file = parsed_data['server_key']

        ca_cert_path = os.path.join(UPLOAD_DIR, 'ca_cert.pem')
        server_cert_path = os.path.join(UPLOAD_DIR, 'server_cert.pem')
        server_key_path = os.path.join(UPLOAD_DIR, 'server_key.pem')

        with open(ca_cert_path, 'wb') as f:
            f.write(ca_cert_file.encode())

        with open(server_cert_path, 'wb') as f:
            f.write(server_cert_file.encode())

        with open(server_key_path, 'wb') as f:
            f.write(server_key_file.encode())

        # Respond with success message
        self.send_response(200)
        self.send_header('Content-type', 'text/plain')
        self.end_headers()
        self.wfile.write(b'Metadata and files received successfully')

        # Call the function to run the Bash script
        evaluation_file_path = "./indicxnli.sh"
        print(modelID,modelName,datasetID,datasetName,evaluation_file_path, verification_code, ca_cert_path, server_cert_path, server_key_path)
        run_bash_script(modelID, modelName, datasetID, datasetName, evaluation_file_path, verification_code, ca_cert_path, server_cert_path, server_key_path)

def run_bash_script(modelID,modelName,datasetID,datasetName,evaluation_file_path, verification_code, ca_cert_path, server_cert_path, server_key_path):
    # Command to execute the Bash script
    bash_command = f"./ttp_bash.sh {modelID} {modelName} {datasetID} {datasetName} {evaluation_file_path} {verification_code} {ca_cert_path} {server_cert_path} {server_key_path}"

    # Execute the Bash script
    def execute(cmd):
        popen = subprocess.Popen(cmd,shell=True, stdout=subprocess.PIPE, universal_newlines=True)
        for stdout_line in iter(popen.stdout.readline, ""):
            yield stdout_line 
        popen.stdout.close()
        return_code = popen.wait()
        if return_code:
            raise subprocess.CalledProcessError(return_code, cmd)

# Example
    for path in execute(bash_command):
        print(path, end="")


    # Print the output and error messages
    # print("Bash script output:", stdout.decode('utf-8'))
    # print("Bash script errors:", stderr.decode('utf-8'))

def run(server_class=HTTPServer, handler_class=RequestHandler, port=8001):
    server_address = ('', port)
    httpd = server_class(server_address, handler_class)
    
    
    # Load server certificate and private key
    ca_cert_file = 'ca.crt'
    server_cert_file = 'server.crt'
    server_key_file = 'server.key'
    context = ssl.create_default_context(ssl.Purpose.CLIENT_AUTH)
    context.load_cert_chain(certfile=server_cert_file, keyfile=server_key_file)
    context.load_verify_locations(cafile=ca_cert_file)
    context.verify_mode = ssl.CERT_NONE

    httpd.socket = context.wrap_socket(httpd.socket, server_side=True)
    
    print(f'Starting server on port {port}...')
    httpd.serve_forever()


if __name__ == '__main__':
    run()