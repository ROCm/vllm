import requests
import json

api_url = "http://10.216.51.98:9000/upload"

def send_data_to_host(api_url, data):
    """
    Sends JSON data to the host machine using the specified API endpoint.

    Args:
        api_url (str): The full URL of the upload endpoint (e.g., http://host_ip:9000/upload)
        data (dict): The data to send as a JSON payload.

    Returns:
        dict: The server's response in JSON format.
    """
    headers = {"Content-Type": "application/json"}
    try:
        # Convert the data dictionary to JSON and send it using POST
        response = requests.post(api_url, headers=headers, json=data)

        # Raise an error if the request was unsuccessful
        response.raise_for_status()

        print(f"Data successfully sent to {api_url}. Server response:")
        print(response.json())  
        return response.json()
    except requests.exceptions.RequestException as e:
        print(f"Error occurred while sending data: {e}")
        return {"error": str(e)}

# Example callback function to fetch and send data
def example_callback(api_url):
    """
    Simulates data collection and sends it to the host machine.
    """
    # Example data you want to send
    data = {
        "timestamp": ["2024-12-16 08:10:36.164014", "2024-12-16 08:10:36.164027", "2024-12-16 08:10:36.164030"],
        "machine_label": ["machine1", "machine12", "machine3"],
        "operation_result": ["success", "failure", "success"]
    }

    # Call the function to send the data
    send_data_to_host(api_url, data)

if __name__ == "__main__":
    # Run the example callback
    example_callback(api_url)

