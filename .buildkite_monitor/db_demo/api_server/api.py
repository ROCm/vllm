from flask import Flask, jsonify, request
import pandas as pd
from datetime import datetime
import zoneinfo


app = Flask(__name__)

# Example callback: Fetch local data and return it
def fetch_data():
    # Replace this with logic to access your data source
    data = {"timestamp": ['2024-12-16 08:10:36.164014','2024-12-16 08:10:36.164027', '2024-12-16 08:10:36.164030'], 
            "machine_label": ["machine1", "machine12", "machine3"], 
            'operation_result':['success', 'failure', 'success']}
    return data

@app.route("/data", methods=["GET"])
def get_data():
    """API endpoint to fetch data."""
    try:
        data = fetch_data()
        return jsonify(data)
    except Exception as e:
        return jsonify({"error": str(e)}), 500




@app.route("/upload", methods=["POST"])
def upload_data():
    """Endpoint to receive data from the remote machine."""
    try:
        # Assume the remote machine sends JSON data
        data = request.get_json()
        if data is None:
            return jsonify({"error": "No data provided"}), 400

        # Save data to a local Parquet file
        df = pd.DataFrame(data)
        now = datetime.now(zoneinfo.ZoneInfo('Europe/Helsinki')).strftime('%Y-%m-%d_%H:%M:%S')

        file_path = f'/mnt/home/vllm_fresh/.buildkite_monitor/db_demo/api_server/received_data/agent_health_data_{now}.parquet'
        df.to_parquet(file_path, engine="pyarrow")

        return jsonify({"message": f"Data saved to {file_path}"}), 200
    except Exception as e:
        return jsonify({"error": str(e)}), 500

if __name__ == "__main__":
    print("Starting API server...")
    app.run(host="0.0.0.0", port=9000)        