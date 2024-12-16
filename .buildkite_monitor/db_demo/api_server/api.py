from flask import Flask, jsonify, request
import pandas as pd
from datetime import datetime
import zoneinfo


app = Flask(__name__)


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