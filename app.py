from flask import Flask, request, jsonify
from datetime import datetime

app = Flask(__name__)

@app.route('/process_data', methods=['POST'])
def process_data():
    try:
        # Get client name from request
        client_name = request.json.get('client_name')

        # Get data from request
        data = request.json.get('data')

        # Process the data (replace this with your processing logic)
        processed_data = process_data_function(data)

        # Return a JSON response
        response = {
            'status': 'success',
            'client_name': client_name,
            'processed_data': processed_data
        }
        return jsonify(response)

    except Exception as e:
        # Handle exceptions
        response = {'status': 'error', 'message': str(e)}
        return jsonify(response), 500

def process_data_function(data):
    # Implement your data processing logic here
    # For example, you can print the received data

    SendTime = datetime.strptime(data["SendTime"], "%Y-%m-%dT%H:%M:%S").strftime("%H:%M:%S.%f")[:-3]
    nowTime= datetime.now().strftime("%H:%M:%S.%f")[:-3]

    data_array = [
        data["TaskID"],
        SendTime,
        nowTime,
        data["TaskFileSize"],
        data["TaskOutputFileSize"],
        data["TaskFileLength"]
        ]
    # Replace this with your actual processing logic
    processed_data = {'message': 'Data processed successfully',
                      "TaskID": data["TaskID"],
                      "ArrivalTime": nowTime,
                      "SendTime": SendTime,
                      "TaskFileSize":data["TaskFileSize"],
                      "TaskOutputFileSize":data["TaskOutputFileSize"],
                      "TaskFileLength":data["TaskFileLength"],
                      
                      }
    return processed_data

if __name__ == '__main__':
    app.run(debug=True)
