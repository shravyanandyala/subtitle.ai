from flask import Flask, request, jsonify
from flask_cors import CORS
from translate import run

app = Flask(__name__)
CORS(app)

@app.route('/process-audio', methods=['POST'])
def process_audio():
    audio_file = request.files['file']
    # Process video, transcribe, translate, align
    results = run(audio_file)
    
    return jsonify(results)

if __name__ == '__main__':
    app.run(debug=True)