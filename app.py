import os
from flask import Flask, request, jsonify
from flask.helpers import send_from_directory
from flask_cors import CORS, cross_origin
from werkzeug.utils import secure_filename
import ffmpeg
from translate import run
from time import time

upload_dir = '/Users/shravya/Projects/speech.ru/userFiles/'

app = Flask(__name__, static_folder='translate/build', static_url_path='')
CORS(app)

@app.route('/process-audio', methods=['POST'])
@cross_origin()
def process_audio():
    # Save video file from user
    file = request.files['file']
    curr_time = time()

    filename = secure_filename(f'{curr_time}.mp4')
    video_path = os.path.join(upload_dir, filename)
    file.save(video_path)
    app.logger.info('Saved video file to %s', video_path)

    # Extract the audio
    filename = secure_filename(f'{curr_time}.mp3')
    audio_path = os.path.join(upload_dir, filename)
    input = ffmpeg.input(video_path)
    input.output(audio_path, acodec='mp3').run(overwrite_output=True)
    
    # Process video, transcribe, translate, align
    results = run([audio_path])

    # Delete saved files
    os.remove(video_path)
    os.remove(audio_path)

    return jsonify(results[audio_path])

@app.route('/')
@cross_origin()
def serve():
    return send_from_directory(app.static_folder, 'index.html')

if __name__ == '__main__':
    app.run(debug=True)