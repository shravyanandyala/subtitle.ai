import React, { useState } from 'react';
import ReactPlayer from 'react-player';
import { RotatingLines } from 'react-loader-spinner';
import axios from 'axios';
import './Subtitle.css';

export default function Subtitle() {
  const [file, setFile] = useState();
  const [videoURL, setVideoURL] = useState();
  const [results, setResults] = useState();
  const [isLoading, setLoading] = useState(false);

  function handleChange(event) {
    setFile(event.target.files[0]);
    setVideoURL(URL.createObjectURL(event.target.files[0]));
  }

  function handleSubmit(event) {
    setLoading(true);
    setResults(null);
    event.preventDefault();
    const url = 'http://localhost:5000/process-audio';
    const formData = new FormData();
    formData.append('file', file)
    formData.append('video_path', videoURL);
    axios.post(url, formData).then((response) => {
      setResults(response.data);
    }).catch((error) => {
      console.error('Error:', error.response);
    }).finally(() => {setLoading(false);});
  }

  return (
    <div>
      <h2>Video Transcription and Translation</h2>
      <form onSubmit={handleSubmit}>
        <input type="file" onChange={handleChange} accept="video/*" />
        {/* If waiting for an API response, disable upload button and
          * display loading spinner instead. */}
        {isLoading
          ? <RotatingLines
          strokeColor="grey"
          strokeWidth="5"
          width="20"
          visible={true}/>
          : <button type="submit">Upload and Process</button>
        }
      </form>
      <br/>
      {results &&
        <div id="container">
          <ReactPlayer
            className='react-player fixed-bottom'
            url={videoURL}
            width='100%'
            height='100%'
            controls = {true}
          />
          {results.transcription && <div><h2>Transcription</h2><p>{results.transcription}</p></div>}
          {results.translation && <div><h2>Translation</h2><p>{results.translation}</p></div>}
          {results.alignment && <div><h2>Alignment</h2>
            {results.alignment.map(([input, output]) =>
              <p>({input}, {output})</p>
            )}
          </div>}
        </div>
      }
    </div>
  );
}