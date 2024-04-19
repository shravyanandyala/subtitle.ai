import React, { useState } from 'react';
import axios from 'axios';

export default function Subtitle() {
  const [file, setFile] = useState(null);
  const [results, setResults] = useState({transcription: '', translation: '', alignment: ''});

  function handleChange(event) {
    setFile(event.target.files[0]);
  }

  function handleSubmit(event) {
    event.preventDefault()
    const url = 'http://localhost:5000/process-audio';
    const formData = new FormData();
    const reader = new FileReader();
    
    reader.readAsDataURL(file);
    reader.onload = () => formData.append('file', reader.result);
    formData.append('filename', file.name)

    const config = {
      headers: {
        'Access-Control-Allow-Origin': '*',
        'Access-Control-Allow-Headers': '*',
        'Authorization': "http://localhost:3000/login",
        'content-type': 'multipart/form-data',
      },
    };

    try {
      axios.post(url, formData, config).then((response) => {
        setResults(response.data);
      });
    } catch (error) {
        console.error('Error with uploading / processing file:', error);
    }
  }

  return (
    <div>
      <h1>Audio Transcription and Translation</h1>
      <form onSubmit={handleSubmit}>
        <input type="file" onChange={handleChange} accept="audio/*" />
        <button type="submit">Upload and Process</button>
      </form>
      {results.transcription && <div><h2>Transcription</h2><p>{results.transcription}</p></div>}
      {results.translation && <div><h2>Translation</h2><p>{results.translation}</p></div>}
      {results.alignment && <div><h2>Alignment</h2><p>{results.alignment}</p></div>}
    </div>
  );
}