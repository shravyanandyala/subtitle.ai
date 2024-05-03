import React, { useState } from 'react';
import { useNavigate } from 'react-router-dom';
import ReactPlayer from 'react-player';
import { TiDownload } from "react-icons/ti";
import { Stack } from '@mui/material';
import { TypeAnimation } from 'react-type-animation';
import axios from 'axios';
import './Subtitle.css';

export default function Subtitle() {
  const navigate = useNavigate();
  const [file, setFile] = useState();
  const [videoURL, setVideoURL] = useState();
  const [subtitles, setSubtitles] = useState([]);
  const [results, setResults] = useState();
  const [isLoading, setLoading] = useState(false);

  const handleChange = (event) => {
    setResults(null);
    setFile(event.target.files[0]);
    setVideoURL(URL.createObjectURL(event.target.files[0]));
  };

  const handleSubmit = (event) => {
    setLoading(true);
    setResults(null);
    event.preventDefault();
    const url = 'http://localhost:5000/process-audio';
    const formData = new FormData();
    formData.append('file', file)
    formData.append('video_path', videoURL);
    axios.post(url, formData).then((response) => {
      setResults(response.data);
      if (response.data) {
        setSubtitles([
          {
            kind: 'subtitles',
            src: process.env.PUBLIC_URL + '/' + response.data.ru_subs,
            srcLang: 'ru',
            label: 'Russian',
            mode: 'showing',
            default: true
          },
          {
            kind: 'subtitles',
            src: process.env.PUBLIC_URL + '/' + response.data.en_subs,
            srcLang: 'en',
            label: 'English',
            mode: 'showing'
          }
        ]);
      }
    }).catch((error) => {
      console.error('Error:', error.response);
    }).finally(() => {setLoading(false);});
  };

  return (
    <div>
    <button onClick={() => {navigate('/login');}} className='logout-button'>log out</button>
    <button onClick={() => {navigate('/home');}} className='back-button'>home</button>
    <Stack spacing={2} className='container'>
      <div>
        <h1>
          <TypeAnimation
            sequence={['subtitle.ai']}
            wrapper='span'
            speed={5}
            style={{fontSize: '45px', display: 'inline-block'}}
            cursor={false}
          />
        </h1>
        {isLoading
        ? <TypeAnimation
            style={{'font-weight': '400', 'font-size': '30px'}}
            sequence={['Working some magic...', 1000, '', 500]}
            wrapper='span'
            speed={5}
            repeat={Infinity}
          />
        : <h2 style={{'font-weight': '400'}}>Generate smart subtitles using the power of machine learning.</h2>
        }
      </div>
      {!results && <form onSubmit={handleSubmit}>
        {!isLoading && 
          <div>
            <input type='file' onChange={handleChange} accept='video/*' />
            <button className='button' type='submit' disabled={!videoURL}>
              Upload and Process
            </button>
          </div>
        }
      </form>}
      {results &&
        <Stack spacing={5}>
          <Stack direction='row' spacing={10} className='center'>
            {results.ru_subs &&
              <a href={process.env.PUBLIC_URL + '/' + results.ru_subs}
                download={results.ru_subs}
                target='_blank'>
                <div className='download-button'>
                    <TiDownload style={{'margin-right':'2%'}}/>
                    Download Russian subtitles
                </div>
              </a>
            }
            {results.en_subs &&
              <a href={process.env.PUBLIC_URL + '/' + results.en_subs}
                  download={results.en_subs}
                  target='_blank'>
                <div className='download-button'>
                    <TiDownload style={{'margin-right':'2%'}}/>
                    Download English subtitles
                </div>
              </a>
            }
          </Stack>
          <ReactPlayer
            className='react-player fixed-bottom'
            url={videoURL}
            width='100%'
            height='100%'
            controls={true}
            config={{
              file: {
                attributes: {
                  crossOrigin: 'true',
                },
                tracks: subtitles
              },
            }}
          />
          {results.align && results.align[currSub].map(([ruWord, enWord]) =>
            <div className='subs'>
            </div>
          )}
        </Stack>
      }
    </Stack>
    </div>
  );
}