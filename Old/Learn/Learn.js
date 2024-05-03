import React, { useState } from 'react';
import ReactPlayer from 'react-player';
import { useNavigate } from 'react-router-dom';
import { Stack } from '@mui/material';
import { TypeAnimation } from 'react-type-animation';
import { FaArrowRightLong } from "react-icons/fa6";
import axios from 'axios';
import './Learn.css';

export default function Learn() {
  const navigate = useNavigate();
  const [file, setFile] = useState();
  const [videoURL, setVideoURL] = useState();
  const [subtitles, setSubtitles] = useState([]);
  const [enabledLang, setEnabledLang] = useState();
  const [stage, setStage] = useState(0);
  const [videoEnded, setVideoEnded] = useState(false);
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
            mode: 'showing',
            default: true
          }
        ]);
      }
      setStage(stage + 1);
    }).catch((error) => {
      console.error('Error:', error.response);
    }).finally(() => {setLoading(false);});
  };

  const titles = [
    <h1>
      <TypeAnimation
        sequence={['Word Matching', 1000, () => {setStage(stage + 1);}]}
        wrapper='span'
        speed={5}
        style={{fontSize: '45px', display: 'inline-block'}}
        cursor={false}
      />
    </h1>,
    <h2 className='title'>To begin, upload a video in Russian.</h2>,
    <div>
      <h2 className='title'>Let's watch with Russian subtitles.</h2>
      <button onClick={() => {setEnabledLang('ru'); setStage(stage + 1);}} className='next-button'>
        <FaArrowRightLong />
      </button>
    </div>,
    <></>,
    <div>
      <h2 className='title'>Now let's watch with English subtitles.</h2>
      <button onClick={() => {
        setEnabledLang('en');
        setVideoEnded(false);
        setStage(stage + 1);
        }} className='next-button'>
        <FaArrowRightLong />
      </button>
    </div>,
    <></>,
    <div>
      <h2 className='title'>Let's review the word alignment.</h2>
      <button onClick={() => {setStage(stage + 1);}} className='next-button'>
        <FaArrowRightLong />
      </button>
    </div>,
    <h2 className='title'>Word Alignment</h2>
  ]

  return (
    <div>
    <button onClick={() => {navigate('/login');}} className='logout-button'>log out</button>
    <button onClick={() => {navigate('/home');}} className='back-button'>home</button>
    <Stack spacing={5} className='container'>
      <div>
        {isLoading
        ? <TypeAnimation
            style={{'font-weight': '400', 'font-size': '30px'}}
            sequence={['Working some magic...', 1000, '']}
            wrapper='span'
            speed={5}
            repeat={Infinity}
          />
        : titles[stage]
        }
      </div>
      {stage == 1 && 
        <form onSubmit={handleSubmit}>
          {!isLoading && 
            <div>
              <input type='file' onChange={handleChange} accept='video/*' />
              <button className='button' type='submit' disabled={!videoURL}>
                Upload and Process
              </button>
            </div>
          }
        </form>
      }
      {(stage == 3 || stage == 5) && results &&
        <div>
          <ReactPlayer
            className='react-player fixed-bottom'
            url={videoURL}
            width='100%'
            height='100%'
            onEnded={() => setVideoEnded(true)}
            controls={true}
            config={{
              file: {
                attributes: {
                  crossOrigin: 'true',
                },
                tracks: enabledLang == 'ru' ? [subtitles[0]] : [subtitles[1]]
              },
            }}
          />
          <button onClick={() => {setStage(stage + 1);}} className='next-button' disabled={!videoEnded}>
            <FaArrowRightLong />
          </button>
        </div>
      }
      {stage == 7 && results && results.align.map((segment) =>
        <div>{segment.map(([w1, w2]) => `(${w1}, ${w2})`)}</div>
      )}
    </Stack>
    </div>
  );
}