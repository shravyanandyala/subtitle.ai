import React, { useState } from 'react';
import { useNavigate } from 'react-router-dom';
import { Stack } from '@mui/material';
import { TypeAnimation } from 'react-type-animation';
import './Home.css';

export default function Home() {
  const navigate = useNavigate(); 
  const [showButtons, setShowButtons] = useState(false);
  
  return (
    <div>
      <button onClick={() => {navigate('/login');}} className='logout-button'>log out</button>
      <div className='centered'>
        <Stack spacing={10}>
          <h1>
            <TypeAnimation
              sequence={[
                'Language Tools',
                500, /* Wait 500 ms before showing buttons */
                () => { setShowButtons(true); }
              ]}
              speed={5}
              style={{fontSize: '1.5em', display: 'inline-block'}}
            />
          </h1>
          {showButtons &&
            <div className='options'>
              <Stack direction="row" spacing={10}>
                <button className='button2' onClick={() => navigate('/subtitle')}>I want to subtitle</button>
                <button className='button2' onClick={() => navigate('/learn')}>I want to learn</button>
              </Stack>
            </div>
          }
        </Stack>
      </div>
    </div>
  );
}