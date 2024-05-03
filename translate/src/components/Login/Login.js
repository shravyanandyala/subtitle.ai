import { useState } from 'react';
import { GoogleLogin } from '@react-oauth/google';
import { TypeAnimation } from 'react-type-animation';
import { Stack } from '@mui/material';
import PropTypes from 'prop-types';
import './Login.css';

export default function Login({ login }) {
  const [ showLogin, setShowLogin ] = useState(false);

  return (
    <div>
      <div className='centered'>
        <Stack spacing={5}>
          <div className='login-title'>
            <h1>
              <TypeAnimation
                sequence={['Login', 500, () => {setShowLogin(true);}]}
                wrapper='span'
                speed={5}
                cursor={false}
                style={{fontSize: '1.5em', display: 'inline-block'}}
              />
            </h1>
          </div>
          {showLogin &&
            <div className='login-button'>
              <GoogleLogin
                onSuccess={credentialResponse => {
                  console.log('Successfully logged in');
                  login(credentialResponse);
                }}
                onError={() => {
                  console.log('Login Failed');
                }}
                auto_select={true}
                useOneTap
              />
            </div>
          }
        </Stack>
      </div>
      <div className='note'>
        <div>Currently restricted to the Carnegie Mellon University community.</div>
        <br/>
        <div>Made by Shravya Nandyala ❤️</div>
      </div>
    </div>
  );
}

Login.propTypes = {
  login: PropTypes.func.isRequired
};