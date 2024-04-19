import { GoogleLogin } from '@react-oauth/google';
import PropTypes from 'prop-types';
import './Login.css';

export default function Login({ setToken }) {
  return (
    <div className="login-wrapper">
      <GoogleLogin
        onSuccess={credentialResponse => {
          setToken(credentialResponse);
          console.log('Successfully logged in');
        }}
        onError={() => {
          console.log('Login Failed');
        }}
      />
    </div>
  );
}

Login.propTypes = {
  setToken: PropTypes.func.isRequired
};