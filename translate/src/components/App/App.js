import React, { useState } from 'react';
import { Routes, Route, Navigate, useNavigate } from 'react-router-dom';
import Login from '../Login/Login';
import Home from '../Home/Home';
import Subtitle from '../Subtitle/Subtitle';
import './App.css';

export default function App() {
  const [token, setToken] = useState();
  const navigate = useNavigate();

  const login = (credentials) => {
    setToken(credentials);
    navigate('/home');
  };

  return (
    <div className='wrapper'>
      <Routes>
        <Route
          path='/login'
          element={<Login login={login} />}
        />
        <Route
          path='/subtitle'
          element={token ? <Subtitle /> : <Navigate to='/login' />}
        />
        {/* Redirect all other paths to login/dashboard depending on authentication */}
        <Route
          path='*'
          element={token ? <Home /> : <Navigate to='/login' />}
        />
      </Routes>
    </div>
  );
}