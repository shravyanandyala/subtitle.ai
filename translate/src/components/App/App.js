import React, { useState } from 'react';
import { Routes, Route, Navigate } from 'react-router-dom';

import './App.css';
import Subtitle from '../Subtitle/Subtitle';
import Login from '../Login/Login';

function App() {
  const [token, setToken] = useState();

  return (
      <div className="wrapper">
        <h1>subtitle.ai</h1>
        <Routes>
          <Route 
            path="/subtitle" 
            element={token ? <Subtitle /> : <Navigate to="/login" />} 
          />
          {/* Redirect all other paths to login/dashboard depending on authentication */}
          <Route 
            path="*" 
            element={token ? <Navigate to="/subtitle" /> : <Login setToken={setToken} />} 
          />
        </Routes>
      </div>
  );
}

export default App;