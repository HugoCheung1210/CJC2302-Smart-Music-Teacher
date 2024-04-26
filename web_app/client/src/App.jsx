import React from 'react';
import Homepage from './components/Homepage';
import PieceOverview from './components/PieceOverview';
import RecordingOverview from './components/RecordingOverview';
import Playback from './components/Playback';
import EmotionAnalysis from './components/EmotionAnalysis';
import StyleTransfer from './components/StyleTransfer';
import { BrowserRouter as Router, Route, Routes } from 'react-router-dom';

function App() {
  return (
    <Router>
      <Routes>
        <Route path="/" element={<Homepage />} />

        <Route path="/pieces/:pieceId" element={<PieceOverview />}/>

        <Route path="/emotion" element={<EmotionAnalysis />} />

        <Route path="/style" element={<StyleTransfer />} />

        <Route path="/recordings/:recordingId" element={<RecordingOverview />} />

        <Route path="/playback/:recordingId" element={<Playback />} />

        <Route path="*" element={<div>404 Not Found</div>} />
      </Routes>      
    </Router>
  );
}

export default App;
