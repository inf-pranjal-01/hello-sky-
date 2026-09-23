import React from 'react';
import { AppRouter } from './router/AppRouter';
import { AppErrorBoundary } from './components/common/AppErrorBoundary';
import './styles/global.css';

export const App: React.FC = () => {
  return <AppErrorBoundary><AppRouter /></AppErrorBoundary>;
};

export default App;
