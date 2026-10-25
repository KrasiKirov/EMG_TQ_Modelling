import '@mantine/core/styles.css';
import './styles.css';
import React from 'react';
import ReactDOM from 'react-dom/client';
import { MantineProvider } from '@mantine/core';
import { schemeManager, theme } from './theme';
import { App } from './App';

const manager = schemeManager();

ReactDOM.createRoot(document.getElementById('root')!).render(
  <React.StrictMode>
    <MantineProvider theme={theme} defaultColorScheme="auto" colorSchemeManager={manager}>
      <App />
    </MantineProvider>
  </React.StrictMode>,
);
