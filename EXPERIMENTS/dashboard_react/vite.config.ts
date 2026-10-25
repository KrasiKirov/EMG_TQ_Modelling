import { defineConfig } from 'vite';
import react from '@vitejs/plugin-react';
import { viteSingleFile } from 'vite-plugin-singlefile';

// One self-contained dist/index.html (JS, CSS and the results data inlined) that opens from file:// with no server.
export default defineConfig({
  base: './',
  plugins: [react(), viteSingleFile({ removeViteModuleLoader: true })],
  json: { stringify: true },
  build: {
    target: 'es2020',
    chunkSizeWarningLimit: 10000,
    reportCompressedSize: false,
  },
});
