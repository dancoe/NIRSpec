const { contextBridge, ipcRenderer } = require('electron');

contextBridge.exposeInMainWorld('electronAPI', {
  saveFile: (filename, content) => ipcRenderer.invoke('save-file', { filename, content }),
  loadBackup: (filename) => ipcRenderer.invoke('load-backup', { filename }),
});
