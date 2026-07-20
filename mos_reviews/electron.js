import { app, BrowserWindow, nativeImage, ipcMain } from 'electron';
import { spawn } from 'child_process';
import path from 'path';
import { fileURLToPath } from 'url';
import net from 'net';
import fs from 'fs';

const __filename = fileURLToPath(import.meta.url);
const __dirname = path.dirname(__filename);

// Subdirectory in Library for backups
app.name = 'MOS Reviews';
const DATA_DIR = path.join(app.getPath('home'), 'Library', 'Application Support', 'Out of This World', 'MOS Reviews');

if (!fs.existsSync(DATA_DIR)) {
  fs.mkdirSync(DATA_DIR, { recursive: true });
}

let serverProcess = null;
let mainWindow = null;
const spawnEnv = { ...process.env };

// Spawn backend server
function startServer() {
  const isDev = fs.existsSync(path.join(__dirname, 'server.ts'));
  if (isDev) {
    serverProcess = spawn('npx', ['tsx', 'server.ts'], { cwd: __dirname, stdio: 'inherit', shell: true, env: spawnEnv });
  } else {
    serverProcess = spawn('node', ['dist/server.cjs'], { cwd: __dirname, stdio: 'inherit', env: spawnEnv });
  }
}

// Wait for port to be ready and load URL
function checkPortAndLoad(port) {
  const client = new net.Socket();
  client.once('connect', () => {
    client.end();
    if (mainWindow) mainWindow.loadURL(`http://localhost:${port}`);
  });
  client.once('error', () => setTimeout(() => checkPortAndLoad(port), 200));
  client.connect({ port });
}

function createWindow() {
  const iconPath = path.join(__dirname, 'public', 'icon-512.png');
  const hasIcon = fs.existsSync(iconPath);
  
  // Set Dock icon programmatically for macOS
  if (process.platform === 'darwin' && hasIcon) {
    try {
      app.dock.setIcon(nativeImage.createFromPath(iconPath));
    } catch (e) {
      console.error(e);
    }
  }

  // Read app version from package.json if exists
  let version = '0.0.0';
  try {
    const pkg = JSON.parse(fs.readFileSync(path.join(__dirname, 'package.json'), 'utf8'));
    version = pkg.version || '0.0.0';
  } catch (e) {
    console.error('Failed to read package.json version:', e);
  }

  mainWindow = new BrowserWindow({
    width: 1400,
    height: 900,
    icon: hasIcon ? iconPath : undefined,
    title: `MOS Reviews: version ${version}`,
    webPreferences: {
      nodeIntegration: false,
      contextIsolation: true,
      sandbox: false, // Allow preload script to use Node require
      preload: path.join(__dirname, 'preload.js')
    }
  });

  // Ensure child windows inherit the preload settings
  mainWindow.webContents.setWindowOpenHandler(() => {
    return {
      action: 'allow',
      overrideBrowserWindowOptions: {
        webPreferences: {
          preload: path.join(__dirname, 'preload.js'),
          contextIsolation: true,
          nodeIntegration: false,
          sandbox: false
        }
      }
    };
  });

  checkPortAndLoad(5187); // Port of MOS Reviews
  mainWindow.on('closed', () => { mainWindow = null; });
}

app.whenReady().then(() => {
  // Handle save-file IPC call from renderer
  ipcMain.handle('save-file', async (event, { filename, content }) => {
    try {
      const filePath = path.join(DATA_DIR, filename);
      fs.writeFileSync(filePath, content, 'utf-8');
      console.log(`Saved local file to: ${filePath}`);
      return { success: true, path: filePath };
    } catch (error) {
      console.error('Failed to save file:', error);
      return { success: false, error: error.message };
    }
  });

  // Handle load-backup IPC call from renderer
  ipcMain.handle('load-backup', async (event, { filename }) => {
    try {
      const filePath = path.join(DATA_DIR, filename);
      if (!fs.existsSync(filePath)) {
        return { success: false, error: 'File does not exist' };
      }
      const content = fs.readFileSync(filePath, 'utf-8');
      return { success: true, content };
    } catch (error) {
      console.error('Failed to load file:', error);
      return { success: false, error: error.message };
    }
  });

  startServer();
  createWindow();
});

app.on('window-all-closed', () => {
  app.quit();
});

app.on('will-quit', () => {
  if (serverProcess) serverProcess.kill('SIGTERM');
});
