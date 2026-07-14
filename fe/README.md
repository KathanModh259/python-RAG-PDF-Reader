# Nyaya Mitra Frontend Desktop App UI

This is the user interface for **Nyaya Mitra**, built with React, TypeScript, Vite, TailwindCSS v4, and Tauri. It operates as a local desktop application wrapping the web frontend.

---

## Features

- **Tauri Integration**: Leverages Tauri for a lightweight desktop container that runs on Windows, macOS, and Linux.
- **TailwindCSS v4**: Uses the latest TailwindCSS styling engine for maximum utility-first efficiency and modern aesthetics.
- **Interactive Screens**:
  - **Chat Interface**: Speak or type queries directly to the local model.
  - **Document Library**: Drag-and-drop or browse documents to upload and index into the local vector database.
  - **Guided Flows**: Custom wizard workflows for exploring common legal scenarios.
  - **Panic Mode**: Quick action dashboard displaying local legal aid helplines (NALSA/DLSA) and direct assistance info.
  - **Anti-Exploitation Scanner**: Visual indicators for contract scams and unfair legal fees.
- **Multilingual Localization**: Integrated system configuration for regional languages.
- **Local Voice Actions**: Speech-to-text input and text-to-speech feedback buttons.

---

## Installation & Development

### Prerequisites

- Node.js (v18+)
- npm
- Rust & Tauri system prerequisites (e.g. C++ Build Tools on Windows)

### Dev Setup

1. **Install Node Dependencies**:
   ```bash
   npm install
   ```

2. **Run Dev Server (Web Mode)**:
   Runs the application in the web browser at `http://localhost:5173`:
   ```bash
   npm run dev
   ```

3. **Run Dev Server (Tauri Desktop Mode)**:
   Runs the application inside the Tauri desktop window:
   ```bash
   npm run tauri dev
   ```

---

## Production Build

To build the standalone executable package:

```bash
npm run tauri build
```

This compiles both the React frontend and the Rust backend, generating a native desktop installer under `src-tauri/target/release/bundle/`.

