use std::sync::Mutex;
use std::process::{Child, Command as SysCommand};
use std::env;
use tauri::Manager;

struct PythonBackend {
    child: Mutex<Option<Child>>,
}

impl Drop for PythonBackend {
    fn drop(&mut self) {
        if let Ok(mut guard) = self.child.lock() {
            if let Some(ref mut child) = *guard {
                let _ = child.kill();
                let _ = child.wait();
            }
        }
    }
}

fn find_python() -> String {
    // Try common Python executable names
    for name in &["py -3.11", "python3", "python", "py"] {
        let parts: Vec<&str> = name.split_whitespace().collect();
        if parts.is_empty() { continue; }
        if let Ok(_) = SysCommand::new(parts[0]).arg("--version").output() {
            return name.to_string();
        }
    }
    "python".to_string()
}

#[tauri::command]
fn get_api_url() -> String {
    "http://127.0.0.1:8765".to_string()
}

#[cfg_attr(mobile, tauri::mobile_entry_point)]
pub fn run() {
    tauri::Builder::default()
        .setup(|app| {
            if cfg!(debug_assertions) {
                app.handle().plugin(
                    tauri_plugin_log::Builder::default()
                        .level(log::LevelFilter::Info)
                        .build(),
                )?;
            }

            // Resolve project root (where pyproject.toml lives)
            let root = app.path().resource_dir()
                .unwrap_or_else(|_| env::current_dir().unwrap())
                .parent()
                .map(|p| p.to_path_buf())
                .unwrap_or_else(|| env::current_dir().unwrap());

            let python = find_python();
            log::info!("Starting Python backend with: {} from {:?}", python, root);

            // Start the FastAPI server
            let child = SysCommand::new("py")
                .args(["-3.11", "-m", "poetry", "run", "python", "app/main.py", "--mode", "api"])
                .current_dir(&root)
                .spawn();

            match child {
                Ok(c) => {
                    log::info!("Python backend started, PID: {}", c.id());
                    app.manage(PythonBackend {
                        child: Mutex::new(Some(c)),
                    });
                }
                Err(e) => {
                    log::error!("Failed to start Python backend: {}", e);
                    app.manage(PythonBackend {
                        child: Mutex::new(None),
                    });
                }
            }

            Ok(())
        })
        .invoke_handler(tauri::generate_handler![get_api_url])
        .run(tauri::generate_context!())
        .expect("error while running tauri application");
}
