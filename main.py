import argparse, multiprocessing, os, signal, sys, time
import dotenv
dotenv.load_dotenv()

def run_api(host: str, port: int):
    import uvicorn
    from server.app_server import app
    uvicorn.run(app, host=host, port=port, log_level="info")

def run_ui(host: str, port: int, backend_url: str):
    os.environ["BACKEND_URL"] = backend_url
    from user_interface.main_ui import launch_ui
    launch_ui(server_name=host, server_port=port)

def parse_args():
    p = argparse.ArgumentParser(description="Multi-Head AR TTS: API (+ optional Gradio UI)")
    p.add_argument("--api-host", default="0.0.0.0")
    p.add_argument("--api-port", type=int, default=8888)
    p.add_argument("--with-ui", action="store_true", help="Enable Gradio UI in a separate process")
    p.add_argument("--ui-host", default="0.0.0.0")
    p.add_argument("--ui-port", type=int, default=7860)
    p.add_argument("--backend-url", default=None, help="UI -> API base url; default http://<api-host>:<api-port>")
    return p.parse_args()

def main():
    multiprocessing.set_start_method("spawn", force=True)  # 兼容 Windows
    args = parse_args()
    backend_url = args.backend_url or f"http://127.0.0.1:{args.api_port}"

    api_proc = multiprocessing.Process(target=run_api, args=(args.api_host, args.api_port), daemon=False)
    api_proc.start()
    print(f"[main] API started on {args.api_host}:{args.api_port} (pid={api_proc.pid})")

    ui_proc = None
    if args.with_ui:
        ui_proc = multiprocessing.Process(target=run_ui, args=(args.ui_host, args.ui_port, backend_url), daemon=False)
        ui_proc.start()
        print(f"[main] UI  started on {args.ui_host}:{args.ui_port} (pid={ui_proc.pid}) -> backend {backend_url}")

    def shutdown(*_):
        print("[main] Shutting down...")
        for p in (ui_proc, api_proc):
            if p and p.is_alive():
                p.terminate()
        for p in (ui_proc, api_proc):
            if p:
                p.join(timeout=5)
        sys.exit(0)

    signal.signal(signal.SIGINT, shutdown)
    signal.signal(signal.SIGTERM, shutdown)

    # 等待 API 退出（或 Ctrl+C）
    api_proc.join()

if __name__ == "__main__":
    main()
