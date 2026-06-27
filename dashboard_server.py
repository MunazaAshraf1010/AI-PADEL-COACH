"""
Dashboard Server
Serves the match analytics web dashboard and provides the report data via API.
"""

import http.server
import socketserver
import json
import os
import sys
import webbrowser
from pathlib import Path
from urllib.parse import urlparse
import threading


DASHBOARD_DIR = Path(__file__).resolve().parent / "dashboard"
REPORT_FILE = "match_report.json"
PORT = 8080


class DashboardHandler(http.server.SimpleHTTPRequestHandler):
    """Custom HTTP handler that serves dashboard files and report API"""

    def __init__(self, *args, report_path=None, dashboard_dir=None, **kwargs):
        self.report_path = report_path
        self._dashboard_dir = dashboard_dir or DASHBOARD_DIR
        super().__init__(*args, directory=str(self._dashboard_dir), **kwargs)

    def do_GET(self):
        parsed = urlparse(self.path)

        # API endpoint for report data
        if parsed.path == '/api/report':
            self._serve_report()
            return

        # Serve match_report.json from project root
        if parsed.path == '/match_report.json':
            self._serve_report()
            return

        # Default: serve static files from dashboard/
        super().do_GET()

    def _serve_report(self):
        """Serve the match report JSON"""
        report_path = self.report_path or Path.cwd() / REPORT_FILE

        if not report_path.exists():
            self.send_error(404, "Report not found. Run the analysis first.")
            return

        try:
            with open(report_path, 'r') as f:
                report_data = f.read()

            self.send_response(200)
            self.send_header('Content-Type', 'application/json')
            self.send_header('Access-Control-Allow-Origin', '*')
            self.send_header('Content-Length', len(report_data.encode()))
            self.end_headers()
            self.wfile.write(report_data.encode())
        except Exception as e:
            self.send_error(500, f"Error reading report: {str(e)}")

    def log_message(self, format, *args):
        """Suppress default logging, use custom format"""
        print(f"  [dashboard] {args[0]}")


def create_handler(report_path):
    """Factory to create handler with report_path bound"""
    dashboard_dir = DASHBOARD_DIR
    def handler(*args, **kwargs):
        return DashboardHandler(*args, report_path=report_path, dashboard_dir=dashboard_dir, **kwargs)
    return handler


def start_dashboard(
    report_path: str = None, 
    port: int = 8080, 
    open_browser: bool = True,
    background: bool = False,
):
    """
    Start the dashboard web server.
    
    Args:
        report_path: Path to the match_report.json file
        port: HTTP port to serve on (default 8080)
        open_browser: Whether to automatically open the browser
        background: Whether to run in a background thread
    """
    if report_path:
        report_path = Path(report_path)
    else:
        report_path = Path.cwd() / REPORT_FILE

    print(f"\n{'='*60}")
    print(f"🏓 AI Padel Coach - Match Dashboard")
    print(f"{'='*60}")
    print(f"  Report: {report_path}")
    print(f"  Dashboard: http://localhost:{port}")
    print(f"{'='*60}\n")

    handler = create_handler(report_path)

    if background:
        server = socketserver.TCPServer(("", port), handler)
        thread = threading.Thread(target=server.serve_forever, daemon=True)
        thread.start()
        if open_browser:
            webbrowser.open(f"http://localhost:{port}")
        return server
    else:
        with socketserver.TCPServer(("", port), handler) as httpd:
            if open_browser:
                # Open browser after a small delay
                def _open():
                    import time
                    time.sleep(0.5)
                    webbrowser.open(f"http://localhost:{port}")
                threading.Thread(target=_open, daemon=True).start()

            print(f"  Serving at http://localhost:{port}")
            print(f"  Press Ctrl+C to stop\n")
            try:
                httpd.serve_forever()
            except KeyboardInterrupt:
                print("\n  Dashboard server stopped.")
                httpd.shutdown()


if __name__ == "__main__":
    # Allow passing report path as argument
    report = sys.argv[1] if len(sys.argv) > 1 else None
    port = int(sys.argv[2]) if len(sys.argv) > 2 else PORT
    start_dashboard(report_path=report, port=port, open_browser=True)
