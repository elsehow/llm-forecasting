#!/usr/bin/env python3
"""Simple dev server for scenario explorer with file listing API."""

import json
from http.server import HTTPServer, SimpleHTTPRequestHandler
from pathlib import Path
import os

# Serve from scenario-construction directory
ROOT = Path(__file__).parent.parent
os.chdir(ROOT)


class Handler(SimpleHTTPRequestHandler):
    def do_GET(self):
        if self.path == '/api/files':
            self.send_response(200)
            self.send_header('Content-Type', 'application/json')
            self.send_header('Cache-Control', 'no-cache')
            self.end_headers()

            files = []
            seen = set()  # Track (target, approach) to dedupe
            results_dir = Path('results')
            if results_dir.exists():
                # Sorted reverse = newest first (timestamps in filename)
                for json_file in sorted(results_dir.glob('**/*.json'), reverse=True):
                    rel_path = str(json_file)
                    try:
                        with open(json_file) as f:
                            data = json.load(f)
                            # Must have scenarios and signals to be a scenario set
                            if 'scenarios' not in data or 'signals' not in data:
                                continue
                            label = data.get('name', json_file.stem)
                            target = data.get('target', 'unknown')
                            approach = data.get('approach', 'unknown')
                    except:
                        continue

                    # Only keep newest version of each (target, approach)
                    key = (target, approach)
                    if key in seen:
                        continue
                    seen.add(key)
                    files.append({
                        'path': rel_path,
                        'label': label,
                        'target': target,
                        'approach': approach,
                    })

            self.wfile.write(json.dumps(files).encode())
            return

        super().do_GET()

    def log_message(self, format, *args):
        # Quieter logging - only show errors
        if '404' in str(args) or '500' in str(args):
            super().log_message(format, *args)


if __name__ == '__main__':
    port = 8000
    print(f"Serving at http://localhost:{port}")
    print(f"Open http://localhost:{port}/web/index.html")
    HTTPServer(('', port), Handler).serve_forever()
