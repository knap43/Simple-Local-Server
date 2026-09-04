#!/usr/bin/env python3
"""
File server backend for the mobile/controller-friendly file-explorer
frontend (file-explorer-remote.html). This is a standalone copy of
server.py's original companion, pointed at the alternate frontend file
instead — the API surface is otherwise identical, so both frontends can
talk to either backend if you ever want to mix and match.

Serves the following from a single Flask process:
  - the frontend itself, at /
  - GET  /api/list?path=<rel>      -> JSON listing of one directory
  - GET  /api/read?path=<rel>      -> raw text of a small text file (.txt/.md)
  - POST /api/write                -> save edited .txt/.md content
                                       (JSON body: {"path": ..., "content": ...})
  - GET  /api/thumbnail?path=<rel> -> a resized JPEG for an image file
  - GET  /api/image?path=<rel>     -> full-resolution image, for the viewer
  - GET  /api/stream?path=<rel>    -> range-request-capable file stream,
                                       for the built-in audio/video player
  - GET  /api/download?path=<rel>  -> the file itself, or a folder zipped
                                       on the fly, as an attachment
  - POST /api/upload                -> form fields: "path" (target dir) and
                                       one or more "files"; a folder upload's
                                       relative paths (webkitRelativePath)
                                       are preserved and recreated on disk

`path` is always a path *relative to ROOT_DIR*, using forward slashes,
e.g. "Photos/Trip 2024". Every request is resolved against ROOT_DIR and
checked to make sure it didn't escape it (symlink tricks and "../"
included) before anything touches the filesystem.

Binds to 0.0.0.0 (all interfaces) by default, so it's reachable from other
devices on your LAN as soon as you start it — no --host flag needed. That
also means anyone who can reach this machine on the network can browse,
download, upload to, and edit files in the served directory: there is no
authentication. This is meant for a trusted home LAN behind your own router,
not the open internet. For remote access, use a reverse proxy with auth
(Caddy/nginx + basic auth) or a Tailscale/WireGuard tunnel rather than
forwarding a port to it. Pass --host 127.0.0.1 to restrict it to this
machine only.

Install (Arch, via pacman — matches what's already in the repos):
    sudo pacman -S python-flask python-pillow

Or via pip in a virtualenv (Arch's system Python is externally managed,
so plain `pip install` will refuse to run outside one):
    python -m venv venv && source venv/bin/activate
    pip install -r requirements.txt

Run:
    python server.py /path/to/share
    python server.py /path/to/share --port 8000
    python server.py /path/to/share --host 127.0.0.1   # this machine only
"""

from __future__ import annotations

import argparse
import hashlib
import mimetypes
import os
import tempfile
import threading
import time
import zipfile
from datetime import datetime
from pathlib import Path
from urllib.parse import quote

from flask import Flask, Response, abort, jsonify, request, send_file, stream_with_context

try:
    from PIL import Image
except ImportError:
    Image = None  # thumbnails will fall back to serving the original file

# ---------------------------------------------------------------------------
# Config (set in __main__ before app.run(); see configure())
# ---------------------------------------------------------------------------
ROOT_DIR: Path | None = None
THUMB_DIR: Path | None = None
SIZE_CACHE_TTL = float(os.environ.get("FILESERVER_SIZE_CACHE_TTL", "60"))
THUMB_MAX_SIZE = (320, 320)
MAX_TEXT_BYTES = 2 * 1024 * 1024  # don't render anything bigger than 2 MB inline

IMAGE_EXT = {".jpg", ".jpeg", ".png", ".gif", ".webp"}
AUDIO_EXT = {".mp3", ".flac", ".wav", ".ogg", ".m4a"}
VIDEO_EXT = {".mp4", ".webm", ".mov", ".mkv", ".avi"}
TEXT_EXT = {".md", ".txt"}
MIME_OVERRIDES = {
    ".flac": "audio/flac",
    ".mkv": "video/x-matroska",
    ".webm": "video/webm",
    ".m4a": "audio/mp4",
}

app = Flask(__name__, static_folder=None)

_size_cache: dict[str, tuple[int, float]] = {}
_size_cache_lock = threading.Lock()


def configure(root: str | os.PathLike) -> None:
    """Set the directory this server exposes. Must be called before app.run()."""
    global ROOT_DIR, THUMB_DIR
    ROOT_DIR = Path(root).resolve()
    if not ROOT_DIR.is_dir():
        raise SystemExit(f"Not a directory: {ROOT_DIR}")
    THUMB_DIR = ROOT_DIR / ".thumbnails"
    THUMB_DIR.mkdir(exist_ok=True)


# ---------------------------------------------------------------------------
# Path safety
# ---------------------------------------------------------------------------
def safe_path(rel: str) -> Path:
    """Resolve a client-supplied relative path and confirm it stays inside ROOT_DIR."""
    rel = (rel or "").strip("/")
    candidate = (ROOT_DIR / rel).resolve()
    try:
        candidate.relative_to(ROOT_DIR)
    except ValueError:
        abort(403, "Path escapes the served directory")
    return candidate


def is_visible(name: str) -> bool:
    """Hidden files/dirs (dotfiles, and our own thumbnail cache) are omitted from
    listings — same convention as GNOME Files, Finder, and Explorer's default view."""
    return not name.startswith(".")


def visible_children(path: Path) -> list[os.DirEntry]:
    try:
        return sorted(
            (e for e in os.scandir(path) if is_visible(e.name)),
            key=lambda e: e.name.lower(),
        )
    except OSError:
        return []


def ext_of(name: str) -> str:
    return Path(name).suffix.lower()


# ---------------------------------------------------------------------------
# Recursive, cached directory sizing
# ---------------------------------------------------------------------------
def get_dir_size(path: Path, _visited: set[Path] | None = None) -> int:
    """Sum of all file sizes under `path`, recursively.

    Cached per-directory with a short TTL (FILESERVER_SIZE_CACHE_TTL, default
    60s) — walking a large tree on every request is too slow for anything
    but a small share. `_visited` guards against symlink cycles, which show
    up more often than you'd expect on a NAS with a media-library symlink farm.
    """
    if _visited is None:
        _visited = set()
    try:
        real = path.resolve()
    except OSError:
        return 0
    if real in _visited:
        return 0
    _visited.add(real)

    key = str(real)
    now = time.time()
    with _size_cache_lock:
        cached = _size_cache.get(key)
        if cached and cached[1] > now:
            return cached[0]

    total = 0
    for entry in visible_children(path):
        try:
            if entry.is_dir():
                total += get_dir_size(Path(entry.path), _visited)
            elif entry.is_file():
                total += entry.stat().st_size
        except OSError:
            continue  # permission errors, broken symlinks, races — just skip

    with _size_cache_lock:
        _size_cache[key] = (total, now + SIZE_CACHE_TTL)
    return total


def invalidate_size_cache(path: Path) -> None:
    """Drop cached sizes for `path` and every ancestor up to ROOT_DIR, so an
    upload is reflected immediately instead of waiting out the TTL."""
    try:
        p = path.resolve()
    except OSError:
        return
    with _size_cache_lock:
        while True:
            _size_cache.pop(str(p), None)
            if p == ROOT_DIR or p.parent == p:
                break
            p = p.parent


def lan_addresses() -> list[str]:
    """Best-effort list of this machine's LAN IPs, just for a friendlier
    startup message when bound to 0.0.0.0 — not used for anything functional."""
    import socket
    addrs = set()
    try:
        hostname = socket.gethostname()
        for info in socket.getaddrinfo(hostname, None):
            ip = info[4][0]
            if ":" not in ip and not ip.startswith("127."):
                addrs.add(ip)
    except OSError:
        pass
    return sorted(addrs)


# ---------------------------------------------------------------------------
# Routes
# ---------------------------------------------------------------------------
@app.route("/")
def index():
    frontend = Path(__file__).parent / "file-explorer-remote.html"
    if not frontend.exists():
        return (
            "file-explorer-remote.html not found next to server-remote.py. "
            "Place the frontend file alongside this script.",
            500,
        )
    return send_file(frontend)


@app.route("/api/list")
def api_list():
    target = safe_path(request.args.get("path", ""))
    if not target.is_dir():
        abort(404, "Not a directory")

    items = []
    for entry in visible_children(target):
        entry_path = Path(entry.path)
        rel = entry_path.relative_to(ROOT_DIR).as_posix()
        try:
            stat = entry.stat()
        except OSError:
            continue
        modified = datetime.fromtimestamp(stat.st_mtime).isoformat()

        if entry.is_dir():
            items.append({
                "name": entry.name,
                "type": "dir",
                "size": get_dir_size(entry_path),
                "modified": modified,
                "isImage": False,
                "thumb": top_level_thumb(entry_path),
                "url": None,
                "downloadUrl": download_url(rel),
                "itemCount": len(visible_children(entry_path)),
            })
        else:
            is_image = ext_of(entry.name) in IMAGE_EXT
            is_media = ext_of(entry.name) in AUDIO_EXT | VIDEO_EXT
            items.append({
                "name": entry.name,
                "type": "file",
                "size": stat.st_size,
                "modified": modified,
                "isImage": is_image,
                "thumb": thumb_url(rel) if is_image else None,
                "url": image_url(rel) if is_image else (stream_url(rel) if is_media else None),
                "downloadUrl": download_url(rel),
            })
    return jsonify(items)


def top_level_thumb(dir_path: Path) -> str | None:
    for entry in visible_children(dir_path):
        if entry.is_file() and ext_of(entry.name) in IMAGE_EXT:
            rel = Path(entry.path).relative_to(ROOT_DIR).as_posix()
            return thumb_url(rel)
    return None


def thumb_url(rel: str) -> str:
    return f"/api/thumbnail?path={quote(rel)}"


def stream_url(rel: str) -> str:
    return f"/api/stream?path={quote(rel)}"


def image_url(rel: str) -> str:
    return f"/api/image?path={quote(rel)}"


def download_url(rel: str) -> str:
    return f"/api/download?path={quote(rel)}"


@app.route("/api/read")
def api_read():
    target = safe_path(request.args.get("path", ""))
    if not target.is_file():
        abort(404)
    if ext_of(target.name) not in TEXT_EXT:
        abort(400, "Not a readable text file")
    if target.stat().st_size > MAX_TEXT_BYTES:
        abort(400, "File too large to render inline")
    return target.read_text(encoding="utf-8", errors="replace"), 200, {
        "Content-Type": "text/plain; charset=utf-8"
    }


@app.route("/api/write", methods=["POST"])
def api_write():
    """Save .txt/.md content from the built-in editor. Body: {"path": ..., "content": ...}."""
    data = request.get_json(silent=True) or {}
    target = safe_path(data.get("path", ""))
    content = data.get("content", "")

    if ext_of(target.name) not in TEXT_EXT:
        abort(400, "Only .txt and .md files can be edited here")
    if len(content.encode("utf-8")) > MAX_TEXT_BYTES:
        abort(400, "Content too large to save inline")

    target.parent.mkdir(parents=True, exist_ok=True)
    target.write_text(content, encoding="utf-8")
    invalidate_size_cache(target.parent)
    return jsonify({"ok": True})


@app.route("/api/thumbnail")
def api_thumbnail():
    target = safe_path(request.args.get("path", ""))
    if not target.is_file() or ext_of(target.name) not in IMAGE_EXT:
        abort(404)

    if Image is None:
        # Pillow isn't installed — serve the original rather than fail outright.
        return send_file(target, conditional=True)

    cache_name = hashlib.sha1(str(target).encode()).hexdigest() + ".jpg"
    cache_path = THUMB_DIR / cache_name

    if not cache_path.exists() or cache_path.stat().st_mtime < target.stat().st_mtime:
        try:
            with Image.open(target) as im:
                im = im.convert("RGB")
                im.thumbnail(THUMB_MAX_SIZE)
                im.save(cache_path, "JPEG", quality=82)
        except Exception:
            return send_file(target, conditional=True)

    return send_file(cache_path, conditional=True, mimetype="image/jpeg")


@app.route("/api/stream")
def api_stream():
    target = safe_path(request.args.get("path", ""))
    if not target.is_file():
        abort(404)
    ext = ext_of(target.name)
    if ext not in AUDIO_EXT | VIDEO_EXT:
        abort(400, "Not a streamable media file")
    mimetype = MIME_OVERRIDES.get(ext) or mimetypes.guess_type(target.name)[0] or "application/octet-stream"
    # conditional=True makes Werkzeug honor Range headers (206 Partial Content),
    # which is what lets the player seek/scrub instead of restarting playback.
    return send_file(target, conditional=True, mimetype=mimetype)


@app.route("/api/image")
def api_image():
    """Full-resolution image, served inline (not as an attachment) for the
    built-in viewer — distinct from /api/thumbnail, which is capped at
    THUMB_MAX_SIZE and meant for grid/list icons, not for viewing."""
    target = safe_path(request.args.get("path", ""))
    if not target.is_file() or ext_of(target.name) not in IMAGE_EXT:
        abort(404)
    return send_file(target, conditional=True)


# ---------------------------------------------------------------------------
# Download (files as-is, folders zipped on the fly)
# ---------------------------------------------------------------------------
def content_disposition(filename: str) -> str:
    """RFC 6266 header: an ASCII fallback plus a UTF-8 filename* for
    everything else, since Content-Disposition doesn't allow raw Unicode."""
    ascii_fallback = filename.encode("ascii", "ignore").decode("ascii").strip() or "download"
    return f"attachment; filename=\"{ascii_fallback}\"; filename*=UTF-8''{quote(filename)}"


def zip_directory(dir_path: Path) -> Path:
    """Zip dir_path's visible contents into a temp file and return its path.
    The zip's top-level entry is the folder itself, so extracting it recreates
    the folder rather than dumping its contents loose — the same behavior as
    "compress" in a desktop file manager. Caller is responsible for deleting
    the returned path once it's done streaming it.
    """
    fd, tmp_name = tempfile.mkstemp(suffix=".zip", prefix="fileserver_")
    os.close(fd)
    tmp_path = Path(tmp_name)
    with zipfile.ZipFile(tmp_path, "w", zipfile.ZIP_DEFLATED) as zf:
        for current_dir, dirnames, filenames in os.walk(dir_path):
            dirnames[:] = [d for d in dirnames if is_visible(d)]
            filenames = [f for f in filenames if is_visible(f)]
            current = Path(current_dir)

            if not filenames and not dirnames:
                # An empty leaf directory has no file entry to imply it exists —
                # write it explicitly so extracting the zip recreates it too.
                arcname_dir = current.relative_to(dir_path.parent).as_posix() + "/"
                try:
                    zf.writestr(arcname_dir, "")
                except OSError:
                    pass
                continue

            for fname in filenames:
                file_path = current / fname
                arcname = file_path.relative_to(dir_path.parent).as_posix()
                try:
                    zf.write(file_path, arcname)
                except OSError:
                    continue  # broken symlink, permission error, etc — skip it
    return tmp_path


@app.route("/api/download")
def api_download():
    target = safe_path(request.args.get("path", ""))
    if not target.exists():
        abort(404)

    if target.is_file():
        return send_file(target, as_attachment=True, download_name=target.name, conditional=True)

    # Directory: zip it to a temp file, stream it back, then clean up. Zipping
    # happens synchronously before any bytes go out, so large folders will
    # make the request wait a while before the download starts — acceptable
    # for a home NAS, but worth knowing if you're zipping hundreds of GB.
    tmp_zip = zip_directory(target)
    zip_size = tmp_zip.stat().st_size
    download_name = f"{target.name or 'download'}.zip"

    def generate():
        try:
            with open(tmp_zip, "rb") as f:
                while True:
                    chunk = f.read(1024 * 64)
                    if not chunk:
                        break
                    yield chunk
        finally:
            tmp_zip.unlink(missing_ok=True)

    headers = {
        "Content-Disposition": content_disposition(download_name),
        "Content-Type": "application/zip",
        "Content-Length": str(zip_size),
    }
    return Response(stream_with_context(generate()), headers=headers)


# ---------------------------------------------------------------------------
# Upload
# ---------------------------------------------------------------------------
def unique_path(path: Path) -> Path:
    """If `path` already exists, find a non-colliding sibling name by
    appending " (1)", " (2)", etc. — same convention as most desktop file
    managers. Uploading never silently clobbers an existing file this way."""
    if not path.exists():
        return path
    stem, suffix, parent = path.stem, path.suffix, path.parent
    n = 1
    while True:
        candidate = parent / f"{stem} ({n}){suffix}"
        if not candidate.exists():
            return candidate
        n += 1


@app.route("/api/upload", methods=["POST"])
def api_upload():
    # 'path' is a regular form field (not a query param) since this is a
    # multipart/form-data POST — request.form, not request.args.
    target_dir = safe_path(request.form.get("path", ""))
    if not target_dir.is_dir():
        abort(404, "Target directory does not exist")

    files = request.files.getlist("files")
    if not files:
        abort(400, "No files in upload")

    saved = 0
    for f in files:
        # Folder uploads arrive with the relative path (e.g. "Trip/img.jpg")
        # in place of a plain filename — see webkitRelativePath on the
        # frontend. Rebuild it defensively rather than trusting it outright.
        rel_name = (f.filename or "").replace("\\", "/")
        parts = [p for p in rel_name.split("/") if p not in ("", ".", "..")]
        if not parts:
            continue

        dest = target_dir.joinpath(*parts)
        try:
            dest.resolve().relative_to(ROOT_DIR)
        except ValueError:
            continue  # would have escaped ROOT_DIR — skip it

        dest.parent.mkdir(parents=True, exist_ok=True)
        dest = unique_path(dest)
        f.save(dest)
        saved += 1

    invalidate_size_cache(target_dir)
    return jsonify({"uploaded": saved})


@app.errorhandler(400)
@app.errorhandler(403)
@app.errorhandler(404)
@app.errorhandler(500)
def json_error(err):
    return jsonify({"error": getattr(err, "description", str(err))}), err.code


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument(
        "root", nargs="?", default=os.environ.get("FILESERVER_ROOT", "."),
        help="Directory to serve (default: current directory, or $FILESERVER_ROOT)",
    )
    parser.add_argument("--host", default="0.0.0.0", help="Bind address (default: 0.0.0.0, i.e. reachable on your LAN; use 127.0.0.1 to restrict to this machine)")
    parser.add_argument("--port", type=int, default=8000)
    parser.add_argument("--allow-cors", action="store_true", help="Add permissive CORS headers (only needed if the frontend is served from a different origin/port)")
    args = parser.parse_args()

    configure(args.root)

    if args.allow_cors:
        @app.after_request
        def add_cors_headers(resp):
            resp.headers["Access-Control-Allow-Origin"] = "*"
            return resp

    if Image is None:
        print("Note: Pillow isn't installed — thumbnails will serve full-size originals. "
              "Install it (pacman -S python-pillow, or pip install Pillow) for resized, cached thumbnails.")

    print(f"Serving {ROOT_DIR}")
    print(f"  Local:   http://127.0.0.1:{args.port}")
    if args.host == "0.0.0.0":
        for ip in lan_addresses():
            print(f"  Network: http://{ip}:{args.port}")
    app.run(host=args.host, port=args.port, threaded=True)
