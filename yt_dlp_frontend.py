#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
yt-dlp Interactive Frontend — detailed behaviour and design notes

This script is a small, interactive front-end around yt-dlp designed for two common
workflows:

1) Fast video pre-download (background): when you paste a URL the program
   immediately launches a lightweight background worker that starts a video
   download with metadata embedding. The background workers are intentionally
   non-blocking: they spawn a yt-dlp subprocess whose stdout/stderr are redirected
   to a temporary log file. The main menu is shown right away so you can pick
   an action without waiting for any network I/O or playlist expansion.

   Characteristics:
     - Background jobs are intended for *video* pre-download only (highest-quality
       MP4 by default) and include embedding of thumbnails/metadata/subtitles.
     - Background logging is kept in a temp file. The main process does not stream
       that output unless you explicitly "attach" to a job.
     - Background workers avoid expensive work during construction (e.g. they
       do blocking detection/expansion inside their own thread).
     - If a background job completes successfully, the worker will (only then)
       attempt to identify the main produced file and perform sidecar cleanup
       (remove thumbnails/subs/descriptions, and — unless the user requested
       to keep .info.json — remove the .info.json after embedding).

2) On-demand foreground operations: the UI gives you "Express" choices and a
   "Full custom" mode. These run yt-dlp in the foreground and stream verbose
   output to your console immediately.

   Foreground runs are used for:
     - Explicit audio extraction flows (extract-audio).
     - Any time the user requests a non-video action (or elects full custom).
     - Foreground runs set the console title while active and restore it afterwards.

Important behavioural invariant (explicitly enforced by this code):
   - Background downloads are *only* for video pre-downloads. If, at the menu,
     you choose to obtain audio (express MP3 or Full → Audio), we immediately
     abandon the corresponding background video worker for that URL: the worker
     is terminated (if running), any partially-created files and sidecars are
     removed, its temporary log is deleted, and the background job is removed
     from the background manager. Audio extraction then proceeds in the
     foreground as a fresh, independent operation. This ensures the audio path
     never races with or tries to reuse an in-progress video download.

Concurrency and safety notes:
   - Background workers use a small lock around process access so the main
     thread can safely terminate and inspect running jobs.
   - When a background job is intentionally aborted by the user (audio chosen),
     the code attempts a graceful terminate + short wait then a hard kill if
     needed. After termination we conservatively identify and remove files that
     were created during the job's life (using predicted filenames and/or
     modification-time heuristics), and sidecars are removed as well.
   - The worker's post-run cleanup is suppressed for aborted jobs, preventing
     duplicate attempts to touch the same files.

User-visible behaviour:
   - Paste a URL (or point to a local file of URLs). Video pre-downloads begin
     immediately in the background and you are shown an express menu.
   - Choosing "Express Video" will attach to background jobs (streaming their
     verbose logs) or, if a background job wasn't started, run a foreground
     video download.
   - Choosing "Express MP3" or selecting "Audio" in Full Custom will abandon
     any existing background video job for that URL, remove its partial data,
     and perform a clean foreground audio download/extraction.
   - Choosing "Full custom → Video" will prefer the background pre-download if
     available and otherwise run foreground video downloads.

This file aims to be defensive: we attempt not to remove any unrelated files
outside the download folder, and removal heuristics are based on predicted names
and file modification timestamps anchored on when the background job started.
"""

from __future__ import annotations
import os
import sys
import json
import shutil
import subprocess
import threading
import time
import re
import tempfile
import uuid
from pathlib import Path
from typing import List, Optional, Dict

# -------------------------
# Basic setup / constants
# -------------------------
def get_base_dir() -> str:
    if getattr(sys, "frozen", False):
        return os.path.dirname(sys.executable)
    return os.path.dirname(os.path.abspath(__file__))


BASE_DIR = get_base_dir()
DEFAULT_OUTPUT = os.path.join(BASE_DIR, "downloads")
os.makedirs(DEFAULT_OUTPUT, exist_ok=True)
DEFAULT_OUTPUT_DIR = str(Path(DEFAULT_OUTPUT))
ERROR_LOG = os.path.join(DEFAULT_OUTPUT, "errors.txt")


def ensure_paths():
    Path(BASE_DIR).mkdir(parents=True, exist_ok=True)
    Path(DEFAULT_OUTPUT_DIR).mkdir(parents=True, exist_ok=True)
    Path(ERROR_LOG).touch(exist_ok=True)


def resource_path(name: str) -> str:
    if hasattr(sys, "_MEIPASS"):
        return os.path.join(sys._MEIPASS, name)
    return os.path.abspath(name)


# -------------------------
# Small helpers: console title (best-effort)
# -------------------------
def set_console_title(title: str):
    try:
        if os.name == "nt":
            # Windows
            os.system(f"title {title}")
        else:
            # Most X terminals (best-effort)
            sys.stdout.write(f"\33]0;{title}\a")
            sys.stdout.flush()
    except Exception:
        pass


# -------------------------
# Input sanitisation
# -------------------------
def normalize_youtube_id_or_url(s: str) -> str:
    s = s.strip()
    if not s:
        return s
    if s.lower().startswith("http"):
        return s
    return f"https://www.youtube.com/watch?v={s}"


def sanitize_url(raw: Optional[str]) -> str:
    if raw is None:
        return ""
    cleaned = re.sub(r'[\u200B-\u200D\uFEFF]', '', raw)
    cleaned = cleaned.strip()
    return normalize_youtube_id_or_url(cleaned)


# -------------------------
# yt-dlp helpers (run in background / worker threads)
# -------------------------
def expand_playlist_to_urls(YTDLP_BIN: str, playlist_url: str) -> List[str]:
    try:
        proc = subprocess.run(
            [YTDLP_BIN, "--flat-playlist", "--dump-single-json", playlist_url],
            check=True, capture_output=True, text=True
        )
        data = json.loads(proc.stdout)
        entries = data.get("entries", []) if isinstance(data, dict) else []
        urls = []
        for e in entries:
            u = e.get("url") or e.get("id") or ""
            if u:
                urls.append(normalize_youtube_id_or_url(u))
        return urls
    except Exception:
        return []


def detect_playlist(YTDLP_BIN: str, candidate: str) -> bool:
    try:
        proc = subprocess.run(
            [YTDLP_BIN, "--flat-playlist", "--dump-single-json", candidate],
            capture_output=True, text=True, timeout=8
        )
        data = json.loads(proc.stdout)
        return isinstance(data, dict) and "entries" in data
    except Exception:
        return False


def get_predicted_filename(YTDLP_BIN: str, url: str, outtmpl: str, ffmpeg_location: Optional[str],
                           extra_path_dirs: Optional[List[str]] = None) -> Optional[str]:
    cmd = [YTDLP_BIN, "--get-filename", "--output", outtmpl]
    if ffmpeg_location:
        cmd += ["--ffmpeg-location", ffmpeg_location]
    cmd.append(url)
    try:
        env = os.environ.copy()
        if extra_path_dirs:
            seen = set()
            dirs = []
            for d in extra_path_dirs:
                if not d:
                    continue
                nd = os.path.abspath(d)
                if nd not in seen and os.path.isdir(nd):
                    seen.add(nd)
                    dirs.append(nd)
            if dirs:
                env["PATH"] = os.pathsep.join(dirs) + os.pathsep + env.get("PATH", "")
        proc = subprocess.run(cmd, capture_output=True, text=True, check=True, env=env)
        fn = proc.stdout.strip()
        return fn if fn else None
    except Exception:
        return None


# -------------------------
# Argument builders
# -------------------------
def build_common_args(outtmpl: str, ffmpeg_location: Optional[str],
                      embed: bool = True, keep_infojson: bool = False) -> list:
    """
    Builds common yt-dlp args.

    Note:
      - embed: whether to include embedding flags (subs/metadata/chapters/info-json).
      - keep_infojson: if True -> adds --no-clean-info-json (keeps .info.json). Default False.
    """
    args: List[str] = []
    if ffmpeg_location:
        args += ["--ffmpeg-location", ffmpeg_location]

    args += [
        "--no-overwrites",
        "--no-restrict-filenames",
        "--write-thumbnail",
        "--embed-thumbnail",
        "--write-description",
        "--write-info-json",
        # default: don't include --no-clean-info-json so yt-dlp will clean info.json after embedding
        "--progress",
        "--output", outtmpl,
        "--newline",
        "--concurrent-fragments", "5",
    ]

    # verbosity/diagnostics (we will still redirect background output to log files)
    args += ["--no-quiet", "-v", "--print-traffic"]

    if embed:
        args += [
            "--write-subs",
            "--embed-subs",
            "--embed-metadata",
            "--embed-chapters",
            "--embed-info-json",
            # Convert to JPG for consistent embedding support across containers/players
            "--convert-thumbnails", "jpg",
        ]

    if keep_infojson:
        # Explicitly requested by user to keep .info.json (we don't want this on by default)
        args += ["--no-clean-info-json"]

    return args


def build_video_args(vqual_choice: int, vfmt_choice: int) -> list:
    """
    Build format selection for video.

    Key robustness:
    - When user targets MP4, prefer MP4-friendly streams: avc1 video + m4a (AAC) audio.
      Fall back progressively to progressive MP4, then to generic best pair.
      This prevents silent MP4s caused by trying to mux WebM/Opus audio into MP4.
    """
    # Base quality ceiling
    if vqual_choice == 1:
        base_f = "bv*+ba/b"
    elif vqual_choice == 2:
        base_f = "bv*[height<=1080]+ba/b[height<=1080]"
    elif vqual_choice == 3:
        base_f = "bv*[height<=720]+ba/b[height<=720]"
    else:
        base_f = "bv*[height<=480]+ba/b[height<=480]"

    if vfmt_choice == 3:
        # MP4 target: prefer MP4-friendly codecs
        fstr = (
            "bv*[ext=mp4]+ba[ext=m4a]/"                # ideal: MP4 video + M4A audio
            "bv*[vcodec^=avc1]+ba[acodec^=mp4a]/"      # fallback: avc1 + AAC
            "b[ext=mp4]/"                              # fallback: progressive MP4
            f"{base_f}/"                               # fallback: generic best pair
            "b"                                        # final fallback
        )
    elif vfmt_choice == 2:
        # MKV target: MKV is permissive, base selection is fine
        fstr = base_f
    elif vfmt_choice == 4:
        # WEBM target: prefer native WebM streams
        fstr = f"bv*[ext=webm]+ba[ext=webm]/b[ext=webm]/{base_f}"
    else:
        # Best container (let yt-dlp decide)
        fstr = base_f

    args = ["-f", fstr]
    if vfmt_choice == 2:
        args += ["--merge-output-format", "mkv"]
    elif vfmt_choice == 3:
        args += ["--merge-output-format", "mp4"]
    elif vfmt_choice == 4:
        args += ["--merge-output-format", "webm"]
    return args


def build_audio_args(aqual_choice: int, afmt_choice: int) -> list:
    args = ["--extract-audio", "-f", "bestaudio/best"]
    fmt = None
    if afmt_choice == 2: fmt = "mp3"
    elif afmt_choice == 3: fmt = "m4a"
    elif afmt_choice == 4: fmt = "flac"
    elif afmt_choice == 5: fmt = "opus"
    elif afmt_choice == 1: fmt = "best"
    if fmt:
        args += ["--audio-format", fmt]

    if aqual_choice == 1:
        args += ["--audio-quality", "0"]
    elif aqual_choice == 2:
        args += ["--audio-format", "mp3", "--audio-quality", "0"]
    elif aqual_choice == 3:
        args += ["--audio-format", "m4a", "--audio-quality", "0"]
    elif aqual_choice == 4:
        if "--audio-format" not in " ".join(args):
            args += ["--audio-format", "opus"]

    return args


def build_outtmpl(output_dir: str, naming_choice: int) -> str:
    Path(output_dir).mkdir(parents=True, exist_ok=True)
    if naming_choice == 1:
        return str(Path(output_dir) / "%(title)s.%(ext)s")
    else:
        return str(Path(output_dir) / "%(autonumber)03d.%(ext)s")


# -------------------------
# Sidecar cleanup helpers
#   - All cleanup helpers are now "quiet by default" to avoid interfering with input prompts.
# -------------------------
def remove_file_quiet(path: Path, verbose: bool = False):
    try:
        if path.exists():
            path.unlink()
            if verbose:
                print(f"Removed: {path}")
    except Exception:
        pass


def remove_associated_thumbnails_for_file(main_file: Path, verbose: bool = False):
    thumb_exts = {".jpg", ".jpeg", ".png", ".webp"}
    base = main_file.stem
    outdir = main_file.parent
    for ext in thumb_exts:
        candidate = outdir / (base + ext)
        if candidate.exists():
            remove_file_quiet(candidate, verbose=verbose)


def remove_associated_subs_for_file(main_file: Path, verbose: bool = False):
    subs_exts = {".srt", ".vtt", ".ass", ".ssa", ".ttml"}
    base = main_file.stem
    outdir = main_file.parent
    for f in outdir.iterdir():
        if not f.is_file():
            continue
        if not f.name.startswith(base):
            continue
        if f.suffix.lower() in subs_exts:
            remove_file_quiet(f, verbose=verbose)


def remove_associated_description_for_file(main_file: Path, verbose: bool = False):
    outdir = main_file.parent
    candidate = outdir / (main_file.stem + ".description")
    if candidate.exists():
        remove_file_quiet(candidate, verbose=verbose)


def remove_associated_infojson_for_file(main_file: Path, keep_infojson: bool, verbose: bool = False):
    if keep_infojson:
        return
    outdir = main_file.parent
    candidate = outdir / (main_file.stem + ".info.json")
    if candidate.exists():
        remove_file_quiet(candidate, verbose=verbose)


def remove_associated_sidecars(main_file: Path, keep_infojson: bool = False, verbose: bool = False):
    remove_associated_thumbnails_for_file(main_file, verbose=verbose)
    remove_associated_subs_for_file(main_file, verbose=verbose)
    remove_associated_description_for_file(main_file, verbose=verbose)
    remove_associated_infojson_for_file(main_file, keep_infojson=keep_infojson, verbose=verbose)


def remove_file_if_empty_or_whitespace(path: str):
    try:
        if not os.path.exists(path):
            return
        with open(path, "r", encoding="utf-8") as f:
            content = f.read()
        if not content.strip():
            os.remove(path)
            # harmless at end of program; leave quiet to reduce noise
            # print(f"Removed empty error log: {path}")
    except Exception:
        pass


# -------------------------
# Post-process reinforcement
# -------------------------
def reinforce_mkv_thumbnail_embed(ffmpeg_bin: Optional[str], main_file: Path, verbose: bool = False) -> bool:
    """
    For MKV files, some players don't show the embedded thumbnail even when yt-dlp says it embedded it.
    Reinforce by attaching the thumbnail explicitly via ffmpeg as an MKV attachment.

    Returns True if a re-mux was performed successfully.
    """
    try:
        if ffmpeg_bin is None:
            return False
        if main_file.suffix.lower() != ".mkv":
            return False

        # Seek a matching thumbnail file by stem
        stem = main_file.stem
        outdir = main_file.parent
        thumb: Optional[Path] = None
        for ext in (".jpg", ".jpeg", ".png", ".webp"):
            cand = outdir / (stem + ext)
            if cand.exists():
                thumb = cand
                break
        if thumb is None:
            return False

        tmp_out = main_file.with_suffix(".tmp." + main_file.suffix.lstrip("."))
        cmd = [
            ffmpeg_bin,
            "-y",
            "-i", str(main_file),
            "-map", "0",
            "-c", "copy",
            "-attach", str(thumb),
            "-metadata:s:t", "mimetype=image/jpeg",
            "-metadata:s:t", f"filename={thumb.name}",
            str(tmp_out),
        ]
        proc = subprocess.run(cmd, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
        if proc.returncode != 0 or not tmp_out.exists():
            try:
                if tmp_out.exists():
                    tmp_out.unlink()
            except Exception:
                pass
            return False

        # Replace original atomically
        try:
            os.replace(str(tmp_out), str(main_file))
        except Exception:
            # Attempt cleanup on failure
            if tmp_out.exists():
                tmp_out.unlink()
            return False

        if verbose:
            print(f"Reinforced MKV thumbnail embed: {main_file.name}")
        return True
    except Exception:
        return False


# -------------------------
# Background worker classes
# -------------------------
class BackgroundDownload:
    """
    Background download object:
     - creates a unique temp log file
     - starts yt-dlp subprocess with stdout/stderr -> log file
     - does not perform heavy subprocess work in constructor (keeps startup fast)
     - after process completes and rc == 0 (and not aborted), tries to identify the main
       downloaded file and cleans sidecars (unless keep_infojson True).
     - if aborted externally (user chose audio), the job will be terminated and
       any produced files removed.
    """

    def __init__(self, url: str, YTDLP_BIN: str, output_dir: str,
                 ffmpeg_location: Optional[str], extra_path_dirs: Optional[List[str]] = None,
                 embed: bool = True, keep_infojson: bool = False):
        self.url = url
        self.YTDLP_BIN = YTDLP_BIN
        self.output_dir = output_dir
        self.ffmpeg_location = ffmpeg_location
        self.extra_path_dirs = extra_path_dirs or []
        self.embed = embed
        self.keep_infojson = keep_infojson

        self.id = uuid.uuid4().hex[:12]
        self.log_path = Path(tempfile.gettempdir()) / f"ytdlp_bg_{self.id}.log"
        self.process: Optional[subprocess.Popen] = None
        self.thread: Optional[threading.Thread] = None
        self.start_time: Optional[float] = None
        self.completed: bool = False
        self.rc: Optional[int] = None
        self._lock = threading.Lock()
        self.aborted: bool = False  # mark if job was intentionally aborted

        # Build static args early (cheap)
        self.outtmpl = build_outtmpl(self.output_dir, naming_choice=1)
        # default: embed True for background video pre-downloads
        self.common_args = build_common_args(self.outtmpl, self.ffmpeg_location, embed=self.embed, keep_infojson=self.keep_infojson)
        # target mp4 by default for pre-download video (robust mp4-friendly selection defined in builder)
        self.video_args = build_video_args(vqual_choice=1, vfmt_choice=3)

    def start(self):
        self.thread = threading.Thread(target=self._run, daemon=True)
        self.thread.start()

    def _run(self):
        self.start_time = time.time()
        # detect playlist in background (this is intentionally done here so main thread isn't blocked)
        try:
            is_playlist = detect_playlist(self.YTDLP_BIN, self.url)
        except Exception:
            is_playlist = False

        # if is_playlist True we pass the playlist URL directly to yt-dlp and it will download entries;
        # otherwise it downloads the single video.
        try:
            base_dir = os.path.dirname(os.path.abspath(__file__))
            local_ytdlp = os.path.join(base_dir, "yt-dlp.exe")
            exe = local_ytdlp if os.path.exists(local_ytdlp) else self.YTDLP_BIN
        except Exception:
            exe = self.YTDLP_BIN

        cmd = [exe] + self.common_args + self.video_args + [self.url]
        # Write log bytes
        try:
            self.log_path.parent.mkdir(parents=True, exist_ok=True)
        except Exception:
            pass

        try:
            # Keep this as a binary file so we capture yt-dlp's raw output.
            # Note: subprocess writes directly to the file descriptor, so Python-side buffering is not the limiting factor here.
            with open(self.log_path, "ab") as logf:
                proc = subprocess.Popen(cmd, stdout=logf, stderr=subprocess.STDOUT, env=self._build_env())
                with self._lock:
                    self.process = proc
                proc.wait()
                with self._lock:
                    self.rc = proc.returncode
                    self.completed = True
        except Exception:
            with self._lock:
                self.rc = 1
                self.completed = True

        # If successful and not aborted, attempt to identify main file and clean sidecars (only if rc == 0)
        if not self.aborted and self.rc == 0:
            try:
                main = self.identify_main_file_after_run()
                if main:
                    # remove sidecars quietly to avoid messing with any prompts
                    remove_associated_sidecars(main, keep_infojson=self.keep_infojson, verbose=False)
            except Exception:
                pass

    def _build_env(self):
        env = os.environ.copy()
        if self.extra_path_dirs:
            seen = set()
            dirs = []
            for d in self.extra_path_dirs:
                if not d: continue
                nd = os.path.abspath(d)
                if nd not in seen and os.path.isdir(nd):
                    seen.add(nd)
                    dirs.append(nd)
            if dirs:
                env["PATH"] = os.pathsep.join(dirs) + os.pathsep + env.get("PATH", "")
        return env

    def is_running(self) -> bool:
        with self._lock:
            if self.process is None:
                return False
            return self.process.poll() is None

    def identify_main_file_after_run(self) -> Optional[Path]:
        # If yt-dlp wrote predictable filename, try to use it
        predicted = get_predicted_filename(self.YTDLP_BIN, self.url, self.outtmpl, self.ffmpeg_location, self.extra_path_dirs)
        if predicted:
            cand = Path(predicted)
            if not cand.is_absolute():
                cand = Path(self.output_dir) / cand.name
            if cand.exists():
                return cand
            # fallback by stem match
            pred_stem = Path(predicted).stem
            for f in Path(self.output_dir).iterdir():
                if not f.is_file():
                    continue
                if f.stem == pred_stem:
                    return f

        # fallback: choose largest file modified since start_time
        candidates = []
        try:
            for f in Path(self.output_dir).iterdir():
                if not f.is_file():
                    continue
                try:
                    if self.start_time and f.stat().st_mtime >= (self.start_time - 1.0):
                        candidates.append(f)
                except Exception:
                    continue
        except Exception:
            pass

        if candidates:
            return max(candidates, key=lambda f: (f.stat().st_size if f.exists() else 0))
        return None

    def attach_and_stream(self):
        """
        Attach to the log and stream it to the console in real-time.
        While streaming, set the console title to indicate active download.
        When finished, restore a neutral title.

        IMPORTANT:
         - This will block until the background process finishes and the background thread
           has completed its post-run cleanup (sidecar removal), so sequencing is preserved.
         - We stream the whole log (from the start) so the user sees the full verbose yt-dlp output.
        """
        # set attaching title
        set_console_title("yt-dlp — downloading (attached)")

        # Wait until process attribute set or up to a short timeout
        waited = 0.0
        while True:
            with self._lock:
                p = self.process
            if p is not None:
                break
            time.sleep(0.05)
            waited += 0.05
            if waited > 5.0:
                break

        # Stream log content
        try:
            # Open as text for easier streaming to console; replace errors to avoid crashing on binary sequences.
            with open(self.log_path, "r", encoding="utf-8", errors="replace") as f:
                # read entire existing log from start (replay) then follow
                f.seek(0, os.SEEK_SET)
                content = f.read()
                if content:
                    sys.stdout.write(content)
                    sys.stdout.flush()

                while True:
                    with self._lock:
                        p = self.process
                    if p is None:
                        break
                    if p.poll() is not None:
                        # final tail
                        rest = f.read()
                        if rest:
                            sys.stdout.write(rest)
                            sys.stdout.flush()
                        break
                    new = f.read()
                    if new:
                        sys.stdout.write(new)
                        sys.stdout.flush()
                    else:
                        time.sleep(0.12)
        except FileNotFoundError:
            # if no log, wait for process quietly
            if self.process:
                self.process.wait()
        except Exception:
            if self.process:
                self.process.wait()

        # Wait for the background thread to finish its post-run cleanup (so sidecar removal messages
        # and other prints produced by the background thread appear before we return).
        try:
            if self.thread and self.thread is not threading.current_thread():
                # join without timeout to ensure cleanup finished
                self.thread.join()
        except Exception:
            pass

        # restore neutral title
        set_console_title("yt-dlp Interactive Loader")

    def terminate_and_remove_partial(self):
        """
        Intentionally abort this background job and remove any partial files/logs.

        Behaviour:
         - Mark job as aborted so post-run cleanup doesn't try to re-run embedding/sidecar logic.
         - If the yt-dlp process is running, attempt a graceful terminate, then a kill.
         - Attempt to identify produced files via the predicted filename and by modification-time
           heuristics relative to job start_time; remove matching files and sidecars.
         - Remove the temporary log file.
         - Mark the job as completed with a non-zero rc.

        All removals are quiet to avoid interfering with user prompts.
        """
        with self._lock:
            self.aborted = True
            p = self.process

        if p is not None:
            try:
                # attempt graceful termination
                p.terminate()
            except Exception:
                pass
            # wait briefly
            try:
                p.wait(timeout=3)
            except Exception:
                try:
                    p.kill()
                except Exception:
                    pass
                try:
                    p.wait(timeout=2)
                except Exception:
                    pass

        # mark as finished/aborted
        with self._lock:
            self.rc = 1
            self.completed = True
            # clear process reference (safe to set; thread will still handle its own cleanup)
            self.process = None

        # Remove files that look like they were created by this job.
        candidates = set()
        # try predicted filename first (most specific)
        try:
            predicted = get_predicted_filename(self.YTDLP_BIN, self.url, self.outtmpl, self.ffmpeg_location, self.extra_path_dirs)
        except Exception:
            predicted = None

        if predicted:
            cand = Path(predicted)
            if not cand.is_absolute():
                cand = Path(self.output_dir) / cand.name
            if cand.exists():
                candidates.add(cand)
            pred_stem = Path(predicted).stem
            try:
                for f in Path(self.output_dir).iterdir():
                    if not f.is_file():
                        continue
                    if f.name.startswith(pred_stem):
                        candidates.add(f)
            except Exception:
                pass

        # fallback: files modified since start_time (conservative window)
        if self.start_time:
            try:
                for f in Path(self.output_dir).iterdir():
                    if not f.is_file():
                        continue
                    try:
                        if f.stat().st_mtime >= (self.start_time - 5.0):
                            candidates.add(f)
                    except Exception:
                        continue
            except Exception:
                pass

        # Remove the collected candidate files and their sidecars (quietly).
        for f in list(candidates):
            try:
                remove_associated_sidecars(f, keep_infojson=self.keep_infojson, verbose=False)
            except Exception:
                pass
            try:
                if f.exists():
                    f.unlink()
            except Exception:
                pass

        # remove fragments/part files with common patterns that might not match predicted names
        try:
            for f in Path(self.output_dir).iterdir():
                if not f.is_file():
                    continue
                name = f.name.lower()
                # common temporary/partial suffixes
                if any(name.endswith(s) for s in (".part", ".part.tmp", ".part_", ".ytdl", ".ytdlpart", ".crdownload", ".part0")):
                    try:
                        f.unlink()
                    except Exception:
                        pass
        except Exception:
            pass

        # remove the log file for this background job
        try:
            if self.log_path.exists():
                self.log_path.unlink()
        except Exception:
            pass


class BackgroundManager:
    def __init__(self):
        self.downloads: Dict[str, BackgroundDownload] = {}
        self._lock = threading.Lock()

    def start_for_inputs(self, inputs: List[str], YTDLP_BIN: str, output_dir: str,
                         ffmpeg_location: Optional[str], extra_path_dirs: Optional[List[str]],
                         embed: bool = True, keep_infojson: bool = False):
        """
        Start background downloads for each provided input (each input may be a single url or a playlist url).
        This function is intentionally lightweight and returns immediately after starting background threads.
        """
        for inp in inputs:
            bd = BackgroundDownload(inp, YTDLP_BIN, output_dir, ffmpeg_location, extra_path_dirs, embed=embed, keep_infojson=keep_infojson)
            with self._lock:
                self.downloads[inp] = bd
            bd.start()

    def get(self, url: str) -> Optional[BackgroundDownload]:
        with self._lock:
            return self.downloads.get(url)

    def any_running(self) -> bool:
        with self._lock:
            return any((d.process is not None and d.is_running()) for d in self.downloads.values())

    def finalize_and_attach_all(self) -> bool:
        """
        For all known background downloads:
         - if running: attach (this streams the log and waits for cleanup)
         - otherwise, wait for the background thread to join so any post-run cleanup finishes
        Returns True if we attached to at least one running job (meaning verbose logs were streamed).
        """
        attached_any = False
        with self._lock:
            items = list(self.downloads.items())
        for url, bd in items:
            if bd is None:
                continue
            try:
                if bd.is_running():
                    attached_any = True
                    print(f"\nAttaching to background download (finalizing): {url}")
                    bd.attach_and_stream()
                else:
                    # If the background thread exists but hasn't fully finished, join to wait for cleanup.
                    if bd.thread and bd.thread.is_alive():
                        bd.thread.join()
            except Exception:
                # ignore individual failures but continue finalising others
                continue
        return attached_any

    def terminate_for_url(self, url: str) -> bool:
        """
        Terminate and remove any background job and its partial files for the given URL.
        Returns True if a background job was found and requested to terminate.
        """
        with self._lock:
            bd = self.downloads.get(url)
        if not bd:
            return False
        try:
            bd.terminate_and_remove_partial()
        except Exception:
            pass
        with self._lock:
            # remove from registry - we intentionally forget aborted jobs
            if url in self.downloads:
                del self.downloads[url]
        return True


# -------------------------
# Foreground runner
# -------------------------
def run_YTDLP_BIN_foreground(YTDLP_BIN: str, url: str, args: list, error_log_path: str,
                             extra_path_dirs: Optional[List[str]] = None) -> int:
    env = os.environ.copy()
    if extra_path_dirs:
        seen = set()
        dirs = []
        for d in extra_path_dirs:
            if not d:
                continue
            nd = os.path.abspath(d)
            if nd not in seen and os.path.isdir(nd):
                seen.add(nd)
                dirs.append(nd)
        if dirs:
            env["PATH"] = os.pathsep.join(dirs) + os.pathsep + env.get("PATH", "")

    try:
        base_dir = os.path.dirname(os.path.abspath(__file__))
        local_ytdlp = os.path.join(base_dir, "yt-dlp.exe")
        exe = local_ytdlp if os.path.exists(local_ytdlp) else YTDLP_BIN
    except Exception:
        exe = YTDLP_BIN

    # Ensure console title flag is present for foreground runs (so UI appears the same as usual)
    args_for_run = list(args)
    if "--console-title" not in args_for_run:
        args_for_run = ["--console-title"] + args_for_run

    cmd = [exe] + args_for_run + [url]
    try:
        # Set console title to indicate foreground download
        set_console_title("yt-dlp — downloading")
        proc = subprocess.run(cmd, env=env)
        rc = proc.returncode
    except FileNotFoundError:
        print(f"yt-dlp executable not found: {exe}", file=sys.stderr)
        rc = 1
    except Exception as e:
        print(f"Failed to run yt-dlp: {e}", file=sys.stderr)
        rc = 1
    finally:
        # Reset title back to neutral
        set_console_title("yt-dlp Interactive Loader")

    if rc != 0:
        try:
            with open(error_log_path, "a", encoding="utf-8") as f:
                f.write(f"ERROR downloading: {url}\n")
        except Exception:
            pass
    return rc


# -------------------------
# File utilities used by UI flows
# -------------------------
def read_urls_from_file(path: str) -> List[str]:
    urls: List[str] = []
    with open(path, "r", encoding="utf-8-sig") as f:
        for line in f:
            s = line.strip()
            if not s or s.startswith("#"):
                continue
            urls.append(s)
    return urls


def list_files_in_dir(dirpath: str) -> List[Path]:
    p = Path(dirpath)
    if not p.exists():
        return []
    return [f for f in p.iterdir() if f.is_file()]


def find_existing_media_for_predicted(predicted: Optional[str], output_dir: str) -> Optional[Path]:
    if not predicted:
        return None
    cand = Path(predicted)
    if not cand.is_absolute():
        cand = Path(output_dir) / cand.name
    if cand.exists():
        return cand
    stem = Path(predicted).stem
    for f in Path(output_dir).iterdir():
        if not f.is_file():
            continue
        if f.stem == stem:
            return f
    return None


def ffmpeg_available(local_ffmpeg: str, local_ffprobe: str) -> Optional[str]:
    if local_ffmpeg and os.path.exists(local_ffmpeg):
        return os.path.abspath(local_ffmpeg)
    if shutil.which("ffmpeg"):
        return shutil.which("ffmpeg")
    return None


def extract_audio_with_ffmpeg(ffmpeg_bin: str, input_file: Path, target_format: str = "mp3") -> bool:
    outfile = input_file.with_suffix("." + target_format)
    try:
        if target_format == "mp3":
            cmd = [ffmpeg_bin, "-y", "-i", str(input_file), "-vn", "-c:a", "libmp3lame", "-q:a", "0", str(outfile)]
        elif target_format == "m4a":
            cmd = [ffmpeg_bin, "-y", "-i", str(input_file), "-vn", "-c:a", "aac", "-b:a", "320k", str(outfile)]
        elif target_format == "flac":
            cmd = [ffmpeg_bin, "-y", "-i", str(input_file), "-vn", "-c:a", "flac", str(outfile)]
        else:
            cmd = [ffmpeg_bin, "-y", "-i", str(input_file), "-vn", "-c:a", "libmp3lame", "-q:a", "0", str(outfile)]
        proc = subprocess.run(cmd, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
        return proc.returncode == 0
    except Exception:
        return False


# -------------------------
# Prompts
# -------------------------
def prompt_choice(title: str, options: List[str], default_idx: int = 1) -> int:
    print(title)
    for i, opt in enumerate(options, 1):
        print(f"  [{i}] {opt}")
    while True:
        raw = input(f"Choose 1-{len(options)} [{default_idx}]: ").strip()
        if raw == "":
            return default_idx
        if raw.isdigit():
            n = int(raw)
            if 1 <= n <= len(options):
                return n
        print("Invalid selection. Try again.")


def prompt_text(prompt: str, allow_empty: bool = False, default=None) -> str:
    while True:
        raw = input(prompt).strip()
        if raw == "" and default is not None:
            return default
        if raw == "" and allow_empty:
            return ""
        if raw:
            return raw
        print("Input cannot be empty.")


# -------------------------
# Main program flow
# -------------------------
def main():
    ensure_paths()

    local_ytdlp = resource_path("yt-dlp.exe")
    local_ffmpeg = resource_path("ffmpeg.exe")
    local_ffprobe = resource_path("ffprobe.exe")

    if os.path.exists(local_ytdlp):
        YTDLP_BIN = local_ytdlp
    else:
        YTDLP_BIN = shutil.which("yt-dlp") or "yt-dlp"

    # Only pass --ffmpeg-location when we actually have a local ffmpeg/ffprobe.
    # Otherwise let yt-dlp find ffmpeg in PATH.
    ffmpeg_location: Optional[str] = None
    if os.path.exists(local_ffmpeg):
        ffmpeg_location = os.path.dirname(os.path.abspath(local_ffmpeg))
    elif os.path.exists(local_ffprobe):
        ffmpeg_location = os.path.dirname(os.path.abspath(local_ffprobe))

    extra_path_dirs: List[str] = []
    if os.path.exists(local_ytdlp):
        extra_path_dirs.append(os.path.dirname(os.path.abspath(local_ytdlp)))
    if os.path.exists(local_ffmpeg):
        extra_path_dirs.append(os.path.dirname(os.path.abspath(local_ffmpeg)))
    if os.path.exists(local_ffprobe):
        extra_path_dirs.append(os.path.dirname(os.path.abspath(local_ffprobe)))

    # For optional post-processing (MKV reinforcement)
    ffmpeg_bin_for_post = ffmpeg_available(local_ffmpeg, local_ffprobe)

    print("\n============================================")
    print("          yt-dlp Interactive Loader")
    print("============================================\n")

    initial_input = prompt_text("Paste a URL / playlist / YouTube ID / or full path to a file of URLs: ").strip()
    if not initial_input:
        print("No input provided. Exiting.", file=sys.stderr)
        sys.exit(1)

    # Quick file check (very fast). If it's a local file, expand URLs immediately.
    urls: List[str] = []
    if os.path.isfile(initial_input):
        raw_urls = read_urls_from_file(initial_input)
        urls = [sanitize_url(u) for u in raw_urls if sanitize_url(u)]
        if not urls:
            print("No valid URLs found in the provided file.", file=sys.stderr)
            sys.exit(1)
    else:
        # Do not run yt-dlp detection on main thread (that caused pauses).
        # Instead treat candidate as a single input; background threads will detect/expand playlists.
        candidate = sanitize_url(initial_input)
        if not candidate:
            print("No URL/ID provided.", file=sys.stderr)
            sys.exit(1)
        urls = [candidate]

    # Start background downloads IMMEDIATELY (no heavy work in main thread).
    bg_manager = BackgroundManager()
    # For pre-download we want embedding enabled to embed metadata into the video and ensure thumbnail conversion to jpg.
    # We do not set keep_infojson True: we want yt-dlp to clean .info.json by default after embed.
    bg_manager.start_for_inputs(urls, YTDLP_BIN, DEFAULT_OUTPUT_DIR, ffmpeg_location, extra_path_dirs, embed=True, keep_infojson=False)

    # Do not print anything about background activity. Immediately present express menu.
    print("\nExpress options (default 1):")
    print("  [1] Express Video (highest-quality MP4 with embeds) — default")
    print("  [2] Express MP3 (extract audio)")
    print("  [3] Full custom (choose options)")
    choice = prompt_text("Choose express option [1]: ", allow_empty=True, default="1").strip()
    if choice == "":
        choice = "1"
    if choice not in {"1", "2", "3"}:
        print("Invalid choice; defaulting to 1 (Express Video).")
        choice = "1"

    # Helper: attempt to attach to background job for a given input and stream its log.
    def attach_if_background_present(inp: str):
        bd = bg_manager.get(inp)
        if bd is None:
            return False
        # If running, attach and stream
        if bd.is_running():
            bd.attach_and_stream()
        else:
            # if already completed, optionally show a short confirmation
            main = bd.identify_main_file_after_run()
            if main:
                print(f"Available: {main.name} (in {main.parent})")
            else:
                # nothing found
                pass
        return True

    # Express behaviors
    if choice == "1":
        # Express Video: for each input, if background exists attach (will block until finished)
        for u in urls:
            attached = attach_if_background_present(u)
            if not attached:
                # No background job: run a standard foreground highest-quality mp4 download
                outtmpl = build_outtmpl(DEFAULT_OUTPUT_DIR, naming_choice=1)
                common = build_common_args(outtmpl, ffmpeg_location, embed=True, keep_infojson=False)
                video_args = build_video_args(vqual_choice=1, vfmt_choice=3)
                rc = run_YTDLP_BIN_foreground(YTDLP_BIN, u, common + video_args, ERROR_LOG, extra_path_dirs)
                if rc == 0:
                    # try to find main file and clean up sidecars
                    start_time = time.time() - 5
                    # pick newest large file
                    cand = None
                    for f in list_files_in_dir(DEFAULT_OUTPUT_DIR):
                        try:
                            if f.stat().st_mtime >= start_time:
                                if cand is None or f.stat().st_size > cand.stat().st_size:
                                    cand = f
                        except Exception:
                            continue
                    if cand:
                        # For MP4 path only; MKV not expected here but safe to reinforce if occurs
                        if cand.suffix.lower() == ".mkv":
                            reinforce_mkv_thumbnail_embed(ffmpeg_bin_for_post, cand, verbose=False)
                        remove_associated_sidecars(cand, keep_infojson=False, verbose=False)
    elif choice == "2":
        # Express MP3: the policy here is explicit: background jobs are video-only.
        # If a background job exists for this URL, abandon it (terminate and remove partials)
        # and then run a fresh foreground extract-audio run.
        for u in urls:
            bd = bg_manager.get(u)
            if bd:
                print(f"Abandoning background video download for: {u}")
                bg_manager.terminate_for_url(u)

            # Now run foreground extract-audio with yt-dlp (standard behaviour)
            outtmpl = build_outtmpl(DEFAULT_OUTPUT_DIR, naming_choice=1)
            common = build_common_args(outtmpl, ffmpeg_location, embed=False, keep_infojson=False)
            audio_args = build_audio_args(aqual_choice=1, afmt_choice=2)  # mp3
            rc = run_YTDLP_BIN_foreground(YTDLP_BIN, u, common + audio_args, ERROR_LOG, extra_path_dirs)

            # After a successful audio run, remove lingering sidecars for the produced file
            if rc == 0:
                predicted = get_predicted_filename(YTDLP_BIN, u, outtmpl, ffmpeg_location, extra_path_dirs)
                main = find_existing_media_for_predicted(predicted, DEFAULT_OUTPUT_DIR)
                if main:
                    remove_associated_sidecars(main, keep_infojson=False, verbose=False)
    else:
        # Full custom: If any background jobs are still running, bring them in now (attach & stream)
        any_running = any((bg_manager.get(u) is not None and bg_manager.get(u).is_running()) for u in urls)
        if any_running:
            for u in urls:
                bd = bg_manager.get(u)
                if bd and bd.is_running():
                    bd.attach_and_stream()

        # Now full options similar to earlier script
        mode_idx = prompt_choice("Select mode:", ["Audio (extract best by default)", "Video (bestvideo+bestaudio by default)"], default_idx=1)
        mode = "AUDIO" if mode_idx == 1 else "VIDEO"

        # If user explicitly chooses AUDIO in full custom, abandon any background video jobs for these URLs
        if mode == "AUDIO":
            for u in urls:
                bd = bg_manager.get(u)
                if bd:
                    print(f"Abandoning background video download for: {u}")
                    bg_manager.terminate_for_url(u)

        print("\nOutput folder (Enter to accept default):")
        print(f"  Default: {DEFAULT_OUTPUT_DIR}")
        tmp_out = input("Custom output folder (optional): ").strip()
        output_dir = tmp_out if tmp_out else DEFAULT_OUTPUT_DIR
        Path(output_dir).mkdir(parents=True, exist_ok=True)
        error_log = os.path.join(output_dir, "errors.txt")
        Path(error_log).touch(exist_ok=True)

        if mode == "AUDIO":
            aqual_idx = prompt_choice("Audio quality:", ["Highest (default)", "MP3 320k", "M4A AAC High", "Opus (best)"], default_idx=1)
        else:
            vqual_idx = prompt_choice("Video quality ceiling:", ["Best available (default)", "Up to 1080p", "Up to 720p", "Up to 480p"], default_idx=1)

        if mode == "AUDIO":
            afmt_idx = prompt_choice("Audio format container:", ["Best (default; keep native if possible)", "MP3", "M4A (AAC)", "FLAC", "Opus"], default_idx=1)
        else:
            vfmt_idx = prompt_choice("Video merge container:", ["Best (default; let yt-dlp decide)", "MKV", "MP4", "WEBM"], default_idx=1)

        name_idx = prompt_choice("Naming scheme:", ["Original video title (%(title)s.%(ext)s)", "Numeric 001, 002... (%(autonumber)03d.%(ext)s)"], default_idx=1)

        outtmpl = build_outtmpl(output_dir, name_idx)
        if mode == "AUDIO":
            mode_args = build_audio_args(aqual_idx, afmt_idx)
            common = build_common_args(outtmpl, ffmpeg_location, embed=False, keep_infojson=False)
        else:
            mode_args = build_video_args(vqual_idx, vfmt_idx)
            common = build_common_args(outtmpl, ffmpeg_location, embed=True, keep_infojson=False)

        failures = 0
        for u in urls:
            print("------------------------------------------------------")
            print(f"Downloading: {u}")
            print("------------------------------------------------------")
            rc = run_YTDLP_BIN_foreground(YTDLP_BIN, u, common + mode_args, error_log, extra_path_dirs)
            if rc != 0:
                failures += 1
            else:
                # attempt to clean sidecars for produced file(s)
                # scan for recently modified files (simple heuristic)
                start_time = time.time() - 30
                candidates = []
                for f in list_files_in_dir(output_dir):
                    try:
                        if f.stat().st_mtime >= start_time:
                            candidates.append(f)
                    except Exception:
                        continue
                if candidates:
                    main = max(candidates, key=lambda f: f.stat().st_size)

                    # Reinforce MKV thumbnail embedding if needed BEFORE sidecar cleanup
                    if main.suffix.lower() == ".mkv":
                        reinforce_mkv_thumbnail_embed(ffmpeg_bin_for_post, main, verbose=False)

                    # Clean sidecars quietly
                    remove_associated_sidecars(main, keep_infojson=False, verbose=False)

                # Third issue fix: If the user chose a non-default output_dir for VIDEO,
                # remove any background pre-downloaded .mp4 that landed in DEFAULT_OUTPUT_DIR.
            if rc == 0 and mode == "VIDEO" and (output_dir != DEFAULT_OUTPUT_DIR):
                bd = bg_manager.get(u)
                if bd:
                    try:
                        bg_main = bd.identify_main_file_after_run()
                    except Exception:
                        bg_main = None
                    if bg_main and bg_main.exists():
                        # Only clear if it's in DEFAULT_OUTPUT_DIR to avoid touching unrelated files
                        if str(bg_main.parent.resolve()) == str(Path(DEFAULT_OUTPUT_DIR).resolve()):
                            # Remove sidecars then the file itself, quietly
                            remove_associated_sidecars(bg_main, keep_infojson=False, verbose=False)
                            try:
                                bg_main.unlink()
                            except Exception:
                                pass

        remove_file_if_empty_or_whitespace(error_log)
        if failures:
            print(f"Finished with {failures} failure(s). Check {error_log} for details.")
        else:
            print("All requested downloads completed.")

    # finish: ensure background downloads are finalized and cleanup performed
    attached_any = bg_manager.finalize_and_attach_all()

    # clean up any empty error logs
    remove_file_if_empty_or_whitespace(ERROR_LOG)

    # If we attached and streamed verbose output, that output already served as the completion trace.
    # Print a short confirmation only (avoid an early "Done" message that occurs before sidecar removal).
    if attached_any:
        print("\nDownloads finished. Check the downloads folder for results.")
    else:
        print("\nDone. Check the downloads folder for results.")
    input()

if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        print("\nAborted by user.")
        sys.exit(130)