#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Interactive yt-dlp Frontend (Audio/Video) — Python edition
- Mirrors the intended behavior of the batch version, minus the Windows parsing headaches.
- Picks audio/video, input source (single URL / playlist URL / text file of URLs),
  output directory (default inside BASE_DIR), quality & format, naming scheme.
- Uses subprocess to call yt-dlp with safe, arg-list execution (no fragile shell quoting).
- Expands playlist URLs without jq by reading yt-dlp's JSON.
- Logs failures to errors.txt. Never deletes .webm.
"""

import os
import sys
import json
import shutil
import subprocess
from pathlib import Path

# === Keep these readable/same ===
BASE_DIR = Path(__file__).resolve().parent
FFMPEG_PATH = str(BASE_DIR)   # directory, not binary
# the run_yt_dlp function prefers local yt-dlp.exe if present

def get_base_dir() -> str:
    """Return the directory of the running EXE or script."""
    if getattr(sys, 'frozen', False):
        # Running from a PyInstaller bundle
        return os.path.dirname(sys.executable)
    return os.path.dirname(os.path.abspath(__file__))

BASE_DIR = get_base_dir()
DEFAULT_OUTPUT = os.path.join(BASE_DIR, "downloads")
os.makedirs(DEFAULT_OUTPUT, exist_ok=True)

# === Defaults ===
DEFAULT_OUTPUT_DIR = os.path.join(DEFAULT_OUTPUT)
ERROR_LOG          = os.path.join(DEFAULT_OUTPUT, "errors.txt")

# === Helpers ===
def ensure_paths():
    Path(BASE_DIR).mkdir(parents=True, exist_ok=True)
    Path(DEFAULT_OUTPUT_DIR).mkdir(parents=True, exist_ok=True)
    Path(ERROR_LOG).touch(exist_ok=True)

def resource_path(name: str) -> str:
    """Return absolute path to a bundled resource (works inside PyInstaller EXE)."""
    if hasattr(sys, "_MEIPASS"):
        return os.path.join(sys._MEIPASS, name)
    return os.path.abspath(name)

# Paths for bundled executables
YTDLP_BIN = resource_path("yt-dlp.exe")
FFMPEG_BIN = resource_path("ffmpeg.exe")
JQ_BIN = resource_path("jq.exe")


def which_or_die(cmd):
    exe = shutil.which(cmd)
    if not exe:
        print(f'Error: "{cmd}" is not on PATH. Install it or add to PATH.', file=sys.stderr)
        sys.exit(1)
    return exe

def prompt_choice(title, options, default_idx=1):
    """
    options: list of strings; returns selected 1-based index
    """
    print(title)
    for i, opt in enumerate(options, 1):
        print(f"  [{i}] {opt}")
    while True:
        raw = input(f"Choose 1-{len(options)} [{default_idx}]: ").strip()
        if raw == "": return default_idx
        if raw.isdigit():
            n = int(raw)
            if 1 <= n <= len(options):
                return n
        print("Invalid selection. Try again.")

def prompt_text(prompt, allow_empty=False, default=None):
    while True:
        raw = input(prompt).strip()
        if raw == "" and default is not None:
            return default
        if raw == "" and allow_empty:
            return ""
        if raw:
            return raw
        print("Input cannot be empty.")

def normalize_youtube_id_or_url(s: str) -> str:
    s = s.strip()
    if not s:
        return s
    if s.lower().startswith("http"):
        return s
    # Assume bare ID
    return f"https://www.youtube.com/watch?v={s}"

def expand_playlist_to_urls(YTDLP_BIN: str, playlist_url: str):
    """
    Uses yt-dlp to dump a flat playlist JSON and returns a list of normalized URLs.
    Avoids jq; robust across platforms.
    """
    try:
        proc = subprocess.run(
            [YTDLP_BIN, "--flat-playlist", "--dump-single-json", playlist_url],
            check=True, capture_output=True, text=True
        )
        data = json.loads(proc.stdout)
        entries = data.get("entries", [])
        urls = []
        for e in entries:
            u = e.get("url") or e.get("id") or ""
            if u:
                urls.append(normalize_youtube_id_or_url(u))
        return urls
    except subprocess.CalledProcessError as e:
        print("Failed to fetch playlist data via yt-dlp.", file=sys.stderr)
        if e.stderr:
            print(e.stderr, file=sys.stderr)
        return []
    except json.JSONDecodeError:
        print("Failed to parse playlist JSON.", file=sys.stderr)
        return []

def build_common_args(outtmpl: str) -> list:
    return [
        "--ffmpeg-location", FFMPEG_PATH,
        "--no-overwrites",
        "--restrict-filenames",
        "--write-thumbnail",
        "--embed-thumbnail",
        "--progress",
        "--console-title",
        "--output", outtmpl,
        "--newline",
    ]

def build_audio_args(aqual_choice: int, afmt_choice: int) -> list:
    """
    aqual_choice:
      1 Highest (default) => --audio-quality 0
      2 MP3 320k          => mp3 + --audio-quality 0
      3 M4A AAC High      => m4a + --audio-quality 0
      4 Opus (best)       => opus (no explicit quality flag)
    afmt_choice:
      1 Best (default; keep native if possible)
      2 MP3
      3 M4A (AAC)
      4 FLAC
      5 Opus
    """
    args = ["--extract-audio", "-f", "bestaudio/best"]

    # Container selection (AFMT) baseline
    fmt = None
    if afmt_choice == 2: fmt = "mp3"
    elif afmt_choice == 3: fmt = "m4a"
    elif afmt_choice == 4: fmt = "flac"
    elif afmt_choice == 5: fmt = "opus"
    elif afmt_choice == 1: fmt = "best"
    if fmt: args += ["--audio-format", fmt]

    # Quality overlay (AQUAL)
    if aqual_choice == 1:
        args += ["--audio-quality", "0"]
    elif aqual_choice == 2:
        args += ["--audio-format", "mp3", "--audio-quality", "0"]
    elif aqual_choice == 3:
        args += ["--audio-format", "m4a", "--audio-quality", "0"]
    elif aqual_choice == 4:
        # Opus best; no explicit audio-quality
        if "--audio-format" not in " ".join(args):
            args += ["--audio-format", "opus"]

    return args

def build_video_args(vqual_choice: int, vfmt_choice: int) -> list:
    """
    vqual_choice:
      1 Best available
      2 Up to 1080p
      3 Up to 720p
      4 Up to 480p
    vfmt_choice:
      1 Best (let yt-dlp decide)
      2 MKV
      3 MP4
      4 WEBM
    """
    # Format selector with ceilings
    if vqual_choice == 1:
        fstr = "bestvideo+bestaudio/best"
    elif vqual_choice == 2:
        fstr = "bv*[height<=1080]+ba/b[height<=1080]"
    elif vqual_choice == 3:
        fstr = "bv*[height<=720]+ba/b[height<=720]"
    else:
        fstr = "bv*[height<=480]+ba/b[height<=480]"

    args = ["-f", fstr]

    # Merge container
    if vfmt_choice == 2:
        args += ["--merge-output-format", "mkv"]
    elif vfmt_choice == 3:
        args += ["--merge-output-format", "mp4"]
    elif vfmt_choice == 4:
        args += ["--merge-output-format", "webm"]
    # vfmt_choice == 1 => let yt-dlp decide

    return args

def build_outtmpl(output_dir: str, naming_choice: int) -> str:
    Path(output_dir).mkdir(parents=True, exist_ok=True)
    if naming_choice == 1:
        return str(Path(output_dir) / "%(title)s.%(ext)s")
    else:
        return str(Path(output_dir) / "%(autonumber)03d.%(ext)s")

def run_YTDLP_BIN(YTDLP_BIN: str, url: str, args: list, error_log_path: str) -> int:
    """
    Run yt-dlp so its progress output uses carriage-returns (overwrites the same line).
    This does NOT capture stdout/stderr; it lets yt-dlp talk directly to the terminal,
    which is what preserves the "single-line updating" progress bars.

    - YTDLP_BIN: full path to yt-dlp or "yt-dlp" if on PATH
    - url: the URL to download
    - args: list of yt-dlp arguments (not a single joined string)
    - error_log_path: where to append failing URLs
    Returns the process returncode (0 good, non-zero failure).
    """
    import os
    import subprocess
    from shutil import which

    # Build command as list (safer than shell quoting)
    # Put executable first, then URL, then the rest of args (so we preserve how the rest of your script built args)
    cmd = [YTDLP_BIN, url] + args

    # Ensure ffmpeg path (if you used FFMPEG_PATH in module) is available in the child env PATH.
    # If your FFMPEG_PATH variable points at a binary (ffmpeg.exe) convert to its folder.
    env = os.environ.copy()
    try:
        ff = FFMPEG_PATH  # uses the FFMPEG_PATH constant from the module
    except NameError:
        ff = None

    if ff:
        # if user provided the path to the binary, get its folder
        if os.path.isfile(ff):
            ff_dir = os.path.dirname(os.path.abspath(ff))
        else:
            ff_dir = os.path.abspath(ff)
            # Prepend bundled binary directory
            bin_dir = os.path.dirname(YTDLP_BIN)
            env["PATH"] = bin_dir + os.pathsep + env.get("PATH", "")


    # If the user provided a local yt-dlp.exe in the same folder, prefer it (optional convenience).
    # This avoids relying on system PATH if you package yt-dlp.exe beside the script.
    try:
        base_dir = os.path.dirname(os.path.abspath(__file__))
        local_ytdlp = os.path.join(base_dir, "yt-dlp.exe")
        if os.path.exists(local_ytdlp):
            cmd[0] = local_ytdlp
    except Exception:
        pass

    # Run the command WITHOUT redirecting stdout/stderr so yt-dlp sees a real console/TTY.
    # This preserves the \r carriage-return updating behaviour for progress lines.
    try:
        # subprocess.run will inherit parent's stdin/stdout/stderr when these are None.
        # On normal Windows CMD/PowerShell terminals this will show the nice single-line progress.
        proc = subprocess.run(cmd, env=env)
        rc = proc.returncode
    except FileNotFoundError:
        print(f"yt-dlp executable not found: {cmd[0]}", file=sys.stderr)
        rc = 1
    except Exception as e:
        print(f"Failed to run yt-dlp: {e}", file=sys.stderr)
        rc = 1

    if rc != 0:
        try:
            with open(error_log_path, "a", encoding="utf-8") as f:
                f.write(f"ERROR downloading: {url}\n")
        except Exception:
            # don't crash if logging fails
            pass

    return rc

def read_urls_from_file(path: str):
    urls = []
    with open(path, "r", encoding="utf-8-sig") as f:
        for line in f:
            s = line.strip()
            if not s: continue
            if s.startswith("#"): continue
            urls.append(s)
    return urls

def remove_associated_thumbnails(downloaded_files: list):
    """
    Remove thumbnails that match the downloaded files' basenames.
    e.g. track.mp3 -> track.webp, track.jpg, etc.
    """
    thumb_exts = (".jpg", ".jpeg", ".png", ".webp")
    for f in downloaded_files:
        base, _ = os.path.splitext(f)
        for ext in thumb_exts:
            thumb = base + ext
            if os.path.exists(thumb):
                try:
                    os.remove(thumb)
                    print(f"Removed: {thumb}")
                except Exception as e:
                    print(f"Could not remove {thumb}: {e}")

def main():
    ensure_paths()
    YTDLP_BIN = resource_path("yt-dlp.exe")
    FFMPEG_BIN = resource_path("ffmpeg.exe")
    JQ_BIN = resource_path("jq.exe")


    print("\n============================================")
    print("          yt-dlp Interactive Loader")
    print("============================================\n")

    # 1) Audio or Video
    mode_idx = prompt_choice("Select mode:", ["Audio (extract best by default)", "Video (bestvideo+bestaudio by default)"], default_idx=1)
    mode = "AUDIO" if mode_idx == 1 else "VIDEO"
    print(f"  -> Mode: {mode}\n")

    # 2) Input Source
    src_idx = prompt_choice("Input source:", ["Single URL", "Playlist URL (expand)", "Text file of URLs (one per line)"], default_idx=1)
    insrc = ["SINGLE", "PLAYLIST", "FILE"][src_idx - 1]
    print(f"  -> Source: {insrc}\n")

    # Gather source specifics
    urls = []
    if insrc == "SINGLE":
        single = prompt_text("Paste the URL: ")
        urls = [single]
    elif insrc == "PLAYLIST":
        pl = prompt_text("Paste the playlist URL: ")
        print("Expanding playlist via yt-dlp...")
        urls = expand_playlist_to_urls(YTDLP_BIN, pl)
        if not urls:
            print("No entries found or expansion failed.", file=sys.stderr)
            sys.exit(1)
    else:
        list_path = prompt_text("Enter full path to text file of URLs: ")
        if not os.path.exists(list_path):
            print(f'File not found: "{list_path}"', file=sys.stderr)
            sys.exit(1)
        urls = read_urls_from_file(list_path)
        if not urls:
            print("No URLs found in the provided file.", file=sys.stderr)
            sys.exit(1)

    # 3) Output Folder
    print("\nOutput folder (Enter to accept default):")
    print(f"  Default: {DEFAULT_OUTPUT_DIR}")
    tmp_out = input("Custom output folder (optional): ").strip()
    output_dir = tmp_out if tmp_out else DEFAULT_OUTPUT_DIR
    Path(output_dir).mkdir(parents=True, exist_ok=True)
    print(f"  -> Output: {output_dir}\n")

    # 4) Quality
    if mode == "AUDIO":
        aqual_idx = prompt_choice(
            "Audio quality:",
            ["Highest (default)", "MP3 320k", "M4A AAC High", "Opus (best)"],
            default_idx=1
        )
        print(f"  -> Audio quality option: {aqual_idx}")
    else:
        vqual_idx = prompt_choice(
            "Video quality ceiling:",
            ["Best available (default)", "Up to 1080p", "Up to 720p", "Up to 480p"],
            default_idx=1
        )
        print(f"  -> Video quality option: {vqual_idx}")
    print()

    # 5) Format
    if mode == "AUDIO":
        afmt_idx = prompt_choice(
            "Audio format container:",
            ["Best (default; keep native if possible)", "MP3", "M4A (AAC)", "FLAC", "Opus"],
            default_idx=1
        )
        print(f"  -> Audio format option: {afmt_idx}")
    else:
        vfmt_idx = prompt_choice(
            "Video merge container:",
            ["Best (default; let yt-dlp decide)", "MKV", "MP4", "WEBM"],
            default_idx=1
        )
        print(f"  -> Video container option: {vfmt_idx}")
    print()

    # 6) Naming
    name_idx = prompt_choice(
        "Naming scheme:",
        ["Original video title (%(title)s.%(ext)s)", "Numeric 001, 002... (%(autonumber)03d.%(ext)s)"],
        default_idx=1
    )
    print(f"  -> Naming: {'TITLE' if name_idx == 1 else 'NUMERIC'}\n")

    # Build args
    outtmpl = build_outtmpl(output_dir, name_idx)
    common = build_common_args(outtmpl)

    if mode == "AUDIO":
        mode_args = build_audio_args(aqual_idx, afmt_idx)
    else:
        mode_args = build_video_args(vqual_idx, vfmt_idx)

    # Summary
    print("Final options summary:")
    print(f"  Mode       : {mode}")
    print(f"  Output dir : {output_dir}")
    print(f"  Outtmpl    : {outtmpl}")
    if mode == "AUDIO":
        print(f"  AQUAL/AFMT : {aqual_idx}/{afmt_idx}")
    else:
        print(f"  VQUAL/VFMT : {vqual_idx}/{vfmt_idx}")
    print()

    # Dispatch
    failures = 0
    downloaded_files = []

    for u in urls:
        url = u.strip()
        if not url:
            continue
        print("------------------------------------------------------")
        print(f"Downloading: {url}")
        print("------------------------------------------------------")

        # Get exact output filename
        getname_cmd = [YTDLP_BIN, "--get-filename"] + (common + mode_args) + [url]
        try:
            proc = subprocess.run(getname_cmd, capture_output=True, text=True, check=True)
            filename = proc.stdout.strip()
        except subprocess.CalledProcessError:
            filename = None

        # Run the actual download
        rc = run_YTDLP_BIN(YTDLP_BIN, url, common + mode_args, ERROR_LOG)
        if rc == 0 and filename:
            downloaded_files.append(filename)
        else:
            failures += 1

        # Clean up thumbnails for successfully downloaded files
        remove_associated_thumbnails(downloaded_files)


    print("\n================================")
    if failures:
        print(f"Done with {failures} failure(s). Check errors.txt.")
    else:
        print("Done. All downloads completed.")
    print("================================")

if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        print("\nAborted by user.")
        sys.exit(130)
