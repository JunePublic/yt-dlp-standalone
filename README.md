# yt-dlp-standalone
A standalone and fully running version of a command line CLI, featuring yt-dlp, ffmpeg, jq, and some vibe-coded python code that calls those tools all together to download any video / music avalibe on YouTube, in any quality, however you like. 
Made by Juno 2025. 
First git repo, please excuse for any mistakes. 

## Usage:

### Dependencies:

jq, yt-dlp, ffmpeg. 

Download the standalone executables here:

https://github.com/jqlang/jq/releases/download

https://www.ffmpeg.org/download.html

https://github.com/yt-dlp/yt-dlp

Clone the run.bat and the fronend python script into a folder. 

Copy the jq executable, **only** the ffmpeg executable binary, and the yt-dlp main executable into the same folder. 

Exeute run.bat to initiate the program. 

To **compile**, use the command:

``` python
  python -m PyInstaller --onefile ^
  --add-binary "yt-dlp.exe;." ^
  --add-binary "ffmpeg.exe;." ^
  --add-binary "jq.exe;." ^
  yt_dlp_frontend.py
```

to compile. The result will be a standalone executable in \dist.

Next aim for the project, is to modulate the code, and enforce more necessary comments, as well as implement some more features. 
