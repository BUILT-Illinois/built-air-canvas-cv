import json
import os
import subprocess
import warnings


def load_config():
    """Load config.json from the project directory."""
    config_path = os.path.join(os.path.dirname(__file__), os.pardir, "config.json")
    if not os.path.exists(config_path):
        return {"streaming": {"enabled": False}}
    try:
        with open(config_path, "r") as f:
            return json.load(f)
    except (OSError, json.JSONDecodeError) as e:
        warnings.warn(f"Failed to load config from {config_path}: {e}. "
                       "Using default configuration.")
        return {"streaming": {"enabled": False}}


def _ffmpeg_available():
    try:
        subprocess.run(
            ["ffmpeg", "-version"],
            stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL,
            check=True,
        )
        return True
    except (FileNotFoundError, subprocess.CalledProcessError):
        return False


def start_streaming(config, frame_width, frame_height):
    """Start an ffmpeg subprocess for YouTube RTMP streaming.

    Returns the Popen pipe, or None if streaming is not configured/available.
    """
    streaming_cfg = config.get("streaming", {})
    if not streaming_cfg.get("enabled", False):
        return None

    if not _ffmpeg_available():
        print("Warning: Streaming enabled in config but ffmpeg not found. "
              "Streaming disabled.")
        return None

    youtube_cfg = config.get("youtube") if isinstance(config, dict) else None
    if (not isinstance(youtube_cfg, dict)
            or "stream_url" not in youtube_cfg
            or "stream_key" not in youtube_cfg):
        print("Streaming disabled: missing YouTube 'stream_url' or "
              "'stream_key' in configuration.")
        return None

    full_url = f"{youtube_cfg['stream_url']}{youtube_cfg['stream_key']}"
    bitrate = streaming_cfg.get("bitrate", "3000k")
    preset = streaming_cfg.get("preset", "ultrafast")

    ffmpeg_cmd = [
        "ffmpeg", "-y",
        "-f", "rawvideo",
        "-vcodec", "rawvideo",
        "-pix_fmt", "bgr24",
        "-s", f"{frame_width}x{frame_height}",
        "-r", "30",
        "-i", "pipe:0",
        "-f", "lavfi",
        "-i", "anullsrc=channel_layout=stereo:sample_rate=44100",
        "-c:v", "libx264",
        "-pix_fmt", "yuv420p",
        "-preset", preset,
        "-b:v", bitrate,
        "-maxrate", bitrate,
        "-bufsize", "6000k",
        "-g", "60",
        "-c:a", "aac",
        "-b:a", "128k",
        "-ar", "44100",
        "-f", "flv",
        full_url,
    ]

    try:
        pipe = subprocess.Popen(
            ffmpeg_cmd, stdin=subprocess.PIPE, bufsize=10**8
        )
        print("YouTube streaming started!")
        print(f"Streaming to: {youtube_cfg['stream_url']}")
        print(f"Resolution: {frame_width}x{frame_height}")
        return pipe
    except Exception as e:
        print(f"Failed to start streaming: {e}")
        import traceback
        traceback.print_exc()
        return None


def write_frame(pipe, frame):
    """Write a frame to the streaming pipe. Returns the pipe (or None on failure)."""
    if pipe is None:
        return None
    try:
        pipe.stdin.write(frame.tobytes())
        return pipe
    except (BrokenPipeError, IOError) as e:
        print(f"Streaming error: {e}")
        stop_streaming(pipe)
        return None


def stop_streaming(pipe):
    """Gracefully shut down the ffmpeg streaming process."""
    if pipe is None:
        return
    try:
        if pipe.stdin:
            pipe.stdin.close()
    except Exception:
        pass
    try:
        pipe.terminate()
    except Exception:
        pass
    try:
        pipe.wait(timeout=5)
    except subprocess.TimeoutExpired:
        pipe.kill()
        try:
            pipe.wait()
        except Exception:
            pass
    except Exception:
        pass
