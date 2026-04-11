import json
import os
import platform
import queue
import re
import socket
import subprocess
import threading
import time
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


def _has_videotoolbox() -> bool:
    """Check if ffmpeg was built with VideoToolbox (Apple Silicon HW encoder)."""
    if platform.system() != "Darwin":
        return False
    try:
        result = subprocess.run(
            ["ffmpeg", "-encoders"],
            stdout=subprocess.PIPE, stderr=subprocess.DEVNULL,
            text=True, timeout=5,
        )
        return "h264_videotoolbox" in result.stdout
    except Exception:
        return False


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


# ---------------------------------------------------------------------------
# Audio capture — sounddevice-based (cross-platform, no device names needed)
# ---------------------------------------------------------------------------

class AudioCapture:
    """Captures the system default microphone via sounddevice and serves
    raw PCM over a local TCP socket that ffmpeg reads from.

    sounddevice finds the OS default input device automatically on every
    platform (Windows WASAPI/MME, macOS CoreAudio, Linux ALSA/PulseAudio)
    without needing to know any device names.

    Requires: pip install sounddevice
    """

    SAMPLE_RATE = 44100
    CHANNELS = 1
    BLOCK_SIZE = 1024  # ~23 ms per callback at 44100 Hz

    def __init__(self):
        # Bind first so the port is known before ffmpeg is launched.
        self._server_sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        self._server_sock.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
        self._server_sock.bind(("127.0.0.1", 0))
        self._server_sock.listen(1)
        self._server_sock.settimeout(15)   # ffmpeg must connect within 15 s
        self.port: int = self._server_sock.getsockname()[1]
        self._running = True
        self._thread = threading.Thread(
            target=self._run, daemon=True, name="AudioCapture"
        )
        self._thread.start()

    @property
    def ffmpeg_input_args(self) -> list:
        """ffmpeg input arguments that read our PCM audio stream."""
        return [
            # Both audio and video have independent clocks; each input needs
            # its own enlarged packet queue or ffmpeg silently drops audio.
            "-thread_queue_size", "512",
            "-f", "s16le",                       # signed 16-bit little-endian PCM
            "-ar", str(self.SAMPLE_RATE),
            "-ac", str(self.CHANNELS),
            "-i", f"tcp://127.0.0.1:{self.port}",
        ]

    def stop(self) -> None:
        self._running = False
        try:
            self._server_sock.close()
        except Exception:
            pass
        self._thread.join(timeout=3)

    def _run(self) -> None:
        try:
            import sounddevice as sd
        except ImportError:
            print("AudioCapture: sounddevice not installed.")
            print("  Run: pip install sounddevice")
            return

        try:
            conn, _ = self._server_sock.accept()
        except socket.timeout:
            print("AudioCapture: ffmpeg did not connect within 15 s")
            return
        except OSError:
            return  # stop() was called before ffmpeg connected

        conn.settimeout(2.0)

        # Small bounded queue — drop oldest audio on overflow so we never
        # accumulate latency (live audio: a gap is better than stale data).
        audio_q: queue.Queue = queue.Queue(maxsize=80)

        def _callback(indata, frames, time_info, status):
            if not self._running:
                raise sd.CallbackStop()
            try:
                audio_q.put_nowait(bytes(indata))
            except queue.Full:
                pass

        try:
            with sd.InputStream(
                samplerate=self.SAMPLE_RATE,
                channels=self.CHANNELS,
                dtype="int16",
                blocksize=self.BLOCK_SIZE,
                callback=_callback,
            ):
                while self._running:
                    try:
                        chunk = audio_q.get(timeout=1.0)
                        conn.sendall(chunk)
                    except queue.Empty:
                        continue
                    except (BrokenPipeError, ConnectionResetError, OSError):
                        break
        except Exception as e:
            print(f"AudioCapture error: {e}")
        finally:
            try:
                conn.close()
            except Exception:
                pass


# ---------------------------------------------------------------------------
# Platform-native audio fallback (when sounddevice is not installed)
# ---------------------------------------------------------------------------

def _detect_dshow_device() -> str | None:
    """Return the first DirectShow audio device name found, or None."""
    try:
        result = subprocess.run(
            ["ffmpeg", "-list_devices", "true", "-f", "dshow", "-i", "dummy"],
            stdout=subprocess.DEVNULL, stderr=subprocess.PIPE,
            text=True, timeout=5,
        )
        in_audio = False
        for line in result.stderr.splitlines():
            if "audio" in line.lower() and "devices" in line.lower():
                in_audio = True
                continue
            if not in_audio:
                continue
            if "alternative name" in line.lower():
                continue
            m = re.search(r'"([^"]+)"', line)
            if m:
                return m.group(1)
    except Exception:
        pass
    return None


def _native_audio_args() -> tuple:
    """Best-effort platform-native ffmpeg audio args without sounddevice."""
    sys = platform.system()
    if sys == "Darwin":
        return (
            ["-thread_queue_size", "512", "-f", "avfoundation", "-i", ":0"],
            "mic (avfoundation :0)",
        )
    if sys == "Linux":
        return (
            ["-thread_queue_size", "512", "-f", "pulse", "-i", "default"],
            "mic (PulseAudio default)",
        )
    # Windows — dshow requires a device name
    device = _detect_dshow_device()
    if device:
        return (
            ["-thread_queue_size", "512", "-f", "dshow",
             "-rtbufsize", "100M", "-i", f"audio={device}"],
            f"mic (dshow): {device}",
        )
    return (
        ["-thread_queue_size", "512", "-f", "lavfi",
         "-i", "anullsrc=channel_layout=stereo:sample_rate=44100"],
        "silent — run 'pip install sounddevice' for mic support",
    )


# ---------------------------------------------------------------------------
# Audio input selector
# ---------------------------------------------------------------------------

def _build_audio_input(streaming_cfg: dict) -> tuple:
    """Return (AudioCapture_or_None, ffmpeg_input_args, description)."""
    mic_cfg = streaming_cfg.get("mic_device", "auto")

    if mic_cfg in (None, "none", "silent"):
        return (
            None,
            ["-thread_queue_size", "512", "-f", "lavfi",
             "-i", "anullsrc=channel_layout=stereo:sample_rate=44100"],
            "silent (anullsrc)",
        )

    # sounddevice: cross-platform, uses system default mic, no device names
    try:
        import sounddevice as sd
        dev = sd.query_devices(kind="input")
        capture = AudioCapture()
        return (
            capture,
            capture.ffmpeg_input_args,
            f"mic (sounddevice): {dev['name']}",
        )
    except ImportError:
        print("sounddevice not installed — falling back to platform-native audio.")
        print("  For reliable cross-platform mic: pip install sounddevice")
    except Exception as e:
        print(f"sounddevice unavailable ({e}) — falling back to platform-native audio.")

    args, desc = _native_audio_args()
    return (None, args, desc)


# ---------------------------------------------------------------------------
# Streaming thread
# ---------------------------------------------------------------------------

class Streamer:
    """Wraps an ffmpeg process and a dedicated thread for constant-rate delivery.

    The main loop calls push_frame() freely at whatever rate it runs.
    The internal thread consumes frames at exactly `fps` using a
    high-precision sleep + spin-wait loop, completely independent of the
    display loop timing. If the main loop is faster than fps, extra frames
    are dropped. If slower, the last frame is repeated — both are correct
    for live CBR streaming.
    """

    def __init__(self, pipe: subprocess.Popen, fps: float = 30.0,
                 audio_capture=None):
        self._pipe = pipe
        self._audio_capture = audio_capture
        self._interval = 1.0 / fps
        self._queue: queue.Queue = queue.Queue(maxsize=1)
        self._running = True
        self._thread = threading.Thread(
            target=self._run, daemon=True, name="StreamingThread"
        )
        self._thread.start()

    def push_frame(self, frame) -> None:
        """Non-blocking. Replaces any pending frame with the latest one."""
        if not self._running:
            return
        try:
            self._queue.get_nowait()
        except queue.Empty:
            pass
        try:
            self._queue.put_nowait(frame.tobytes())
        except queue.Full:
            pass

    def stop(self) -> None:
        self._running = False
        self._thread.join(timeout=3)
        _stop_pipe(self._pipe)
        if self._audio_capture is not None:
            self._audio_capture.stop()

    @staticmethod
    def _sleep_until(target: float) -> None:
        """Coarse sleep + spin-wait for sub-millisecond accuracy on Windows."""
        slack = target - time.monotonic()
        if slack > 0.002:
            time.sleep(slack - 0.001)
        while time.monotonic() < target:
            pass

    def _run(self) -> None:
        next_time = time.monotonic() + self._interval
        last_frame_bytes = None

        while self._running:
            self._sleep_until(next_time)
            next_time += self._interval

            now = time.monotonic()
            if next_time < now:
                next_time = now + self._interval

            try:
                last_frame_bytes = self._queue.get_nowait()
            except queue.Empty:
                pass  # repeat last frame to maintain CBR

            if last_frame_bytes is None:
                continue

            try:
                self._pipe.stdin.write(last_frame_bytes)
                self._pipe.stdin.flush()
            except (BrokenPipeError, IOError) as e:
                print(f"Streaming error: {e}")
                self._running = False
                break


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def start_streaming(config, frame_width, frame_height):
    """Start an ffmpeg subprocess and return a Streamer, or None if disabled."""
    streaming_cfg = config.get("streaming", {})
    if not streaming_cfg.get("enabled", False):
        return None

    if not _ffmpeg_available():
        print("Warning: ffmpeg not found — streaming disabled.")
        return None

    youtube_cfg = config.get("youtube") if isinstance(config, dict) else None
    if (not isinstance(youtube_cfg, dict)
            or "stream_url" not in youtube_cfg
            or "stream_key" not in youtube_cfg):
        print("Streaming disabled: missing 'stream_url' or 'stream_key'.")
        return None

    full_url = f"{youtube_cfg['stream_url']}{youtube_cfg['stream_key']}"
    bitrate = streaming_cfg.get("bitrate", "3000k")
    preset = streaming_cfg.get("preset", "ultrafast")
    fps = int(streaming_cfg.get("fps", 30))

    try:
        bufsize = f"{int(bitrate.lower().rstrip('k')) * 2}k"
    except ValueError:
        bufsize = "6000k"

    audio_capture, audio_input_args, audio_desc = _build_audio_input(streaming_cfg)

    use_vtb = _has_videotoolbox()

    if use_vtb:
        video_enc_args = [
            "-c:v", "h264_videotoolbox",
            "-pix_fmt", "yuv420p",
            "-realtime", "1",
            "-b:v", bitrate,
            "-maxrate", bitrate,
            "-bufsize", bufsize,
            "-g", str(fps * 2),
        ]
        encoder_name = "h264_videotoolbox (Apple Silicon GPU)"
    else:
        video_enc_args = [
            "-c:v", "libx264",
            "-pix_fmt", "yuv420p",
            "-preset", preset,
            "-tune", "zerolatency",
            "-b:v", bitrate,
            "-maxrate", bitrate,
            "-bufsize", bufsize,
            "-g", str(fps * 2),
        ]
        encoder_name = f"libx264 (CPU, preset={preset})"

    ffmpeg_cmd = [
        "ffmpeg", "-y",
        "-threads", "0",            # 0 = auto-detect, use all cores
        # Input 0: video from stdin
        "-thread_queue_size", "1024",
        "-f", "rawvideo",
        "-vcodec", "rawvideo",
        "-pix_fmt", "bgr24",
        "-s", f"{frame_width}x{frame_height}",
        "-r", str(fps),
        "-use_wallclock_as_timestamps", "1",
        "-i", "pipe:0",
        # Input 1: audio
        *audio_input_args,
        # Explicit stream mapping
        "-map", "0:v",
        "-map", "1:a",
        # Video encoding (GPU or CPU)
        *video_enc_args,
        "-filter_threads", "2",  # parallel pixel format conversion
        "-vsync", "cfr",
        # Audio encoding
        "-c:a", "aac",
        "-b:a", "128k",
        "-ar", "44100",
        "-flvflags", "no_duration_filesize",
        "-f", "flv",
        full_url,
    ]

    try:
        pipe = subprocess.Popen(ffmpeg_cmd, stdin=subprocess.PIPE, bufsize=0)
        streamer = Streamer(pipe, fps=fps, audio_capture=audio_capture)
        print("YouTube streaming started!")
        print(f"  URL:    {youtube_cfg['stream_url']}")
        print(f"  Encoder: {encoder_name}")
        print(f"  Video:  {frame_width}x{frame_height} @ {fps} fps | {bitrate}")
        print(f"  Audio:  {audio_desc}")
        print("  Tip: Set YouTube Studio latency to 'Ultra Low' to cut ~5-8 s of delay.")
        return streamer
    except Exception as e:
        print(f"Failed to start streaming: {e}")
        if audio_capture is not None:
            audio_capture.stop()
        import traceback
        traceback.print_exc()
        return None


def _stop_pipe(pipe) -> None:
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


# ---------------------------------------------------------------------------
# Legacy shims
# ---------------------------------------------------------------------------

def write_frame(pipe, frame):
    """Deprecated: use Streamer.push_frame() instead.

    Accepts either a raw Popen pipe (legacy) or a Streamer instance so that
    existing callers (e.g. air_canvas_combined.py) work without modification
    after start_streaming() was updated to return a Streamer.
    """
    if pipe is None:
        return None
    if isinstance(pipe, Streamer):
        pipe.push_frame(frame)
        return pipe
    try:
        pipe.stdin.write(frame.tobytes())
        pipe.stdin.flush()
        return pipe
    except (BrokenPipeError, IOError) as e:
        print(f"Streaming error: {e}")
        _stop_pipe(pipe)
        return None


def stop_streaming(pipe_or_streamer):
    """Stop a Streamer or a raw pipe."""
    if pipe_or_streamer is None:
        return
    if isinstance(pipe_or_streamer, Streamer):
        pipe_or_streamer.stop()
    else:
        _stop_pipe(pipe_or_streamer)
