"""YT clipper: finds new videos from channel.txt, makes captioned vertical clips,
saves them to Google Drive (kept 30 days) and uploads them to YouTube."""
import datetime as dt
import json
import os
import pathlib
import re
import subprocess
import time
import xml.etree.ElementTree as ET

import requests

# ---------- SETTINGS (safe to edit) ----------
MAX_AGE_DAYS = 3          # only look at videos newer than this
MAX_VIDEOS_PER_RUN = 2    # videos processed per run
CLIPS_PER_VIDEO = 3       # clips made per video
MAX_UPLOADS_PER_RUN = 5   # YouTube API free quota is about 6 uploads/day
MAX_VIDEO_MINUTES = 60    # skip videos longer than this
MIN_VIDEO_SECONDS = 180   # skip anything shorter (this skips Shorts)
SAVE_TO_DRIVE = True      # copy every clip to Google Drive (needs the Google keys)
DRIVE_FOLDER = "YT Clips" # folder name in your Drive (the script creates it)
DRIVE_KEEP_DAYS = 30      # clips older than this are deleted from Drive
UPLOAD_TO_YOUTUBE = True  # upload clips to your YouTube channel (needs the Google keys)
PRIVACY = "private"       # "private", "unlisted" or "public" (new API projects are forced private)
GEMINI_MODEL = "gemini-3.5-flash"          # change here if Google retires it
GEMINI_BACKUP_MODEL = "gemini-3.1-flash-lite"
# ---------------------------------------------

ROOT = pathlib.Path(__file__).parent
WORK = ROOT / "work"
OUT = ROOT / "out"
STATE = ROOT / "processed.json"
COOKIES = WORK / "cookies.txt"
NS = {"a": "http://www.w3.org/2005/Atom", "yt": "http://www.youtube.com/xml/schemas/2015"}

ASS_HEAD = """[Script Info]
ScriptType: v4.00+
PlayResX: 1080
PlayResY: 1920

[V4+ Styles]
Format: Name, Fontname, Fontsize, PrimaryColour, SecondaryColour, OutlineColour, BackColour, Bold, Italic, Underline, StrikeOut, ScaleX, ScaleY, Spacing, Angle, BorderStyle, Outline, Shadow, Alignment, MarginL, MarginR, MarginV, Encoding
Style: Default,DejaVu Sans,80,&H00FFFFFF,&H000000FF,&H00000000,&H64000000,-1,0,0,0,100,100,0,0,1,7,2,2,60,60,480,1

[Events]
Format: Layer, Start, End, Style, Name, MarginL, MarginR, MarginV, Effect, Text
"""


def load_channels():
    channels = []
    f = ROOT / "channel.txt"
    for line in f.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if not line or line.startswith("#"):
            continue
        parts = [p.strip() for p in line.split("|")]
        if len(parts) < 2 or not parts[0].startswith(("UC", "@")):
            print(f"SKIPPED (needs: @handle | name | optional note): {line}")
            continue
        channels.append({"id": parts[0], "name": parts[1]})
    return channels


def resolve_channel_id(ref):
    """Turns @handle into a channel ID (UC...). Prints the channel title so you can check it."""
    if ref.startswith("UC"):
        return ref
    r = requests.get(
        f"https://www.youtube.com/{ref}",
        headers={"User-Agent": "Mozilla/5.0", "Accept-Language": "en-US,en;q=0.9"},
        cookies={"CONSENT": "YES+1", "SOCS": "CAI"},
        timeout=30,
    )
    r.raise_for_status()
    m = re.search(r'"channelId":"(UC[\w-]{22})"', r.text) or re.search(
        r'youtube\.com/channel/(UC[\w-]{22})', r.text
    )
    if not m:
        raise ValueError("channel not found")
    t = re.search(r'<meta property="og:title" content="([^"]*)"', r.text)
    print(f"Resolved {ref} -> {m.group(1)} (channel title found: {t.group(1) if t else '?'})")
    return m.group(1)


def new_videos(channels, done):
    found = []
    cutoff = dt.datetime.now(dt.timezone.utc) - dt.timedelta(days=MAX_AGE_DAYS)
    for ch in channels:
        try:
            ch_id = resolve_channel_id(ch["id"])
            r = requests.get(
                f"https://www.youtube.com/feeds/videos.xml?channel_id={ch_id}", timeout=30
            )
            r.raise_for_status()
            root = ET.fromstring(r.content)
        except Exception as e:
            print(f"Could not read feed for {ch['name']}: {e}")
            continue
        for e in root.findall("a:entry", NS):
            vid = e.find("yt:videoId", NS).text
            title = e.find("a:title", NS).text
            published = dt.datetime.fromisoformat(e.find("a:published", NS).text)
            if vid in done or published < cutoff:
                continue
            found.append({"id": vid, "title": title, "channel": ch["name"], "published": published})
    found.sort(key=lambda v: v["published"], reverse=True)
    return found


def ytdlp(args):
    cmd = ["yt-dlp"] + (["--cookies", str(COOKIES)] if COOKIES.exists() else []) + args
    return subprocess.run(cmd, capture_output=True, text=True)


def explain_block(err):
    low = (err or "").lower()
    if "sign in" in low or "bot" in low or "429" in low or "403" in low:
        print("YOUTUBE IS BLOCKING THE GITHUB SERVER (it thinks this is a bot). "
              "Nothing is broken in your setup. The script will retry on the next run. "
              "The real fix is the optional YT_COOKIES secret.")
    else:
        print("yt-dlp said:", (err or "").strip()[-400:])


def get_duration(vid):
    """Video length in seconds, or None if YouTube didn't answer."""
    r = ytdlp(["--no-playlist", "--skip-download", "--print", "%(duration)s",
               f"https://www.youtube.com/watch?v={vid}"])
    try:
        return int(float(r.stdout.strip().splitlines()[-1]))
    except (ValueError, IndexError):
        explain_block(r.stderr)
        return None


def download(vid):
    target = WORK / f"{vid}.mp4"
    if target.exists():
        return target
    r = ytdlp([
        "-f", "bv*[height<=720]+ba/b[height<=720]",
        "--merge-output-format", "mp4",
        "--no-playlist",
        "-o", str(target),
        f"https://www.youtube.com/watch?v={vid}",
    ])
    if not target.exists():
        explain_block(r.stderr)
        return None
    return target


def transcribe(path):
    from faster_whisper import WhisperModel

    model = WhisperModel("base", device="cpu", compute_type="int8")
    segments, _ = model.transcribe(str(path), word_timestamps=True, vad_filter=True)
    words, lines = [], []
    for s in segments:
        lines.append(f"[{int(s.start)}] {s.text.strip()}")
        for w in s.words or []:
            words.append({"s": w.start, "e": w.end, "t": w.word.strip()})
    return words, "\n".join(lines)


def pick_clips(title, transcript):
    key = os.environ.get("GEMINI_API_KEY")
    if not key:
        raise SystemExit("GEMINI_API_KEY secret is missing")
    prompt = (
        f"Video title: {title}\n"
        f"Below is a transcript. Each line starts with [seconds from video start].\n"
        f"Pick the {CLIPS_PER_VIDEO} best self-contained moments for short vertical clips "
        f"(funny, surprising, insightful or emotional). Each clip must be 25 to 55 seconds, "
        f"start at the beginning of a sentence and end at the end of a sentence, "
        f"and make sense without the rest of the video.\n"
        f'Reply ONLY with a JSON array like [{{"start": 120.0, "end": 160.0, "title": "short honest title under 80 characters"}}]. '
        f"Titles must accurately describe the clip, no fake claims.\n\n"
        f"TRANSCRIPT:\n{transcript}"
    )
    r = None
    for model in (GEMINI_MODEL, GEMINI_BACKUP_MODEL):
        url = f"https://generativelanguage.googleapis.com/v1beta/models/{model}:generateContent"
        r = requests.post(
            url,
            headers={"x-goog-api-key": key, "Content-Type": "application/json"},
            json={
                "contents": [{"parts": [{"text": prompt}]}],
                "generationConfig": {"responseMimeType": "application/json", "temperature": 0.4},
            },
            timeout=180,
        )
        if r.status_code == 200:
            break
        print(f"Gemini model {model} failed ({r.status_code}): {r.text[:300]}")
    if r is None or r.status_code != 200:
        raise RuntimeError("Gemini failed with both models (check GEMINI_API_KEY and the model names at the top)")
    text = r.json()["candidates"][0]["content"]["parts"][0]["text"]
    text = re.sub(r"^```(?:json)?|```$", "", text.strip()).strip()
    data = json.loads(text)
    picks = []
    for p in data:
        try:
            s, e = float(p["start"]), float(p["end"])
        except (KeyError, TypeError, ValueError):
            continue
        if 20 <= e - s <= 60:
            picks.append({"start": s, "end": e, "title": str(p.get("title", "Clip"))[:90]})
    return picks


def ts(t):
    t = max(0.0, t)
    return f"{int(t // 3600)}:{int(t % 3600 // 60):02d}:{t % 60:05.2f}"


def build_ass(ws, start):
    events = []
    for i in range(0, len(ws), 3):
        g = ws[i:i + 3]
        a = g[0]["s"] - start
        own_end = g[-1]["e"] - start
        b = (ws[i + 3]["s"] - start) if i + 3 < len(ws) else own_end
        b = max(min(b, own_end + 0.6), a + 0.3)
        text = " ".join(w["t"] for w in g).replace("{", "").replace("}", "").upper()
        events.append(f"Dialogue: 0,{ts(a)},{ts(b)},Default,,0,0,0,,{text}")
    return ASS_HEAD + "\n".join(events) + "\n"


def make_clip(src, words, start, end, name):
    # snap the cut to real word boundaries
    ws = [w for w in words if w["s"] >= start - 0.1 and w["e"] <= end + 0.1]
    if len(ws) < 8:
        return None
    start = max(0.0, ws[0]["s"] - 0.2)
    end = ws[-1]["e"] + 0.4
    if end - start > 59:
        end = start + 59
    ass = WORK / f"{name}.ass"
    ass.write_text(build_ass(ws, start), encoding="utf-8")
    out = OUT / f"{name}.mp4"
    vf = f"crop=trunc(ih*9/32)*2:ih,scale=1080:1920,subtitles={ass.name}"
    cmd = [
        "ffmpeg", "-y", "-ss", f"{start:.2f}", "-t", f"{end - start:.2f}",
        "-i", str(src.resolve()), "-vf", vf,
        "-c:v", "libx264", "-preset", "veryfast", "-crf", "23",
        "-c:a", "aac", "-b:a", "128k", "-movflags", "+faststart",
        str(out.resolve()),
    ]
    subprocess.run(cmd, cwd=WORK, check=True)
    return out


_token = {"value": None, "until": 0.0}


def google_token():
    """Google login token; refreshed automatically when it is about to expire."""
    if _token["value"] and time.time() < _token["until"]:
        return _token["value"]
    r = requests.post(
        "https://oauth2.googleapis.com/token",
        data={
            "client_id": os.environ["YT_CLIENT_ID"],
            "client_secret": os.environ["YT_CLIENT_SECRET"],
            "refresh_token": os.environ["YT_REFRESH_TOKEN"],
            "grant_type": "refresh_token",
        },
        timeout=60,
    )
    if r.status_code != 200:
        print("Google login failed:", r.text[:300])
        r.raise_for_status()
    j = r.json()
    _token["value"] = j["access_token"]
    _token["until"] = time.time() + int(j.get("expires_in", 3600)) - 300
    return _token["value"]


def yt_upload(path, title, desc):
    meta = {
        "snippet": {"title": title[:100], "description": desc, "categoryId": "22"},
        "status": {"privacyStatus": PRIVACY, "selfDeclaredMadeForKids": False},
    }
    init = requests.post(
        "https://www.googleapis.com/upload/youtube/v3/videos?uploadType=resumable&part=snippet,status",
        headers={
            "Authorization": f"Bearer {google_token()}",
            "Content-Type": "application/json; charset=UTF-8",
            "X-Upload-Content-Type": "video/mp4",
        },
        json=meta,
        timeout=60,
    )
    if init.status_code != 200:
        print("YouTube upload start failed:", init.text[:500])
        init.raise_for_status()
    with open(path, "rb") as f:
        r = requests.put(
            init.headers["Location"], data=f, headers={"Content-Type": "video/mp4"}, timeout=900
        )
    r.raise_for_status()
    return r.json()["id"]


def drive_setup():
    """Finds or creates the Drive folder and deletes clips older than DRIVE_KEEP_DAYS."""
    h = {"Authorization": f"Bearer {google_token()}"}
    q = (f"name='{DRIVE_FOLDER}' and mimeType='application/vnd.google-apps.folder' "
         f"and trashed=false")
    r = requests.get("https://www.googleapis.com/drive/v3/files", headers=h,
                     params={"q": q, "fields": "files(id)"}, timeout=60)
    r.raise_for_status()
    found = r.json().get("files", [])
    if found:
        folder = found[0]["id"]
    else:
        r = requests.post("https://www.googleapis.com/drive/v3/files", headers=h, timeout=60,
                          json={"name": DRIVE_FOLDER, "mimeType": "application/vnd.google-apps.folder"})
        r.raise_for_status()
        folder = r.json()["id"]
        print(f"Created Drive folder '{DRIVE_FOLDER}'")
    cutoff = (dt.datetime.now(dt.timezone.utc) - dt.timedelta(days=DRIVE_KEEP_DAYS)).strftime(
        "%Y-%m-%dT%H:%M:%SZ")
    r = requests.get("https://www.googleapis.com/drive/v3/files", headers=h, timeout=60, params={
        "q": f"'{folder}' in parents and createdTime < '{cutoff}' and trashed=false",
        "fields": "files(id,name)", "pageSize": 100})
    r.raise_for_status()
    old = r.json().get("files", [])
    for f in old:
        requests.delete(f"https://www.googleapis.com/drive/v3/files/{f['id']}", headers=h, timeout=60)
    if old:
        print(f"Deleted {len(old)} clip(s) older than {DRIVE_KEEP_DAYS} days from Drive")
    return folder


def drive_upload(path, folder):
    h = {"Authorization": f"Bearer {google_token()}",
         "Content-Type": "application/json; charset=UTF-8",
         "X-Upload-Content-Type": "video/mp4"}
    init = requests.post(
        "https://www.googleapis.com/upload/drive/v3/files?uploadType=resumable",
        headers=h, json={"name": path.name, "parents": [folder]}, timeout=60)
    if init.status_code != 200:
        print("Drive upload start failed:", init.text[:500])
        init.raise_for_status()
    with open(path, "rb") as f:
        r = requests.put(init.headers["Location"], data=f,
                         headers={"Content-Type": "video/mp4"}, timeout=900)
    r.raise_for_status()
    return r.json()["id"]


def main():
    WORK.mkdir(exist_ok=True)
    OUT.mkdir(exist_ok=True)
    done = set(json.loads(STATE.read_text())) if STATE.exists() else set()
    if os.environ.get("YT_COOKIES"):
        COOKIES.write_text(os.environ["YT_COOKIES"])

    manual = (os.environ.get("VIDEO_URL") or "").strip()
    has_google = all(
        os.environ.get(k) for k in ("YT_CLIENT_ID", "YT_CLIENT_SECRET", "YT_REFRESH_TOKEN")
    )
    use_drive = has_google and SAVE_TO_DRIVE
    can_upload = has_google and UPLOAD_TO_YOUTUBE and not manual
    report = []

    try:
        if manual:
            m = re.search(r"(?:v=|youtu\.be/|shorts/)([\w-]{11})", manual)
            if not m:
                raise SystemExit("Could not find a video ID in that link")
            videos = [{"id": m.group(1), "title": "Test video", "channel": "manual test"}]
        else:
            videos = new_videos(load_channels(), done)
        print(f"{len(videos)} new video(s) found. "
              f"Save to Drive: {'ON' if use_drive else 'OFF'}. "
              f"Upload to YouTube: {'ON' if can_upload else 'OFF'}")

        uploads, folder = 0, None
        if use_drive:
            try:
                folder = drive_setup()
            except Exception as e:
                print(f"Google Drive is not working, skipping Drive this run: {e}")
                report.append(f"DRIVE PROBLEM: {e}")

        processed, failures = 0, 0
        for v in videos:
            if processed >= MAX_VIDEOS_PER_RUN:
                break
            if failures >= 3:
                print("3 failures in a row, stopping this run. Will retry next run.")
                break
            print(f"\n=== {v['channel']}: {v['title']}")
            try:
                dur = get_duration(v["id"])
                if dur is None:
                    print("Could not read video info. Will retry next run.")
                    report.append(f"INFO FAILED: {v['title']}")
                    failures += 1
                    continue
                if dur < MIN_VIDEO_SECONDS or dur > MAX_VIDEO_MINUTES * 60:
                    print(f"Skipping: length {dur}s is outside the allowed range "
                          f"(Shorts and very long videos are skipped).")
                    if not manual:
                        done.add(v["id"])
                    continue
                src = download(v["id"])
                if not src:
                    print("Download failed. Will retry next run.")
                    report.append(f"DOWNLOAD FAILED: {v['title']}")
                    failures += 1
                    continue
                words, transcript = transcribe(src)
                if len(words) < 50:
                    print("Not enough speech, skipping.")
                    if not manual:
                        done.add(v["id"])
                    continue
                picks = pick_clips(v["title"], transcript)
                for i, p in enumerate(picks[:CLIPS_PER_VIDEO], 1):
                    clip = make_clip(src, words, p["start"], p["end"], f"{v['id']}_{i}")
                    if not clip:
                        continue
                    line = f"CLIP: {clip.name} | {p['title']}"
                    if folder:
                        try:
                            drive_upload(clip, folder)
                            line += " | SAVED TO DRIVE"
                        except Exception as e:
                            line += f" | DRIVE FAILED: {e}"
                    if can_upload and uploads < MAX_UPLOADS_PER_RUN:
                        try:
                            desc = (
                                f'Clip from "{v["title"]}" by {v["channel"]}.\n'
                                f"Original video: https://www.youtube.com/watch?v={v['id']}\n"
                                f"All credit to the original creator.\n\n#Shorts"
                            )
                            new_id = yt_upload(clip, p["title"], desc)
                            uploads += 1
                            line += f" | UPLOADED https://youtu.be/{new_id} ({PRIVACY})"
                        except Exception as e:
                            line += f" | YOUTUBE UPLOAD FAILED: {e}"
                    print(line)
                    report.append(line)
                if not manual:
                    done.add(v["id"])
                processed += 1
                src.unlink(missing_ok=True)
            except Exception as e:
                print(f"Problem with this video, will retry next run: {e}")
                report.append(f"ERROR on {v['title']}: {e}")
                failures += 1
    finally:
        STATE.write_text(json.dumps(sorted(done)))
        summary = os.environ.get("GITHUB_STEP_SUMMARY")
        if summary and report:
            with open(summary, "a", encoding="utf-8") as f:
                f.write("## Clipper report\n" + "\n".join(f"- {r}" for r in report) + "\n")


if __name__ == "__main__":
    main()
