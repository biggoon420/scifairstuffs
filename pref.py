import os
import json
import csv
import random
import shutil
import time
import colorsys
import subprocess
import itertools
import math
import numpy as np
import sounddevice as sd
from pydub import AudioSegment
import tkinter as tk
from tkinter import filedialog, messagebox
from tkinter import font as tkfont
import torch

from PIL import Image, ImageTk, ImageSequence


# ============================================================
# LOAD SILERO VAD MODEL
# ============================================================

sd.default.device = None
sd.default.reset()

STATE_FILE = "session_state.json"

vad_model, vad_utils = torch.hub.load(
    repo_or_dir="snakers4/silero-vad",
    model="silero_vad",
    force_reload=False
)

(get_speech_timestamps,
 read_audio,
 save_audio,
 VADIterator,
 collect_chunks) = vad_utils


# ============================================================
# AUDIO HELPERS
# ============================================================


def get_macos_output_volume():
    try:
        out = subprocess.check_output(
            ["osascript", "-e", "output volume of (get volume settings)"]
        )
        return int(out.strip()) / 100.0
    except Exception:
        return 1.0

def load_audio(path, sr=16000):
    if not hasattr(load_audio, "_cache"):
        load_audio._cache = {}

    if path in load_audio._cache:
        return load_audio._cache[path]

    audio = AudioSegment.from_file(path)
    audio = audio.set_channels(1).set_frame_rate(sr)
    y = np.array(audio.get_array_of_samples()).astype(np.float32) / 32768.0

    load_audio._cache[path] = (y, sr)
    return y, sr



def extract_speech_vad(y, sr):
    audio_t = torch.tensor(y, dtype=torch.float32)
    speech_ts = get_speech_timestamps(audio_t, vad_model, sampling_rate=sr)

    if not speech_ts:
        return None

    chunks = [y[ts["start"]:ts["end"]] for ts in speech_ts]
    return np.concatenate(chunks)
    
def top_k_segments(y, sr, seconds=10, k=5):

    target_len = seconds * sr

    speech = extract_speech_vad(y, sr)
    if speech is None or len(speech) < target_len:
        return [y[:target_len]]

    step = target_len   # jump by full segment length

    scored = []

    for i in range(0, len(speech) - target_len, step):
        seg = speech[i:i + target_len]
        score = float(np.sum(np.abs(seg)))
        scored.append((score, seg))

    if not scored:
        return [speech[:target_len]]

    scored.sort(reverse=True, key=lambda x: x[0])

    segments = [seg for _, seg in scored[:k]]

    return segments

def num_chunks_for_duration(duration_sec):
    n = int(2 + 1.5 * math.log2(max(duration_sec, 2)))
    return max(2, min(n, 7))


def best_10_seconds(y, sr, seconds=10):
    target_len = seconds * sr
    speech_audio = extract_speech_vad(y, sr)

    if speech_audio is None:
        return y[:target_len]

    if len(speech_audio) <= target_len:
        return speech_audio

    best_sum = -1
    best_idx = 0
    step = sr // 2

    for i in range(0, len(speech_audio) - target_len, step):
        chunk = speech_audio[i:i + target_len]
        score = np.sum(np.abs(chunk))
        if score > best_sum:
            best_sum, best_idx = score, i

    return speech_audio[best_idx: best_idx + target_len]


# ============================================================
# ROUNDED BUTTON CLASS
# ============================================================

class RoundedButton(tk.Canvas):
    def __init__(
        self, parent, text, command=None,
        radius=10, padding_x=20, padding_y=10,
        bg_color="#E36D5A", fg_color="black",
        font=("Arial", 12, "bold"),
        border_color="#F3D4CF",
        border_width=4
    ):
        self.text = text
        self.command = command
        self.radius = radius
        self.bg_color = bg_color
        self.fg_color = fg_color
        self.font = font
        self.border_color = border_color
        self.border_width = border_width

        fnt = tkfont.Font(font=font)
        text_w = fnt.measure(text)
        text_h = fnt.metrics("linespace")

        width = text_w + 2 * padding_x
        height = text_h + 2 * padding_y

        super().__init__(
            parent,
            width=width,
            height=height,
            highlightthickness=0,
            bd=0
        )

        self._draw_button()
        self.bind("<Button-1>", lambda e: self.command() if self.command else None)
        self.bind("<Enter>", lambda e: self._hover(True))
        self.bind("<Leave>", lambda e: self._hover(False))

    def _round_rect(self, x1, y1, x2, y2, r, **kwargs):
        pts = [
            x1 + r, y1,
            x2 - r, y1,
            x2, y1, x2, y1 + r,
            x2, y2 - r,
            x2, y2, x2 - r, y2,
            x1 + r, y2,
            x1, y2, x1, y2 - r,
            x1, y1 + r,
            x1, y1
        ]
        return self.create_polygon(pts, smooth=True, **kwargs)

    def _draw_button(self):
        w = int(self.cget("width"))
        h = int(self.cget("height"))
        r = self.radius

        self.rect = self._round_rect(
            1, 1, w - 1, h - 1, r,
            fill=self.bg_color,
            outline=self.border_color,
            width=self.border_width
        )
        self.txt = self.create_text(
            w // 2, h // 2,
            text=self.text,
            fill=self.fg_color,
            font=self.font
        )

    def _hover(self, state):
        if state:
            self.itemconfigure(self.rect, fill=self._lighten(self.bg_color, 1.08))
        else:
            self.itemconfigure(self.rect, fill=self.bg_color)

    def _lighten(self, color, factor):
        c = color.lstrip("#")
        r = int(c[0:2], 16)
        g = int(c[2:4], 16)
        b = int(c[4:6], 16)
        r = min(int(r * factor), 255)
        g = min(int(g * factor), 255)
        b = min(int(b * factor), 255)
        return f"#{r:02x}{g:02x}{b:02x}"

    def set_fill(self, color):
        self.bg_color = color
        self.itemconfigure(self.rect, fill=color)


# ============================================================
# MAIN APPLICATION
# ============================================================

class PreferenceApp:

    def __init__(self, root):
        self.has_started = False

        self.root = root
        root.title("Audio Preference Trainer")
        self.base_volume = 0.6  # your app’s max loudness
        self.all_pairs_mode = False
        menubar = tk.Menu(root)
        self.chunk_cache = {}
        self.current_chunk_info = {}


        # App menu (macOS requires at least one)
        app_menu = tk.Menu(menubar, tearoff=0)
        app_menu.add_command(label="Quit", command=root.quit)
        menubar.add_cascade(label="App", menu=app_menu)

        # Mode menu
        mode_menu = tk.Menu(menubar, tearoff=0)
        mode_menu.add_checkbutton(
            label="All-Pairs Mode (n choose 2)",
            command=self.toggle_all_pairs_mode
        )
        menubar.add_cascade(label="Mode", menu=mode_menu)

        root.config(menu=menubar)
        self.root.bind("1", lambda e: self.next_segment(0))  # first audio
        self.root.bind("2", lambda e: self.next_segment(1))  # second audio
        self.root.bind("<Shift-E>", lambda e: self.force_export())





        self.COL_SALMON = "#EA8C7B"
        self.COL_PEACH = "#F6B29A"
        self.COL_BLUSH = "#F3D4CF"
        self.COL_PERI = "#9DA8C8"
        self.COL_DARK = "#2F2F2F"

        self.source_folder = None
        self.folder = None
        self.pool = []
        self.current_pair = None
        self.cache = {}
        self.results = []
        self.index = 0
        self.total_pairs = 0

        self.playing = False
        self.play_start = None
        self.play_duration = None
        self.playhead_line = None

        self.debug_open = False
        self.debug_log = []
        self.fun_mode = False
        self.fun_hue = 0

        self.bg_frames = []
        self.bg_index = 0
        self.bg_tkimg = None
        self.bg_image_id = None

        try:
            root.state("zoomed")
        except Exception:
            root.geometry("1200x800")

        self.canvas = tk.Canvas(root, highlightthickness=0, bd=0, bg=self.COL_BLUSH)
        self.canvas.pack(fill="both", expand=True)

        self._load_or_convert_gif()

        self.text_items = {}

        self._create_text_box(
            "title",
            "Audio Preference Trainer",
            ("Arial", 36, "bold"),
            self.COL_DARK
        )
        self._create_text_box(
            "subtitle",
            "Pairwise 10-second VAD-trimmed comparisons",
            ("Arial", 16),
            self.COL_DARK
        )
        self._create_text_box(
            "folder",
            "No source folder selected",
            ("Arial", 14),
            self.COL_DARK
        )
        self._create_text_box(
            "subset_label",
            "Create subset of",
            ("Arial", 12),
            self.COL_DARK
        )
        self._create_text_box(
            "files_label",
            "files",
            ("Arial", 12),
            self.COL_DARK
        )
        self._create_text_box(
            "progress",
            "Progress: 0/0",
            ("Arial", 16),
            self.COL_DARK
        )

        self.num_entry = tk.Entry(
            root,
            width=6,
            justify="center",
            font=("Arial", 12)
        )

        self.choose_btn = RoundedButton(
            root,
            "Choose Folder",
            command=self._wrap(self.choose_source_folder),
            radius=10,
            bg_color=self.COL_PERI,
            fg_color="white",
            font=("Arial", 12, "bold"),
            border_color=self.COL_BLUSH,
            border_width=4
        )

        self.make_subset_btn = RoundedButton(
            root,
            "Create Subset",
            command=self._wrap(self.create_subset),
            radius=10,
            bg_color=self.COL_PERI,
            fg_color="white",
            font=("Arial", 12, "bold"),
            border_color=self.COL_BLUSH,
            border_width=4
        )

        self.start_btn = RoundedButton(
            root,
            "Start Training",
            command=self._wrap(self.start),
            radius=10,
            padding_x=40,
            padding_y=12,
            bg_color=self.COL_SALMON,
            fg_color="black",
            font=("Arial", 18, "bold"),
            border_color=self.COL_BLUSH,
            border_width=4
        )

        self.play1 = RoundedButton(
            root, "▶ Play First",
            command=self._wrap(self.play_first),
            radius=10,
            bg_color=self.COL_PERI,
            fg_color="white",
            font=("Arial", 14, "bold"),
            border_color=self.COL_BLUSH,
            border_width=4
        )

        self.play2 = RoundedButton(
            root, "▶ Play Second",
            command=self._wrap(self.play_second),
            radius=10,
            bg_color=self.COL_PERI,
            fg_color="white",
            font=("Arial", 14, "bold"),
            border_color=self.COL_BLUSH,
            border_width=4
        )

        self.skip1 = RoundedButton(
            root, "Skip First",
            command=self._wrap(lambda: self.skip(0)),
            radius=10,
            bg_color=self.COL_PEACH,
            fg_color="black",
            font=("Arial", 12, "bold"),
            border_color=self.COL_BLUSH,
            border_width=4
        )

        self.skip2 = RoundedButton(
            root, "Skip Second",
            command=self._wrap(lambda: self.skip(1)),
            radius=10,
            bg_color=self.COL_PEACH,
            fg_color="black",
            font=("Arial", 12, "bold"),
            border_color=self.COL_BLUSH,
            border_width=4
        )

        PREF_COLOR = self.COL_SALMON

        self.vote1 = RoundedButton(
            root, "FIRST IS BETTER",
            command=self._wrap(lambda: self.vote(1)),
            radius=10, padding_x=50, padding_y=15,
            bg_color=PREF_COLOR,
            fg_color="black",
            font=("Arial", 20, "bold"),
            border_color=self.COL_BLUSH,
            border_width=4
        )

        self.vote2 = RoundedButton(
            root, "SECOND IS BETTER",
            command=self._wrap(lambda: self.vote(2)),
            radius=10, padding_x=50, padding_y=15,
            bg_color=PREF_COLOR,
            fg_color="black",
            font=("Arial", 20, "bold"),
            border_color=self.COL_BLUSH,
            border_width=4
        )

        self.waveform_canvas = tk.Canvas(
            root,
            height=120,
            bg=self.COL_PERI,
            highlightthickness=0,
            bd=0
        )

        self.debug_btn = RoundedButton(
            root, "Debug Panel",
            command=self._wrap(self.toggle_debug),
            radius=10,
            bg_color="#555577",
            fg_color="white",
            font=("Arial", 11, "bold"),
            border_color=self.COL_BLUSH,
            border_width=4
        )

        self.fun_btn = RoundedButton(
            root, "fun mode",
            command=self._wrap(self.toggle_fun_mode),
            radius=10,
            bg_color="#333333",
            fg_color="white",
            font=("Arial", 11, "bold"),
            border_color=self.COL_BLUSH,
            border_width=4
        )

        self.debug_panel = tk.Frame(root, bg="#1b1b29")
        self.debug_label = tk.Label(
            self.debug_panel,
            text="",
            fg="#A8FFB0",
            bg="#1b1b29",
            font=("Consolas", 11),
            justify="left"
        )
        self.debug_label.pack(padx=10, pady=5)

        self.debug_panel_window = None
        self.win_items = {}

        self._relayout()
        self.root.bind("<Configure>", lambda e: self._relayout())
        self.root.bind("<Command-Shift-f>", lambda e: self.finish_early())


        self.root.after(40, self._update_playhead)
        self.root.after(22, self._update_fun_mode)
        self.load_state()
        self.debug("Shortcut: ⌘ + Shift + F → Finish now")


    # ========================================================
    # TEXT BOX HELPERS
    # ========================================================
    def toggle_all_pairs_mode(self):
        self.all_pairs_mode = not self.all_pairs_mode

        mode = "ALL-PAIRS" if self.all_pairs_mode else "TOURNAMENT"
        self.debug(f"Mode switched to {mode}")

        messagebox.showinfo(
        "Mode Changed",
        f"Comparison mode set to:\n\n{mode}"
        )
    def force_export(self):
        if not self.results:
            messagebox.showinfo("Nothing to export", "No data to export yet.")
            return

    # CSV
        with open("preferences.csv", "w", newline="") as f:
            w = csv.writer(f)
            w.writerow(["file1", "file2", "choice"])
            w.writerows(self.results)

    # JSONL
        with open("preference_data.jsonl", "w") as f:
            for f1, f2, c in self.results:
                if c == 1:
                    f.write(json.dumps({"winner": f1, "loser": f2}) + "\n")
                else:
                    f.write(json.dumps({"winner": f2, "loser": f1}) + "\n")

        messagebox.showinfo("Exported", "Data exported to CSV and JSONL.")
        
    def _precompute_chunks(self, files):
        for f in files:
            if f not in self.chunk_cache:
                try:
                    self.extract_chunks(f)
                except Exception as e:
                    self.debug(f"Chunk precompute failed: {os.path.basename(f)}")

    def extract_chunks(self, path):
        if path in self.chunk_cache:
            return self.chunk_cache[path]

        audio, sr = load_audio(path)
        duration = len(audio) / sr
        k = min(12, num_chunks_for_duration(duration))


        speech_ts = get_speech_timestamps(
            audio, vad_model, sampling_rate=sr
        )
    
        chunks = [
            audio[t["start"]:t["end"]]
            for t in speech_ts
            if (t["end"] - t["start"]) / sr >= 1.0
        ]

        random.shuffle(chunks)
        chunks = chunks[:k]

        self.chunk_cache[path] = {
            "sr": sr,
            "chunks": chunks
        }
        return self.chunk_cache[path]
    def get_random_chunk(self, path):
        path = self._normalize_path(path)
        data = self.extract_chunks(path)

        if not data["chunks"]:
            return None, None, None, None

        idx = random.randrange(len(data["chunks"]))
        return (
            data["chunks"][idx],
            data["sr"],
            idx + 1,
            len(data["chunks"])
        )

    def _round_rect(self, x1, y1, x2, y2, r, **kwargs):
        pts = [
            x1 + r, y1,
            x2 - r, y1,
            x2, y1, x2, y1 + r,
            x2, y2 - r,
            x2, y2, x2 - r, y2,
            x1 + r, y2,
            x1, y2, x1, y2 - r,
            x1, y1 + r,
            x1, y1
        ]
        return self.canvas.create_polygon(pts, smooth=True, **kwargs)

    def _create_text_box(self, name, text, font, fill):
        text_id = self.canvas.create_text(
            0, 0,
            text=text,
            font=font,
            fill=fill,
            anchor="n"
        )
        rect_id = self.canvas.create_polygon(0, 0, 0, 0, fill=self.COL_BLUSH, outline=self.COL_BLUSH, width=4)
        self.canvas.tag_lower(rect_id, text_id)
        self.text_items[name] = {
            "text": text_id,
            "rect": rect_id,
            "font": font,
            "fill": fill
        }

    def _move_text_box(self, name, x, y, pad_x=16, pad_y=8, radius=14):
        item = self.text_items[name]
        text_id = item["text"]
        rect_id = item["rect"]

        self.canvas.coords(text_id, x, y)
        bbox = self.canvas.bbox(text_id)
        if not bbox:
            return
        x1, y1, x2, y2 = bbox
        x1 -= pad_x
        x2 += pad_x
        y1 -= pad_y
        y2 += pad_y

        self.canvas.delete(rect_id)
        rect_id_new = self._round_rect(
            x1, y1, x2, y2, radius,
            fill=self.COL_BLUSH,
            outline=self.COL_BLUSH,
            width=4
        )
        self.canvas.tag_lower(rect_id_new, text_id)
        item["rect"] = rect_id_new

    def _set_text(self, name, text):
        item = self.text_items[name]
        self.canvas.itemconfigure(item["text"], text=text)

    # ========================================================
    # LAYOUT
    # ========================================================
    def save_state(self):
        if not self.current_pair:
            return

        state = {
            "source_folder": self.source_folder,
            "folder": self.folder,
            "pool": self.pool,
            "current_pair": self.current_pair,
            "results": self.results,
            "index": self.index,
            "total_pairs": self.total_pairs,
        }

        with open(STATE_FILE, "w") as f:
            json.dump(state, f)

        self.debug("Session saved")


    def load_state(self):
        if not os.path.exists(STATE_FILE):
            return False

        with open(STATE_FILE, "r") as f:
            state = json.load(f)

        self.source_folder = state["source_folder"]
        self.folder = state["folder"]
        self.pool = state["pool"]
        self.current_pair = tuple(state["current_pair"])
        self.results = state["results"]
        self.index = state["index"]
        self.total_pairs = state["total_pairs"]

        self.cache = {}

        self._set_text("folder", f"Subset: {self.folder}")
        self._set_text("progress", f"Progress: {self.index}/{self.total_pairs}")
        self.waveform_canvas.delete("all")
    
        self.debug("Session restored")
        return True


    def finish_early(self):
        if not self.results:
            messagebox.showinfo("Nothing to save", "No comparisons made yet.")
            return

        self.debug("FORCED FINISH")
        self.finish()


    def _place_widget(self, name, widget, x, y, anchor="n"):
        if name in self.win_items:
            self.canvas.coords(self.win_items[name], x, y)
        else:
            self.win_items[name] = self.canvas.create_window(
                x, y, window=widget, anchor=anchor
            )

    def _relayout(self):
        w = self.canvas.winfo_width()
        h = self.canvas.winfo_height()
        if w <= 1 or h <= 1:
            return

        center_x = w // 2
        y = 40

        self._move_text_box("title", center_x, y)
        y += 60

        self._move_text_box("subtitle", center_x, y)
        y += 50

        self._move_text_box("folder", center_x - 22, y)
        self._place_widget("choose_btn", self.choose_btn, center_x + 200, y - 8, anchor="n")
        y += 50

        subset_y = y
        self._move_text_box("subset_label", center_x - 80, subset_y)
        self._place_widget("num_entry", self.num_entry, center_x + 18, subset_y - 8, anchor="n")
        self._move_text_box("files_label", center_x + 70, subset_y)
        self._place_widget("make_subset_btn", self.make_subset_btn, center_x + 200, subset_y - 8, anchor="n")
        y += 70

        self._place_widget("start_btn", self.start_btn, center_x, y, anchor="n")
        y += 90

        self._place_widget("play1", self.play1, center_x - 200, y, anchor="n")
        self._place_widget("play2", self.play2, center_x + 200, y, anchor="n")
        y += 60

        self._place_widget("skip1", self.skip1, center_x - 200, y, anchor="n")
        self._place_widget("skip2", self.skip2, center_x + 200, y, anchor="n")
        y += 90

        self._place_widget("vote1", self.vote1, center_x - 200, y, anchor="n")
        self._place_widget("vote2", self.vote2, center_x + 200, y, anchor="n")
        y += 110

        self._move_text_box("progress", center_x, y)
        y += 40

        wf_width = int(w * 0.9)
        self.waveform_canvas.config(width=wf_width)
        self._place_widget("waveform", self.waveform_canvas, center_x, y + 80, anchor="center")

        bottom_y = h - 40
        self._place_widget("debug_btn", self.debug_btn, 80, bottom_y, anchor="w")
        self._place_widget("fun_btn", self.fun_btn, w - 80, bottom_y, anchor="e")

        if self.debug_open:
            if self.debug_panel_window is None:
                self.debug_panel_window = self.canvas.create_window(
                    center_x, h - 140, window=self.debug_panel, anchor="n"
                )
            else:
                self.canvas.coords(self.debug_panel_window, center_x, h - 140)

    # ========================================================
    # GIF BACKGROUND
    # ========================================================

    def _load_or_convert_gif(self):
        if os.path.exists("bg.gif"):
            try:
                self.bg_frames = [
                    frame.copy().convert("RGBA")
                    for frame in ImageSequence.Iterator(Image.open("bg.gif"))
                ]
                self.bg_index = 0
                return
            except Exception as e:
                print("GIF load failed:", e)

        if os.path.exists("background.mp4"):
            try:
                subprocess.run(
                    ["ffmpeg", "-y", "-i", "background.mp4",
                     "-vf", "fps=12,scale=1280:-1:flags=lanczos",
                     "bg.gif"],
                    check=True
                )
                gif = Image.open("bg.gif")
                self.bg_frames = [
                    frame.copy().convert("RGBA")
                    for frame in ImageSequence.Iterator(gif)
                ]
                self.bg_index = 0
                return
            except Exception as e:
                print("Auto conversion failed:", e)

        self.bg_frames = []
        self.bg_index = 0

    def _play_gif_background(self):
        if not self.fun_mode or not self.bg_frames:
            return

        w = self.canvas.winfo_width()
        h = self.canvas.winfo_height()
        if w < 10 or h < 10:
            self.root.after(70, self._play_gif_background)
            return

        frame = self.bg_frames[self.bg_index].resize((w, h), Image.LANCZOS)
        self.bg_tkimg = ImageTk.PhotoImage(frame)

        if self.bg_image_id is None:
            self.bg_image_id = self.canvas.create_image(
                0, 0, image=self.bg_tkimg, anchor="nw"
            )
        else:
            self.canvas.itemconfigure(self.bg_image_id, image=self.bg_tkimg)

        self.canvas.tag_lower(self.bg_image_id)

        self.bg_index = (self.bg_index + 1) % len(self.bg_frames)
        self.root.after(70, self._play_gif_background)

    # ========================================================
    # GENERAL HELPERS
    # ========================================================

    def _wrap(self, f):
        def inner(*args, **kwargs):
            self.stop_audio()
            return f(*args, **kwargs)
        return inner
    def reset_audio(self):
        try:
            sd.stop()
            sd.default.device = None  # force system default output
        except Exception as e:
            self.debug(f"Audio reset failed: {e}")


    # ========================================================
    # FOLDER / SUBSET
    # ========================================================

    def choose_source_folder(self):
        folder = filedialog.askdirectory()
        if folder:
            self.source_folder = folder
            self._set_text("folder", f"Source: {folder}")
            self.debug(f"Selected folder: {folder}")

    def create_subset(self):
        if not self.source_folder:
            messagebox.showerror("Error", "Pick a folder first.")
            return

        try:
            n = int(self.num_entry.get())
        except Exception:
            messagebox.showerror("Error", "Invalid number.")
            return

        EXT = (".mp3", ".wav", ".m4a", ".flac", ".aac", ".ogg")

        files = [
            os.path.join(self.source_folder, f)
            for f in os.listdir(self.source_folder)
            if f.lower().endswith(EXT)
        ]

        if n < 2:
            messagebox.showerror("Error", "N must be ≥ 2.")
            return
        if n > len(files):
            messagebox.showerror("Error", f"Only {len(files)} available.")
            return

        subset = os.path.join(self.source_folder, "training_subset")
        if os.path.exists(subset):
            shutil.rmtree(subset)
        os.makedirs(subset)

        chosen = random.sample(files, n)
        for f in chosen:
            shutil.copy(f, subset)

        self.folder = subset
        self._set_text("folder", f"Subset: {subset}")
        self.debug(f"Created subset: {n} files")

    # ========================================================
    # TRAINING
    # ========================================================
    def reset_progress(self):

        self.results = []
        self.index = 0
        self.total_pairs = 0
        self.current_chunk_info = {}
        self.chunk_cache = {}
        self.waveform_canvas.delete("all")
        self._set_text("progress", "Progress: 0/0")


    def start(self):
        if self.has_started:
            self.reset_progress()

        self.has_started = True

        if not self.folder:
            messagebox.showerror("Error", "No subset created.")
            return

        EXT = (".mp3", ".wav", ".m4a", ".flac", ".aac", ".ogg")

        files = [
            os.path.join(self.folder, f)
            for f in os.listdir(self.folder)
            if f.lower().endswith(EXT)
        ]

        if len(files) < 2:
            messagebox.showerror("Error", "Need 2+ files.")
            return



        if self.all_pairs_mode:
            self.all_pairs = list(itertools.combinations(files, 2))
            random.shuffle(self.all_pairs)

            self.current_pair = self.all_pairs.pop(0)
            self.pool = self.all_pairs
            self.total_pairs = len(self.pool) + 1
        else:
            random.shuffle(files)
            f1 = files.pop()
            f2 = files.pop()

            self.current_pair = (f1, f2)
            self.pool = files
            self.total_pairs = 1 + len(files)
            # PRECOMPUTE VAD CHUNKS IN BACKGROUND
        self.debug("Precomputing audio chunks…")
        MAX_PRELOAD = 12
        self.debug(f"Precomputing audio chunks (max {MAX_PRELOAD})…")
        self.root.after(
            10,
            lambda: self._precompute_chunks(files[:MAX_PRELOAD])
        )



        self._set_text("progress", f"Progress: {self.index}/{self.total_pairs}")
        self.waveform_canvas.delete("all")
        a, b = self.current_pair
        self.debug(f"Starting: {os.path.basename(a)} vs {os.path.basename(b)}")


    # ========================================================
    # PLAYBACK
    # ========================================================

    def start_playback(self, clip, sr, label):
        if clip is None or len(clip) == 0:
            self.debug(f"Empty clip for {label}")
            return

        self.draw_waveform(clip)

        self.reset_audio()

        system_vol = get_macos_output_volume()
        effective_vol = self.base_volume * system_vol

        safe_clip = np.clip(clip * effective_vol, -1.0, 1.0)
        sd.play(safe_clip.astype(np.float32), sr)


        self.playing = True
        self.play_start = time.time()
        self.play_duration = len(clip) / sr

        self.debug(f"Playing {label}")

    def play_first(self):
        if not self.current_pair:
            return
        f1, _ = self.current_pair
        clip, sr, c, n = self.get_random_chunk(f1)
        if clip is None:
            return

        self.current_chunk_info[0] = (
            os.path.basename(f1), c, n
        )
        self.start_playback(clip, sr, "FIRST")


    def play_second(self):
        if not self.current_pair:
            return
        _, f2 = self.current_pair
        clip, sr, c, n = self.get_random_chunk(f2)
        if clip is None:
            return

        self.current_chunk_info[1] = (
            os.path.basename(f2), c, n
        )
        self.start_playback(clip, sr, "SECOND")
    def stop_audio(self):
        sd.stop()
        self.playing = False
        if self.playhead_line:
            self.waveform_canvas.delete(self.playhead_line)
            self.playhead_line = None

    # ========================================================
    # CLIPS / WAVEFORM
    # ========================================================

    def get_clip(self, path):
        if path not in self.cache:
            y, sr = load_audio(path)
            segments = top_k_segments(y, sr, seconds=10, k=5)

            self.cache[path] = {
                "segments": segments,
                "idx": 0,
                "sr": sr
            }

        entry = self.cache[path]
        clip = entry["segments"][entry["idx"]]
        return clip, entry["sr"]
    def next_segment(self, which):
        if not self.current_pair:
            return

        path = self.current_pair[which]
        if path not in self.cache:
            return

        entry = self.cache[path]
        entry["idx"] = (entry["idx"] + 1) % len(entry["segments"])

        label = "FIRST" if which == 0 else "SECOND"
        self.debug(f"{label} → switched to segment {entry['idx'] + 1}")

        # replay automatically
        clip, sr = self.get_clip(path)
        self.start_playback(clip, sr, label)



    def draw_waveform(self, clip):
        self.waveform_canvas.delete("all")

        w = self.waveform_canvas.winfo_width()
        h = self.waveform_canvas.winfo_height()

        if not w or not h:
            return

        mid = h // 2
        amp = h * 0.4

        num_pts = min(w, len(clip))
        idx = np.linspace(0, len(clip) - 1, num_pts).astype(int)
        vals = clip[idx]

        m = np.max(np.abs(vals))
        if m > 0:
            vals = vals / m

        coords = []
        for i, v in enumerate(vals):
            coords.extend([i, mid - v * amp])

        color = "white"
        if self.fun_mode:
            r, g, b = colorsys.hsv_to_rgb((self.fun_hue + 0.33) % 1, 1, 1)
            color = "#%02x%02x%02x" % (int(r * 255), int(g * 255), int(b * 255))

        self.waveform_canvas.create_line(coords, fill=color, width=2)

    def _update_playhead(self):
        if self.playing and self.play_duration:
            t = time.time() - self.play_start
            prog = min(1, max(0, t / self.play_duration))

            w = self.waveform_canvas.winfo_width()
            x = int(prog * w)

            if self.playhead_line:
                self.waveform_canvas.delete(self.playhead_line)

            color = "#FFEB3B"
            if self.fun_mode:
                r, g, b = colorsys.hsv_to_rgb(self.fun_hue, 1, 1)
                color = "#%02x%02x%02x" % (int(r * 255), int(g * 255), int(b * 255))

            self.playhead_line = self.waveform_canvas.create_line(
                x, 0, x, self.waveform_canvas.winfo_height(),
                fill=color, width=3
            )

            if t >= self.play_duration:
                self.playing = False

        self.root.after(40, self._update_playhead)

    # ========================================================
    # FUN MODE
    # ========================================================

    def toggle_fun_mode(self):
        self.fun_mode = not self.fun_mode

        if self.fun_mode:
            self.debug("FUN MODE ON")
            self.fun_btn.set_fill("#8800FF")
            self._play_gif_background()
        else:
            self.debug("FUN MODE OFF")
            self.fun_btn.set_fill("#333333")
            if self.bg_image_id is not None:
                self.canvas.delete(self.bg_image_id)
                self.bg_image_id = None
            self.canvas.configure(bg=self.COL_BLUSH)
            self._restore_pastel()

    def _restore_pastel(self):
        self.vote1.set_fill(self.COL_SALMON)
        self.vote2.set_fill(self.COL_SALMON)
        self.skip1.set_fill(self.COL_PEACH)
        self.skip2.set_fill(self.COL_PEACH)
        self.play1.set_fill(self.COL_PERI)
        self.play2.set_fill(self.COL_PERI)
        self.debug_btn.set_fill("#555577")
        self.start_btn.set_fill(self.COL_SALMON)

    def _update_fun_mode(self):
        if self.fun_mode:
            self.fun_hue = (self.fun_hue + 0.01) % 1.0
        self.root.after(22, self._update_fun_mode)

    # ========================================================
    # DEBUG
    # ========================================================

    def toggle_debug(self):
        self.debug_open = not self.debug_open
        if self.debug_open:
            self.debug("Debug ON")
            if self.debug_panel_window is None:
                self.debug_panel_window = self.canvas.create_window(
                    self.canvas.winfo_width() // 2,
                    self.canvas.winfo_height() - 140,
                    window=self.debug_panel,
                    anchor="n"
                )
            else:
                self.canvas.itemconfigure(self.debug_panel_window, state="normal")
            self._relayout()
        else:
            if self.debug_panel_window is not None:
                self.canvas.itemconfigure(self.debug_panel_window, state="hidden")

    def debug(self, msg):
        entry = f"[{time.strftime('%H:%M:%S')}] {msg}"
        self.debug_log.append(entry)
        self.debug_log = self.debug_log[-20:]
        if self.debug_open:
            self.debug_label.config(text="\n".join(self.debug_log))

    # ========================================================
    # VOTING
    # ========================================================
    def _normalize_path(self, x):
        if isinstance(x, (list, tuple)):
            return x[0]
        return x


    def skip(self, which):
        if not self.current_pair or not self.source_folder:
            return

        f1, f2 = self.current_pair
        bad = f1 if which == 0 else f2
        keep = f2 if which == 0 else f1

        EXT = (".mp3", ".wav", ".m4a", ".flac", ".aac", ".ogg")

        pool = [
            os.path.join(self.source_folder, f)
            for f in os.listdir(self.source_folder)
            if f.lower().endswith(EXT)
            and os.path.join(self.source_folder, f) != keep
        ]

        if not pool:
            messagebox.showerror("Error", "No replacement files.")
            return

        newf = random.choice(pool)

        if which == 0:
            self.current_pair = (newf, keep)
        else:
            self.current_pair = (keep, newf)

        if bad in self.cache:
            del self.cache[bad]

        self.waveform_canvas.delete("all")
        self.debug(f"Skipped {os.path.basename(bad)} → {os.path.basename(newf)}")
        self.save_state()


    def vote(self, choice):
        if not self.current_pair:
            return

        f1, f2 = self.current_pair
        winner = f1 if choice == 1 else f2


        # chunk info
        name1, c1, t1 = self.current_chunk_info.get(
            0, (os.path.basename(f1), None, None)
        )
        name2, c2, t2 = self.current_chunk_info.get(
            1, (os.path.basename(f2), None, None)
        )

        self.results.append([
            os.path.basename(f1), f"{c1}/{t1}",
            os.path.basename(f2), f"{c2}/{t2}",
            choice
        ])




        if not self.pool:
            self.index += 1
            self._set_text("progress", f"Progress: {self.index}/{self.total_pairs}")
            self.finish()
            return

        if self.all_pairs_mode:
            if not self.pool:
                self.finish()
                return
            self.current_pair = self.pool.pop(0)
        else:
            new = self.pool.pop()
            self.current_pair = (winner, new)


        self.index += 1
        self._set_text("progress", f"Progress: {self.index}/{self.total_pairs}")
        self.waveform_canvas.delete("all")

        self.debug(f"Winner: {os.path.basename(winner)}")
        self.save_state()


    # ========================================================
    # FINISH
    # ========================================================

    def finish(self):
        with open("preferences.csv", "w", newline="") as f:
            w = csv.writer(f)
            w.writerow(["file1","chunk1", "file2","chunk2", "choice"])
            w.writerows(self.results)

        with open("preference_data.jsonl", "w") as f:
            for f1, f2, c in self.results:
                if c == 1:
                    f.write(json.dumps({"winner": f1, "loser": f2}) + "\n")
                else:
                    f.write(json.dumps({"winner": f2, "loser": f1}) + "\n")

        messagebox.showinfo("Done", "Saved preferences!")
        self.debug("FINISHED!")
        if os.path.exists(STATE_FILE):
            os.remove(STATE_FILE)



if __name__ == "__main__":
    root = tk.Tk()
    app = PreferenceApp(root)
    root.mainloop()
