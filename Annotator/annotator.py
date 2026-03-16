import cv2
import os
import json
import tkinter as tk
import numpy as np
from tkinter import messagebox
from PIL import Image, ImageTk


class TrainingReadyAnnotator:
    def __init__(self, root):
        self.root = root
        self.root.title("Capstone Pipeline: Training-Ready Annotator")
        self.root.geometry("1000x650")

        # Base paths
        self.base_video_dir = "videos"
        self.base_ann_dir = "annotations"

        # Ensure annotation root exists
        os.makedirs(self.base_ann_dir, exist_ok=True)

        # Load classes safely
        if os.path.exists(self.base_video_dir):
            self.classes = sorted([
                d for d in os.listdir(self.base_video_dir)
                if os.path.isdir(os.path.join(self.base_video_dir, d))
            ])
        else:
            self.classes = []

        # State
        self.current_video_idx = 0
        self.video_list = []
        self.cap = None
        self.playing = False
        self.current_frame = 0
        self.total_frames = 0
        self.fps = 0.0
        self.annotation_data = {"start": None, "end": None}
        self.current_image = None
        self.unsaved_changes = False

        self.setup_ui()

        if self.classes:
            self.refresh_sidebar()
            self.video_lb.selection_set(0)
            self.load_current_video()
        else:
            messagebox.showwarning(
                "No Classes Found",
                "No class folders were found inside the 'videos' directory."
            )

        self.root.protocol("WM_DELETE_WINDOW", self.on_close)

    def setup_ui(self):
        self.paned = tk.PanedWindow(self.root, orient=tk.HORIZONTAL)
        self.paned.pack(fill=tk.BOTH, expand=True)

        # -------- LEFT SIDEBAR --------
        self.sidebar_frame = tk.Frame(self.paned, width=250, bg="#2c3e50")
        self.paned.add(self.sidebar_frame)

        tk.Label(
            self.sidebar_frame,
            text="Working Class",
            bg="#2c3e50",
            fg="white",
            font=("Arial", 10, "bold")
        ).pack(pady=5)

        self.class_var = tk.StringVar(value=self.classes[0] if self.classes else "")
        self.class_menu = tk.OptionMenu(
            self.sidebar_frame, self.class_var, *self.classes, command=self.on_class_change
        )
        self.class_menu.pack(fill=tk.X, padx=5)

        self.video_lb = tk.Listbox(
            self.sidebar_frame,
            bg="#34495e",
            fg="white",
            selectbackground="#3498db",
            font=("Consolas", 9)
        )
        self.video_lb.pack(fill=tk.BOTH, expand=True, pady=10, padx=5)
        self.video_lb.bind("<<ListboxSelect>>", self.on_video_select)

        # -------- RIGHT PLAYER --------
        self.player_frame = tk.Frame(self.paned, bg="#ecf0f1")
        self.paned.add(self.player_frame)

        self.canvas = tk.Canvas(self.player_frame, bg="black")
        self.canvas.pack(fill=tk.BOTH, expand=True, pady=10)
        self.canvas.bind("<Configure>", lambda e: self.update_frame())

        self.lbl_info = tk.Label(
            self.player_frame,
            text="Frame: 0 | S: None | E: None",
            bg="#ecf0f1",
            font=("Arial", 11, "bold")
        )
        self.lbl_info.pack()

        self.lbl_meta = tk.Label(
            self.player_frame,
            text="FPS: - | Total Frames: -",
            bg="#ecf0f1",
            fg="#7f8c8d"
        )
        self.lbl_meta.pack()

        self.slider_container = tk.Frame(self.player_frame, bg="#ecf0f1")
        self.slider_container.pack()

        # Marker bar for Start/End positions
        self.marker_canvas = tk.Canvas(
            self.slider_container,
            width=600,
            height=16,
            bg="#ecf0f1",
            highlightthickness=0
        )
        self.marker_canvas.pack()

        self.slider = tk.Scale(
            self.slider_container,
            from_=0,
            to=100,
            orient=tk.HORIZONTAL,
            length=600,
            command=self.seek_video
        )
        self.slider.pack()

        btn_row = tk.Frame(self.player_frame, bg="#ecf0f1")
        btn_row.pack(pady=10)

        tk.Button(btn_row, text="◀ Prev", width=10, command=self.prev_frame).pack(side=tk.LEFT, padx=3)
        tk.Button(btn_row, text="▶ Next", width=10, command=self.next_frame).pack(side=tk.LEFT, padx=3)
        tk.Button(btn_row, text="▶ Play/Pause", width=12, command=self.toggle_play).pack(side=tk.LEFT, padx=5)
        tk.Button(btn_row, text="[S] Mark Start", command=self.mark_start).pack(side=tk.LEFT, padx=5)
        tk.Button(btn_row, text="[E] Mark End", command=self.mark_end).pack(side=tk.LEFT, padx=5)
        tk.Button(
            btn_row,
            text="SAVE [Ctrl+S]",
            bg="#27ae60",
            fg="white",
            width=15,
            command=self.save_data
        ).pack(side=tk.LEFT, padx=10)

        # Key bindings
        self.root.bind("s", lambda e: self.mark_start())
        self.root.bind("e", lambda e: self.mark_end())
        self.root.bind("<Left>", lambda e: self.prev_frame())
        self.root.bind("<Right>", lambda e: self.next_frame())
        self.root.bind("<Control-s>", lambda e: self.save_data())
        self.root.bind("p", lambda e: self.toggle_play())

    def get_current_class_dirs(self):
        cls = self.class_var.get()
        video_dir = os.path.join(self.base_video_dir, cls)
        ann_dir = os.path.join(self.base_ann_dir, cls)
        os.makedirs(ann_dir, exist_ok=True)
        return video_dir, ann_dir

    def refresh_sidebar(self):
        self.video_lb.delete(0, tk.END)
        if not self.class_var.get():
            self.video_list = []
            return

        v_dir, a_dir = self.get_current_class_dirs()

        if not os.path.exists(v_dir):
            self.video_list = []
            return

        self.video_list = sorted([
            f for f in os.listdir(v_dir)
            if f.lower().endswith((".mp4", ".avi", ".mov", ".mkv"))
        ])

        for idx, vid in enumerate(self.video_list):
            ann_file = f"{os.path.splitext(vid)[0]}.json"
            exists = os.path.exists(os.path.join(a_dir, ann_file))
            self.video_lb.insert(tk.END, f"{'[X]' if exists else '[ ]'} {vid}")
            self.video_lb.itemconfig(idx, fg="#2ecc71" if exists else "#e74c3c")

    def on_video_select(self, event):
        sel = self.video_lb.curselection()
        if self.unsaved_changes:
            answer = messagebox.askyesnocancel(
                "Unsaved Annotation",
                "You have unsaved changes.\n\nSave before switching video?"
            )

            if answer is None:  # Cancel
                self.video_lb.selection_clear(0, tk.END)
                self.video_lb.selection_set(self.current_video_idx)
                return

            if answer:  # Yes -> Save
                self.save_data()
        if sel:
            self.current_video_idx = sel[0]
            self.load_current_video()

    def on_class_change(self, *args):
        self.stop_playback()
        self.current_video_idx = 0
        self.refresh_sidebar()
        if self.video_list:
            self.video_lb.selection_clear(0, tk.END)
            self.video_lb.selection_set(0)
            self.load_current_video()
        else:
            self.clear_display("No videos found in this class.")

    def release_cap(self):
        if self.cap is not None:
            self.cap.release()
            self.cap = None

    def load_current_video(self):
        self.stop_playback()

        if not self.video_list:
            self.clear_display("No videos available.")
            return

        if self.current_video_idx < 0 or self.current_video_idx >= len(self.video_list):
            self.clear_display("Invalid video index.")
            return

        cls = self.class_var.get()
        vid_name = self.video_list[self.current_video_idx]
        vid_path = os.path.join(self.base_video_dir, cls, vid_name)
        ann_path = os.path.join(self.base_ann_dir, cls, f"{os.path.splitext(vid_name)[0]}.json")

        self.release_cap()
        self.cap = cv2.VideoCapture(vid_path)

        if not self.cap.isOpened():
            self.clear_display(f"Failed to open video:\n{vid_name}")
            return

        self.total_frames = int(self.cap.get(cv2.CAP_PROP_FRAME_COUNT))
        self.fps = float(self.cap.get(cv2.CAP_PROP_FPS) or 0)
        self.current_frame = 0

        if os.path.exists(ann_path):
            try:
                with open(ann_path, "r") as f:
                    saved = json.load(f)
                    self.annotation_data = {
                        "start": saved.get("start"),
                        "end": saved.get("end")
                    }
            except Exception:
                self.annotation_data = {"start": None, "end": None}
        else:
            self.annotation_data = {"start": None, "end": None}

        self.lbl_meta.config(text=f"FPS: {self.fps:.2f} | Total Frames: {self.total_frames}")
        self.slider.config(to=max(self.total_frames - 1, 0))
        self.slider.set(0)
        self.draw_markers()
        self.update_frame()

    def draw_markers(self):
        self.marker_canvas.delete("all")

        if self.total_frames <= 1:
            return

        width = 600
        h = 16

        s = self.annotation_data.get("start")
        e = self.annotation_data.get("end")

        def frame_to_x(frame_idx):
            return (frame_idx / max(self.total_frames - 1, 1)) * width

        self.marker_canvas.create_line(0, h - 2, width, h - 2, fill="#bdc3c7")

        if s is not None and 0 <= s < self.total_frames:
            x = frame_to_x(s)
            self.marker_canvas.create_line(x, 2, x, h, fill="#27ae60", width=3)
            self.marker_canvas.create_text(
                x + 10, 6, text="S", fill="#27ae60", anchor="w", font=("Arial", 9, "bold")
            )

        if e is not None and 0 <= e < self.total_frames:
            x = frame_to_x(e)
            self.marker_canvas.create_line(x, 2, x, h, fill="#e74c3c", width=3)
            self.marker_canvas.create_text(
                x + 10, 6, text="E", fill="#e74c3c", anchor="w", font=("Arial", 9, "bold")
            )

    def clear_display(self, message=""):
        self.canvas.delete("all")
        self.lbl_info.config(text=message if message else "Frame: - | S: None | E: None")
        self.lbl_meta.config(text="FPS: - | Total Frames: -")
        self.slider.config(to=0)
        self.slider.set(0)
        self.marker_canvas.delete("all")

    def save_data(self):
        if not self.video_list:
            messagebox.showwarning("Warning", "No video loaded.")
            return

        s = self.annotation_data["start"]
        e = self.annotation_data["end"]

        if s is None or e is None:
            messagebox.showwarning("Warning", "Mark both Start and End points.")
            return

        if s > e:
            messagebox.showwarning("Warning", "Start frame cannot be greater than End frame.")
            return

        cls = self.class_var.get()
        _, ann_dir = self.get_current_class_dirs()
        vid_name = self.video_list[self.current_video_idx]
        save_path = os.path.join(ann_dir, f"{os.path.splitext(vid_name)[0]}.json")

        data = {
            "video_name": vid_name,
            "class": cls,
            "start": s,
            "end": e,
            "total_frames": self.total_frames,
            "original_fps": self.fps,
            "duration_frames": e - s + 1,
            "duration_seconds": round((e - s + 1) / self.fps, 3) if self.fps > 0 else None
        }

        try:
            with open(save_path, "w") as f:
                json.dump(data, f, indent=4)
                
            self.unsaved_changes = False
        except Exception as ex:
            messagebox.showerror("Save Error", f"Could not save annotation:\n{ex}")
            return

        self.refresh_sidebar()

        if self.current_video_idx < len(self.video_list) - 1:
            self.current_video_idx += 1
            self.video_lb.selection_clear(0, tk.END)
            self.video_lb.selection_set(self.current_video_idx)
            self.video_lb.activate(self.current_video_idx)
            self.load_current_video()
        else:
            messagebox.showinfo("Done", "Annotation saved.")

    def toggle_play(self):
        if not self.cap:
            return
        self.playing = not self.playing
        if self.playing:
            self.play_loop()

    def stop_playback(self):
        self.playing = False

    def play_loop(self):
        if not self.playing or not self.cap:
            return

        if self.current_frame < self.total_frames - 1:
            self.current_frame += 1
            self.slider.set(self.current_frame)
            self.update_frame()

            delay = int(1000 / self.fps) if self.fps > 0 else 30
            self.root.after(delay, self.play_loop)
        else:
            self.playing = False

    def update_frame(self):
        if not self.cap:
            return

        self.cap.set(cv2.CAP_PROP_POS_FRAMES, self.current_frame)
        ret, frame = self.cap.read()
        if not ret:
            return

        frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

        canvas_w = self.canvas.winfo_width()
        canvas_h = self.canvas.winfo_height()

        if canvas_w < 10 or canvas_h < 10:
            return

        h, w = frame.shape[:2]

        scale = min(canvas_w / w, canvas_h / h)
        new_w = int(w * scale)
        new_h = int(h * scale)

        resized = cv2.resize(frame, (new_w, new_h))

        # Create black background
        display = np.zeros((canvas_h, canvas_w, 3), dtype=np.uint8)

        x_offset = (canvas_w - new_w) // 2
        y_offset = (canvas_h - new_h) // 2

        display[y_offset:y_offset + new_h, x_offset:x_offset + new_w] = resized

        img = Image.fromarray(display)
        self.current_image = ImageTk.PhotoImage(image=img)

        self.canvas.delete("all")
        self.canvas.create_image(0, 0, anchor=tk.NW, image=self.current_image)

        self.lbl_info.config(
            text=f"Frame: {self.current_frame} | S: {self.annotation_data['start']} | E: {self.annotation_data['end']}"
        )

    def seek_video(self, val):
        if not self.cap:
            return
        self.current_frame = max(0, min(int(val), max(self.total_frames - 1, 0)))
        if not self.playing:
            self.update_frame()

    def prev_frame(self):
        if not self.cap:
            return
        self.stop_playback()
        self.current_frame = max(0, self.current_frame - 1)
        self.slider.set(self.current_frame)
        self.update_frame()

    def next_frame(self):
        if not self.cap:
            return
        self.stop_playback()
        self.current_frame = min(self.total_frames - 1, self.current_frame + 1)
        self.slider.set(self.current_frame)
        self.update_frame()

    def mark_start(self):
        if self.cap:
            self.annotation_data["start"] = self.current_frame
            self.unsaved_changes = True
            self.draw_markers()
            self.update_frame()

    def mark_end(self):
        if self.cap:
            self.annotation_data["end"] = self.current_frame
            self.unsaved_changes = True
            self.draw_markers()
            self.update_frame()

    def on_close(self):
        self.stop_playback()
        self.release_cap()
        self.root.destroy()


if __name__ == "__main__":
    root = tk.Tk()
    app = TrainingReadyAnnotator(root)
    root.mainloop()
