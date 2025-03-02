import tkinter as tk
from tkinter import ttk
from PIL import Image, ImageTk
import requests
import cv2
import numpy as np
import threading
import queue
import subprocess
import sys
import os

# Video stream URL
URL = "http://192.168.8.219:5000/video_feed"

class StreamViewer:
    def __init__(self, root):
        self.root = root
        self.root.title("ASTRA UI")

        # Get script directory
        self.script_dir = os.path.dirname(os.path.abspath(__file__))

        # Create main frame (with grid layout)
        main_frame = ttk.Frame(root)
        main_frame.pack(expand=True, fill='both', padx=10, pady=10)

        # Configure grid layout - Make column 0 (camera feed) larger than column 1 (map)
        main_frame.grid_columnconfigure(0, weight=2)  # Changed from weight=1 to weight=2
        main_frame.grid_columnconfigure(1, weight=1)  # Keep weight=1 for map section
        main_frame.grid_rowconfigure(0, weight=5)  
        main_frame.grid_rowconfigure(1, weight=0)  

        # Video frame (left side)
        video_frame = ttk.LabelFrame(main_frame, text="Camera Feed")
        video_frame.grid(row=0, column=0, padx=5, pady=5, sticky='nsew')

        # Create display label for video
        self.display = tk.Label(video_frame)
        self.display.pack(expand=True, fill='both')

        # Status label
        self.status_label = tk.Label(video_frame, text="Initializing...", fg="blue")
        self.status_label.pack(pady=5)

        # Maps frame (right side)
        maps_frame = ttk.LabelFrame(main_frame, text="GPS Map View")
        maps_frame.grid(row=0, column=1, padx=5, pady=5, sticky='nsew')

        # Coordinate label
        self.coord_label = tk.Label(maps_frame, text="Coordinates: Not received", font=('Arial', 12))
        self.coord_label.pack(pady=5)

        # Map display label
        self.map_display = tk.Label(maps_frame)
        self.map_display.pack(expand=True, fill='both')

        # Terminal frame (bottom, spanning both columns)
        terminal_frame = ttk.LabelFrame(main_frame, text="Terminal Output")
        terminal_frame.grid(row=1, column=0, columnspan=2, padx=5, pady=5, sticky='nsew')

        # Terminal output
        self.terminal = tk.Text(terminal_frame, height=5, bg='black', fg='white', font=('Courier', 10))
        self.terminal.pack(fill='both', expand=True)

        # Scrollbar
        scrollbar = ttk.Scrollbar(terminal_frame, command=self.terminal.yview)
        scrollbar.pack(side=tk.RIGHT, fill=tk.Y)
        self.terminal.config(yscrollcommand=scrollbar.set)

        # Clear button
        self.clear_button = ttk.Button(terminal_frame, text="Clear Output", command=self.clear_terminal)
        self.clear_button.pack(pady=5)

        # Setup
        self.frame_queue = queue.Queue(maxsize=2)
        self.running = True

        # Start video stream
        self.video_thread = threading.Thread(target=self.stream_reader)
        self.video_thread.daemon = True
        self.video_thread.start()

        # Start output monitoring (for GPS telemetry)
        self.start_output_monitoring()

        # Start frame updates
        self.update_frame()

    def start_output_monitoring(self):
        """
        Runs 'test_receiver.py' and extracts GPS coordinates from its output.
        """
        script_path = os.path.join(self.script_dir, 'test_receiver.py')

        try:
            # Start the script process
            self.process = subprocess.Popen(
                [sys.executable, "-u", script_path],
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                bufsize=1,
                universal_newlines=True
            )

            def read_output():
                for line in iter(self.process.stdout.readline, ''):
                    if line:
                        self.root.after(0, lambda l=line: self.terminal.insert(tk.END, l))
                        self.root.after(0, self.terminal.see, tk.END)

                        # Extract GPS coordinates if they are printed
                        if "Coordinate:" in line:
                            try:
                                coord_str = line.split("Coordinate:")[1].strip()
                                lat, lon = map(float, coord_str.split(','))
                                self.root.after(0, self.update_coordinate_display, lat, lon)
                            except Exception as e:
                                print(f"Error parsing coordinates: {e}")

            # Start terminal output monitoring
            self.output_thread = threading.Thread(target=read_output, daemon=True)
            self.output_thread.start()

        except Exception as e:
            self.terminal.insert(tk.END, f"Error starting script: {str(e)}\n")
            self.terminal.see(tk.END)

    def update_coordinate_display(self, lat, lon):
        """
        Updates coordinate display and loads an offline map based on GPS coordinates.
        """
        self.coord_label.config(text=f"Coordinates: {lat}, {lon}")
        self.load_offline_map(lat, lon)

    def load_offline_map(self, lat=None, lon=None):
        """
        Loads a pre-downloaded offline map image from the 'maps' folder.
        """
        maps_folder = os.path.join(self.script_dir, "maps")

        # Define coordinate ranges for different maps
        map_files = {
            "UCR.png": (33.9737, -117.3281),  # UCR Coordinates
            "Box_springs.png": (33.9570, -117.2727)  # Box Springs Coordinates
        }

        # Default map
        map_path = os.path.join(maps_folder, "map_default.png")

        if lat is not None and lon is not None:
            for filename, (map_lat, map_lon) in map_files.items():
                if abs(lat - map_lat) < 0.01 and abs(lon - map_lon) < 0.01:
                    map_path = os.path.join(maps_folder, filename)
                    print(f"Loading specific offline map: {filename}")
                    break

        # Load and display the selected map - resize to a smaller size
        if os.path.exists(map_path):
            img = Image.open(map_path)
            img = img.resize((400, 300))  # Reduced from 640x480 to 400x300
            self.map_photo = ImageTk.PhotoImage(img)
            self.map_display.config(image=self.map_photo)
            self.map_display.image = self.map_photo  # Keep a reference
        else:
            self.coord_label.config(text="Offline map not found!")

    def stream_reader(self):
        """
        Reads video stream from the camera.
        """
        while self.running:
            try:
                response = requests.get(URL, stream=True)
                if response.status_code == 200:
                    bytes_data = bytes()
                    for chunk in response.iter_content(chunk_size=1024):
                        if not self.running:
                            break
                        bytes_data += chunk
                        a = bytes_data.find(b'\xff\xd8')
                        b = bytes_data.find(b'\xff\xd9')
                        if a != -1 and b != -1:
                            jpg = bytes_data[a:b+2]
                            bytes_data = bytes_data[b+2:]
                            try:
                                self.frame_queue.put(jpg, timeout=0.1)
                            except queue.Full:
                                continue
            except Exception as e:
                print(f"Stream error: {str(e)}")
                self.root.after(1000, self.stream_reader)
                break

    def update_frame(self):
        """
        Updates the video frame in the Tkinter UI.
        """
        try:
            if not self.frame_queue.empty():
                jpg_data = self.frame_queue.get_nowait()
                img_array = np.frombuffer(jpg_data, dtype=np.uint8)
                img = cv2.imdecode(img_array, cv2.IMREAD_COLOR)

                if img is not None:
                    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
                    img = Image.fromarray(img)
                    img = img.resize((800, 600))  # Increased from 640x480 to 800x600
                    photo = ImageTk.PhotoImage(image=img)

                    self.display.imgtk = photo
                    self.display.configure(image=photo)
                    self.status_label.config(text="Streaming...", fg="green")
        except Exception as e:
            print(f"Display error: {str(e)}")

        if self.running:
            self.root.after(10, self.update_frame)

    def clear_terminal(self):
        self.terminal.delete(1.0, tk.END)

    def on_closing(self):
        self.running = False
        self.root.destroy()

if __name__ == "__main__":
    root = tk.Tk()
    app = StreamViewer(root)
    root.protocol("WM_DELETE_WINDOW", app.on_closing)
    root.mainloop()