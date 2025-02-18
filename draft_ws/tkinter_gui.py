import tkinter as tk
from tkinter import ttk
from PIL import Image, ImageTk
import requests
import cv2
import numpy as np
import threading
import queue
import subprocess
import asyncio
import sys
import os

URL = "http://127.0.0.1:5000/video_feed"

class StreamViewer:
    def __init__(self, root):
        self.root = root
        self.root.title("ASTRA DEMO")
        
        # Create main frame
        main_frame = ttk.Frame(root)
        main_frame.pack(expand=True, fill='both', padx=10, pady=10)
        
        # Video frame
        video_frame = ttk.Frame(main_frame)
        video_frame.pack(fill='both', padx=5, pady=5)
        
        # Create display label for video
        self.display = tk.Label(video_frame)
        self.display.pack()
        
        # Status label
        self.status_label = tk.Label(video_frame, text="Initializing...", fg="blue")
        self.status_label.pack(pady=5)
        
        # Terminal frame
        terminal_frame = ttk.LabelFrame(main_frame, text="Terminal Output")
        terminal_frame.pack(fill='both', expand=True, padx=5, pady=5)
        
        # Terminal output
        self.terminal = tk.Text(terminal_frame, height=10, bg='black', fg='white', font=('Courier', 10))
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
        
        # Start output monitoring
        self.start_output_monitoring()
        
        # Start frame updates
        self.update_frame()

    def start_output_monitoring(self):
        script_path = os.path.join(os.getcwd(), 'test_script.py')
        print(f"Starting script: {script_path}")
        
        # Start the process with a non-blocking pipe
        self.process = subprocess.Popen(
            [sys.executable, script_path],
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,
            bufsize=1,
            universal_newlines=True
        )
        
        # Improved output reading function
        def read_output():
            while self.running and self.process.poll() is None:  # Check if process is still running
                # Read a single line without blocking
                line = self.process.stdout.readline()
                if line:
                    self.root.after(0, self.append_to_terminal, line)
                
                # Check for errors
                error = self.process.stderr.readline()
                if error:
                    self.root.after(0, self.append_to_terminal, f"ERROR: {error}")
        
        # Start reading in a separate thread
        self.output_thread = threading.Thread(target=read_output)
        self.output_thread.daemon = True
        self.output_thread.start()  

    def append_to_terminal(self, text):
        self.terminal.insert(tk.END, text)
        self.terminal.see(tk.END)

    def clear_terminal(self):
        self.terminal.delete(1.0, tk.END)

    def stream_reader(self):
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
        try:
            if not self.frame_queue.empty():
                jpg_data = self.frame_queue.get_nowait()
                img_array = np.frombuffer(jpg_data, dtype=np.uint8)
                img = cv2.imdecode(img_array, cv2.IMREAD_COLOR)
                
                if img is not None:
                    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
                    img = Image.fromarray(img)
                    img = img.resize((640, 480))
                    photo = ImageTk.PhotoImage(image=img)
                    
                    self.display.imgtk = photo
                    self.display.configure(image=photo)
                    self.status_label.config(text="Streaming...", fg="green")
        except Exception as e:
            print(f"Display error: {str(e)}")
            self.status_label.config(text=f"Error: {str(e)}", fg="red")
        
        self.root.after(10, self.update_frame)

    def on_closing(self):
        self.running = False
        if hasattr(self, 'process'):
            self.process.terminate()
        self.root.destroy()

if __name__ == "__main__":
    root = tk.Tk()
    app = StreamViewer(root)
    root.protocol("WM_DELETE_WINDOW", app.on_closing)
    root.mainloop()
