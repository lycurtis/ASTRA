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
#URL = "http://192.168.1.179:5000/video_feed"
URL = "http://192.168.8.219:5000/video_feed"

class StreamViewer:
    def __init__(self, root):
        self.root = root
        self.root.title("ASTRA UI")

        # Get script directory
        self.script_dir = os.path.dirname(os.path.abspath(__file__))

        # Create main frame
        main_frame = ttk.Frame(root)
        main_frame.pack(expand=True, fill='both', padx=10, pady=10)

        # UI grid layout
        main_frame.grid_columnconfigure(0, weight=3)  
        main_frame.grid_columnconfigure(1, weight=1)  #Map section
        main_frame.grid_rowconfigure(0, weight=3)  
        main_frame.grid_rowconfigure(1, weight=1)  

        # Video frame initilization + display label for camera feed
        video_frame = ttk.LabelFrame(main_frame, text="Camera Feed")
        video_frame.grid(row=0, column=0, padx=5, pady=5, sticky='nsew') #sticky expands camera feed in all direc
        self.display = tk.Label(video_frame) # leftbox titled "Camera Feed"
        self.display.pack(expand=True, fill='both', padx=5, pady=5)
        self.status_label = tk.Label(video_frame, text="Initializing...", fg="red")
        self.status_label.pack(pady=5)

        # Maps frame + label (GPS Map View)
        maps_frame = ttk.LabelFrame(main_frame, text="GPS Map View")
        maps_frame.grid(row=0, column=1, padx=5, pady=5, sticky='nsew')
        self.map_display = tk.Label(maps_frame)
        self.map_display.pack(expand=True, fill='both')

        # Coordinate label
        self.coord_label = tk.Label(maps_frame, text="Coordinates: Not received")
        self.coord_label.pack(pady=5)

        # Map type selection
        self.map_type = tk.StringVar(value="online")
        map_options_frame = ttk.Frame(maps_frame)
        map_options_frame.pack(pady=5)
        
        ttk.Radiobutton(map_options_frame, text="Online Map", variable=self.map_type, 
                        value="online", command=self.refresh_map).pack(side=tk.LEFT, padx=5)
        ttk.Radiobutton(map_options_frame, text="Offline Map", variable=self.map_type, 
                        value="offline", command=self.refresh_map).pack(side=tk.LEFT, padx=5)

        # Terminal frame + output
        terminal_frame = ttk.LabelFrame(main_frame, text="Terminal")
        terminal_frame.grid(row=1, column=0, columnspan=2, padx=5, pady=5, sticky='nsew') #columnspan = 2 columns
        self.terminal = tk.Text(terminal_frame, height=5)
        self.terminal.pack(fill='both', expand=True)

        # Setup
        self.frame_queue = queue.Queue(maxsize=2)
        self.running = True
        self.current_lat = None
        self.current_lon = None
        
        self.maps_api_key = "AIzaSyBhDN-_mDpb3k3ZaECxnB1bQPCD2vEemR4"  #<-- API KEY remove when adding to Github
        
        # Start video stream
        self.video_thread = threading.Thread(target=self.stream_reader)
        self.video_thread.daemon = True
        self.video_thread.start()

        # Check internet connection at startup
        if self.check_internet_connection():
            self.map_type.set("online")  # Start in online mode
            self.terminal.insert(tk.END, "Internet detected.\n")
        else:
            self.map_type.set("offline")  # Start in offline mode and load default map initially
            self.terminal.insert(tk.END, "No internet detected.\n")
            self.load_default_map()

        # Start output monitoring (for GPS telemetry)
        self.start_output_monitoring()

        # update frames
        self.update_frame()

    def check_internet_connection(self):
        # Checks if there is an internet connection on startup
        try:
            requests.get("https://www.google.com", timeout=3)  # Ping Google
            return True
        except (requests.ConnectionError, requests.Timeout):
            return False


    def refresh_map(self):
    # Refreshes map based on the current map type selection (online/offline)
        if self.current_lat and self.current_lon:
            self.update_coordinate_display(self.current_lat, self.current_lon)
        else:
            # if no coordinates, load default map
            self.load_default_map()

    def load_default_map(self):
    # Loads the default map image.
        maps_folder = os.path.join(self.script_dir, "maps") #finds map directory inside of ASTRA
        default_map = "map_default.png"
        map_path = os.path.join(maps_folder, default_map)
        
        # Check if default map exists
        if os.path.exists(map_path):
            img = Image.open(map_path)
            img = img.resize((400, 300))  # Resize to fit the display area
            self.map_photo = ImageTk.PhotoImage(img)
            self.map_display.config(image=self.map_photo)
            self.map_display.image = self.map_photo  # Keep a reference
            self.terminal.insert(tk.END, f"Loaded default map: {default_map}\n")
            self.terminal.see(tk.END)
        else:
            self.terminal.insert(tk.END, f"Default map not found: {map_path}\n")
            self.terminal.see(tk.END)

    def start_output_monitoring(self):
        #Runs receiver and extracts GPS coordinates from its output.
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
                                self.root.after(0, lambda: self.terminal.insert(tk.END, f"Error parsing coordinates: {e}\n"))

            self.output_thread = threading.Thread(target=read_output, daemon=True)
            self.output_thread.start()

        except Exception as e:
            self.terminal.insert(tk.END, f"Error starting script: {str(e)}\n")
            self.terminal.see(tk.END)


    def update_coordinate_display(self, lat, lon):
        #Updates coordinate display and loads the appropriate map based on selection.
        self.current_lat = lat
        self.current_lon = lon
        self.coord_label.config(text=f"Coordinates: {lat:.6f}, {lon:.6f}")
        
        # Load map based on selected type
        if self.map_type.get() == "online":
            self.load_online_map(lat, lon)
        else:
            self.load_offline_map(lat, lon)
            
        # Log the coordinate update
        self.terminal.insert(tk.END, f"Updated map with coordinates: {lat:.6f}, {lon:.6f}\n")
        self.terminal.see(tk.END)

    def load_offline_map(self, lat=None, lon=None):
        # Loads a pre-downloaded offline map image from the 'maps' folder.
        maps_folder = os.path.join(self.script_dir, "maps")

        # Define coordinate ranges for different maps
        map_files = {
            "UCR.png": (33.9737, -117.3281), 
            "box_springs.png": (34.0010, -117.2898)
        }

        # Default map
        default_map = "map_default.png"
        map_path = os.path.join(maps_folder, default_map)

        # If no coordinates provided, use default map
        if lat is None or lon is None:
            self.load_default_map()
            return

        # Select appropriate map based on coordinates
        closest_map = default_map
        min_distance = float('inf')
        
        for filename, (map_lat, map_lon) in map_files.items():
            # Calculate distance to this map's center point
            distance = ((lat - map_lat) ** 2 + (lon - map_lon) ** 2) ** 0.5
            if distance < min_distance:
                min_distance = distance
                closest_map = filename
                
        map_path = os.path.join(maps_folder, closest_map)
        self.terminal.insert(tk.END, f"Selected offline map: {closest_map}\n")
        self.terminal.see(tk.END)

        # Load and display the selected map
        if os.path.exists(map_path):
            img = Image.open(map_path)
            img = img.resize((400, 300))  # Resize to fit the display area
            self.map_photo = ImageTk.PhotoImage(img)
            self.map_display.config(image=self.map_photo)
            self.map_display.image = self.map_photo # keeps a reference photo
                
            # Clear set text
            self.map_display.config(text="", compound="none")
        else:
            self.terminal.insert(tk.END, f"Offline map not found: {map_path}\n")
            self.terminal.see(tk.END)
            self.load_default_map()

    def load_online_map(self, lat, lon):
        #Fetches and displays an online Google Maps image based on GPS coordinates.
        if not lat or not lon:
            self.load_default_map()
            return
            
        zoom = 17  # Adjust zoom level as needed
        map_size = "400x300"  # Size of the map image

        # Create the Google Maps Static API URL
        map_url = (
            f"https://maps.googleapis.com/maps/api/staticmap?"
            f"center={lat},{lon}&zoom={zoom}&size={map_size}"
            f"&markers=color:red%7Clabel:A%7C{lat},{lon}"
            f"&key={self.maps_api_key}"
        )

        try:
            # Check internet connection
            connection_test = requests.get("https://www.google.com", timeout=2)
            
            if connection_test.status_code == 200:
                # Fetch the map image
                response = requests.get(map_url, stream=True)
                
                if response.status_code == 200:
                    img = Image.open(response.raw)
                    img = img.resize((400, 300))  # Resize to fit display area
                    self.map_photo = ImageTk.PhotoImage(img)
                    self.map_display.config(image=self.map_photo)
                    self.map_display.image = self.map_photo  # Keep a reference
                    
                    # Clear any text that might have been set
                    self.map_display.config(text="", compound="none")
                else:
                    error_msg = f"Error loading online map: HTTP {response.status_code}"
                    self.terminal.insert(tk.END, f"{error_msg}\n")
                    self.terminal.see(tk.END)
                    # Fall back to offline map
                    self.map_type.set("offline")
                    self.load_offline_map(lat, lon)
            
        except (requests.ConnectionError, requests.Timeout) as e:
            self.terminal.insert(tk.END, "No internet connection, switching to offline map.\n")
            self.terminal.see(tk.END)
            # Fall back to offline map
            self.map_type.set("offline")
            self.load_offline_map(lat, lon)

    def stream_reader(self):
        #Reads video stream from the camera.
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
                error_msg = f"Stream error: {str(e)}"
                print(error_msg)
                self.root.after(0, lambda: self.terminal.insert(tk.END, f"{error_msg}\n"))
                self.root.after(0, self.terminal.see, tk.END)
                self.root.after(5000, self.stream_reader)  # Retry after 5 seconds
                break

    def update_frame(self):
        #Updates the video frame in the Tkinter UI.
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
            error_msg = f"Display error: {str(e)}"
            print(error_msg)
            self.root.after(0, lambda: self.terminal.insert(tk.END, f"{error_msg}\n"))
            self.root.after(0, self.terminal.see, tk.END)

        if self.running:
            self.root.after(10, self.update_frame)

    def on_closing(self):
        self.running = False
        if hasattr(self, 'process') and self.process:
            self.process.terminate()
        self.root.destroy()

if __name__ == "__main__":
    root = tk.Tk()
    app = StreamViewer(root)
    root.protocol("WM_DELETE_WINDOW", app.on_closing)
    root.geometry("1920x1080")
    root.mainloop()
