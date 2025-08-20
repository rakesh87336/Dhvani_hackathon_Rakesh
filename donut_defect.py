import cv2
import numpy as np
import os
import math
import tkinter as tk
from tkinter import filedialog, ttk, scrolledtext
from typing import Tuple, List, Dict

# A simple class to hold the detection result (for internal use)
class DefectResult:
    """Class to store the result of the defect detection."""
    def __init__(self, is_defective: bool, defect_type: str, defect_area: int):
        self.is_defective = is_defective
        self.defect_type = defect_type
        self.defect_area = defect_area

    def __repr__(self):
        return f"Defective: {self.is_defective}, Type: {self.defect_type}, Area: {self.defect_area} pixels"

def get_donut_parameters(image_path: str) -> Tuple[Tuple[int, int], int, int]:
    """
    Analyzes an image and returns the center, outer radius, and inner radius of the donut shape.

    Args:
        image_path (str): The file path to the image to be analyzed.

    Returns:
        Tuple[Tuple[int, int], int, int]: A tuple containing the center (x, y),
        the average outer radius, and the average inner radius. Returns (None, None, None) on failure.
    """
    image = cv2.imread(image_path)
    if image is None:
        return (None, None, None)

    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    
    # Use adaptive thresholding for robust segmentation
    adaptive_thresh = cv2.adaptiveThreshold(gray, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY_INV, 11, 2)
    
    # Use morphological operations to clean up the binary image
    kernel = np.ones((5, 5), np.uint8)
    binary = cv2.morphologyEx(adaptive_thresh, cv2.MORPH_OPEN, kernel)
    binary = cv2.morphologyEx(binary, cv2.MORPH_CLOSE, kernel)

    contours, _ = cv2.findContours(binary, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if not contours:
        return (None, None, None)
    
    # Find the largest contour and its center of mass
    largest_contour = max(contours, key=cv2.contourArea)
    M = cv2.moments(largest_contour)
    if M["m00"] == 0:
        return (None, None, None)
    
    center_x = int(M["m10"] / M["m00"])
    center_y = int(M["m01"] / M["m00"])
    center = (center_x, center_y)

    # Perform a circular pixel scan to find the radii at various angles
    inner_radii = []
    outer_radii = []
    
    # Iterate through angles to perform a 360-degree scan
    for angle in range(0, 360, 1):
        theta = math.radians(angle)
        
        # Scan outward to find outer radius
        for r in range(min(gray.shape) // 2):
            x = int(center[0] + r * math.cos(theta))
            y = int(center[1] + r * math.sin(theta))
            
            # Check for bounds
            if not (0 <= x < gray.shape[1] and 0 <= y < gray.shape[0]):
                break
            
            # Find the first black pixel (value 255 in the inverted binary image)
            if binary[y, x] > 0:
                outer_radii.append(r)
                
                # Scan inward from the outer radius to find inner radius
                for r_inner in range(r, -1, -1):
                    x_inner = int(center[0] + r_inner * math.cos(theta))
                    y_inner = int(center[1] + r_inner * math.sin(theta))
                    
                    if not (0 <= x_inner < gray.shape[1] and 0 <= y_inner < gray.shape[0]):
                        break
                    
                    # Find the first white pixel (value 0 in the inverted binary image)
                    if binary[y_inner, x_inner] == 0:
                        inner_radii.append(r_inner)
                        break
                break
    
    if not inner_radii or not outer_radii:
        return (None, None, None)

    inner_radius = int(np.mean(inner_radii))
    outer_radius = int(np.mean(outer_radii))
    
    return center, outer_radius, inner_radius

class DefectDetectorApp:
    def __init__(self, master):
        self.master = master
        master.title("Donut Defect Detector")
        master.geometry("600x600")

        self.ref_image_path = ""
        self.test_image_path = ""
        self.defect_threshold = 2

        # Create main frame
        main_frame = ttk.Frame(master, padding="10")
        main_frame.pack(fill=tk.BOTH, expand=True)

        # Reference image selection
        ref_frame = ttk.LabelFrame(main_frame, text="1. Select Reference Image", padding="10")
        ref_frame.pack(fill=tk.X, pady=10)

        self.ref_label = ttk.Label(ref_frame, text="No image selected")
        self.ref_label.pack(side=tk.LEFT, fill=tk.X, expand=True, padx=5)

        ref_button = ttk.Button(ref_frame, text="Browse", command=self.select_reference_image)
        ref_button.pack(side=tk.RIGHT)

        # Test image selection
        test_frame = ttk.LabelFrame(main_frame, text="2. Select Test Image", padding="10")
        test_frame.pack(fill=tk.X, pady=10)

        self.test_label = ttk.Label(test_frame, text="No image selected")
        self.test_label.pack(side=tk.LEFT, fill=tk.X, expand=True, padx=5)

        test_button = ttk.Button(test_frame, text="Browse", command=self.select_test_image)
        test_button.pack(side=tk.RIGHT)

        # Run analysis button
        self.run_button = ttk.Button(main_frame, text="3. Run Analysis", command=self.run_analysis, state=tk.DISABLED)
        self.run_button.pack(fill=tk.X, pady=10)

        # Results display
        result_frame = ttk.LabelFrame(main_frame, text="Analysis Results", padding="10")
        result_frame.pack(fill=tk.BOTH, expand=True, pady=10)

        self.results_text = scrolledtext.ScrolledText(result_frame, wrap=tk.WORD, state='disabled')
        self.results_text.pack(fill=tk.BOTH, expand=True)

    def select_reference_image(self):
        file_path = filedialog.askopenfilename(
            title="Select a reference image",
            filetypes=[("Image files", "*.png;*.jpg;*.jpeg;*.bmp;*.tiff")]
        )
        if file_path:
            self.ref_image_path = file_path
            self.ref_label.config(text=os.path.basename(file_path))
            self.check_run_button_state()

    def select_test_image(self):
        file_path = filedialog.askopenfilename(
            title="Select a test image",
            filetypes=[("Image files", "*.png;*.jpg;*.jpeg;*.bmp;*.tiff")]
        )
        if file_path:
            self.test_image_path = file_path
            self.test_label.config(text=os.path.basename(file_path))
            self.check_run_button_state()

    def check_run_button_state(self):
        if self.ref_image_path and self.test_image_path:
            self.run_button.config(state=tk.NORMAL)
        else:
            self.run_button.config(state=tk.DISABLED)

    def update_results(self, text):
        self.results_text.config(state='normal')
        self.results_text.insert(tk.END, text + "\n")
        self.results_text.config(state='disabled')
        self.results_text.see(tk.END) # Auto-scroll to the bottom

    def run_analysis(self):
        self.update_results("--- Starting Analysis ---")
        self.update_results(f"Using '{os.path.basename(self.ref_image_path)}' as the template.")
        
        # Get perfect parameters from the reference image
        perfect_center, perfect_outer_radius, perfect_inner_radius = get_donut_parameters(self.ref_image_path)
        
        if perfect_center is None:
            self.update_results("Error: Failed to get perfect donut parameters from the reference image.")
            return

        # Process the single test image
        filename = os.path.basename(self.test_image_path)
        self.update_results(f"\nProcessing '{filename}'...")
        
        current_center, current_outer_radius, current_inner_radius = get_donut_parameters(self.test_image_path)
        
        if current_center is None:
            self.update_results(f"  - Status: ERROR ❌")
            self.update_results(f"  - Details: Could not analyze the image.")
            return

        # Calculate difference based on radii
        diff_outer_pixels = abs(current_outer_radius - perfect_outer_radius)
        diff_inner_pixels = abs(current_inner_radius - perfect_inner_radius)
        total_difference = diff_outer_pixels + diff_inner_pixels
        
        if total_difference > self.defect_threshold:
            # Determine defect type
            if diff_outer_pixels > diff_inner_pixels:
                defect_type = "Extra Portion (blob)"
            else:
                defect_type = "Missing Portion (chip)"
                
            self.update_results(f"  - Status: DEFECTIVE! ❌")
            self.update_results(f"  - Defect Type: {defect_type}")
            self.update_results(f"  - Difference Pixels: {total_difference}")
        else:
            self.update_results(f"  - Status: GOOD ✅")
            self.update_results(f"  - Difference Pixels: {total_difference}")

        self.update_results("\n--- Analysis Complete ---")

if __name__ == "__main__":
    root = tk.Tk()
    app = DefectDetectorApp(root)
    root.mainloop()
