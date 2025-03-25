import os
import os.path as osp
import tkinter as tk
from tkinter import filedialog, messagebox
from PIL import Image, ImageTk
import cv2
import numpy as np
import torch
import RRDBNet_arch as arch
import mysql.connector

def initialize_model(model_path, device):
    """Initialize the ESRGAN model"""
    model = arch.RRDBNet(3, 3, 64, 23, gc=32)
    model.load_state_dict(torch.load(model_path), strict=True)
    model.eval()
    model = model.to(device)
    return model

def process_image(model, input_path, device):
    """Process an image through the ESRGAN model"""
    img = cv2.imread(input_path, cv2.IMREAD_COLOR)
    if img is None:
        raise ValueError(f"Unable to read image at {input_path}")

    img = img * 1.0 / 255
    img = torch.from_numpy(np.transpose(img[:, :, [2, 1, 0]], (2, 0, 1))).float()
    img_LR = img.unsqueeze(0)
    img_LR = img_LR.to(device)

    with torch.no_grad():
        output = model(img_LR).data.squeeze().float().cpu().clamp_(0, 1).numpy()

    output = np.transpose(output[[2, 1, 0], :, :], (1, 2, 0))
    output = (output * 255.0).round().astype(np.uint8)
    return output

class ESRGANApp:
    def __init__(self, root, return_to_main_ui, user_id):
        self.root = root
        self.return_to_main_ui = return_to_main_ui
        self.user_id = user_id
        self.root.title("Super-Resolution")
        self.root.geometry("1000x700")
        self.root.config(bg="#2b2a2a")

        # Model initialization
        self.model_path = "models/RRDB_ESRGAN_x4.pth"  # Update this path
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.model = initialize_model(self.model_path, self.device)

        # Application state
        self.input_path = None
        self.output_image = None
        
        # UI Elements
        self.create_widgets()

    def create_widgets(self):
        """Create all UI widgets"""
        self.label = tk.Label(self.root, text="Image Super-Resolution", font=("Arial", 16))
        self.label.pack(pady=10)

        self.input_button = tk.Button(self.root, text="Select Image", command=self.select_input_image)
        self.input_button.pack(pady=10)

        self.preview_button = tk.Button(self.root, text="Preview", command=self.preview_output, state=tk.DISABLED)
        self.preview_button.pack(pady=10)

        self.save_button = tk.Button(self.root, text="Save Image", command=self.save_output, state=tk.DISABLED)
        self.save_button.pack(pady=10)
        
        self.back_button = tk.Button(self.root, text="Back", command=self.go_back, bg="#FF0000", fg="white")
        self.back_button.place(relx=1.0, rely=1.0, anchor="se", x=-10, y=-10)

        self.logout_button = tk.Button(self.root, text="Logout", command=self.logout, bg="#FF0000", fg="white")
        self.logout_button.place(relx=1.0, rely=0.0, anchor="ne", x=-10, y=10)

    def select_input_image(self):
        """Handle input image selection"""
        self.input_path = filedialog.askopenfilename(
            title="Select Input Image",
            filetypes=[("Image Files", "*.png *.jpg *.jpeg *.bmp")]
        )
        if self.input_path:
            self.preview_button.config(state=tk.NORMAL)
            messagebox.showinfo("Input Selected", f"Selected Image: {osp.basename(self.input_path)}")

    def preview_output(self):
        """Preview the super-resolution output"""
        if not self.input_path:
            messagebox.showerror("Error", "No input image selected!")
            return

        try:
            self.output_image = process_image(self.model, self.input_path, self.device)
            output_image_rgb = cv2.cvtColor(self.output_image, cv2.COLOR_BGR2RGB)
            output_image_pil = Image.fromarray(output_image_rgb)
            
            # Calculate aspect ratio and resize for preview
            max_width, max_height = 500, 300
            original_width, original_height = output_image_pil.size
            aspect_ratio = original_width / original_height

            if original_width > original_height:
                new_width = max_width
                new_height = int(new_width / aspect_ratio)
            else:
                new_height = max_height
                new_width = int(new_height * aspect_ratio)

            if new_width > max_width:
                new_width = max_width
                new_height = int(new_width / aspect_ratio)
            if new_height > max_height:
                new_height = max_height
                new_width = int(new_height * aspect_ratio)

            resized_image = output_image_pil.resize((new_width, new_height), Image.Resampling.LANCZOS)            
            output_image_tk = ImageTk.PhotoImage(resized_image)       
            
            # Create preview window
            preview_window = tk.Toplevel(self.root)
            preview_window.title("Preview Output")
            frame = tk.Frame(preview_window, width=max_width, height=max_height)
            frame.pack()

            preview_label = tk.Label(frame, image=output_image_tk)
            preview_label.image = output_image_tk 
            preview_label.pack()
            self.save_button.config(state=tk.NORMAL)

        except Exception as e:
            messagebox.showerror("Error", f"An error occurred:\n{str(e)}")

    def save_output(self):
        """Save the output image to file and database"""
        if self.output_image is None:
            messagebox.showerror("Error", "No output image to save!")
            return

        save_path = filedialog.asksaveasfilename(
            title="Save Output Image",
            defaultextension=".png",
            filetypes=[("PNG Files", "*.png"), ("JPEG Files", "*.jpg"), ("All Files", "*.*")]
        )
        
        if not save_path:
            return

        try:
            # Save to file
            cv2.imwrite(save_path, self.output_image)
            
            # Save to database
            with open(save_path, "rb") as file:
                image_data = file.read()

            conn = mysql.connector.connect(
                host='localhost', 
                user='root', 
                port='3306', 
                password='', 
                database='py_lg_rg_db'
            )
            cursor = conn.cursor()
            query = "INSERT INTO images (image_data, user_id) VALUES (%s, %s)"
            cursor.execute(query, (image_data, self.user_id))
            conn.commit()

            messagebox.showinfo("Success", "Image saved successfully to both file and database!")

        except Exception as e:
            messagebox.showerror("Error", f"Failed to save image:\n{str(e)}")
        finally:
            if 'conn' in locals() and conn.is_connected():
                cursor.close()
                conn.close()

    def go_back(self):
        """Return to main UI"""
        self.root.destroy()
        self.return_to_main_ui()

    def logout(self):
        """Logout and return to login"""
        self.root.destroy()
        if hasattr(self, 'return_to_login'):
            self.return_to_login()

def main():
    root = tk.Tk()
    app = ESRGANApp(root, lambda: None, user_id=1)  # Replace with actual user ID
    root.mainloop()

if __name__ == "__main__":
    main()