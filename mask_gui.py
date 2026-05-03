import tkinter as tk
from tkinter import filedialog, messagebox
from PIL import Image, ImageDraw

CANVAS_SIZE = 512
BRUSH_SIZE = 20

class MaskGUI:
    def __init__(self, root):
        self.root = root
        self.root.title("Binary Mask Creator")
        
        self.canvas = tk.Canvas(root, width=CANVAS_SIZE, height=CANVAS_SIZE, bg='white')
        self.canvas.pack()

        self.image = Image.new("L", (CANVAS_SIZE, CANVAS_SIZE), 0)
        self.draw = ImageDraw.Draw(self.image)

        self.drawing = False
        self.shape_start = None
        self.last_x = None
        self.last_y = None

        self.mode = tk.StringVar(value="Free Draw")  # Modes: Free Draw, Rectangle, Oval, Square, Circle

        self.canvas.bind('<ButtonPress-1>', self.start_draw)
        self.canvas.bind('<ButtonRelease-1>', self.stop_draw)
        self.canvas.bind('<B1-Motion>', self.paint)

        # UI Buttons
        btn_frame = tk.Frame(root)
        btn_frame.pack(fill=tk.X)

        shape_options = ["Free Draw", "Rectangle", "Oval", "Square", "Circle"]
        for shape in shape_options:
            tk.Radiobutton(btn_frame, text=shape, variable=self.mode, value=shape).pack(side=tk.LEFT, padx=5)

        tk.Button(btn_frame, text="Clear", command=self.clear).pack(side=tk.LEFT, padx=5)
        tk.Button(btn_frame, text="Save Mask", command=self.save).pack(side=tk.LEFT, padx=5)

    def start_draw(self, event):
        self.drawing = True
        self.last_x, self.last_y = event.x, event.y
        self.shape_start = (event.x, event.y)
        if self.mode.get() == "Free Draw":
            self.paint(event)

    def stop_draw(self, event):
        if not self.drawing:
            return
        self.drawing = False
        x0, y0 = self.shape_start
        x1, y1 = event.x, event.y
        shape = self.mode.get()

        if shape == "Rectangle":
            self.canvas.create_rectangle(x0, y0, x1, y1, outline='black', fill='black')
            self.draw.rectangle([x0, y0, x1, y1], fill=255)
        elif shape == "Oval":
            self.canvas.create_oval(x0, y0, x1, y1, outline='black', fill='black')
            self.draw.ellipse([x0, y0, x1, y1], fill=255)
        elif shape == "Square":
            size = min(abs(x1 - x0), abs(y1 - y0))
            x1 = x0 + size if x1 >= x0 else x0 - size
            y1 = y0 + size if y1 >= y0 else y0 - size
            self.canvas.create_rectangle(x0, y0, x1, y1, outline='black', fill='black')
            self.draw.rectangle([x0, y0, x1, y1], fill=255)
        elif shape == "Circle":
            size = min(abs(x1 - x0), abs(y1 - y0))
            x1 = x0 + size if x1 >= x0 else x0 - size
            y1 = y0 + size if y1 >= y0 else y0 - size
            self.canvas.create_oval(x0, y0, x1, y1, outline='black', fill='black')
            self.draw.ellipse([x0, y0, x1, y1], fill=255)
        self.last_x = self.last_y = None

    def paint(self, event):
        if not self.drawing or self.mode.get() != "Free Draw":
            return
        x, y = event.x, event.y
        if self.last_x is not None and self.last_y is not None:
            self.canvas.create_line(self.last_x, self.last_y, x, y, fill='black', width=BRUSH_SIZE, capstyle=tk.ROUND)
            self.draw.line([self.last_x, self.last_y, x, y], fill=255, width=BRUSH_SIZE)
        else:
            self.canvas.create_oval(x - BRUSH_SIZE//2, y - BRUSH_SIZE//2, x + BRUSH_SIZE//2, y + BRUSH_SIZE//2, fill='black', outline='black')
            self.draw.ellipse([x - BRUSH_SIZE//2, y - BRUSH_SIZE//2, x + BRUSH_SIZE//2, y + BRUSH_SIZE//2], fill=255)
        self.last_x, self.last_y = x, y

    def clear(self):
        self.canvas.delete('all')
        self.image = Image.new("L", (CANVAS_SIZE, CANVAS_SIZE), 0)
        self.draw = ImageDraw.Draw(self.image)

    def save(self):
        import os
        # Ensure 'masks' directory exists in the script's directory
        script_dir = os.path.dirname(os.path.abspath(__file__))
        masks_dir = os.path.join(script_dir, 'masks')
        os.makedirs(masks_dir, exist_ok=True)
        # Ask only for filename
        filename = tk.simpledialog.askstring("Save Mask", "Enter filename (without extension):")
        if filename:
            if not filename.lower().endswith('.png'):
                filename += '.png'
            file_path = os.path.join(masks_dir, filename)
            mask = self.image.point(lambda p: 255 if p > 127 else 0)
            mask.save(file_path)
            messagebox.showinfo("Saved", f"Mask saved to {file_path}")

if __name__ == "__main__":
    root = tk.Tk()
    app = MaskGUI(root)
    root.mainloop()
