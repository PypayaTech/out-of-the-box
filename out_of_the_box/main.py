import tkinter as tk
from out_of_the_box.roi_selector import ROISelector


def run_app():
    root = tk.Tk()
    root.title("Roy App")
    app = ROISelector(root)
    app.pack(fill=tk.BOTH, expand=True)
    root.mainloop()


if __name__ == "__main__":
    run_app()
