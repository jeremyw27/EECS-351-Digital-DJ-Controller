import tkinter as tk

class SliderApp:
    def __init__(self, master):
        self.master = master
        master.title("LPF Slider")

        self.value_label = tk.Label(master, text="Current cut off freq:")
        self.value_label.pack()

        self.slider = tk.Scale(master, from_=20, to=5000, orient=tk.HORIZONTAL)
        self.slider.pack()

        self.current_value = tk.IntVar(value=self.slider.get())
        self.slider.config(variable=self.current_value)

    def get_current_value(self):
        return self.current_value.get()

def create_slider_app():
    root = tk.Tk()
    app = SliderApp(root)
    return app, root