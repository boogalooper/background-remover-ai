from __future__ import annotations

import tkinter as tk


class ToolTip:
    def __init__(self, widget: tk.Widget, text: str, delay_ms: int = 300):
        self.widget = widget
        self.text = text
        self.delay_ms = delay_ms
        self.tip: tk.Toplevel | None = None
        self.after_id: str | None = None
        widget.bind("<Enter>", self._schedule, add="+")
        widget.bind("<Leave>", self._hide, add="+")
        widget.bind("<ButtonPress>", self._hide, add="+")

    def _schedule(self, _event=None):
        if self.after_id or self.tip or not self.text:
            return
        self.after_id = self.widget.after(self.delay_ms, self._show)

    def _show(self, _event=None):
        self.after_id = None
        if self.tip or not self.text or not self.widget.winfo_exists():
            return
        self.tip = tk.Toplevel(self.widget)
        self.tip.wm_overrideredirect(True)
        try:
            self.tip.attributes("-topmost", True)
        except tk.TclError:
            pass
        label = tk.Label(self.tip, text=self.text, justify="left", relief="solid", borderwidth=1, padx=9, pady=7, wraplength=520)
        label.pack()
        self.tip.update_idletasks()
        x = self.widget.winfo_rootx() + 18
        y = self.widget.winfo_rooty() + self.widget.winfo_height() + 5
        sw, sh = self.widget.winfo_screenwidth(), self.widget.winfo_screenheight()
        tw, th = self.tip.winfo_reqwidth(), self.tip.winfo_reqheight()
        x = min(max(4, x), max(4, sw - tw - 8))
        if y + th > sh - 8:
            y = max(4, self.widget.winfo_rooty() - th - 5)
        self.tip.wm_geometry(f"+{x}+{y}")

    def _hide(self, _event=None):
        if self.after_id:
            try:
                self.widget.after_cancel(self.after_id)
            except tk.TclError:
                pass
            self.after_id = None
        if self.tip:
            self.tip.destroy()
            self.tip = None
