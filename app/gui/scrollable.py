from __future__ import annotations

import tkinter as tk
from tkinter import ttk


class ScrollableFrame(ttk.Frame):
    def __init__(self, parent, **kwargs):
        super().__init__(parent, **kwargs)
        self.canvas = tk.Canvas(self, highlightthickness=0, borderwidth=0)
        self.scrollbar = ttk.Scrollbar(self, orient="vertical", command=self.canvas.yview)
        self.inner = ttk.Frame(self.canvas)
        self.window_id = self.canvas.create_window((0, 0), window=self.inner, anchor="nw")
        self.canvas.configure(yscrollcommand=self.scrollbar.set)
        self.canvas.grid(row=0, column=0, sticky="nsew")
        self.scrollbar.grid(row=0, column=1, sticky="ns")
        self.rowconfigure(0, weight=1)
        self.columnconfigure(0, weight=1)
        self.inner.bind("<Configure>", self._on_inner, add="+")
        self.canvas.bind("<Configure>", self._on_canvas, add="+")
        self._bind_tree(self.inner)

    def _on_inner(self, _event=None):
        self.canvas.configure(scrollregion=self.canvas.bbox("all"))

    def _on_canvas(self, event):
        self.canvas.itemconfigure(self.window_id, width=event.width)

    def _bind_tree(self, widget):
        widget.bind("<MouseWheel>", self._wheel, add="+")
        widget.bind("<Button-4>", lambda e: self._scroll_units(-3), add="+")
        widget.bind("<Button-5>", lambda e: self._scroll_units(3), add="+")
        widget.bind("<Map>", lambda e: self._rebind_children(), add="+")

    def _rebind_children(self):
        stack = [self.inner]
        while stack:
            widget = stack.pop()
            try:
                widget.bind("<MouseWheel>", self._wheel, add="+")
            except tk.TclError:
                pass
            stack.extend(widget.winfo_children())

    def _scroll_units(self, units: int):
        self.canvas.yview_scroll(units, "units")
        return "break"

    def _wheel(self, event):
        delta = -1 if event.delta > 0 else 1
        self.canvas.yview_scroll(delta * 3, "units")
        return "break"
