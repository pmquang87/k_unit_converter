"""kunit GUI - tkinter front-end over the same convert/detect API.

Launch with `kunit gui`, `kunit-gui`, or `python -m kunit.gui`.
"""
from __future__ import annotations

import queue
import threading
import traceback
from pathlib import Path

import tkinter as tk
from tkinter import filedialog, messagebox, ttk

from .cli import parse_curve_overrides
from .convert import ConvertError, convert, report
from .detect import detect
from .units import PRESETS, parse_system


class App:
    def __init__(self, root: tk.Tk):
        self.root = root
        root.title("kunit - LS-DYNA unit-system converter")
        root.minsize(760, 560)
        self.q: "queue.Queue[tuple]" = queue.Queue()
        self.busy = False

        pad = {"padx": 6, "pady": 3}
        frm = ttk.Frame(root)
        frm.pack(fill="both", expand=True, **pad)
        frm.columnconfigure(1, weight=1)

        # deck row
        ttk.Label(frm, text="Deck (.k):").grid(row=0, column=0, sticky="w", **pad)
        self.deck_var = tk.StringVar()
        e = ttk.Entry(frm, textvariable=self.deck_var)
        e.grid(row=0, column=1, sticky="ew", **pad)
        ttk.Button(frm, text="Browse...", command=self.browse_deck
                   ).grid(row=0, column=2, **pad)
        ttk.Button(frm, text="Detect units", command=self.on_detect
                   ).grid(row=0, column=3, **pad)

        # unit systems
        presets = list(PRESETS.keys())
        ttk.Label(frm, text="From:").grid(row=1, column=0, sticky="w", **pad)
        self.from_var = tk.StringVar(value="auto")
        ttk.Combobox(frm, textvariable=self.from_var,
                     values=["auto"] + presets
                     ).grid(row=1, column=1, sticky="w", **pad)
        ttk.Label(frm, text="To:").grid(row=2, column=0, sticky="w", **pad)
        self.to_var = tk.StringVar(value="ton-mm-s")
        ttk.Combobox(frm, textvariable=self.to_var, values=presets
                     ).grid(row=2, column=1, sticky="w", **pad)
        ttk.Label(frm, text="(editable: any MASS-LENGTH-TIME, e.g. g-mm-ms)"
                  ).grid(row=1, column=2, columnspan=2, rowspan=2, sticky="w")

        # output row
        ttk.Label(frm, text="Output:").grid(row=3, column=0, sticky="w", **pad)
        self.out_var = tk.StringVar()
        self.out_entry = ttk.Entry(frm, textvariable=self.out_var)
        self.out_entry.grid(row=3, column=1, sticky="ew", **pad)
        ttk.Button(frm, text="Save as...", command=self.browse_out
                   ).grid(row=3, column=2, **pad)

        # options
        opts = ttk.LabelFrame(frm, text="Options")
        opts.grid(row=4, column=0, columnspan=4, sticky="ew", **pad)
        self.v_inplace = tk.BooleanVar()
        self.v_backup = tk.BooleanVar(value=True)
        self.v_includes = tk.BooleanVar()
        self.v_dry = tk.BooleanVar()
        self.v_blast5 = tk.BooleanVar()
        self.v_unknown = tk.BooleanVar()
        self.v_roundtrip = tk.BooleanVar()
        self.v_selfcheck = tk.BooleanVar(value=True)
        checks = [
            ("Convert in place", self.v_inplace),
            ("Keep .orig backup", self.v_backup),
            ("Follow *INCLUDE tree", self.v_includes),
            ("Dry run (report only)", self.v_dry),
            ("Blast UNIT=5 + CF factors", self.v_blast5),
            ("Allow unknown keywords", self.v_unknown),
            ("Verify round-trip", self.v_roundtrip),
            ("Self-check output units", self.v_selfcheck),
        ]
        for i, (label, var) in enumerate(checks):
            ttk.Checkbutton(opts, text=label, variable=var
                            ).grid(row=i // 4, column=i % 4, sticky="w",
                                   padx=8, pady=2)
        self.v_inplace.trace_add("write", lambda *_: self._toggle_out())

        # curve overrides
        ttk.Label(frm, text="Curve overrides:").grid(row=5, column=0,
                                                     sticky="w", **pad)
        self.curves_var = tk.StringVar()
        ttk.Entry(frm, textvariable=self.curves_var
                  ).grid(row=5, column=1, sticky="ew", **pad)
        ttk.Label(frm, text="e.g. 17=time:accel 18=time:force"
                  ).grid(row=5, column=2, columnspan=2, sticky="w")

        # action row
        self.convert_btn = ttk.Button(frm, text="Convert",
                                      command=self.on_convert)
        self.convert_btn.grid(row=6, column=1, sticky="w", **pad)
        self.status_var = tk.StringVar(value="ready")
        ttk.Label(frm, textvariable=self.status_var
                  ).grid(row=6, column=2, columnspan=2, sticky="w", **pad)

        # log pane
        self.log = tk.Text(frm, height=18, wrap="none",
                           font=("Consolas", 9))
        self.log.grid(row=7, column=0, columnspan=4, sticky="nsew", **pad)
        frm.rowconfigure(7, weight=1)
        ys = ttk.Scrollbar(frm, orient="vertical", command=self.log.yview)
        ys.grid(row=7, column=4, sticky="ns")
        self.log.configure(yscrollcommand=ys.set)

        self.root.after(100, self._poll)

    # ── helpers ─────────────────────────────────────────────────────────────
    def _toggle_out(self):
        state = "disabled" if self.v_inplace.get() else "normal"
        self.out_entry.configure(state=state)

    def _append(self, text: str):
        self.log.insert("end", text + "\n")
        self.log.see("end")

    def _clear(self):
        self.log.delete("1.0", "end")

    def browse_deck(self):
        p = filedialog.askopenfilename(
            title="Select LS-DYNA deck",
            filetypes=[("LS-DYNA keyword", "*.k *.key *.dyn"), ("all", "*.*")])
        if p:
            self.deck_var.set(p)
            self.out_var.set("")

    def browse_out(self):
        p = filedialog.asksaveasfilename(
            title="Output deck", defaultextension=".k",
            filetypes=[("LS-DYNA keyword", "*.k"), ("all", "*.*")])
        if p:
            self.out_var.set(p)

    def _poll(self):
        try:
            while True:
                kind, payload = self.q.get_nowait()
                if kind == "log":
                    self._append(payload)
                elif kind == "status":
                    self.status_var.set(payload)
                elif kind == "done":
                    self.busy = False
                    self.convert_btn.configure(state="normal")
                elif kind == "error":
                    messagebox.showerror("kunit", payload)
        except queue.Empty:
            pass
        self.root.after(100, self._poll)

    def _start(self, target):
        if self.busy:
            return
        deck = self.deck_var.get().strip()
        if not deck or not Path(deck).is_file():
            messagebox.showerror("kunit", "Select an existing .k deck first.")
            return
        self.busy = True
        self.convert_btn.configure(state="disabled")
        self._clear()
        threading.Thread(target=target, args=(deck,), daemon=True).start()

    # ── actions ─────────────────────────────────────────────────────────────
    def on_detect(self):
        self._start(self._detect_worker)

    def _detect_worker(self, deck: str):
        try:
            self.q.put(("status", "detecting..."))
            v = detect(deck, follow_includes=self.v_includes.get())
            self.q.put(("log", v.table()))
            if v.evidence:
                self.q.put(("log", "\nevidence:"))
                for e in v.evidence:
                    self.q.put(("log", f"  - {e}"))
            if v.system is None:
                self.q.put(("log", "\nno usable evidence - set From manually"))
                self.q.put(("status", "no evidence"))
            else:
                tag = "  [AMBIGUOUS - verify!]" if v.ambiguous else ""
                self.q.put(("log", f"\ndetected: {v.system.describe()}{tag}"))
                self.from_var.set(v.system.key)
                self.q.put(("status", f"detected {v.system.key}"
                            + (" (ambiguous)" if v.ambiguous else "")))
        except Exception as e:
            self.q.put(("error", f"detect failed:\n{e}"))
            self.q.put(("log", traceback.format_exc()))
            self.q.put(("status", "detect failed"))
        finally:
            self.q.put(("done", None))

    def on_convert(self):
        self._start(self._convert_worker)

    def _convert_worker(self, deck: str):
        try:
            self.q.put(("status", "converting..."))
            dst = parse_system(self.to_var.get())
            from_spec = self.from_var.get().strip().lower()
            if from_spec in ("", "auto"):
                v = detect(deck, follow_includes=self.v_includes.get())
                if v.system is None or v.ambiguous:
                    self.q.put(("log", v.table()))
                    self.q.put(("error", "Auto-detection is not confident - "
                                "set From explicitly."))
                    self.q.put(("status", "ambiguous source units"))
                    return
                src = v.system
                self.q.put(("log", f"auto-detected source: {src.describe()}\n"))
            else:
                src = parse_system(from_spec)
            if src == dst:
                self.q.put(("log", "source == target, nothing to do"))
                self.q.put(("status", "nothing to do"))
                return
            if self.v_inplace.get():
                out = deck
            else:
                out = self.out_var.get().strip() or str(
                    Path(deck).with_name(f"{Path(deck).stem}__{dst.key}"
                                         f"{Path(deck).suffix}"))
                self.out_var.set(out)
            overrides = parse_curve_overrides(
                self.curves_var.get().split() or None)
            ctx = convert(deck, src, dst, out,
                          blast_unit=5 if self.v_blast5.get() else None,
                          allow_unknown=self.v_unknown.get(),
                          follow_includes=self.v_includes.get(),
                          dry_run=self.v_dry.get(),
                          curve_overrides=overrides,
                          self_check=self.v_selfcheck.get(),
                          verify_roundtrip=self.v_roundtrip.get(),
                          backup=self.v_backup.get())
            txt = report(ctx, src, dst)
            if self.v_dry.get():
                self.q.put(("log", "DRY RUN - no files written\n"))
            self.q.put(("log", txt))
            if not self.v_dry.get():
                log = out + ".kunit.log"
                with open(log, "w", encoding="utf-8") as fh:
                    fh.write(txt + "\n")
                self.q.put(("log", f"\nlog: {log}"))
            bad = ctx.self_check and ctx.self_check.startswith("FAILED")
            self.q.put(("status", "self-check FAILED - inspect output!"
                        if bad else ("dry run complete" if self.v_dry.get()
                                     else "converted OK")))
        except (ConvertError, ValueError, SystemExit) as e:
            self.q.put(("log", f"ERROR: {e}"))
            self.q.put(("error", str(e)))
            self.q.put(("status", "conversion failed"))
        except Exception as e:
            self.q.put(("error", f"unexpected error:\n{e}"))
            self.q.put(("log", traceback.format_exc()))
            self.q.put(("status", "conversion failed"))
        finally:
            self.q.put(("done", None))


def main() -> int:
    root = tk.Tk()
    try:
        ttk.Style().theme_use("vista")
    except tk.TclError:
        pass
    App(root)
    root.mainloop()
    return 0


if __name__ == "__main__":
    main()
