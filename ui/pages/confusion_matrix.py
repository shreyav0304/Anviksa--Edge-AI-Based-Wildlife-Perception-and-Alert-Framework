"""Responsive, single-matrix viewer with explicit evaluation scope and class key."""
from __future__ import annotations
import tkinter as tk
from tkinter import ttk
from typing import Callable
import numpy as np
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
from matplotlib.figure import Figure
from matplotlib.ticker import PercentFormatter
from ..components import SectionCard
from ..services.confusion_data import ACTIVE_MODEL, ConfusionMatrixRecord
from ..services.image_inference import friendly_name
from .base import ScrollablePage

HISTORICAL_GROUP = "All 3 historical models (7 classes)"
HISTORICAL_MODELS = ("Custom CNN", "MobileNetV2", "MobileNetV3 Large")

class ConfusionMatrixPage(ScrollablePage):
    MIN_CONTENT_WIDTH = 360

    def __init__(self, parent: tk.Misc, records: dict[str, ConfusionMatrixRecord], navigate: Callable[[str], None]):
        super().__init__(parent)
        if ACTIVE_MODEL not in records:
            raise ValueError("Active MobileNetV3 evaluation is missing")
        self.records, self.navigate = records, navigate
        self.model_var = tk.StringVar(value=ACTIVE_MODEL)
        self.view_var = tk.StringVar(value="raw")
        self.selection_var = tk.StringVar()
        self.scope_var = tk.StringVar()
        self.headline_var = tk.StringVar()
        self.cell_var = tk.StringVar(value="Hover over a cell for its class names and exact count.")
        self.metric_vars, self.summary_vars = {}, {}
        self._wrap_labels = []
        self._resize_job = None
        self._render_width = 0
        self._render_size = None
        self.historical_charts = {}
        self._history_render_size = None
        self._build()
        self.canvas.bind("<Configure>", self._schedule_layout, add="+")
        self.refresh()

    @staticmethod
    def _pct(value):
        return f"{value:.2%}"

    def _text(self, parent, **kwargs):
        label = ttk.Label(parent, justify="left", wraplength=600, **kwargs)
        self._wrap_labels.append(label)
        return label

    def _build(self):
        self._text(self.content, text="Confusion Matrix", style="PageTitle.TLabel").pack(anchor="w")
        self._text(self.content, textvariable=self.headline_var, style="BodyStrong.TLabel").pack(anchor="w", pady=(4,8))
        controls = ttk.Frame(self.content, style="Card.TFrame", padding=10)
        controls.pack(fill="x")
        ttk.Label(controls, text="Evaluation", style="Body.TLabel").pack(anchor="w")
        self.selector = ttk.Combobox(controls, textvariable=self.model_var, values=(HISTORICAL_GROUP, *self.records), state="readonly", width=20)
        self.selector.pack(fill="x", pady=(4,8))
        self.selector.bind("<<ComboboxSelected>>", lambda _event:self.refresh())
        choices = ttk.Frame(controls, style="Card.TFrame")
        choices.pack(fill="x")
        ttk.Radiobutton(choices,text="Counts",variable=self.view_var,value="raw",command=self.refresh).pack(side="left")
        ttk.Radiobutton(choices,text="Row %",variable=self.view_var,value="normalized",command=self.refresh).pack(side="left",padx=8)
        ttk.Button(controls,text="Show all 3 historical models",command=self.show_historical_models).pack(anchor="w",pady=(8,0))
        ttk.Button(controls,text="Return to active MobileNetV3",command=self.show_active_model).pack(anchor="w",pady=(8,0))
        self._text(self.content,textvariable=self.selection_var,style="BodyStrong.TLabel").pack(anchor="w",pady=(10,4))
        self._text(self.content,textvariable=self.scope_var,style="Notice.TLabel").pack(anchor="w",pady=(0,8))

        self.historical_host = ttk.Frame(self.content,style="Page.TFrame")
        self.matrix_card = SectionCard(self.content, "Actual labels vs predictions")
        self.matrix_card.pack(fill="x")
        self.figure = Figure(figsize=(6.5,4.6),dpi=100,facecolor="white")
        self.axis = self.figure.add_subplot(111)
        self.chart = FigureCanvasTkAgg(self.figure,master=self.matrix_card)
        self.chart.get_tk_widget().configure(width=320,height=340,highlightthickness=0)
        self.chart.get_tk_widget().pack(anchor="center")
        self.chart.mpl_connect("motion_notify_event",self._hover_cell)
        self._text(self.matrix_card,text="Axes use class IDs from the key below. Empty cells mean zero. Rows = actual; columns = predicted.",style="Body.TLabel").pack(anchor="w",pady=(6,4))
        self._text(self.matrix_card,textvariable=self.cell_var,style="Notice.TLabel").pack(anchor="w")

        key = self.class_key = SectionCard(self.content,"Class key and test support")
        key.pack(fill="x",pady=(12,0))
        host = ttk.Frame(key,style="Card.TFrame")
        host.pack(fill="x")
        columns=("id","class","tested","correct","recall")
        self.class_table = ttk.Treeview(host,columns=columns,show="headings",height=7)
        for column,title,width,minimum in (("id","ID",32,28),("class","Class",180,110),("tested","Tested",58,46),("correct","Correct",58,46),("recall","Recall",70,55)):
            self.class_table.heading(column,text=title)
            self.class_table.column(column,width=width,minwidth=minimum,anchor="w" if column=="class" else "center",stretch=column=="class")
        self.class_table.pack(side="left",fill="x",expand=True)
        vertical=ttk.Scrollbar(host,orient="vertical",command=self.class_table.yview)
        vertical.pack(side="right",fill="y")
        horizontal=ttk.Scrollbar(key,orient="horizontal",command=self.class_table.xview)
        horizontal.pack(fill="x")
        self.class_table.configure(yscrollcommand=vertical.set,xscrollcommand=horizontal.set)
        self.class_table.bind("<<TreeviewSelect>>",self._class_selected)
        self._text(key,text="Recall = correct / tested for that actual class. A 6/6 result describes six test images, not guaranteed field performance.",style="Notice.TLabel").pack(anchor="w",pady=(8,0))

        self.panels = ttk.Frame(self.content,style="Page.TFrame")
        self.panels.pack(fill="x",pady=(12,0))
        self._detail_panels = (self._metrics_panel(self.panels), self._interpretation_panel(self.panels))
        for index,card in enumerate(self._detail_panels):
            card.grid(row=index,column=0,sticky="nsew",pady=(0,10))
        self.panels.columnconfigure(0,weight=1)
        self._single_widgets=(self.matrix_card,self.class_key,self.panels)
        note=self.history_note=SectionCard(self.content,"Historical evaluations")
        note.pack(fill="x")
        self._text(note,text="Show all three historical matrices in separate stacked cards, or select a model individually. Older baselines use 7 classes and 545 images; the active model uses 15 classes and 670 images. These scores are not a direct comparison. EfficientNetB0 is incomplete and has no matrix.",style="Body.TLabel").pack(anchor="w")
        ttk.Button(note,text="Model comparison",command=lambda:self.navigate("comparison")).pack(anchor="w",pady=(8,0))

    def _metrics_panel(self,parent):
        card=SectionCard(parent,"Metrics from this matrix")
        for key,label in (("accuracy","Accuracy"),("macro_precision","Macro precision"),("macro_recall","Macro recall"),("macro_f1","Macro F1"),("venomous_recall","Venomous recall"),("snake_macro_recall","Snake macro recall")):
            value=tk.StringVar()
            self.metric_vars[key]=value
            row=ttk.Frame(card,style="Card.TFrame")
            row.pack(fill="x",pady=3)
            ttk.Label(row,text=label,style="Body.TLabel").pack(side="left")
            ttk.Label(row,textvariable=value,style="BodyStrong.TLabel").pack(side="right")
        return card

    def _interpretation_panel(self,parent):
        card=SectionCard(parent,"Counts and errors")
        for key,label in (("total","Test images"),("correct","Correct"),("incorrect","Incorrect"),("matrix_accuracy","Accuracy"),("largest","Largest confusion"),("nv_to_v","Non-venomous to venomous"),("v_to_nv","Venomous to non-venomous")):
            value=tk.StringVar()
            self.summary_vars[key]=value
            row=ttk.Frame(card,style="Card.TFrame")
            row.pack(fill="x",pady=3)
            row.columnconfigure(1,weight=1)
            ttk.Label(row,text=label,style="Small.TLabel",wraplength=130).grid(row=0,column=0,sticky="w")
            self._text(row,textvariable=value,style="BodyStrong.TLabel").grid(row=0,column=1,sticky="e",padx=(10,0))
        self.interpretation_var=tk.StringVar()
        self._text(card,textvariable=self.interpretation_var,style="Notice.TLabel").pack(anchor="w",pady=(8,0))
        return card

    def _schedule_layout(self,_event=None):
        if self._resize_job is not None:
            self.after_cancel(self._resize_job)
        self._resize_job=self.after(100,self._fit_layout)

    def _fit_layout(self):
        self._resize_job=None
        width=max(300,self.canvas.winfo_width()-40)
        for label in self._wrap_labels:
            if label.winfo_exists():
                # Respect the immediate parent's actual width, including nested cards.
                label.configure(wraplength=max(120,min(width,label.master.winfo_width()-36)))
        two_columns=width>=780
        self.panels.columnconfigure(1,weight=1 if two_columns else 0)
        for index,card in enumerate(self._detail_panels):
            card.grid_configure(row=0 if two_columns else index,column=index if two_columns else 0,padx=(0,8) if two_columns and index==0 else (0,0))
        if self.model_var.get()==HISTORICAL_GROUP:
            chart_width=max(240,min(720,width-36))
            height=max(220,min(520,self.canvas.winfo_height()-60,int(chart_width*.78)))
            if (chart_width,height)!=self._history_render_size:
                self._history_render_size=(chart_width,height)
                for name,entry in self.historical_charts.items():
                    entry["chart"].get_tk_widget().configure(width=chart_width,height=height)
                    entry["axis"]=self._draw_chart(entry["figure"],entry["chart"],self.records[name],self.view_var.get()=="normalized",(chart_width,height))
            return
        chart_width=max(240,min(720,self.matrix_card.winfo_width()-36))
        height=max(220,min(520,self.canvas.winfo_height()-60,int(chart_width*.78)))
        self.chart.get_tk_widget().configure(width=chart_width,height=height)
        if (chart_width,height)!=self._render_size:
            self._render_size=(chart_width,height)
            self._render_width=chart_width
            self._render(self.records[self.model_var.get()],self.view_var.get()=="normalized")

    def show_historical_models(self):
        self.model_var.set(HISTORICAL_GROUP)
        self.refresh()
        self.canvas.xview_moveto(0)
        self.canvas.yview_moveto(0)

    def _ensure_historical_charts(self):
        if self.historical_charts:
            return
        for name in HISTORICAL_MODELS:
            record=self.records[name]
            card=SectionCard(self.historical_host,name)
            card.pack(fill="x",pady=(0,14))
            self._text(card,text=f"{record.correct}/{record.total} correct | {record.incorrect} errors | Accuracy {record.matrix_accuracy:.2%}",style="BodyStrong.TLabel").pack(anchor="w",pady=(0,8))
            figure=Figure(figsize=(6.5,4.6),dpi=100,facecolor="white")
            chart=FigureCanvasTkAgg(figure,master=card)
            chart.get_tk_widget().configure(width=320,height=340,highlightthickness=0)
            chart.get_tk_widget().pack(anchor="center")
            key=" | ".join(f"{index}: {friendly_name(label)}" for index,label in enumerate(record.class_names,1))
            self._text(card,text="Class key: "+key,style="Body.TLabel").pack(anchor="w",pady=(8,4))
            detail=tk.StringVar(value="Rows = actual; columns = predicted. Blank cells are zero. Hover for exact class names and counts.")
            self._text(card,textvariable=detail,style="Notice.TLabel").pack(anchor="w")
            self.historical_charts[name]=dict(figure=figure,chart=chart,axis=None,detail=detail)
            chart.mpl_connect("motion_notify_event",lambda event,model=name:self._hover_historical(event,model))

    def _hover_historical(self,event,name):
        entry=self.historical_charts[name]
        if event.inaxes is not entry["axis"] or event.xdata is None or event.ydata is None:
            return
        record=self.records[name]
        row,column=int(round(event.ydata)),int(round(event.xdata))
        if 0<=row<len(record.class_names) and 0<=column<len(record.class_names):
            count,support=int(record.raw[row,column]),int(record.raw[row].sum())
            entry["detail"].set(f"Actual: {friendly_name(record.class_names[row])} | Predicted: {friendly_name(record.class_names[column])}: {count}/{support} ({count/support:.2%})")

    def show_active_model(self):
        if self.model_var.get()!=ACTIVE_MODEL:
            self.model_var.set(ACTIVE_MODEL)
            self.refresh()
        self.canvas.xview_moveto(0)
        self.canvas.yview_moveto(0)

    def on_show(self):
        self.show_active_model()
        self._schedule_layout()

    def refresh(self):
        if self.model_var.get()==HISTORICAL_GROUP:
            self._ensure_historical_charts()
            for widget in self._single_widgets:
                widget.pack_forget()
            self.historical_host.pack(fill="x",before=self.history_note)
            self.headline_var.set("Custom CNN | MobileNetV2 | MobileNetV3 Large")
            self.selection_var.set("HISTORICAL COMPARISON ONLY: all 3 baselines. Inference still uses the active 15-class MobileNetV3.")
            self.scope_var.set("Each matrix uses the same historical 7-class, 545-image frozen test set. Scroll down for the next model. The active 15-class evaluation is separate.")
            self._history_render_size=None
            self._schedule_layout()
            return
        self.historical_host.pack_forget()
        for index,widget in enumerate(self._single_widgets):
            widget.pack(fill="x",pady=(0,0) if index==0 else (12,0),before=self.history_note)
        record=self.records[self.model_var.get()]
        self.headline_var.set(f"{record.correct}/{record.total} correct | {record.incorrect} errors | Accuracy {record.matrix_accuracy:.2%}")
        self.selection_var.set("ACTIVE DEPLOYMENT EVALUATION: MobileNetV3 Large, 15 classes" if record.model==ACTIVE_MODEL else f"HISTORICAL COMPARISON ONLY: {record.model}. Inference still uses MobileNetV3.")
        self.scope_var.set(record.evaluation_scope+". Cell percentages are recall by actual class, not confidence.")
        for key,variable in self.metric_vars.items():variable.set(self._pct(record.metrics[key]))
        source,target,count=record.largest_confusion
        values={"total":str(record.total),"correct":str(record.correct),"incorrect":str(record.incorrect),"matrix_accuracy":self._pct(record.matrix_accuracy),"largest":f"{friendly_name(source)} to {friendly_name(target)} ({count})","nv_to_v":str(record.nonvenomous_to_venomous),"v_to_nv":str(record.venomous_to_nonvenomous)}
        for key,value in values.items():self.summary_vars[key].set(value)
        self.interpretation_var.set(record.interpretation)
        for item in self.class_table.get_children():self.class_table.delete(item)
        for index,name in enumerate(record.class_names):
            support=int(record.raw[index].sum())
            correct=int(record.raw[index,index])
            self.class_table.insert("","end",iid=str(index),values=(index+1,friendly_name(name),support,correct,f"{correct/support:.2%}" if support else "N/A"))
        self.cell_var.set("Hover over a cell, or select a class in the key, for exact errors.")
        self._render(record,self.view_var.get()=="normalized")
        self._schedule_layout()

    def _render(self,record,normalized):
        self.axis=self._draw_chart(self.figure,self.chart,record,normalized,self._render_size)

    def _draw_chart(self,figure,chart,record,normalized,render_size):
        figure.clear()
        if render_size is not None:
            width,height=render_size
            figure.set_size_inches(width/figure.dpi,height/figure.dpi,forward=False)
        width,height=figure.get_size_inches()*figure.dpi
        side=max(100,min(width-110,height-85))
        left=max(45,(width-side-50)/2)
        bottom=45
        axis=figure.add_axes((left/width,bottom/height,side/width,side/height))
        matrix=record.normalized if normalized else record.raw
        image=axis.imshow(matrix,cmap="Blues",vmin=0,vmax=1 if normalized else max(1,int(record.raw.max())),interpolation="nearest")
        size=len(record.class_names)
        ticks=np.arange(size)
        axis.set(xticks=ticks,yticks=ticks,xticklabels=[str(i+1) for i in ticks],yticklabels=[str(i+1) for i in ticks],xlabel="Predicted class ID",ylabel="Actual class ID",title=f"{'Row percentages' if normalized else 'Image counts'} | {record.correct}/{record.total} correct")
        small=(render_size or (0,0))[1]<300
        axis.tick_params(axis="both",labelsize=5 if small else 8,length=0,pad=4)
        axis.xaxis.label.set_size(9)
        axis.yaxis.label.set_size(9)
        axis.title.set_size(10)
        axis.set_xticks(np.arange(-.5,size,1),minor=True)
        axis.set_yticks(np.arange(-.5,size,1),minor=True)
        axis.grid(which="minor",color="#dae2df",linewidth=.35)
        axis.tick_params(which="minor",length=0)
        height=render_size[1] if render_size else chart.get_tk_widget().winfo_height()
        font=4 if height<260 else 5 if height<300 else 6 if height<420 else 8
        for row,column in zip(*np.nonzero(record.raw)):
            value=matrix[row,column]
            text=f"{value*100:.1f}" if normalized else str(int(value))
            axis.text(column,row,text,ha="center",va="center",fontsize=font,color="white" if value>float(matrix.max())*.55 else "#173a2e")
        color_axis=figure.add_axes(((left+side+15)/width,bottom/height,14/width,side/height))
        colorbar=figure.colorbar(image,cax=color_axis)
        colorbar.ax.tick_params(labelsize=7)
        if normalized:colorbar.ax.yaxis.set_major_formatter(PercentFormatter(xmax=1,decimals=0))
        colorbar.set_label("Row %" if normalized else "Images",fontsize=8)
        chart.draw_idle()

        return axis

    def _hover_cell(self,event):
        if self.model_var.get()==HISTORICAL_GROUP:return
        if event.inaxes is not self.axis or event.xdata is None or event.ydata is None:return
        record=self.records[self.model_var.get()]
        row,column=int(round(event.ydata)),int(round(event.xdata))
        if not (0<=row<len(record.class_names) and 0<=column<len(record.class_names)):return
        support=int(record.raw[row].sum())
        count=int(record.raw[row,column])
        self.cell_var.set(f"Actual: {friendly_name(record.class_names[row])} | Predicted: {friendly_name(record.class_names[column])}: {count}/{support} ({count/support:.2%})" if support else "No test support for this class")

    def _class_selected(self,_event=None):
        if self.model_var.get()==HISTORICAL_GROUP:return
        selected=self.class_table.selection()
        if not selected:return
        row=int(selected[0])
        record=self.records[self.model_var.get()]
        errors=[f"{friendly_name(record.class_names[column])}: {int(value)}" for column,value in enumerate(record.raw[row]) if column!=row and value]
        self.cell_var.set(f"{friendly_name(record.class_names[row])} errors: "+("; ".join(errors) if errors else "None on this test split"))

    def close(self):
        if self._resize_job is not None:
            self.after_cancel(self._resize_job)
            self._resize_job=None
        charts=[self.chart,*[entry["chart"] for entry in self.historical_charts.values()]]
        for chart in charts:
            pending=getattr(chart,"_idle_draw_id",None)
            if pending:
                chart.get_tk_widget().after_cancel(pending)
                chart._idle_draw_id=None
