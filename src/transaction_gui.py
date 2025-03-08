import tkinter as tk
from tkinter import ttk, filedialog
import pandas as pd
import numpy as np
from pred_transactions import return_model_and_vectorizer, predict_categories

class TransactionLabellerGUI:
    def __init__(self, root):
        self.root = root
        self.root.title("Transaction Labeller")
        
        # Load model and vectorizer
        self.model, self.vectorizer = return_model_and_vectorizer()
        
        # Setup GUI components
        self.setup_gui()
        
        # Data storage
        self.transactions_df = None
        self.category_dropdowns = {}  # Store references to category dropdowns
        
    def setup_gui(self):
        # Main container
        main_frame = ttk.Frame(self.root, padding="5")
        main_frame.grid(row=0, column=0, sticky=(tk.N, tk.S, tk.E, tk.W))
        
        # Configure weights
        self.root.grid_rowconfigure(0, weight=1)
        self.root.grid_columnconfigure(0, weight=1)
        main_frame.grid_rowconfigure(1, weight=1)
        main_frame.grid_columnconfigure(0, weight=1)
        
        # File loading section
        load_frame = ttk.Frame(main_frame, padding="5")
        load_frame.grid(row=0, column=0, sticky=(tk.W, tk.E))
        ttk.Button(load_frame, text="Load Spreadsheet", command=self.load_spreadsheet).grid(row=0, column=0)
        ttk.Button(load_frame, text="Save", command=self.save_results).grid(row=0, column=1)
        
        # Create canvas and scrollbar for scrolling
        self.canvas = tk.Canvas(main_frame)
        scrollbar = ttk.Scrollbar(main_frame, orient="vertical", command=self.canvas.yview)
        self.scrollable_frame = ttk.Frame(self.canvas)
        
        self.scrollable_frame.bind(
            "<Configure>",
            lambda e: self.canvas.configure(scrollregion=self.canvas.bbox("all"))
        )
        
        self.canvas.create_window((0, 0), window=self.scrollable_frame, anchor="nw")
        self.canvas.configure(yscrollcommand=scrollbar.set)
        
        # Place canvas and scrollbar
        self.canvas.grid(row=1, column=0, sticky=(tk.N, tk.S, tk.E, tk.W))
        scrollbar.grid(row=1, column=1, sticky=(tk.N, tk.S))
        
        # Headers
        ttk.Label(self.scrollable_frame, text="Transaction", font=('Arial', 10, 'bold')).grid(
            row=0, column=0, padx=5, pady=5, sticky=tk.W)
        ttk.Label(self.scrollable_frame, text="Predicted Categories", font=('Arial', 10, 'bold')).grid(
            row=0, column=1, padx=5, pady=5, sticky=tk.W)
        ttk.Label(self.scrollable_frame, text="Selected Category", font=('Arial', 10, 'bold')).grid(
            row=0, column=2, padx=5, pady=5, sticky=tk.W)
        
        # Add mousewheel scrolling
        self.canvas.bind_all("<MouseWheel>", self._on_mousewheel)
        
    def _on_mousewheel(self, event):
        self.canvas.yview_scroll(int(-1*(event.delta/120)), "units")
        
    def load_spreadsheet(self):
        filename = filedialog.askopenfilename(filetypes=[("CSV files", "*.csv"), ("Excel files", "*.xlsx")])
        if filename:
            if filename.endswith('.csv'):
                self.transactions_df = pd.read_csv(filename)
            else:
                self.transactions_df = pd.read_excel(filename)
            
            # Add Category column if it doesn't exist
            if 'Category' not in self.transactions_df.columns:
                self.transactions_df['Category'] = ''
            
            # Clear previous content
            for widget in self.scrollable_frame.winfo_children():
                widget.destroy()
            
            # Recreate headers
            ttk.Label(self.scrollable_frame, text="Transaction", font=('Arial', 10, 'bold')).grid(
                row=0, column=0, padx=5, pady=5, sticky=tk.W)
            ttk.Label(self.scrollable_frame, text="Predicted Categories", font=('Arial', 10, 'bold')).grid(
                row=0, column=1, padx=5, pady=5, sticky=tk.W)
            ttk.Label(self.scrollable_frame, text="Selected Category", font=('Arial', 10, 'bold')).grid(
                row=0, column=2, padx=5, pady=5, sticky=tk.W)
            
            # Get predictions for all transactions
            predictions = predict_categories(self.model, self.vectorizer, self.transactions_df['Name'].values, n=5)
            
            # Create rows for each transaction
            for idx, (transaction, pred_categories) in enumerate(zip(self.transactions_df['Name'], predictions), 1):
                # Transaction name
                ttk.Label(self.scrollable_frame, text=transaction).grid(
                    row=idx, column=0, padx=5, pady=2, sticky=tk.W)
                
                # Predicted categories buttons frame
                pred_frame = ttk.Frame(self.scrollable_frame)
                pred_frame.grid(row=idx, column=1, padx=5, pady=2, sticky=tk.W)
                
                for i, category in enumerate(pred_categories):
                    ttk.Button(pred_frame, text=category,
                             command=lambda row=idx-1, cat=category: self.select_category(row, cat)).pack(
                                 side=tk.LEFT, padx=2)
                
                # Dropdown for all categories
                category_var = tk.StringVar(value=self.transactions_df.at[idx-1, 'Category'])
                category_dropdown = ttk.Combobox(self.scrollable_frame, textvariable=category_var, width=20)
                category_dropdown['values'] = sorted(self.model.classes_)
                category_dropdown.grid(row=idx, column=2, padx=5, pady=2, sticky=tk.W)
                
                # Store reference to dropdown
                self.category_dropdowns[idx-1] = category_dropdown
                
                # Bind dropdown selection
                category_dropdown.bind('<<ComboboxSelected>>',
                                    lambda e, row=idx-1, var=category_var: self.select_category(row, var.get()))
    
    def select_category(self, row_index, category):
        self.transactions_df.at[row_index, 'Category'] = category
        # Update the dropdown to show the selected category
        if row_index in self.category_dropdowns:
            self.category_dropdowns[row_index].set(category)
    
    def save_results(self):
        if self.transactions_df is not None:
            filename = filedialog.asksaveasfilename(
                defaultextension=".csv",
                filetypes=[("CSV files", "*.csv")])
            if filename:
                self.transactions_df.to_csv(filename, index=False)

def main():
    root = tk.Tk()
    app = TransactionLabellerGUI(root)
    root.mainloop()

if __name__ == "__main__":
    main()
