import tkinter as tk
from tkinter import ttk, messagebox, filedialog
import matplotlib.pyplot as plt
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
import pandas as pd
import numpy as np
import torch
import pickle
import os
import threading
from PIL import Image, ImageTk

class StockPredictionApp:
    def __init__(self, root):
        self.root = root
        self.root.title("Stock Prediction System")
        self.root.geometry("1000x700")
        self.root.configure(bg="#f0f0f0")
        
        # Model variables
        self.data = None
        self.model = None
        self.inner_edge = None
        self.outer_edge = None
        self.test_data = None
        self.predictions = None
        self.stocks_df = None
        
        # Create main frames
        self.create_header()
        self.create_sidebar()
        self.create_main_content()
        
        # Try to load stock names
        try:
            self.stocks_df = pd.read_csv("NIFTY50_category.csv")
        except:
            pass
    
    def create_header(self):
        header_frame = tk.Frame(self.root, bg="#2c3e50", height=60)
        header_frame.pack(fill=tk.X)
        
        title_label = tk.Label(header_frame, text="Stock Prediction System", 
                              font=("Arial", 18, "bold"), bg="#2c3e50", fg="white")
        title_label.pack(side=tk.LEFT, padx=20, pady=10)
    
    def create_sidebar(self):
        sidebar_frame = tk.Frame(self.root, bg="#34495e", width=200)
        sidebar_frame.pack(side=tk.LEFT, fill=tk.Y)
        
        # Make sure sidebar maintains its width
        sidebar_frame.pack_propagate(False)
        
        # Create sidebar buttons
        btn_styles = {"font": ("Arial", 12), "bg": "#34495e", "fg": "white", 
                     "activebackground": "#2c3e50", "activeforeground": "white",
                     "bd": 0, "width": 20, "anchor": "w", "padx": 10}
        
        tk.Button(sidebar_frame, text="Load Model", 
                 command=self.load_model, **btn_styles).pack(pady=(30, 10), fill=tk.X)
        
        tk.Button(sidebar_frame, text="Load Data", 
                 command=self.load_data, **btn_styles).pack(pady=10, fill=tk.X)
        
        tk.Button(sidebar_frame, text="Run Prediction", 
                 command=self.run_prediction, **btn_styles).pack(pady=10, fill=tk.X)
        
        tk.Button(sidebar_frame, text="Show Top Stocks", 
                 command=self.show_top_stocks, **btn_styles).pack(pady=10, fill=tk.X)
        
        tk.Button(sidebar_frame, text="Performance Metrics", 
                 command=self.show_metrics, **btn_styles).pack(pady=10, fill=tk.X)
        
        tk.Button(sidebar_frame, text="Exit", 
                 command=self.root.quit, **btn_styles).pack(pady=(50, 10), fill=tk.X)
        
        # Status section
        status_frame = tk.Frame(sidebar_frame, bg="#34495e", pady=20)
        status_frame.pack(side=tk.BOTTOM, fill=tk.X)
        
        tk.Label(status_frame, text="Status:", font=("Arial", 10, "bold"), 
                bg="#34495e", fg="white").pack(anchor="w", padx=10)
        
        self.status_label = tk.Label(status_frame, text="Ready", font=("Arial", 10), 
                                    bg="#34495e", fg="#2ecc71")
        self.status_label.pack(anchor="w", padx=10, pady=5)
    
    def create_main_content(self):
        self.main_frame = tk.Frame(self.root, bg="#f0f0f0")
        self.main_frame.pack(side=tk.RIGHT, fill=tk.BOTH, expand=True, padx=20, pady=20)
        
        # Welcome message
        welcome_frame = tk.Frame(self.main_frame, bg="#f0f0f0")
        welcome_frame.pack(fill=tk.BOTH, expand=True)
        
        tk.Label(welcome_frame, text="Stock Prediction System", 
                font=("Arial", 24, "bold"), bg="#f0f0f0").pack(pady=(50, 20))
        
        tk.Label(welcome_frame, text="A Graph Attention Network Based Tool for Stock Predictions", 
                font=("Arial", 14), bg="#f0f0f0").pack()
        
        instructions = """
        How to use:
        1. Load your trained model using the 'Load Model' button
        2. Load your data file using the 'Load Data' button
        3. Run the prediction using the 'Run Prediction' button
        4. View top stock predictions and performance metrics
        """
        
        tk.Label(welcome_frame, text=instructions, font=("Arial", 12), 
                bg="#f0f0f0", justify=tk.LEFT).pack(pady=30)
        
        # Frame for graphs and results that will be populated later
        self.results_frame = tk.Frame(self.main_frame, bg="#f0f0f0")
        
        # Tabs for different views
        self.tabs = ttk.Notebook(self.main_frame)
        
        # Create tabs
        self.top_stocks_tab = tk.Frame(self.tabs, bg="#f0f0f0")
        self.metrics_tab = tk.Frame(self.tabs, bg="#f0f0f0")
        self.chart_tab = tk.Frame(self.tabs, bg="#f0f0f0")
        
        self.tabs.add(self.top_stocks_tab, text="Top Stocks")
        self.tabs.add(self.metrics_tab, text="Metrics")
        self.tabs.add(self.chart_tab, text="Charts")
    
    def load_model(self):
        try:
            model_path = filedialog.askopenfilename(
                title="Select Model File",
                filetypes=[("PyTorch Model", "*.pth")]
            )
            
            if not model_path:
                return
            
            # Update status
            self.update_status("Loading model...", "orange")
            
            # Import here to avoid loading all modules at startup
            from torch_geometric.nn import GATConv
            import torch.nn as nn
            import torch.nn.functional as F
            
            # Load model classes
            class AttentionBlock(nn.Module):
                def __init__(self, time_step, dim):
                    super(AttentionBlock, self).__init__()
                    self.attention_matrix = nn.Linear(time_step, time_step)

                def forward(self, inputs):  
                    inputs_t = torch.transpose(inputs, 2, 1)
                    attention_weight = self.attention_matrix(inputs_t)
                    attention_probs = F.softmax(attention_weight, dim=-1) 
                    attention_probs = torch.transpose(attention_probs, 2, 1)
                    attention_vec = torch.mul(attention_probs, inputs)
                    attention_vec = torch.sum(attention_vec, dim=1)
                    return attention_vec, attention_probs

            class SequenceEncoder(nn.Module):
                def __init__(self, input_dim, time_step, hidden_dim):
                    super(SequenceEncoder, self).__init__()
                    self.encoder = nn.GRU(input_size=input_dim, hidden_size=hidden_dim, num_layers=1, batch_first=True)
                    self.attention_block = AttentionBlock(time_step, hidden_dim) 
                    self.dropout = nn.Dropout(0.2)
                    self.dim = hidden_dim
                
                def forward(self, seq):
                    seq_vector, _ = self.encoder(seq)
                    seq_vector = self.dropout(seq_vector)
                    attention_vec, _ = self.attention_block(seq_vector)
                    attention_vec = attention_vec.view(-1, 1, self.dim)
                    return attention_vec

            class CategoricalGraphAtt(nn.Module):
                def __init__(self, input_dim, time_step, hidden_dim, inner_edge, outer_edge, no_of_weeks_to_look_back, use_gru, device):
                    super(CategoricalGraphAtt, self).__init__()

                    # basic parameters
                    self.dim = hidden_dim
                    self.input_dim = input_dim
                    self.time_step = time_step
                    self.inner_edge = inner_edge
                    self.outer_edge = outer_edge
                    self.no_of_weeks_to_look_back = no_of_weeks_to_look_back
                    self.use_gru = use_gru
                    self.device = device

                    # hidden layers
                    self.pool_attention = AttentionBlock(10, hidden_dim)
                    if self.use_gru:
                        self.weekly_encoder = nn.GRU(hidden_dim, hidden_dim)
                    self.encoder_list = nn.ModuleList([SequenceEncoder(input_dim, time_step, hidden_dim) for _ in range(no_of_weeks_to_look_back)]) 
                    self.cat_gat = GATConv(hidden_dim, hidden_dim)
                    self.inner_gat = GATConv(hidden_dim, hidden_dim)
                    self.weekly_attention = AttentionBlock(no_of_weeks_to_look_back, hidden_dim)
                    self.fusion = nn.Linear(hidden_dim*3, hidden_dim)

                    # output layer 
                    self.reg_layer = nn.Linear(hidden_dim, 1)
                    self.cls_layer = nn.Linear(hidden_dim, 1)

                def forward(self, weekly_batch):
                    weekly_embedding = self.encoder_list[0](weekly_batch[0].view(-1, self.time_step, self.input_dim))
                    
                    for week_idx in range(1, self.no_of_weeks_to_look_back):
                        weekly_inp = weekly_batch[week_idx]
                        weekly_inp = weekly_inp.view(-1, self.time_step, self.input_dim)
                        week_stock_embedding = self.encoder_list[week_idx](weekly_inp)
                        weekly_embedding = torch.cat((weekly_embedding, week_stock_embedding), dim=1)
                    
                    if self.use_gru:
                        weekly_embedding, _ = self.weekly_encoder(weekly_embedding)
                    weekly_att_vector, _ = self.weekly_attention(weekly_embedding)

                    inner_graph_embedding = self.inner_gat(weekly_att_vector, self.inner_edge)
                    inner_graph_embedding = inner_graph_embedding.view(5, 10, -1)

                    weekly_att_vector = weekly_att_vector.view(5, 10, -1)
                    category_vectors, _ = self.pool_attention(weekly_att_vector)

                    category_vectors = self.cat_gat(category_vectors, self.outer_edge)
                    category_vectors = category_vectors.unsqueeze(1).expand(-1, 10, -1)

                    fusion_vec = torch.cat((weekly_att_vector, category_vectors, inner_graph_embedding), dim=-1)
                    fusion_vec = torch.relu(self.fusion(fusion_vec))

                    reg_output = self.reg_layer(fusion_vec)
                    reg_output = torch.flatten(reg_output)
                    cls_output = torch.sigmoid(self.cls_layer(fusion_vec))
                    cls_output = torch.flatten(cls_output)

                    return reg_output, cls_output

                def predict_toprank(self, test_data, device, top_k=5):
                    y_pred_all_reg, y_pred_all_cls = [], []
                    test_w1, test_w2, test_w3, test_w4 = test_data
                    for idx, _ in enumerate(test_w2):
                        if idx <= 361:
                            batch_x1, batch_x2, batch_x3, batch_x4 = (
                                test_w1[idx].to(self.device),
                                test_w2[idx].to(self.device),
                                test_w3[idx].to(self.device),
                                test_w4[idx].to(self.device),
                            )
                            batch_weekly = [batch_x1, batch_x2, batch_x3, batch_x4][-self.no_of_weeks_to_look_back:]
                            pred_reg, pred_cls = self.forward(batch_weekly)
                            pred_reg, pred_cls = pred_reg.cpu().detach().numpy(), pred_cls.cpu().detach().numpy()
                            y_pred_all_reg.extend(pred_reg.tolist())
                            y_pred_all_cls.extend(pred_cls.tolist())
                    return y_pred_all_reg, y_pred_all_cls
            
            # Load inner and outer edges
            inner_path = filedialog.askopenfilename(
                title="Select Inner Edge File (.npy)",
                filetypes=[("NumPy File", "*.npy")]
            )
            if not inner_path:
                self.update_status("Model loading cancelled", "red")
                return
                
            outer_path = filedialog.askopenfilename(
                title="Select Outer Edge File (.npy)",
                filetypes=[("NumPy File", "*.npy")]
            )
            if not outer_path:
                self.update_status("Model loading cancelled", "red")
                return
            
            # Load edges
            self.inner_edge = np.array(np.load(inner_path))
            self.outer_edge = np.array(np.load(outer_path))
            self.inner_edge = torch.tensor(self.inner_edge.T, dtype=torch.int64)
            self.outer_edge = torch.tensor(self.outer_edge.T, dtype=torch.int64)
            
            # Initialize model
            device = "cpu"
            input_dim = 23  # From your notebook
            time_step = 7   # From your notebook
            hidden_dim = 16
            agg_week_num = 3
            use_gru = False
            
            # Create and load model
            self.model = CategoricalGraphAtt(
                input_dim=input_dim,
                time_step=time_step,
                hidden_dim=hidden_dim,
                inner_edge=self.inner_edge,
                outer_edge=self.outer_edge,
                no_of_weeks_to_look_back=agg_week_num,
                use_gru=use_gru,
                device=device
            )
            
            self.model.load_state_dict(torch.load(model_path))
            self.model.eval()
            
            self.update_status("Model loaded successfully", "green")
            messagebox.showinfo("Success", "Model loaded successfully!")
        
        except Exception as e:
            self.update_status(f"Error loading model: {str(e)}", "red")
            messagebox.showerror("Error", f"Failed to load model: {str(e)}")
    
    def load_data(self):
        try:
            data_path = filedialog.askopenfilename(
                title="Select Data File (.pkl)",
                filetypes=[("Pickle File", "*.pkl")]
            )
            
            if not data_path:
                return
            
            self.update_status("Loading data...", "orange")
            
            # Load data
            with open(data_path, "rb") as f:
                self.data = pickle.load(f)
            
            # Process test data
            test_w1 = torch.Tensor(self.data["test"]["x1"].astype(float)).float()
            test_w2 = torch.Tensor(self.data["test"]["x2"].astype(float)).float()
            test_w3 = torch.Tensor(self.data["test"]["x3"].astype(float)).float()
            test_w4 = torch.Tensor(self.data["test"]["x4"].astype(float)).float()
            self.test_data = [test_w1, test_w2, test_w3, test_w4]
            
            self.update_status("Data loaded successfully", "green")
            messagebox.showinfo("Success", "Data loaded successfully!")
        
        except Exception as e:
            self.update_status(f"Error loading data: {str(e)}", "red")
            messagebox.showerror("Error", f"Failed to load data: {str(e)}")
    
    def run_prediction(self):
        if self.model is None or self.data is None:
            messagebox.showerror("Error", "Please load both model and data first!")
            return
        
        try:
            self.update_status("Running prediction...", "orange")
            
            # Run in a separate thread to avoid freezing UI
            threading.Thread(target=self._run_prediction_thread).start()
        
        except Exception as e:
            self.update_status(f"Error running prediction: {str(e)}", "red")
            messagebox.showerror("Error", f"Failed to run prediction: {str(e)}")
    
    def _run_prediction_thread(self):
        try:
            # Run prediction
            y_pred_reg, y_pred_cls = self.model.predict_toprank(self.test_data, "cpu", top_k=5)
            
            # Store predictions
            self.predictions = np.array(y_pred_reg)[:18050]
            self.test_y = np.array(self.data["test"]["y_return_ratio"]).ravel()
            self.test_cls = np.array(self.data["test"]["y_up_or_down"])
            
            # Update UI from main thread
            self.root.after(0, self._prediction_done)
        
        except Exception as e:
            self.root.after(0, lambda: self.update_status(f"Error: {str(e)}", "red"))
    
    def _prediction_done(self):
        self.update_status("Prediction completed successfully", "green")
        messagebox.showinfo("Success", "Prediction completed successfully!")
        
        # Show the notebook tabs now that we have predictions
        self.tabs.pack(fill=tk.BOTH, expand=True)
        
        # Update the tabs with results
        self.show_top_stocks()
        self.show_metrics()
        self.show_charts()
    
    def show_top_stocks(self):
        if self.predictions is None or self.stocks_df is None:
            if self.predictions is None:
                messagebox.showerror("Error", "Please run prediction first!")
            else:
                messagebox.showerror("Error", "Stock data file not found!")
            return
        
        # Clear previous content
        for widget in self.top_stocks_tab.winfo_children():
            widget.destroy()
        
        # Create a frame for the stocks table
        frame = tk.Frame(self.top_stocks_tab, bg="#f0f0f0")
        frame.pack(fill=tk.BOTH, expand=True, padx=20, pady=20)
        
        tk.Label(frame, text="Top Performing Stocks Prediction", 
                font=("Arial", 16, "bold"), bg="#f0f0f0").pack(pady=(0, 20))
        
        # Create treeview widget
        columns = ("Rank", "Company", "Sector", "Predicted Return")
        tree = ttk.Treeview(frame, columns=columns, show="headings", height=10)
        
        # Define column headings
        for col in columns:
            tree.heading(col, text=col)
            width = 100 if col != "Company" else 200
            tree.column(col, width=width, anchor="center")
        
        # Create indices with predicted values
        with_indices = [(self.predictions[i], i) for i in range(len(self.predictions))]
        
        # Sort by prediction values in descending order
        with_indices.sort(key=lambda x: x[0], reverse=True)
        
        # Get top 20 stocks
        top_k = 20
        for i in range(min(top_k, len(with_indices))):
            pred_value, idx = with_indices[i]
            stock_idx = idx % 50  # Assuming 50 stocks as in your notebook
            
            if self.stocks_df is not None and stock_idx < len(self.stocks_df):
                company = self.stocks_df.iloc[stock_idx]['company']
                sector = self.stocks_df.iloc[stock_idx]['category']
            else:
                company = f"Stock {stock_idx}"
                sector = "Unknown"
            
            tree.insert("", "end", values=(
                i+1, 
                company, 
                sector, 
                f"{pred_value:.4f}"
            ))
        
        # Add scrollbar
        scrollbar = ttk.Scrollbar(frame, orient="vertical", command=tree.yview)
        tree.configure(yscrollcommand=scrollbar.set)
        scrollbar.pack(side="right", fill="y")
        tree.pack(fill="both", expand=True)
        
        # Select the first tab
        self.tabs.select(0)
    
    def show_metrics(self):
        if self.predictions is None:
            messagebox.showerror("Error", "Please run prediction first!")
            return
        
        # Clear previous content
        for widget in self.metrics_tab.winfo_children():
            widget.destroy()
        
        # Create metrics frame
        frame = tk.Frame(self.metrics_tab, bg="#f0f0f0")
        frame.pack(fill=tk.BOTH, expand=True, padx=20, pady=20)
        
        tk.Label(frame, text="Performance Metrics", 
                font=("Arial", 16, "bold"), bg="#f0f0f0").pack(pady=(0, 20))
        
        # Calculate metrics
        from sklearn.metrics import mean_absolute_error
        
        mae = round(mean_absolute_error(self.test_y[:len(self.predictions)], self.predictions), 4)
        
        # Calculate accuracy
        pred_cls = (self.predictions > 0) * 1
        acc_sum = sum(1 for i in range(len(pred_cls)) if self.test_cls.ravel()[i] == pred_cls[i])
        acc_score = acc_sum / len(pred_cls)
        
        # Create a styled frame for metrics
        metrics_frame = tk.Frame(frame, bg="white", bd=1, relief=tk.RIDGE)
        metrics_frame.pack(fill=tk.X, padx=50, pady=20)
        
        # Display metrics
        metrics = [
            ("Mean Absolute Error (MAE)", f"{mae:.4f}"),
            ("Prediction Accuracy", f"{acc_score:.4f}"),
        ]
        
        for i, (label, value) in enumerate(metrics):
            row_frame = tk.Frame(metrics_frame, bg="white" if i % 2 == 0 else "#f8f8f8")
            row_frame.pack(fill=tk.X)
            
            tk.Label(row_frame, text=label, font=("Arial", 12), 
                    bg=row_frame["bg"], anchor="w").pack(side=tk.LEFT, padx=20, pady=10)
            
            tk.Label(row_frame, text=value, font=("Arial", 12, "bold"), 
                    bg=row_frame["bg"]).pack(side=tk.RIGHT, padx=20, pady=10)
    
    def show_charts(self):
        if self.predictions is None:
            messagebox.showerror("Error", "Please run prediction first!")
            return
        
        # Clear previous content
        for widget in self.chart_tab.winfo_children():
            widget.destroy()
        
        # Create charts frame
        frame = tk.Frame(self.chart_tab, bg="#f0f0f0")
        frame.pack(fill=tk.BOTH, expand=True, padx=20, pady=20)
        
        tk.Label(frame, text="Prediction Visualization", 
                font=("Arial", 16, "bold"), bg="#f0f0f0").pack(pady=(0, 20))
        
        # Create figure for the chart
        fig = plt.Figure(figsize=(10, 6), dpi=100)
        ax = fig.add_subplot(111)
        
        # Use only the first 100 predictions to make the chart readable
        sample_size = 100
        indices = np.arange(sample_size)
        
        # Sort predictions
        sorted_indices = np.argsort(self.predictions[:sample_size])[::-1]
        sorted_preds = self.predictions[sorted_indices]
        sorted_actual = self.test_y[sorted_indices]
        
        # Plot
        ax.bar(indices, sorted_preds, alpha=0.7, label='Predicted')
        ax.plot(indices, sorted_actual[:sample_size], 'ro-', label='Actual', linewidth=2)
        
        ax.set_xlabel('Stock Index (Sorted by Prediction Value)')
        ax.set_ylabel('Return Ratio')
        ax.set_title('Predicted vs Actual Return Ratio (Top Predictions)')
        ax.legend()
        
        # Embed the chart in the tkinter window
        canvas = FigureCanvasTkAgg(fig, master=frame)
        canvas.draw()
        canvas.get_tk_widget().pack(fill=tk.BOTH, expand=True)
    
    def update_status(self, message, color="black"):
        self.status_label.config(text=message, fg=color)
        self.root.update_idletasks()

if __name__ == "__main__":
    root = tk.Tk()
    app = StockPredictionApp(root)
    root.mainloop()

def load_weights(self):
    try:
        file_path = filedialog.askopenfilename(title="Select Model Weights File", filetypes=[("PyTorch Weights", "*.pt")])
        if not file_path:
            return
            
        # Load state dict with weights_only=False to fix the PyTorch 2.6 compatibility issue
        state_dict = torch.load(file_path, map_location=torch.device('cpu'), weights_only=False)
        
        # Check if this is a full model or just weights
        if isinstance(state_dict, dict) and 'model_config' in state_dict:
            # Load configuration if available
            self.model_config = state_dict['model_config']
            state_dict = state_dict['state_dict']
            
            # Update UI with loaded config
            self.hidden_dim_var.set(str(self.model_config['hidden_dim']))
            self.use_gru_var.set(self.model_config['use_gru'])
        
        # We'll initialize the model later when we have all necessary parameters
        self.model_weights = state_dict
        self.status_label.config(text="Status: Model weights loaded successfully")
    except Exception as e:
        messagebox.showerror("Error", f"Failed to load model weights: {e}")