import numpy as np
import torch
from typing import Dict, Tuple
import plotly.graph_objects as go
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from tabulate import tabulate

class EvaluateAtmoSeer:
    def __init__(
        self, 
        gas_type: str, 
        model: torch.nn.Module, 
        test_loader: torch.utils.data.DataLoader
    ) -> None:
        self.gas_type = gas_type.upper()
        self.model = model
        self.test_loader = test_loader
        self.true_values, self.predictions = self._get_predictions()
        self.metrics = self._calculate_metrics()
        self._display_metrics_table()
        self._create_performance_plot()
    
    def _get_predictions(self) -> Tuple[np.ndarray, np.ndarray]:
        self.model.eval()
        true_values = []
        predictions = []
        
        with torch.no_grad():
            for X_batch, y_batch in self.test_loader:
                X_batch = X_batch.to(next(self.model.parameters()).device)
                y_pred = self.model(X_batch)
                true_values.extend(y_batch.numpy())
                predictions.extend(y_pred.cpu().numpy())
        
        return np.array(true_values).reshape(-1), np.array(predictions).reshape(-1)
    
    def _calculate_metrics(self) -> Dict[str, float]:
        metrics = {
            'R² Score': r2_score(self.true_values, self.predictions),
            'MSE': mean_squared_error(self.true_values, self.predictions),
            'RMSE': np.sqrt(mean_squared_error(self.true_values, self.predictions)),
            'MAE': mean_absolute_error(self.true_values, self.predictions),
            'MAPE': np.mean(np.abs((self.true_values - self.predictions) / self.true_values)) * 100,
            'Explained Variance': np.var(self.predictions) / np.var(self.true_values)
        }
        
        residuals = self.true_values - self.predictions
        metrics.update({
            'Mean Bias': np.mean(residuals),
            'Residual Std': np.std(residuals),
            'Max Error': np.max(np.abs(residuals))
        })
        
        return metrics
    
    def _display_metrics_table(self) -> None:
        table_data = []
        for metric, value in self.metrics.items():
            if abs(value) < 0.01 or abs(value) > 1000:
                formatted_value = f"{value:.2e}"
            else:
                formatted_value = f"{value:.4f}"
            table_data.append([metric, formatted_value])
        
        print(f"\n{self.gas_type} Model Performance Metrics")
        print("=" * 40)
        print(tabulate(table_data, headers=['Metric', 'Value'], 
                      tablefmt='fancy_grid', numalign='right'))
    
    def _create_performance_plot(self) -> None:
        z = np.polyfit(self.true_values, self.predictions, 1)
        regression_line = np.poly1d(z)
        residuals = self.predictions - self.true_values
        std_residuals = np.std(residuals)
        prediction_interval = 1.96 * std_residuals
        
        fig = go.Figure()
        
        fig.add_trace(go.Scatter(
            x=self.true_values,
            y=self.predictions,
            mode='markers',
            name='Predictions',
            marker=dict(
                color='orange',
                size=8,
                opacity=0.6
            )
        ))
        
        min_val = min(self.true_values.min(), self.predictions.min())
        max_val = max(self.true_values.max(), self.predictions.max())
        line_range = np.linspace(min_val, max_val, 100)
        
        fig.add_trace(go.Scatter(
            x=line_range,
            y=line_range,
            mode='lines',
            name='Perfect Prediction',
            line=dict(color='white', dash='dash')
        ))
        
        fig.add_trace(go.Scatter(
            x=line_range,
            y=regression_line(line_range),
            mode='lines',
            name='Regression Line',
            line=dict(color='orange')
        ))
        
        fig.add_trace(go.Scatter(
            x=line_range,
            y=line_range + prediction_interval,
            mode='lines',
            name='95% Prediction Interval',
            line=dict(color='orange', dash='dot'),
            showlegend=False
        ))
        
        fig.add_trace(go.Scatter(
            x=line_range,
            y=line_range - prediction_interval,
            mode='lines',
            line=dict(color='orange', dash='dot'),
            fill='tonexty',
            fillcolor='rgba(255, 165, 0, 0.1)',
            name='95% Prediction Interval'
        ))
        
        fig.update_layout(
            title=dict(
                text=f'AtmoSeer {self.gas_type} Model Performance',
                font=dict(size=24, color='white')
            ),
            xaxis_title=f"Actual {self.gas_type} Concentration (ppm)",
            yaxis_title=f"Predicted {self.gas_type} Concentration (ppm)",
            plot_bgcolor='black',
            paper_bgcolor='black',
            font=dict(color='white'),
            showlegend=True,
            legend=dict(
                bgcolor='rgba(0,0,0,0)',
                bordercolor='white',
                borderwidth=1
            ),
            width=1000,
            height=800
        )
        
        fig.update_xaxes(
            gridcolor='rgba(128,128,128,0.2)',
            zerolinecolor='rgba(128,128,128,0.2)'
        )
        fig.update_yaxes(
            gridcolor='rgba(128,128,128,0.2)',
            zerolinecolor='rgba(128,128,128,0.2)'
        )
        
        r2 = self.metrics['R² Score']
        rmse = self.metrics['RMSE']
        fig.add_annotation(
            text=f'R² = {r2:.4f}<br>RMSE = {rmse:.4f}',
            xref='paper',
            yref='paper',
            x=0.02,
            y=0.98,
            showarrow=False,
            font=dict(size=16, color='white'),
            bgcolor='rgba(0,0,0,0.5)',
            bordercolor='white',
            borderwidth=1
        )
        
        fig.show()
    
    def get_metrics(self) -> Dict[str, float]:
        return self.metrics