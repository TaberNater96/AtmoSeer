import numpy as np
import pandas as pd
import torch
from typing import Dict, List, Tuple, Optional
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
import plotly.graph_objects as go
from datetime import datetime, timedelta

class AtmoSeerEvaluator:
    def __init__(
        self,
        model: torch.nn.Module,
        data_loaders: Dict,
        device: Optional[torch.device] = None
    ):
        self.model = model
        self.data_loaders = data_loaders
        self.device = device or torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.model.to(self.device)
        self.target_scaler = data_loaders.get('target_scaler')
        
    def _evaluate_dataloader(
        self,
        dataloader: torch.utils.data.DataLoader
    ) -> Tuple[np.ndarray, np.ndarray]:
        self.model.eval()
        y_true_list = []
        y_pred_list = []
        
        with torch.no_grad():
            for X_batch, y_batch in dataloader:
                X_batch = X_batch.to(self.device)
                y_batch = y_batch.to(self.device)
                y_pred = self.model(X_batch)
                
                # Move predictions and targets back to CPU for numpy conversion
                y_true_list.extend(y_batch.cpu().numpy())
                y_pred_list.extend(y_pred.cpu().numpy())
        
        y_true = np.array(y_true_list)
        y_pred = np.array(y_pred_list)
        
        if self.target_scaler:
            y_true = self.target_scaler.inverse_transform(y_true.reshape(-1, 1)).flatten()
            y_pred = self.target_scaler.inverse_transform(y_pred.reshape(-1, 1)).flatten()
        
        return y_true, y_pred
    
    def evaluate(self) -> Dict[str, Dict[str, float]]:
        metrics = {}
        
        # Evaluate train set
        train_true, train_pred = self._evaluate_dataloader(self.data_loaders['train_loader'])
        metrics['train'] = {
            'mse': mean_squared_error(train_true, train_pred),
            'rmse': np.sqrt(mean_squared_error(train_true, train_pred)),
            'mae': mean_absolute_error(train_true, train_pred),
            'mape': np.mean(np.abs((train_true - train_pred) / train_true)) * 100,
            'r2': r2_score(train_true, train_pred)
        }
        
        # Evaluate test set
        test_true, test_pred = self._evaluate_dataloader(self.data_loaders['test_loader'])
        metrics['test'] = {
            'mse': mean_squared_error(test_true, test_pred),
            'rmse': np.sqrt(mean_squared_error(test_true, test_pred)),
            'mae': mean_absolute_error(test_true, test_pred),
            'mape': np.mean(np.abs((test_true - test_pred) / test_true)) * 100,
            'r2': r2_score(test_true, test_pred)
        }
        
        self.train_data = {'true': train_true, 'pred': train_pred}
        self.test_data = {'true': test_true, 'pred': test_pred}
        
        return metrics
    
    def plot_results(
        self,
        gas_type: str,
        dates: List[datetime],
        forecast_length: int = 365
    ) -> go.Figure:
        # Generate forecast
        last_sequence = next(iter(self.data_loaders['test_loader']))[0][-1:]
        forecast = self.model.generate_forecast(
            initial_sequence=last_sequence,
            forecast_length=forecast_length,
            device=self.device
        )
        
        if self.target_scaler:
            forecast = {
                k: self.target_scaler.inverse_transform(
                    v.reshape(-1, 1)
                ).flatten() for k, v in forecast.items()
            }
        
        # Create figure
        fig = go.Figure()
        
        # Add training data
        fig.add_trace(go.Scatter(
            x=dates[:len(self.train_data['true'])],
            y=self.train_data['true'],
            name='Train',
            line=dict(color='#00B5F7', width=1.5)
        ))
        
        # Add test data
        fig.add_trace(go.Scatter(
            x=dates[len(self.train_data['true']):len(self.train_data['true'])+len(self.test_data['true'])],
            y=self.test_data['true'],
            name='Test',
            line=dict(color='#FFA500', width=1.5)
        ))
        
        # Add forecast
        forecast_dates = pd.date_range(dates[-1], dates[-1] + timedelta(days=forecast_length))
        
        fig.add_trace(go.Scatter(
            x=forecast_dates,
            y=forecast['predictions'],
            name='Forecast',
            line=dict(color='#32CD32', width=1.5)
        ))
        
        # Add confidence intervals
        fig.add_trace(go.Scatter(
            x=forecast_dates,
            y=forecast['upper_bound'],
            fill=None,
            mode='lines',
            line=dict(color='rgba(50, 205, 50, 0)'),
            showlegend=False
        ))
        
        fig.add_trace(go.Scatter(
            x=forecast_dates,
            y=forecast['lower_bound'],
            fill='tonexty',
            mode='lines',
            line=dict(color='rgba(50, 205, 50, 0)'),
            fillcolor='rgba(50, 205, 50, 0.2)',
            showlegend=False
        ))
        
        # Update layout
        fig.update_layout(
            title=f'{gas_type.upper()} Concentration Forecast',
            template='plotly_dark',
            xaxis_title='Date',
            yaxis_title=f'{gas_type.upper()} (ppm)',
            legend=dict(
                yanchor="top",
                y=0.99,
                xanchor="left",
                x=0.01,
                bgcolor='rgba(0,0,0,0.5)'
            ),
            margin=dict(l=20, r=20, t=40, b=20),
            showlegend=True,
            plot_bgcolor='rgba(0,0,0,0)',
            paper_bgcolor='rgba(0,0,0,0)'
        )
        
        # Update axes
        fig.update_xaxes(
            showgrid=True,
            gridwidth=0.5,
            gridcolor='rgba(128,128,128,0.2)',
            showline=True,
            linewidth=1,
            linecolor='rgba(128,128,128,0.8)'
        )
        
        fig.update_yaxes(
            showgrid=True,
            gridwidth=0.5,
            gridcolor='rgba(128,128,128,0.2)',
            showline=True,
            linewidth=1,
            linecolor='rgba(128,128,128,0.8)'
        )
        
        return fig