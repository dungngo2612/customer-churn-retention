"""Visual improvements for better UX and reduced eye strain."""

import plotly.graph_objects as go
import plotly.express as px
import pandas as pd
import streamlit as st


def create_risk_pie_chart(predictions_df: pd.DataFrame) -> go.Figure:
    """Create a pie chart showing risk distribution with better colors."""
    # Map probabilities to risk categories
    risk_labels = []
    for prob in predictions_df['churn_prob']:
        if prob > 0.7:
            risk_labels.append('Cao')
        elif prob >= 0.4:
            risk_labels.append('Trung bình')
        else:
            risk_labels.append('Thấp')
    
    # Count each category
    risk_counts = pd.Series(risk_labels).value_counts().reindex(['Thấp', 'Trung bình', 'Cao'], fill_value=0)
    
    # Softer, more accessible colors matching the improved theme
    colors = ['#10b981', '#f59e0b', '#ef4444']  # green, amber, red (softer shades)
    
    fig = go.Figure(data=[go.Pie(
        labels=risk_counts.index,
        values=risk_counts.values,
        hole=0.4,  # Donut chart for better modern look
        marker=dict(colors=colors),
        textinfo='percent+label',
        textfont_size=13,
        hovertemplate='<b>%{label}</b><br>%{value} khách (%{percent})<extra></extra>'
    )])
    
    dark = st.session_state.get("ui_dark_mode", False)
    lang = st.session_state.get("ui_lang", "vi")
    text_color = '#f1f5f9' if dark else '#1a202c'
    legend_color = '#cbd5e1' if dark else '#4a5568'
    chart_title = 'Phân bố nguy cơ rời bỏ' if lang == 'vi' else 'Churn Risk Distribution'

    fig.update_layout(
        title=dict(
            text=chart_title,
            font=dict(size=15, color=text_color, family='Inter, sans-serif'),
            x=0.5
        ),
        showlegend=True,
        legend=dict(
            orientation="v",
            yanchor="middle",
            y=0.5,
            xanchor="left",
            x=1.05,
            font=dict(size=12, color=legend_color)
        ),
        margin=dict(l=20, r=120, t=40, b=20),
        paper_bgcolor='rgba(0,0,0,0)',
        plot_bgcolor='rgba(0,0,0,0)',
        height=300
    )
    
    return fig


def create_budget_allocation_chart(allocations_df: pd.DataFrame, budget: float) -> go.Figure:
    """Create a horizontal bar chart showing budget allocation by intervention type."""
    if allocations_df.empty:
        # Return empty figure
        fig = go.Figure()
        fig.update_layout(
            title=dict(
                text='Phân bổ ngân sách',
                font=dict(size=15, color='#1a202c', family='Inter, sans-serif'),
                x=0.5
            ),
            height=200,
            paper_bgcolor='rgba(0,0,0,0)',
            plot_bgcolor='rgba(0,0,0,0)'
        )
        return fig
    
    # Group by action type and calculate total cost
    action_costs = allocations_df.groupby('action_name')['cost'].sum().reset_index()
    action_costs = action_costs.sort_values('cost', ascending=True)
    
    # Calculate percentage of budget
    action_costs['pct_of_budget'] = (action_costs['cost'] / budget * 100).round(1)
    
    # Softer colors for bars
    colors = ['#4f46e5', '#7c3aed', '#a855f7', '#d946ef']  # Indigo shades
    
    fig = go.Figure(data=[go.Bar(
        y=action_costs['action_name'],
        x=action_costs['cost'],
        orientation='h',
        marker=dict(color=colors[:len(action_costs)], line=dict(width=0)),
        text=action_costs['pct_of_budget'].astype(str) + '%',
        textposition='auto',
        hovertemplate='<b>%{y}</b><br>$%{x:,.0f}<br>%{text} ngân sách<extra></extra>'
    )])
    
    fig.update_layout(
        title=dict(
            text='Phân bổ ngân sách theo loại can thiệp',
            font=dict(size=15, color='#1a202c', family='Inter, sans-serif'),
            x=0.5
        ),
        xaxis=dict(
            title='Chi phí (USD)',
            title_font=dict(size=12, color='#718096'),
            tickformat='$,.0f',
            gridcolor='#e2e8f0',
            color='#4a5568'
        ),
        yaxis=dict(
            title='Loại can thiệp',
            title_font=dict(size=12, color='#718096'),
            color='#4a5568'
        ),
        margin=dict(l=10, r=10, t=40, b=30),
        paper_bgcolor='rgba(0,0,0,0)',
        plot_bgcolor='rgba(0,0,0,0)',
        height=max(200, len(action_costs) * 40 + 80),
        showlegend=False
    )
    
    return fig


def create_customer_segmentation_chart(customers_df: pd.DataFrame) -> go.Figure:
    """Create a bar chart showing customer segmentation by key attributes."""
    # Select key attributes for segmentation
    attributes = ['Contract', 'InternetService', 'PaymentMethod']
    
    # Create subplot figure
    fig = go.Figure()
    
    for i, attr in enumerate(attributes):
        if attr in customers_df.columns:
            # Count customers by attribute
            counts = customers_df[attr].value_counts().reset_index()
            counts.columns = ['value', 'count']
            
            # Add bar trace
            fig.add_trace(go.Bar(
                x=counts['value'],
                y=counts['count'],
                name=attr,
                visible=(i == 0),  # Only show first one initially
                marker_color=f'rgba({79 + i*30}, {70 + i*10}, {229 - i*40}, 0.8)',
                hovertemplate=f'<b>{attr}: %{{x}}</b><br>%{{y}} khách<extra></extra>'
            ))
    
    # Add dropdown menu
    fig.update_layout(
        updatemenus=[dict(
            buttons=list([
                dict(
                    args=[{"visible": [i == j for j in range(len(attributes))]}],
                    label=attributes[i],
                    method="update"
                ) for i in range(len(attributes)) if attributes[i] in customers_df.columns
            ]),
            direction="down",
            pad={"r": 10, "t": 10},
            showactive=True,
            x=0.1,
            xanchor="left",
            y=1.15,
            yanchor="top",
            bgcolor="#f8fafc",
            bordercolor="#e2e8f0",
            font=dict(color="#4a5568", size=12)
        )],
        title=dict(
            text='Phân đoạn khách hàng',
            font=dict(size=15, color='#1a202c', family='Inter, sans-serif'),
            x=0.5
        ),
        xaxis=dict(
            title='Giá trị',
            title_font=dict(size=12, color='#718096'),
            color='#4a5568',
            gridcolor='#e2e8f0'
        ),
        yaxis=dict(
            title='Số lượng khách hàng',
            title_font=dict(size=12, color='#718096'),
            color='#4a5568',
            gridcolor='#e2e8f0'
        ),
        margin=dict(l=10, r=10, t=60, b=40),
        paper_bgcolor='rgba(0,0,0,0)',
        plot_bgcolor='rgba(0,0,0,0)',
        height=300,
        showlegend=False
    )
    
    return fig


def create_improved_bar_chart(data, x_col, y_col, title, x_label, y_label):
    """Create a modern, improved bar chart with better styling."""
    fig = go.Figure(data=[go.Bar(
        x=data[x_col],
        y=data[y_col],
        marker_color='#4f46e5',
        marker_line_width=0,
        text=data[y_col],
        textposition='auto',
        hovertemplate=f'<b>%{{x}}</b><br>%{{y}}{y_label}<extra></extra>'
    )])
    
    fig.update_layout(
        title=dict(
            text=title,
            font=dict(size=15, color='#1a202c', family='Inter, sans-serif'),
            x=0.5
        ),
        xaxis=dict(
            title=x_label,
            title_font=dict(size=12, color='#718096'),
            color='#4a5568',
            gridcolor='#e2e8f0'
        ),
        yaxis=dict(
            title=y_label,
            title_font=dict(size=12, color='#718096'),
            color='#4a5568',
            gridcolor='#e2e8f0'
        ),
        margin=dict(l=10, r=10, t=40, b=40),
        paper_bgcolor='rgba(0,0,0,0)',
        plot_bgcolor='rgba(0,0,0,0)',
        height=300
    )
    
    return fig