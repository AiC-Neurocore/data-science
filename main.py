# Установка необходимых библиотек
!pip install networkx pyvis ipywidgets community plotly

import networkx as nx
from pyvis.network import Network
from google.colab import files
import io
import json
import ipywidgets as widgets
from IPython.display import display, HTML
import numpy as np
from community import community_louvain
import plotly.express as px
import pandas as pd

# Загрузка файла GraphML
uploaded = files.upload()
graphml_file = list(uploaded.keys())[0]

# Чтение графа из файла GraphML
G = nx.read_graphml(graphml_file)

# Преобразование направленного графа в ненаправленный для кластеризации
if G.is_directed():
    G_undirected = G.to_undirected()
else:
    G_undirected = G

# Кластеризация графа
partition = community_louvain.best_partition(G_undirected)

# Экспоненциальное масштабирование размеров узлов
def get_node_size(size, min_size=10, max_size=100, exponent=4):
    if size is None or float(size) <= 0:
        return min_size
    sizes = [float(G.nodes[n].get('size', 10)) for n in G.nodes if float(G.nodes[n].get('size', 10)) > 0]
    min_size_val, max_size_val = min(sizes, default=1), max(sizes, default=1)
    if max_size_val == min_size_val:
        return min_size
    normalized = (float(size) - min_size_val) / (max_size_val - min_size_val)
    scaled = normalized ** exponent
    return min_size + (max_size - min_size) * scaled

# Нормализация толщины рёбер
def get_edge_width(weight, min_width=0.5, max_width=5):
    if weight is None or float(weight) <= 0:
        return min_width
    weights = [float(G.edges[e].get('weight', 0)) for e in G.edges if float(G.edges[e].get('weight', 0)) > 0]
    min_weight, max_weight = min(weights, default=1), max(weights, default=1)
    if max_weight == min_weight:
        return min_width
    normalized = (float(weight) - min_weight) / (max_weight - min_weight)
    return min_width + (max_width - min_width) * normalized

# Виджеты для настройки
min_size_filter = widgets.FloatSlider(value=10, min=10, max=max([float(G.nodes[n].get('size', 10)) for n in G.nodes], default=10), step=0.1, description='Min Wallet Size:')
min_weight_filter = widgets.FloatSlider(value=0, min=0, max=max([float(G.edges[e].get('weight', 0)) for e in G.edges], default=1), step=0.1, description='Min Tx Weight:')
min_color_filter = widgets.FloatSlider(value=0, min=0, max=max([float(G.nodes[n].get('color', 0)) for n in G.nodes], default=1), step=0.1, description='Min Color:')
node_size_scale = widgets.IntSlider(value=200, min=20, max=500, step=10, description='Max Node Size:')
size_exponent = widgets.FloatSlider(value=4.5, min=1.0, max=7.0, step=0.1, description='Size Exponent:')
edge_width_scale = widgets.FloatSlider(value=5.0, min=0.5, max=15.0, step=0.5, description='Max Edge Width:')
font_size = widgets.IntSlider(value=14, min=8, max=24, step=2, description='Font Size:')
bg_color = widgets.ColorPicker(value='#0a0a23', description='Background:')
font_color = widgets.ColorPicker(value='#ffffff', description='Font Color:')
physics_enabled = widgets.Checkbox(value=True, description='Enable Physics')
gravity = widgets.FloatSlider(value=-60, min=-100, max=0, step=5, description='Gravity:')
highlight_cluster = widgets.Dropdown(options=['All'] + list(set(partition.values())), value='All', description='Highlight Cluster:')
show_top_wallets = widgets.Checkbox(value=True, description='Highlight Top Wallets')
top_wallets_count = widgets.IntSlider(value=5, min=1, max=20, step=1, description='Top Wallets:')
color_mode = widgets.Dropdown(options=['Cluster', 'Color Attribute', 'Degree'], value='Color Attribute', description='Color Mode:')
focus_wallet = widgets.Text(value='', description='Focus Wallet:', placeholder='Enter wallet address')

# Функция для обновления и отображения графа
def update_visualization(change=None):
    # Индикатор загрузки
    display(HTML("<p style='color: white;'>Loading graph...</p>"))

    # Создание объекта Network
    net = Network(notebook=True, height="800px", width="100%", 
                  bgcolor=bg_color.value, font_color=font_color.value, cdn_resources='remote')

    # Фильтрация узлов и рёбер
    filtered_nodes = [n for n in G.nodes if float(G.nodes[n].get('size', 10)) >= min_size_filter.value and float(G.nodes[n].get('color', 0)) >= min_color_filter.value]
    if focus_wallet.value.strip():
        try:
            focus_node = next(n for n in filtered_nodes if G.nodes[n].get('label', n) == focus_wallet.value.strip())
            # Ограничение узлов до соседей 1–2 степени
            neighbors = set(nx.ego_graph(G, focus_node, radius=2).nodes())
            filtered_nodes = [n for n in filtered_nodes if n in neighbors]
        except StopIteration:
            pass
    filtered_edges = [(u, v) for u, v in G.edges if u in filtered_nodes and v in filtered_nodes and float(G.edges[u, v].get('weight', 0)) >= min_weight_filter.value]

    # Определение топ-кошельков
    if show_top_wallets.value:
        sizes = [(n, float(G.nodes[n].get('size', 10))) for n in filtered_nodes]
        top_wallets = [n for n, _ in sorted(sizes, key=lambda x: x[1], reverse=True)[:top_wallets_count.value]]
    else:
        top_wallets = []

    # Вычисление степеней узлов для тепловой карты
    degrees = dict(G.degree(filtered_nodes))

    # Добавление узлов
    for node in filtered_nodes:
        size_val = float(G.nodes[node].get('size', 10))
        label = G.nodes[node].get('label', node)[:8] if node not in top_wallets else G.nodes[node].get('label', node)[:16]
        color_val = float(G.nodes[node].get('color', 0))
        cluster = partition.get(node, 0)
        node_size = get_node_size(size_val, max_size=node_size_scale.value, exponent=size_exponent.value)

        # Цвет узла в зависимости от режима
        if color_mode.value == 'Cluster':
            color = {'background': f'hsl({(cluster * 60) % 360}, 70%, 50%)',
                     'border': '#ffffff',
                     'highlight': {'background': f'hsl({(cluster * 60) % 360}, 70%, 60%)', 'border': '#ffffff'}}
        elif color_mode.value == 'Color Attribute':
            color = {'background': f'hsl({color_val % 360}, 70%, 50%)',
                     'border': '#ffffff',
                     'highlight': {'background': f'hsl({color_val % 360}, 70%, 60%)', 'border': '#ffffff'}}
        else:  # Degree
            degree = degrees[node]
            max_degree = max(degrees.values(), default=1)
            color = {'background': f'hsl({(degree/max_degree) * 240}, 70%, 50%)',
                     'border': '#ffffff',
                     'highlight': {'background': f'hsl({(degree/max_degree) * 240}, 70%, 60%)', 'border': '#ffffff'}}
        
        if highlight_cluster.value != 'All' and int(highlight_cluster.value) != cluster:
            color = {'background': '#444444', 'border': '#666666'}
            opacity = 0.3
        else:
            opacity = 1.0
        if node in top_wallets:
            color = {'background': 'radial-gradient(circle, #ff4d4d, #cc0000)',
                     'border': '#ffffff',
                     'highlight': {'background': '#ff8080', 'border': '#ffffff'}}
            node_size *= 1.6

        # Всплывающая подсказка
        title = f"Address: {G.nodes[node].get('label', node)}<br>Size: {size_val:.2f}<br>Color: {color_val:.2f}<br>Cluster: {cluster}<br>Degree: {degrees[node]}"
        net.add_node(node, label=label, size=node_size, title=title, color=color, opacity=opacity,
                     borderWidth=2, borderWidthSelected=5, mass=size_val/10)

    # Добавление рёбер
    for u, v in filtered_edges:
        weight = float(G.edges[u, v].get('weight', 1))
        width = get_edge_width(weight, max_width=edge_width_scale.value)
        opacity = 1.0 if highlight_cluster.value == 'All' or (partition.get(u) == partition.get(v) == int(highlight_cluster.value)) else 0.3
        net.add_edge(u, v, width=width, title=f"Tx Weight: {weight:.2f}", 
                     color={'color': '#aaaaaa', 'opacity': opacity, 'highlight': '#ffffff'},
                     arrows='to')

    # Настройка параметров визуализации
    options = {
        "nodes": {
            "shape": "dot",
            "scaling": {
                "min": 10,
                "max": node_size_scale.value
            },
            "font": {
                "size": font_size.value,
                "face": "arial",
                "color": font_color.value
            },
            "shadow": {
                "enabled": True,
                "size": 10,
                "x": 5,
                "y": 5
            },
            "borderWidth": 2
        },
        "edges": {
            "color": {
                "inherit": False
            },
            "smooth": {
                "type": "dynamic",
                "roundness": 0.5
            },
            "shadow": {
                "enabled": True
            },
            "hoverWidth": 3,
            "arrows": {
                "to": {
                    "enabled": True,
                    "scaleFactor": 0.5
                }
            }
        },
        "physics": {
            "enabled": physics_enabled.value,
            "forceAtlas2Based": {
                "gravitationalConstant": gravity.value,
                "centralGravity": 0.01,
                "springLength": 100,
                "avoidOverlap": 0.5
            },
            "minVelocity": 0.75,
            "solver": "forceAtlas2Based"
        },
        "interaction": {
            "hover": True,
            "tooltipDelay": 100,
            "zoomView": True,
            "dragView": True,
            "dragNodes": True,
            "hoverConnectedEdges": True,
            "selectConnectedEdges": True
        }
    }

    # Анимация и стили
    custom_css = f"""
    <style>
        .vis-network canvas {{
            background: linear-gradient(135deg, {bg_color.value} 0%, #16213e 100%);
            transition: background 0.5s;
        }}
        .vis-network .vis-node {{
            transition: all 0.3s;
        }}
        .vis-network .vis-node:hover {{
            animation: pulse 1.2s infinite;
        }}
        .vis-network .vis-node.top-wallet {{
            animation: wave 2s infinite;
        }}
        @keyframes pulse {{
            0% {{ transform: scale(1); }}
            50% {{ transform: scale(1.15); }}
            100% {{ transform: scale(1); }}
        }}
        @keyframes wave {{
            0% {{ box-shadow: 0 0 10px #ff4d4d; }}
            50% {{ box-shadow: 0 0 25px #ff4d4d; }}
            100% {{ box-shadow: 0 0 10px #ff4d4d; }}
        }}
    </style>
    """

    # Динамическая легенда
    total_nodes = len(filtered_nodes)
    total_edges = len(filtered_edges)
    max_size = max([float(G.nodes[n].get('size', 10)) for n in filtered_nodes], default=10)
    top_wallet = max([(n, float(G.nodes[n].get('size', 10))) for n in filtered_nodes], key=lambda x: x[1], default=('None', 0))[0]
    cluster_count = len(set(partition[n] for n in filtered_nodes))
    legend_html = f"""
    <div style='background: rgba(0,0,0,0.85); color: white; padding: 15px; border-radius: 10px; position: absolute; top: 10px; left: 10px; z-index: 1000; font-family: Arial;'>
        <b>Tokenomics Visualization</b><br>
        - <b>Nodes</b>: {total_nodes}<br>
        - <b>Edges</b>: {total_edges}<br>
        - <b>Max Wallet Size</b>: {max_size:.2f} (Address: {top_wallet[:8]})<br>
        - <b>Clusters</b>: {cluster_count}<br>
        - <b>Node Size</b>: Exponentially scaled (adjust with Size Exponent)<br>
        - <b>Edge Width</b>: Transaction weight<br>
        - <b>Node Color</b>: {color_mode.value}<br>
        - <b>Tooltip</b>: Hover for details<br>
        - <b>Top Wallets</b>: Red nodes with full labels
    </div>
    """

    # Сохранение и отображение графа
    net.show("graph.html")
    with open("graph.html", "r") as f:
        html_content = f.read()
    display(HTML(custom_css + legend_html + html_content))

    # График распределения размеров кошельков
    sizes = [float(G.nodes[n].get('size', 10)) for n in filtered_nodes]
    fig = px.histogram(x=sizes, nbins=20, title="Wallet Size Distribution", labels={'x': 'Wallet Size'})
    fig.update_layout(showlegend=False, plot_bgcolor='rgba(0,0,0,0)', paper_bgcolor='rgba(0,0,0,0)', font_color='white')
    fig.show()

# Привязка виджетов
min_size_filter.observe(update_visualization, names='value')
min_weight_filter.observe(update_visualization, names='value')
min_color_filter.observe(update_visualization, names='value')
node_size_scale.observe(update_visualization, names='value')
size_exponent.observe(update_visualization, names='value')
edge_width_scale.observe(update_visualization, names='value')
font_size.observe(update_visualization, names='value')
bg_color.observe(update_visualization, names='value')
font_color.observe(update_visualization, names='value')
physics_enabled.observe(update_visualization, names='value')
gravity.observe(update_visualization, names='value')
highlight_cluster.observe(update_visualization, names='value')
show_top_wallets.observe(update_visualization, names='value')
top_wallets_count.observe(update_visualization, names='value')
color_mode.observe(update_visualization, names='value')
focus_wallet.observe(update_visualization, names='value')

# Отображение виджетов
display(widgets.VBox([
    min_size_filter, min_weight_filter, min_color_filter, node_size_scale, size_exponent,
    edge_width_scale, font_size, bg_color, font_color, physics_enabled, gravity,
    highlight_cluster, show_top_wallets, top_wallets_count, color_mode, focus_wallet
]))

# Кнопка для скачивания HTML
download_button = widgets.Button(description="Download HTML")
def on_download_button_clicked(b):
    files.download("graph.html")
download_button.on_click(on_download_button_clicked)
display(download_button)

# Первоначальная визуализация
update_visualization()
