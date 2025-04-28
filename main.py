# Установка необходимых библиотек
!pip install networkx pyvis ipywidgets community

import networkx as nx
from pyvis.network import Network
from google.colab import files
import io
import json
import ipywidgets as widgets
from IPython.display import display, HTML
import numpy as np
from community import community_louvain

# Загрузка файла GraphML
uploaded = files.upload()
graphml_file = list(uploaded.keys())[0]  # Получаем имя загруженного файла

# Чтение графа из файла GraphML
G = nx.read_graphml(graphml_file)

# Преобразование направленного графа в ненаправленный для кластеризации
if G.is_directed():
    G_undirected = G.to_undirected()
else:
    G_undirected = G

# Кластеризация графа
partition = community_louvain.best_partition(G_undirected)

# Экспоненциальное масштабирование размеров узлов на основе атрибута 'size'
def get_node_size(size, min_size=10, max_size=100, exponent=4):
    if size is None or float(size) <= 0:
        return min_size
    sizes = [float(G.nodes[n].get('size', 10)) for n in G.nodes if float(G.nodes[n].get('size', 10)) > 0]
    min_size_val, max_size_val = min(sizes, default=1), max(sizes, default=1)
    if max_size_val == min_size_val:
        return min_size
    # Экспоненциальное масштабирование
    normalized = (float(size) - min_size_val) / (max_size_val - min_size_val)
    scaled = normalized ** exponent  # Экспонента для контраста
    return min_size + (max_size - min_size) * scaled

# Нормализация толщины рёбер на основе 'weight'
def get_edge_width(weight, min_width=0.5, max_width=5):
    if weight is None or float(weight) <= 0:
        return min_width
    weights = [float(G.edges[e].get('weight', 0)) for e in G.edges if float(G.edges[e].get('weight', 0)) > 0]
    min_weight, max_weight = min(weights, default=1), max(weights, default=1)
    if max_weight == min_weight:
        return min_width
    normalized = (float(weight) - min_weight) / (max_weight - min_weight)
    return min_width + (max_width - min_width) * normalized

# Виджеты для настройки визуализации
min_size_filter = widgets.FloatSlider(value=10, min=10, max=max([float(G.nodes[n].get('size', 10)) for n in G.nodes], default=10), step=0.1, description='Min Wallet Size:')
min_weight_filter = widgets.FloatSlider(value=0, min=0, max=max([float(G.edges[e].get('weight', 0)) for e in G.edges], default=1), step=0.1, description='Min Tx Weight:')
min_color_filter = widgets.FloatSlider(value=0, min=0, max=max([float(G.nodes[n].get('color', 0)) for n in G.nodes], default=1), step=0.1, description='Min Color:')
node_size_scale = widgets.IntSlider(value=150, min=20, max=400, step=10, description='Max Node Size:')
size_exponent = widgets.FloatSlider(value=4.0, min=1.0, max=6.0, step=0.1, description='Size Exponent:')
edge_width_scale = widgets.FloatSlider(value=5.0, min=0.5, max=15.0, step=0.5, description='Max Edge Width:')
font_size = widgets.IntSlider(value=14, min=8, max=24, step=2, description='Font Size:')
bg_color = widgets.ColorPicker(value='#0a0a23', description='Background:')
font_color = widgets.ColorPicker(value='#ffffff', description='Font Color:')
physics_enabled = widgets.Checkbox(value=True, description='Enable Physics')
gravity = widgets.FloatSlider(value=-60, min=-100, max=0, step=5, description='Gravity:')
highlight_cluster = widgets.Dropdown(options=['All'] + list(set(partition.values())), value='All', description='Highlight Cluster:')
show_top_wallets = widgets.Checkbox(value=True, description='Highlight Top Wallets')
top_wallets_count = widgets.IntSlider(value=5, min=1, max=20, step=1, description='Top Wallets:')

# Функция для обновления и отображения графа
def update_visualization(change=None):
    # Создание объекта Network
    net = Network(notebook=True, height="800px", width="100%", 
                  bgcolor=bg_color.value, font_color=font_color.value, cdn_resources='remote')

    # Фильтрация узлов и рёбер
    filtered_nodes = [n for n in G.nodes if float(G.nodes[n].get('size', 10)) >= min_size_filter.value and float(G.nodes[n].get('color', 0)) >= min_color_filter.value]
    filtered_edges = [(u, v) for u, v in G.edges if u in filtered_nodes and v in filtered_nodes and float(G.edges[u, v].get('weight', 0)) >= min_weight_filter.value]

    # Определение топ-кошельков
    if show_top_wallets.value:
        sizes = [(n, float(G.nodes[n].get('size', 10))) for n in filtered_nodes]
        top_wallets = [n for n, _ in sorted(sizes, key=lambda x: x[1], reverse=True)[:top_wallets_count.value]]
    else:
        top_wallets = []

    # Добавление узлов
    for node in filtered_nodes:
        size_val = float(G.nodes[node].get('size', 10))
        label = G.nodes[node].get('label', node)[:8] if node not in top_wallets else G.nodes[node].get('label', node)[:12]
        color_val = float(G.nodes[node].get('color', 0))
        cluster = partition.get(node, 0)
        node_size = get_node_size(size_val, max_size=node_size_scale.value, exponent=size_exponent.value)

        # Цвет и стиль узла
        if highlight_cluster.value == 'All' or int(highlight_cluster.value) == cluster:
            color = {'background': f'hsl({color_val % 360}, 70%, 50%)',
                     'border': '#ffffff',
                     'highlight': {'background': f'hsl({color_val % 360}, 70%, 60%)', 'border': '#ffffff'}}
            opacity = 1.0
        else:
            color = {'background': '#444444', 'border': '#666666'}
            opacity = 0.3
        if node in top_wallets:
            color = {'background': 'radial-gradient(circle, #ff4d4d, #cc0000)',
                     'border': '#ffffff',
                     'highlight': {'background': '#ff8080', 'border': '#ffffff'}}
            node_size *= 1.5  # Увеличение размера для топ-кошельков

        # Всплывающая подсказка
        title = f"Address: {G.nodes[node].get('label', node)}<br>Size: {size_val:.2f}<br>Color: {color_val:.2f}<br>Cluster: {cluster}"
        net.add_node(node, label=label, size=node_size, title=title, color=color, opacity=opacity,
                     borderWidth=2, borderWidthSelected=5, mass=size_val/10)

    # Добавление рёбер
    for u, v in filtered_edges:
        weight = float(G.edges[u, v].get('weight', 1))
        width = get_edge_width(weight, max_width=edge_width_scale.value)
        opacity = 1.0 if highlight_cluster.value == 'All' or (partition.get(u) == partition.get(v) == int(highlight_cluster.value)) else 0.3
        net.add_edge(u, v, width=width, title=f"Tx Weight: {weight:.2f}", 
                     color={'color': '#aaaaaa', 'opacity': opacity, 'highlight': '#ffffff'})

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
            "hoverWidth": 3
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

    # Добавление анимации и стилей
    custom_css = f"""
    <style>
        .vis-network canvas {{
            background: linear-gradient(135deg, {bg_color.value} 0%, #16213e 100%);
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
            50% {{ box-shadow: 0 0 20px #ff4d4d; }}
            100% {{ box-shadow: 0 0 10px #ff4d4d; }}
        }}
    </style>
    """

    # Легенда
    legend_html = """
    <div style='background: rgba(0,0,0,0.85); color: white; padding: 15px; border-radius: 10px; position: absolute; top: 10px; left: 10px; z-index: 1000; font-family: Arial;'>
        <b>Tokenomics Visualization</b><br>
        - <b>Node Size</b>: Exponentially scaled wallet size (adjust with Size Exponent)<br>
        - <b>Edge Width</b>: Transaction weight<br>
        - <b>Node Color</b>: Based on color attribute or red gradient for top wallets<br>
        - <b>Tooltip</b>: Hover for details (address, size, color, cluster)<br>
        - <b>Cluster</b>: Select to highlight community<br>
        - <b>Top Wallets</b>: Red nodes with full labels for top wallets by size
    </div>
    """

    # Сохранение и отображение графа
    net.show("graph.html")
    with open("graph.html", "r") as f:
        html_content = f.read()
    display(HTML(custom_css + legend_html + html_content))

# Привязка виджетов к функции обновления
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

# Отображение виджетов
display(widgets.VBox([
    min_size_filter, min_weight_filter, min_color_filter, node_size_scale, size_exponent,
    edge_width_scale, font_size, bg_color, font_color, physics_enabled, gravity,
    highlight_cluster, show_top_wallets, top_wallets_count
]))

# Кнопка для скачивания HTML
download_button = widgets.Button(description="Download HTML")
def on_download_button_clicked(b):
    files.download("graph.html")
download_button.on_click(on_download_button_clicked)
display(download_button)

# Первоначальная визуализация
update_visualization()
