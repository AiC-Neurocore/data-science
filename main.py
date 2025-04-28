# Установка необходимых библиотек
!pip install networkx pyvis ipywidgets community plotly imageio Pillow

import networkx as nx
from pyvis.network import Network
from google.colab import files
import io
import json
import ipywidgets as widgets
from IPython.display import display, HTML
import numpy as np
from community import community_louvain
import pandas as pd
import plotly.express as px
import pickle
import os
import imageio
from PIL import Image
import base64
from datetime import datetime

# Загрузка файла GraphML
uploaded = files.upload()
graphml_file = list(uploaded.keys())[0]

# Чтение графа из файла GraphML
try:
    G = nx.read_graphml(graphml_file)
except Exception as e:
    display(HTML(f"<p style='color: red;'>Error loading GraphML: {str(e)}</p>"))
    raise

# Кэширование кластеризации
cache_file = "partition_cache.pkl"
if os.path.exists(cache_file):
    with open(cache_file, 'rb') as f:
        partition = pickle.load(f)
else:
    if G.is_directed():
        G_undirected = G.to_undirected()
    else:
        G_undirected = G
    partition = community_louvain.best_partition(G_undirected)
    with open(cache_file, 'wb') as f:
        pickle.dump(partition, f)

# Экспоненциальное масштабирование размеров узлов
def get_node_size(size, min_size=10, max_size=100, exponent=5):
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
min_size_filter = widgets.FloatSlider(value=10, min=10, max=max([float(G.nodes[n].get('size', 10)) for n in G.nodes], default=10), step=0.1, description='Min Size:', tooltip='Filter wallets by minimum size')
min_weight_filter = widgets.FloatSlider(value=0, min=0, max=max([float(G.edges[e].get('weight', 0)) for e in G.edges], default=1), step=0.1, description='Min Weight:', tooltip='Filter transactions by minimum weight')
min_color_filter = widgets.FloatSlider(value=0, min=0, max=max([float(G.nodes[n].get('color', 0)) for n in G.nodes], default=1), step=0.1, description='Min Color:', tooltip='Filter nodes by minimum color attribute')
node_size_scale = widgets.IntSlider(value=300, min=20, max=600, step=10, description='Node Size:', tooltip='Maximum size for nodes')
size_exponent = widgets.FloatSlider(value=5.0, min=1.0, max=8.0, step=0.1, description='Exponent:', tooltip='Exponent for node size scaling')
edge_width_scale = widgets.FloatSlider(value=5.0, min=0.5, max=15.0, step=0.5, description='Edge Width:', tooltip='Maximum width for edges')
font_size = widgets.IntSlider(value=14, min=8, max=24, step=2, description='Font Size:', tooltip='Font size for labels')
bg_color = widgets.ColorPicker(value='#0a0a23', description='Background:', tooltip='Background color')
font_color = widgets.ColorPicker(value='#ffffff', description='Font Color:', tooltip='Font color for labels')
physics_enabled = widgets.Checkbox(value=True, description='Physics', tooltip='Enable physics simulation')
gravity = widgets.FloatSlider(value=-60, min=-100, max=0, step=5, description='Gravity:', tooltip='Physics gravity strength')
highlight_cluster = widgets.Dropdown(options=['All'] + list(set(partition.values())), value='All', description='Cluster:', tooltip='Highlight a specific cluster')
show_top_wallets = widgets.Checkbox(value=True, description='Top Wallets', tooltip='Highlight top wallets by size')
top_wallets_count = widgets.IntSlider(value=5, min=1, max=20, step=1, description='Top Count:', tooltip='Number of top wallets to highlight')
color_mode = widgets.Dropdown(options=['Cluster', 'Color Attribute', 'Degree'], value='Color Attribute', description='Color Mode:', tooltip='Node color mode')
focus_wallet = widgets.Text(value='', description='Focus Wallet:', placeholder='Enter wallet address', tooltip='Focus on a specific wallet')
render_mode = widgets.Dropdown(options=['2D', '3D'], value='2D', description='Render:', tooltip='Switch between 2D and 3D rendering')
rotate_top_wallets = widgets.Checkbox(value=False, description='Rotate Wallets', tooltip='Enable rotation for top wallets')
rotation_direction = widgets.Dropdown(options=['Clockwise', 'Counterclockwise'], value='Clockwise', description='Direction:', tooltip='Rotation direction', disabled=True)
high_quality = widgets.Checkbox(value=True, description='High Quality', tooltip='Enable high-quality rendering with effects')
theme_mode = widgets.Dropdown(options=['Dark', 'Light'], value='Dark', description='Theme:', tooltip='Switch between dark and light themes')

# Управление видимостью Rotation Direction
def update_rotation_direction(change):
    rotation_direction.disabled = not rotate_top_wallets.value
rotate_top_wallets.observe(update_rotation_direction, names='value')

# Apply Settings button
apply_button = widgets.Button(description='Apply', button_style='success', tooltip='Apply all settings')
apply_button.layout = widgets.Layout(width='150px', height='40px', margin='10px')

# Организация виджетов в аккордеон
filters_accordion = widgets.VBox([min_size_filter, min_weight_filter, min_color_filter, focus_wallet], layout=widgets.Layout(padding='5px'))
visualization_accordion = widgets.VBox([node_size_scale, size_exponent, edge_width_scale, font_size], layout=widgets.Layout(padding='5px'))
physics_accordion = widgets.VBox([physics_enabled, gravity, highlight_cluster], layout=widgets.Layout(padding='5px'))
appearance_accordion = widgets.VBox([bg_color, font_color, color_mode, show_top_wallets, top_wallets_count, rotate_top_wallets, rotation_direction, high_quality, theme_mode, render_mode], layout=widgets.Layout(padding='5px'))
accordion = widgets.Accordion(children=[filters_accordion, visualization_accordion, physics_accordion, appearance_accordion])
accordion.set_title(0, '⚙️ Filters')
accordion.set_title(1, '📊 Visualization')
accordion.set_title(2, '🌐 Physics')
accordion.set_title(3, '🎨 Appearance')

# CSS для стилизации меню и графа
menu_css = """
<style>
    .widget-label { font-weight: 600; color: #222; font-size: 14px; }
    .widget-vbox { background: #fafafa; border-radius: 6px; padding: 8px; box-shadow: 0 2px 4px rgba(0,0,0,0.1); }
    .widget-accordion .p-Accordion-child { background: #fff; border-radius: 4px; margin: 4px; }
    .widget-accordion .p-Accordion-title { background: #00aaff; color: #fff; border-radius: 4px; padding: 6px; font-size: 14px; }
    .widget-button { font-size: 14px; font-weight: 600; border-radius: 4px; transition: all 0.2s; }
    .widget-button:hover { background: #00cc00 !important; transform: scale(1.05); }
    .dark-theme .widget-vbox { background: #1a1a1a; }
    .dark-theme .widget-label { color: #ddd; }
    .dark-theme .widget-accordion .p-Accordion-child { background: #2a2a2a; }
    .dark-theme .widget-accordion .p-Accordion-title { background: #0077cc; }
    .notification { position: fixed; top: 60px; left: 10px; background: #00cc00; color: #fff; padding: 8px 16px; border-radius: 4px; z-index: 1000; font-size: 14px; box-shadow: 0 2px 4px rgba(0,0,0,0.2); display: none; }
    .particles { position: absolute; width: 100%; height: 800px; pointer-events: none; }
    .particle { position: absolute; background: #fff; border-radius: 50%; opacity: 0.4; animation: move 8s linear infinite; }
    .dark-theme .particle { background: #fff; }
    .light-theme .particle { background: #000; }
    @keyframes move {
        0% { transform: translateY(0); opacity: 0.4; }
        100% { transform: translateY(-800px); opacity: 0; }
    }
</style>
"""

# Функция для 2D-визуализации
def render_2d_visualization(filtered_nodes, filtered_edges, top_wallets, degrees):
    net = Network(notebook=True, height="800px", width="100%", bgcolor=bg_color.value, font_color=font_color.value, cdn_resources='remote')
    for node in filtered_nodes[:1000 if not high_quality.value else len(filtered_nodes)]:
        size_val = float(G.nodes[node].get('size', 10))
        label = G.nodes[node].get('label', node)[:8] if node not in top_wallets else G.nodes[node].get('label', node)[:16]
        color_val = float(G.nodes[node].get('color', 0))
        cluster = partition.get(node, 0)
        node_size = get_node_size(size_val, max_size=node_size_scale.value, exponent=size_exponent.value)

        if color_mode.value == 'Cluster':
            color = {'background': f'hsl({(cluster * 60) % 360}, 70%, 50%)', 'border': '#fff', 'highlight': {'background': f'hsl({(cluster * 60) % 360}, 70%, 60%)', 'border': '#fff'}}
        elif color_mode.value == 'Color Attribute':
            color = {'background': f'hsl({color_val % 360}, 70%, 50%)', 'border': '#fff', 'highlight': {'background': f'hsl({color_val % 360}, 70%, 60%)', 'border': '#fff'}}
        else:
            degree = degrees[node]
            max_degree = max(degrees.values(), default=1)
            color = {'background': f'hsl({(degree/max_degree) * 240}, 70%, 50%)', 'border': '#fff', 'highlight': {'background': f'hsl({(degree/max_degree) * 240}, 70%, 60%)', 'border': '#fff'}}
        
        if highlight_cluster.value != 'All' and int(highlight_cluster.value) != cluster:
            color = {'background': '#444', 'border': '#666'}
            opacity = 0.3
        else:
            opacity = 1.0
        if node in top_wallets:
            color = {'background': 'radial-gradient(circle, #ff4d4d, #cc0000)', 'border': '#fff', 'highlight': {'background': '#ff8080', 'border': '#fff'}}
            node_size *= 1.7

        title = f"Address: {G.nodes[node].get('label', node)}<br>Size: {size_val:.2f}<br>Color: {color_val:.2f}<br>Cluster: {cluster}<br>Degree: {degrees[node]}"
        net.add_node(node, label=label, size=node_size, title=title, color=color, opacity=opacity, borderWidth=2, borderWidthSelected=5, mass=size_val/10)

    top_edges = sorted(filtered_edges, key=lambda e: float(G.edges[e].get('weight', 1)), reverse=True)[:min(10, len(filtered_edges))]
    for u, v in filtered_edges[:1000 if not high_quality.value else len(filtered_edges)]:
        weight = float(G.edges[u, v].get('weight', 1))
        width = get_edge_width(weight, max_width=edge_width_scale.value)
        opacity = 1.0 if highlight_cluster.value == 'All' or (partition.get(u) == partition.get(v) == int(highlight_cluster.value)) else 0.3
        edge_class = 'top-edge' if (u, v) in top_edges and high_quality.value else ''
        net.add_edge(u, v, width=width, title=f"Tx Weight: {weight:.2f}", color={'color': '#aaa', 'opacity': opacity, 'highlight': '#fff'}, arrows='to', custom_class=edge_class)

    options = {
        "nodes": {
            "shape": "dot",
            "scaling": {"min": 10, "max": node_size_scale.value},
            "font": {"size": font_size.value, "face": "arial", "color": font_color.value},
            "shadow": {"enabled": high_quality.value, "size": 10, "x": 5, "y": 5},
            "borderWidth": 2
        },
        "edges": {
            "color": {"inherit": False},
            "smooth": {"type": "dynamic", "roundness": 0.5},
            "shadow": {"enabled": high_quality.value},
            "hoverWidth": 3,
            "arrows": {"to": {"enabled": True, "scaleFactor": 0.5}}
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
    net.set_options(json.dumps(options))
    return net

# Функция для 3D-визуализации
def render_3d_visualization(filtered_nodes, filtered_edges, top_wallets, degrees):
    nodes_data = []
    for node in filtered_nodes[:500 if not high_quality.value else len(filtered_nodes)]:
        size_val = float(G.nodes[node].get('size', 10))
        color_val = float(G.nodes[node].get('color', 0))
        cluster = partition.get(node, 0)
        node_size = get_node_size(size_val, max_size=node_size_scale.value/20, exponent=size_exponent.value)
        if color_mode.value == 'Cluster':
            color = f'hsl({(cluster * 60) % 360}, 70%, 50%)'
        elif color_mode.value == 'Color Attribute':
            color = f'hsl({color_val % 360}, 70%, 50%)'
        else:
            degree = degrees[node]
            max_degree = max(degrees.values(), default=1)
            color = f'hsl({(degree/max_degree) * 240}, 70%, 50%)'
        if node in top_wallets:
            color = '#ff4d4d'
        nodes_data.append({
            'id': node,
            'size': node_size,
            'color': color,
            'label': G.nodes[node].get('label', node)[:16] if node in top_wallets else G.nodes[node].get('label', node)[:8]
        })

    links_data = []
    for u, v in filtered_edges[:500 if not high_quality.value else len(filtered_edges)]:
        weight = float(G.edges[u, v].get('weight', 1))
        links_data.append({
            'source': u,
            'target': v,
            'value': get_edge_width(weight, max_width=edge_width_scale.value/2)
        })

    html_content = f"""
    <html>
    <head>
        <script src="https://unpkg.com/3d-force-graph@1.70.10/dist/3d-force-graph.min.js"></script>
        <style>
            body {{ margin: 0; background: {bg_color.value}; }}
            #graph {{ width: 100%; height: 800px; }}
        </style>
    </head>
    <body>
        <div id="graph"></div>
        <script>
            const nodes = {json.dumps(nodes_data)};
            const links = {json.dumps(links_data)};
            const Graph = ForceGraph3D()
                (document.getElementById('graph'))
                .graphData({{ nodes: nodes, links: links }})
                .nodeLabel('label')
                .nodeVal('size')
                .nodeColor('color')
                .linkWidth('value')
                .linkDirectionalArrowLength(3)
                .linkDirectionalArrowRelPos(1)
                .backgroundColor('{bg_color.value}')
                .nodeOpacity(0.9)
                .linkOpacity(0.7)
                .d3Force('charge', d3.forceManyBody().strength(-100));
        </script>
    </body>
    </html>
    """
    with open("graph_3d.html", "w") as f:
        f.write(html_content)
    return html_content

# Функция для создания легенды
def create_legend(filtered_nodes, filtered_edges, top_wallet):
    total_nodes = len(filtered_nodes)
    total_edges = len(filtered_edges)
    max_size = max([float(G.nodes[n].get('size', 10)) for n in filtered_nodes], default=10)
    cluster_count = len(set(partition[n] for n in filtered_nodes))
    rotation_status = 'Off' if not rotate_top_wallets.value else rotation_direction.value
    return f"""
    <div style='background: rgba(0,0,0,0.7); color: #fff; padding: 12px; border-radius: 6px; position: absolute; top: 10px; left: 10px; z-index: 1000; font-family: Arial; font-size: 14px;'>
        <b>Tokenomics ({render_mode.value})</b><br>
        - Nodes: {total_nodes}<br>
        - Edges: {total_edges}<br>
        - Max Size: {max_size:.2f} (<a href='javascript:void(0)' onclick="document.getElementById('focus_wallet').value='{top_wallet}';document.getElementById('apply_button').click();">Focus</a>)<br>
        - Clusters: {cluster_count}<br>
        - Node Size: Exponential scaling<br>
        - Edge Width: Tx weight<br>
        - Color: {color_mode.value}<br>
        - Rotation: {rotation_status}<br>
        - Tooltip: Hover for details<br>
        - Top Wallets: Red nodes
    </div>
    """

# Функция для аналитики
def render_analytics(filtered_nodes):
    sizes = [float(G.nodes[n].get('size', 10)) for n in filtered_nodes]
    fig = px.histogram(x=sizes, nbins=20, title="Wallet Size Distribution", labels={'x': 'Wallet Size'})
    fig.update_layout(showlegend=False, plot_bgcolor='rgba(0,0,0,0)', paper_bgcolor='rgba(0,0,0,0)', font_color='white' if theme_mode.value == 'Dark' else 'black')
    fig.show()

# Функция для создания GIF
def save_gif():
    display(HTML("""
    <div id='gif_notification' class='notification'>Creating GIF...</div>
    <script src="https://html2canvas.hertzen.com/dist/html2canvas.min.js"></script>
    <script>
        function captureFrames(count, callback) {{
            if (count <= 0) return callback();
            html2canvas(document.querySelector(".vis-network") || document.querySelector("#graph")).then(canvas => {{
                var img = canvas.toDataURL('image/png');
                var xhr = new XMLHttpRequest();
                xhr.open('POST', '/_message', true);
                xhr.setRequestHeader('Content-Type', 'application/json');
                xhr.send(JSON.stringify({{ frame: img }}));
                setTimeout(() => captureFrames(count - 1, callback), 200);
            }});
        }}
        captureFrames(10, () => {{
            document.getElementById('gif_notification').innerText = 'GIF Created!';
            setTimeout(() => document.getElementById('gif_notification').style.display = 'none', 1500);
        }});
    </script>
    """))

# Обработчик кадров для GIF
frames = []
def handle_frame(data):
    global frames
    frame_data = json.loads(data)['frame']
    img_data = base64.b64decode(frame_data.split(',')[1])
    img = Image.open(io.BytesIO(img_data))
    frames.append(np.array(img))
    if len(frames) == 10:
        output_path = 'graph_animation.gif'
        imageio.mimsave(output_path, frames, duration=0.2, loop=0)
        files.download(output_path)
        frames = []
from google.colab import output
output.register_callback('notebook.handle_frame', handle_frame)

# Основная функция обновления
def update_visualization(b=None):
    # Показать уведомление
    display(HTML("""
    <div id='notification' class='notification'>Settings Applied</div>
    <script>
        const notification = document.getElementById('notification');
        notification.style.display = 'block';
        setTimeout(() => notification.style.display = 'none', 1500);
    </script>
    """))

    # Адаптация темы
    if theme_mode.value == 'Light':
        bg_color.value, font_color.value = '#f0f0f0', '#000000'
    else:
        bg_color.value, font_color.value = '#0a0a23', '#ffffff'

    # Фильтрация узлов и рёбер
    filtered_nodes = [n for n in G.nodes if float(G.nodes[n].get('size', 10)) >= min_size_filter.value and float(G.nodes[n].get('color', 0)) >= min_color_filter.value]
    if focus_wallet.value.strip():
        try:
            focus_node = next(n for n in filtered_nodes if G.nodes[n].get('label', n) == focus_wallet.value.strip())
            neighbors = set(nx.ego_graph(G, focus_node, radius=2).nodes())
            filtered_nodes = [n for n in filtered_nodes if n in neighbors]
        except StopIteration:
            pass
    filtered_edges = [(u, v) for u, v in G.edges if u in filtered_nodes and v in filtered_nodes and float(G.edges[u, v].get('weight', 0)) >= min_weight_filter.value]

    # Определение топ-кошельков
    top_wallets = []
    if show_top_wallets.value:
        sizes = [(n, float(G.nodes[n].get('size', 10))) for n in filtered_nodes]
        top_wallets = [n for n, _ in sorted(sizes, key=lambda x: x[1], reverse=True)[:top_wallets_count.value]]

    # Вычисление степеней узлов
    degrees = dict(G.degree(filtered_nodes))

    # Рендеринг
    if render_mode.value == '2D':
        net = render_2d_visualization(filtered_nodes, filtered_edges, top_wallets, degrees)
        net.show("graph.html")
        with open("graph.html", "r") as f:
            html_content = f.read()
    else:
        html_content = render_3d_visualization(filtered_nodes, filtered_edges, top_wallets, degrees)

    # Стили и анимации
    animations = ['wave 2s infinite'] if show_top_wallets.value else []
    if rotate_top_wallets.value and high_quality.value and show_top_wallets.value:
        rotation_keyframe = 'rotate' if rotation_direction.value == 'Clockwise' else 'rotate-reverse'
        animations.append(f'{rotation_keyframe} 20s linear infinite')
    animation_css = ', '.join(animations) if animations else 'none'
    custom_css = f"""
    <style>
        .vis-network canvas {{
            background: linear-gradient(135deg, {bg_color.value} 0%, {'#e0e0e0' if theme_mode.value == 'Light' else '#16213e'} 100%);
            transition: background 0.5s;
        }}
        .vis-network .vis-node {{
            transition: all 0.3s;
        }}
        .vis-network .vis-node:hover {{
            animation: pulse 1.2s infinite;
        }}
        .vis-network .vis-node.top-wallet {{
            animation: {animation_css};
        }}
        .vis-network .vis-edge.top-edge {{
            animation: pulse-edge 2s infinite;
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
        @keyframes rotate {{
            0% {{ transform: rotate(0deg); }}
            100% {{ transform: rotate(360deg); }}
        }}
        @keyframes rotate-reverse {{
            0% {{ transform: rotate(0deg); }}
            100% {{ transform: rotate(-360deg); }}
        }}
        @keyframes pulse-edge {{
            0% {{ stroke-opacity: 0.7; }}
            50% {{ stroke-opacity: 1; }}
            100% {{ stroke-opacity: 0.7; }}
        }}
        .particles {{ position: absolute; width: 100%; height: 800px; pointer-events: none; }}
        .particle {{ position: absolute; background: {'#000000' if theme_mode.value == 'Light' else '#ffffff'}; border-radius: 50%; opacity: 0.4; animation: move 8s linear infinite; }}
        @keyframes move {{
            0% {{ transform: translateY(0); opacity: 0.4; }}
            100% {{ transform: translateY(-800px); opacity: 0; }}
        }}
    </style>
    <script>
        if ({'true' if render_mode.value == '2D' and high_quality.value else 'false'}) {{
            const particleContainer = document.createElement('div');
            particleContainer.className = 'particles';
            document.body.appendChild(particleContainer);
            for (let i = 0; i < 50; i++) {{
                const particle = document.createElement('div');
                particle.className = 'particle';
                particle.style.width = '3px';
                particle.style.height = '3px';
                particle.style.left = Math.random() * 100 + '%';
                particle.style.animationDelay = Math.random() * 8 + 's';
                particleContainer.appendChild(particle);
            }}
        }}
        document.querySelectorAll('.vis-node.top-wallet').forEach(node => {{
            node.style.animation = '{animation_css}';
        }});
    </script>
    """

    # Легенда
    top_wallet = max([(n, float(G.nodes[n].get('size', 10))) for n in filtered_nodes], key=lambda x: x[1], default=('None', 0))[0]
    legend_html = create_legend(filtered_nodes, filtered_edges, top_wallet)

    # Сохранение скриншота
    screenshot_button = widgets.Button(description="Screenshot", button_style='info', tooltip='Save graph as PNG')
    def on_screenshot_button_clicked(b):
        display(HTML("""
        <script src="https://html2canvas.hertzen.com/dist/html2canvas.min.js"></script>
        <script>
            html2canvas(document.querySelector(".vis-network") || document.querySelector("#graph")).then(canvas => {{
                var link = document.createElement('a');
                link.download = 'graph_screenshot.png';
                link.href = canvas.toDataURL();
                link.click();
            }});
        </script>
        """))
    screenshot_button.on_click(on_screenshot_button_clicked)

    # Сохранение GIF
    gif_button = widgets.Button(description="Save GIF", button_style='info', tooltip='Save graph as animated GIF')
    def on_gif_button_clicked(b):
        save_gif()
    gif_button.on_click(on_gif_button_clicked)

    # Экспорт данных
    export_button = widgets.Button(description="Export CSV", button_style='info', tooltip='Export graph data as CSV')
    def on_export_button_clicked(b):
        df = pd.DataFrame([(n, G.nodes[n].get('size', 10), G.nodes[n].get('color', 0), partition.get(n, 0), degrees[n]) for n in filtered_nodes],
                          columns=['Address', 'Size', 'Color', 'Cluster', 'Degree'])
        df.to_csv("graph_data.csv", index=False)
        files.download("graph_data.csv")
    export_button.on_click(on_export_button_clicked)

    # Отображение
    display(HTML(f"""
    <div class='particles' style='position: absolute; width: 100%; height: 800px;'></div>
    {menu_css}
    {custom_css}
    {legend_html}
    {html_content}
    <script>
        document.getElementById('focus_wallet').id = 'focus_wallet';
        document.getElementById('apply_button').id = 'apply_button';
    </script>
    """))
    display(widgets.HBox([screenshot_button, gif_button, export_button]))

    # Аналитика
    render_analytics(filtered_nodes)

# Привязка кнопки Apply
apply_button.on_click(update_visualization)

# Отображение меню и кнопки
display(widgets.VBox([accordion, apply_button]))

# Кнопка для скачивания HTML
download_button = widgets.Button(description="Download HTML", button_style='info', tooltip='Download graph as HTML')
def on_download_button_clicked(b):
    files.download("graph.html" if render_mode.value == '2D' else "graph_3d.html")
download_button.on_click(on_download_button_clicked)
display(download_button)

# Первоначальная визуализация
update_visualization()
