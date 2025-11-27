from graphviz import Digraph

# Create a directed graph
g = Digraph('MSW_Flow_Gray', format='png')
g.attr(rankdir='TB', bgcolor='white', splines='ortho', ranksep='0.6')

# Global node style
g.attr('node', shape='box', style='filled,rounded', color='#333333',
       fontname='Helvetica', fontsize='11', fontcolor='black', width='2.8')

# --- Process & Financial Chain (uniform gray) ---
fill = '#E5E7EB'  # light gray (Tailwind gray-200)

g.node('A', 'MSW TPD (Feedstock Input) Increases', fillcolor=fill)
g.node('B', 'Dry MSW = MSW * (1 - Moisture Loss)', fillcolor=fill)
g.node('C', 'RDF Yield = Dry MSW * 0.8187', fillcolor=fill)
g.node('D', 'Syngas Yield = RDF * 0.45', fillcolor=fill)
g.node('E', 'Power Output (MWe) = (Syngas/450.3)*36.5 + (TDF/85)*9.8', fillcolor=fill)
g.node('F', 'Revenue = Power * Power Price', fillcolor=fill)
g.node('G', 'Variable O&M Costs scale with throughput', fillcolor=fill)
g.node('H', 'Cash Flow = Revenue - (Fixed + Variable Costs)', fillcolor=fill)
g.node('I', 'NPV recalculated using new Cash Flows (12% WACC)', fillcolor=fill)
g.node('J', 'IRR recalculated where NPV = 0', fillcolor=fill)
g.node('K', 'Equity IRR updates (leverage amplified)', fillcolor=fill)
g.node('L', 'Updated Project KPIs & Summary', fillcolor=fill)

# --- Edges (connections) ---
edges = [
    ('A','B'), ('B','C'), ('C','D'), ('D','E'),
    ('E','F'), ('F','G'), ('G','H'), ('H','I'),
    ('H','J'), ('J','K'), ('I','L')
]
for u, v in edges:
    g.edge(u, v, color='#4B5563', penwidth='1.3')  # medium gray lines

# Render and display
g.render('msw_flowchart_gray', view=True)
