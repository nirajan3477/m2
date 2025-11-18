import streamlit as st
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import os
import math

st.set_page_config(layout="wide", page_title="Quantum Confinement + Extras")

# --- Constants ---
h = 6.626e-34
m_e = 9.11e-31
eV = 1.602e-19

# --- Default color for plots/objects (neutral gray) ---
DEFAULT_OBJ_COLOR = "#646464"

# --- Materials ---
MATERIALS = {
    "Gallium Arsenide (GaAs)": {"m_star": 0.067 * m_e, "Eg0": 1.519, "varshni_alpha": 5.405e-4, "varshni_beta": 204},
    "Cadmium Selenide (CdSe)": {"m_star": 0.13 * m_e, "Eg0": 1.84, "varshni_alpha": 4.5e-4, "varshni_beta": 170},
    "Silicon (Si)": {"m_star": 0.19 * m_e, "Eg0": 1.17, "varshni_alpha": 4.73e-4, "varshni_beta": 636}
}

BULK_MATERIAL_IMAGES = {
    "Gallium Arsenide (GaAs)": "images/gaas.jpg",
    "Cadmium Selenide (CdSe)": "images/cdse.jpg",
    "Silicon (Si)": "images/si.jpg"
}

# --- Helper Functions ---

@st.cache_data
def calculate_energy_levels(L_nm, m_star, dimensionality):
    L_m = L_nm * 1e-9
    if L_m <= 0:
        return [], 0.0

    constant_factor_J = (h**2) / (8.0 * m_star * L_m**2)
    constant_factor_eV = constant_factor_J / eV
    energies = []

    if dimensionality == "Quantum Wire (1D Freedom)":
        for n in range(1, 21):
            energies.append(((n,), (n**2) * constant_factor_eV))

    elif dimensionality == "Quantum Well (2D Freedom)":
        for nx in range(1, 11):
            for ny in range(1, 11):
                energies.append(((nx, ny), (nx**2 + ny**2) * constant_factor_eV))

    elif dimensionality == "Quantum Dot (0D Freedom)":
        for nx in range(1, 7):
            for ny in range(1, 7):
                for nz in range(1, 7):
                    energies.append(((nx, ny, nz), (nx**2 + ny**2 + nz**2) * constant_factor_eV))

    elif dimensionality == "Bulk (3D Freedom)":
        return [], 0.0

    energies.sort(key=lambda x: x[1])
    unique_levels = []
    seen = []

    for q_numbers, E in energies:
        if not any(math.isclose(E, s, rel_tol=1e-9, abs_tol=1e-12) for s in seen):
            unique_levels.append((q_numbers, E))
            seen.append(E)
        if len(unique_levels) >= 12:
            break

    band_gap_eV = unique_levels[0][1] if unique_levels else 0.0
    return unique_levels, band_gap_eV

def varshni_bandgap(Eg0, T, alpha, beta):
    return Eg0 - (alpha * T * T) / (T + beta)

def create_confinement_plot(dimensionality, color_hex, is_selected, material_name=None):
    """
    Creates a 3D matplotlib plot or displays an image.
    """
    
    # Try to plot the real material image for Bulk
    if dimensionality == "Bulk (3D Freedom)":
        try:
            if material_name and material_name in BULK_MATERIAL_IMAGES:
                img = plt.imread(BULK_MATERIAL_IMAGES[material_name]) 
                fig, ax = plt.subplots(figsize=(5, 5))
                ax.imshow(img)
                ax.axis('off') 
                ax.set_title(f"{material_name.split(' (')[0]} Bulk", color='white', fontsize=14)
                
                if is_selected:
                    fig.patch.set_edgecolor('green')
                    fig.patch.set_linewidth(5)
                fig.set_facecolor('#0E1117')
                return fig 
                
        except FileNotFoundError:
            print(f"Warning: Image file not found for {material_name} at {BULK_MATERIAL_IMAGES[material_name]}. Check folder/filename.")
            pass 
        except Exception as e:
            print(f"Warning: Failed to load bulk image {material_name}. Drawing 3D cube instead. Error: {e}")
            pass 
    
    # Fallback 3D Plot code (for Well, Wire, Dot, and Bulk fallback)
    fig = plt.figure(figsize=(5, 5))
    
    if is_selected:
        fig.patch.set_edgecolor('green')
        fig.patch.set_linewidth(5)
    else:
        fig.patch.set_edgecolor('none')

    ax = fig.add_subplot(111, projection='3d')
    ax.set_facecolor('#0E1117') 
    fig.set_facecolor('#0E1117')
    
    ax.set_xlim([0, 10]); ax.set_ylim([0, 10]); ax.set_zlim([0, 10])
    ax.set_xlabel('X', color='white'); ax.set_ylabel('Y', color='white'); ax.set_zlabel('Z', color='white')
    ax.set_xticks([]); ax.set_yticks([]); ax.set_zticks([])
    ax.xaxis.line.set_color("gray"); ax.yaxis.line.set_color("gray"); ax.zaxis.line.set_color("gray")
    ax.grid(False)

    def draw_arrow(ax, start, end, color):
        ax.quiver(start[0], start[1], start[2],
                  end[0]-start[0], end[1]-start[1], end[2]-start[2],
                  color=color, arrow_length_ratio=0.3, linewidth=3)
    
    object_color = color_hex 
    object_alpha = 0.8 if dimensionality != "Bulk (3D Freedom)" else 0.3

    if dimensionality == "Bulk (3D Freedom)": # This is now the fallback
        ax.bar3d(1, 1, 1, 8, 8, 8, color=DEFAULT_BULK_COLOR_HEX, alpha=0.3) 
        draw_arrow(ax, (5, 5, 5), (9, 5, 5), 'green') # X
        draw_arrow(ax, (5, 5, 5), (5, 9, 5), 'green') # Y
        draw_arrow(ax, (5, 5, 5), (5, 5, 9), 'green') # Z
        
    elif dimensionality == "Quantum Well (2D Freedom)":
        ax.bar3d(1, 1, 4.5, 8, 8, 1, color=object_color, alpha=object_alpha)
        draw_arrow(ax, (5, 5, 5), (9, 5, 5), 'green') # X
        draw_arrow(ax, (5, 5, 5), (5, 9, 5), 'green') # Y
        draw_arrow(ax, (5, 5, 8), (5, 5, 6), 'red') # Z-in
        draw_arrow(ax, (5, 5, 2), (5, 5, 4), 'red') # Z-in

    elif dimensionality == "Quantum Wire (1D Freedom)":
        ax.bar3d(1, 4.5, 4.5, 8, 1, 1, color=object_color, alpha=object_alpha)
        draw_arrow(ax, (5, 5, 5), (9, 5, 5), 'green') # X
        draw_arrow(ax, (5, 8, 5), (5, 6, 5), 'red') # Y-in
        draw_arrow(ax, (5, 2, 5), (5, 4, 5), 'red') # Y-in
        draw_arrow(ax, (5, 5, 8), (5, 5, 6), 'red') # Z-in
        draw_arrow(ax, (5, 5, 2), (5, 5, 4), 'red') # Z-in

    elif dimensionality == "Quantum Dot (0D Freedom)":
        ax.bar3d(4, 4, 4, 2, 2, 2, color=object_color, alpha=object_alpha)
        draw_arrow(ax, (8, 5, 5), (6, 5, 5), 'red') # X-in
        draw_arrow(ax, (2, 5, 5), (4, 5, 5), 'red') # X-in
        draw_arrow(ax, (5, 8, 5), (5, 6, 5), 'red') # Y-in
        draw_arrow(ax, (5, 2, 5), (5, 4, 5), 'red') # Y-in
        draw_arrow(ax, (5, 5, 8), (5, 5, 6), 'red') # Z-in
        draw_arrow(ax, (5, 5, 2), (5, 5, 4), 'red') # Z-in
        
    return fig

def plot_electronic_structure(dim_choice, energy_levels, L_nm):
    fig, ax = plt.subplots(figsize=(8, 5))
    fig.patch.set_facecolor('#0E1117')
    ax.set_facecolor('#0E1117')
    ax.set_ylabel("Energy (eV)", color='white')
    ax.tick_params(colors='white')

    if dim_choice == "Bulk (3D Freedom)":
        E = np.linspace(0.001, 2.0, 300)
        dos = np.sqrt(E)
        ax.plot(dos, E, linewidth=3)
        ax.set_xlabel("Density of States g(E)", color='white')
        ax.set_title("Continuous DOS (Bulk)", color='white')
        ax.fill_between(dos, E, alpha=0.25)
        ax.set_xticks([])

    elif dim_choice == "Quantum Well (2D Freedom)":
        E_levels = [E for q, E in energy_levels] or [0.1]
        plot_limit = E_levels[-1] * 1.2
        energies = [0] + E_levels + [plot_limit]
        dos_steps = [0]
        for i in range(1, len(energies)):
            dos_steps.append(i - 1)
        ax.step(dos_steps, energies, where='post', linewidth=3)
        ax.set_ylim(0, plot_limit)
        ax.set_xlim(0, len(E_levels) * 1.1)
        ax.set_xticks([])
        ax.set_title("Staircase DOS (Quantum Well)", color='white')

    elif dim_choice == "Quantum Wire (1D Freedom)":
        E_levels = [E for q, E in energy_levels] or [0.1]
        plot_limit = E_levels[-1] * 1.2
        ax.set_ylim(0, plot_limit)
        ax.set_xlim(0, 10)
        ax.set_xticks([])
        if len(E_levels) > 1:
            e_scale = max((E_levels[1] - E_levels[0]) * 0.8, plot_limit / 5)
        else:
            e_scale = plot_limit / 5
        e_axis_rel = np.linspace(0.01, e_scale, 120)
        dos_shape = (1.0 / np.sqrt(e_axis_rel))
        dos_shape = (dos_shape / np.max(dos_shape)) * 8
        for q, E_n in energy_levels:
            if E_n < plot_limit:
                ax.plot(dos_shape, E_n + e_axis_rel, linewidth=2)
                ax.hlines(E_n, 0, dos_shape[0], linestyle='--', linewidth=1)
        ax.set_title("1D DOS (Quantum Wire)", color='white')

    elif dim_choice == "Quantum Dot (0D Freedom)":
        ax.set_xticks([])
        ax.set_xlim(0, 1.4)
        if energy_levels:
            maxE = energy_levels[-1][1]
            for q, E in energy_levels:
                ax.hlines(E, 0, 1, linewidth=3)
                ax.text(1.02, E, f"{q} ({E:.3f} eV)", color='white', fontsize=9)
            ax.set_ylim(0, maxE * 1.2)
        else:
            ax.set_ylim(0, 1)
            ax.text(0.5, 0.5, "No levels", color='white')
        ax.set_title("0D Levels (Quantum Dot)", color='white')

    return fig

def compute_SA_V(shape, size_nm):
    L = size_nm * 1e-9
    if shape == "Sphere":
        r = L / 2
        SA = 4 * math.pi * r*r
        V = (4/3) * math.pi * r*r*r
    elif shape == "Cube":
        SA = 6 * L*L
        V = L*L*L
    elif shape == "Rod (cylinder)":
        r = L/6
        h = L
        SA = 2*math.pi*r*h + 2*math.pi*r*r
        V = math.pi*r*r*h
    elif shape == "Plate":
        t = L/10
        a = L
        SA = 2*a*a + 4*a*t
        V = a*a*t
    else:
        SA = V = 0
    return SA, V, (SA/V if V > 0 else float("inf"))

def coulomb_blockade_iv(C_total_f, T_K, Vg, Vbias_array):
    e = 1.602e-19
    kB = 1.380649e-23
    Ec = e*e / (2*C_total_f) if C_total_f > 0 else 0.0
    beta = 1/(kB*T_K) if T_K > 0 else 0.0
    I = []
    for V in Vbias_array:
        delta = Ec - e*Vg
        expo = beta*abs(delta - e*V) if beta != 0 else 0.0
        rate = math.exp(-expo) if expo < 700 and expo > 0 else (1.0 if expo == 0 else 0.0)
        I.append(rate*(V + 1e-12))
    return np.array(I), Ec

st.title("Quantum-Confinement-and-Nanoscale-Electronic-Structure-Simulator")

st.sidebar.header("Simulation Controls")
dim_choice = st.sidebar.radio("Dimensionality",
    ["Bulk (3D Freedom)", "Quantum Well (2D Freedom)",
     "Quantum Wire (1D Freedom)", "Quantum Dot (0D Freedom)"], index=3)

mat_choice = st.sidebar.selectbox("Material", list(MATERIALS.keys()))
use_real = st.sidebar.checkbox("Use real Varshni parameters", value=True)

L_MIN, L_MAX = 1.0, 25.0
L_nm = st.sidebar.slider("Size L (nm)", L_MIN, L_MAX, 7.0, 0.1)

# Temperature
st.sidebar.markdown("---")
st.sidebar.header("Temperature & Bandgap")
T_K = st.sidebar.slider("Temperature (K)", 0, 600, 300)
mat = MATERIALS[mat_choice]
Eg0 = mat["Eg0"]
alpha = mat["varshni_alpha"]
beta = mat["varshni_beta"]
Eg_T = varshni_bandgap(Eg0, T_K, alpha, beta)
st.sidebar.write(f"Bulk Eg(T) ≈ {Eg_T:.3f} eV")

# SA/V
st.sidebar.markdown("---")
st.sidebar.header("Surface-to-Volume Ratio")
shape = st.sidebar.selectbox("Shape", ["Sphere", "Cube", "Rod (cylinder)", "Plate"])
sv_size = st.sidebar.slider("Shape size (nm)", 1.0, 200.0, 50.0)

# Coulomb blockade
st.sidebar.markdown("---")
st.sidebar.header("Coulomb Blockade")
C_total_f = st.sidebar.slider("Total island capacitance (aF)", 0.1, 1000.0, 10.0) * 1e-18
Vg = st.sidebar.slider("Gate potential", -1.0, 1.0, 0.0)
T_CB = st.sidebar.slider("SET Temperature (K)", 0.1, 100.0, 4.0)

# Compute energy levels
m_star_val = MATERIALS[mat_choice]["m_star"]
energy_levels, band_gap = calculate_energy_levels(L_nm, m_star_val, dim_choice)

# Use default neutral color for visual elements
object_color_hex = DEFAULT_OBJ_COLOR

# -----------------------------------------------------------
# Confinement Visual
# -----------------------------------------------------------
st.markdown("---")
st.header("Visualizing Confinement")

col1, col2, col3, col4 = st.columns(4)

with col1:
    st.subheader("Bulk")
    fig = create_confinement_plot("Bulk (3D Freedom)", color_hex=object_color_hex,
                                  is_selected=(dim_choice == "Bulk (3D Freedom)"), material_name=mat_choice)
    st.pyplot(fig)
    plt.close(fig)

with col2:
    st.subheader("Quantum Well")
    fig = create_confinement_plot("Quantum Well (2D Freedom)", color_hex=object_color_hex,
                                  is_selected=(dim_choice == "Quantum Well (2D Freedom)"), material_name=mat_choice)
    st.pyplot(fig)
    plt.close(fig)

with col3:
    st.subheader("Quantum Wire")
    fig = create_confinement_plot("Quantum Wire (1D Freedom)", color_hex=object_color_hex,
                                  is_selected=(dim_choice == "Quantum Wire (1D Freedom)"), material_name=mat_choice)
    st.pyplot(fig)
    plt.close(fig)

with col4:
    st.subheader("Quantum Dot")
    fig = create_confinement_plot("Quantum Dot (0D Freedom)", color_hex=object_color_hex,
                                  is_selected=(dim_choice == "Quantum Dot (0D Freedom)"), material_name=mat_choice)
    st.pyplot(fig)
    plt.close(fig)

# -----------------------------------------------------------
# Simulation Results
# -----------------------------------------------------------
st.markdown("---")
st.header("Simulation Results")

res_col1, res_col2 = st.columns([1, 1.4])

with res_col1:
    st.subheader("Material & Bandgap")
    st.write(f"Material: **{mat_choice.split(' (')[0]}**")
    if dim_choice != "Bulk (3D Freedom)":
        st.metric("Calculated Band Gap", f"{band_gap:.3f} eV")
    else:
        st.info("Bulk: continuum DOS, band-like behavior.")

    st.markdown("### Surface-to-Volume")
    SA, V, SV = compute_SA_V(shape, sv_size)
    st.write(f"Shape: **{shape}**")
    st.write(f"Surface area ≈ {SA:.3e} m²")
    st.write(f"Volume ≈ {V:.3e} m³")
    st.write(f"S/V ratio ≈ {SV:.3e} m⁻¹")

with res_col2:
    st.subheader("Electronic Structure")
    fig = plot_electronic_structure(dim_choice, energy_levels, L_nm)
    st.pyplot(fig)
    plt.close(fig)

# -----------------------------------------------------------
# Coulomb Blockade Section
# -----------------------------------------------------------
st.markdown("---")
st.header("Coulomb Blockade Demo")

Vbias = np.linspace(-0.1, 0.1, 200)
I_cb, Ec = coulomb_blockade_iv(C_total_f, T_CB, Vg, Vbias)

fig, ax = plt.subplots(figsize=(6, 3))
fig.patch.set_facecolor('#0E1117')
ax.set_facecolor('#0E1117')
ax.plot(Vbias, I_cb, linewidth=2)
ax.set_xlabel("Bias Voltage (V)", color='white')
ax.set_ylabel("Current (arb.)", color='white')
ax.tick_params(colors='white')
st.write(f"Charging Energy E₍C₎ ≈ {Ec/eV:.3f} eV")
st.pyplot(fig)
plt.close(fig)