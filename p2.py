import streamlit as st
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import os
import math

st.set_page_config(layout="wide", page_title="Quantum Confinement Simulator")

# --- Constants ---
h = 6.626e-34
m_e = 9.11e-31
eV = 1.602e-19

# --- Default color for plots/objects (neutral gray) ---
DEFAULT_OBJ_COLOR = "#646464"

# --- Materials ---
MATERIALS = {
    "Gallium Arsenide (GaAs)": {"m_star": 0.067 * m_e, "Eg0": 1.519},
    "Cadmium Selenide (CdSe)": {"m_star": 0.13 * m_e, "Eg0": 1.74}, # Adjusted for visible range
    "Silicon (Si)": {"m_star": 0.19 * m_e, "Eg0": 1.17},
    "Gold (Au) [Metal]": {"m_star": 1.0 * m_e, "Eg0": 0.0} # No Bandgap
}

# --- Bulk Material Appearance (Reflected Color) ---
BULK_APPEARANCE = {
    "Gallium Arsenide (GaAs)": "#505050",  # Dark Metallic Grey
    "Cadmium Selenide (CdSe)": "#421C02",  # Dark Reddish-Black
    "Silicon (Si)": "#686C70",             # Metallic Light Grey
    "Gold (Au) [Metal]": "#FFD700"         # Gold
}

# --- Helper Functions ---

def wavelength_to_rgb(wavelength_nm, gamma=0.8):
    """
    Converts a wavelength in nm to an RGB tuple (0-255).
    Approximates the color of *emitted* light.
    """
    if wavelength_nm < 380:
        return (75, 0, 130) # Indigo/Violet for UV
    if wavelength_nm > 780:
        return (128, 0, 0) # Dark Red for IR

    if wavelength_nm >= 380 and wavelength_nm < 440:
        attenuation = 0.3 + 0.7 * (wavelength_nm - 380) / (440 - 380)
        R = ((-(wavelength_nm - 440) / (440 - 380)) * attenuation)**gamma
        G = 0.0
        B = (1.0 * attenuation)**gamma
    elif wavelength_nm >= 440 and wavelength_nm < 490:
        R = 0.0
        G = ((wavelength_nm - 440) / (490 - 440))**gamma
        B = 1.0
    elif wavelength_nm >= 490 and wavelength_nm < 510:
        R = 0.0
        G = 1.0
        B = (-(wavelength_nm - 510) / (510 - 490))**gamma
    elif wavelength_nm >= 510 and wavelength_nm < 580:
        R = ((wavelength_nm - 510) / (580 - 510))**gamma
        G = 1.0
        B = 0.0
    elif wavelength_nm >= 580 and wavelength_nm < 645:
        R = 1.0
        G = (-(wavelength_nm - 645) / (645 - 580))**gamma
        B = 0.0
    elif wavelength_nm >= 645 and wavelength_nm <= 780:
        attenuation = 0.3 + 0.7 * (780 - wavelength_nm) / (780 - 645)
        R = (1.0 * attenuation)**gamma
        G = 0.0
        B = 0.0
    else:
        R = 0.0; G = 0.0; B = 0.0
    return (int(R * 255), int(G * 255), int(B * 255))

@st.cache_data
def calculate_energy_levels(L_nm, m_star, dimensionality):
    L_m = L_nm * 1e-9
    if L_m <= 0:
        return [], 0.0

    # Particle in a box constant
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

    # Sort and filter close energy levels
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

def create_confinement_plot(dimensionality, color_hex, is_selected, material_name=None):
    """Creates a 3D matplotlib plot."""
    fig = plt.figure(figsize=(5, 5))
    
    if is_selected:
        fig.patch.set_edgecolor('#00FF00') # Bright Green border for selection
        fig.patch.set_linewidth(3)
    else:
        fig.patch.set_edgecolor('none')

    ax = fig.add_subplot(111, projection='3d')
    ax.set_facecolor('#0E1117') 
    fig.set_facecolor('#0E1117')
    
    ax.set_xlim([0, 10]); ax.set_ylim([0, 10]); ax.set_zlim([0, 10])
    ax.set_xlabel('X', color='white'); ax.set_ylabel('Y', color='white'); ax.set_zlabel('Z', color='white')
    ax.set_xticks([]); ax.set_yticks([]); ax.set_zticks([])

    def draw_arrow(ax, start, end, color):
        ax.quiver(start[0], start[1], start[2],
                  end[0]-start[0], end[1]-start[1], end[2]-start[2],
                  color=color, arrow_length_ratio=0.3, linewidth=3)
    
    object_color = color_hex if color_hex is not None else DEFAULT_OBJ_COLOR
    object_alpha = 0.4 if dimensionality != "Bulk (3D Freedom)" else 0.3

    if dimensionality == "Bulk (3D Freedom)":
        # Draw a large transparent block
        ax.bar3d(1, 1, 1, 8, 8, 8, color=DEFAULT_OBJ_COLOR, alpha=0.3) 
        # Arrows in all directions indicating freedom
        draw_arrow(ax, (5, 5, 5), (9, 5, 5), 'cyan')
        draw_arrow(ax, (5, 5, 5), (5, 9, 5), 'cyan')
        draw_arrow(ax, (5, 5, 5), (5, 5, 9), 'cyan')
        
    elif dimensionality == "Quantum Well (2D Freedom)":
        # Plate shape
        ax.bar3d(1, 1, 4.5, 8, 8, 1, color=object_color, alpha=object_alpha)
        draw_arrow(ax, (5, 5, 5), (9, 5, 5), 'cyan') # Freedom X
        draw_arrow(ax, (5, 5, 5), (5, 9, 5), 'cyan') # Freedom Y
        draw_arrow(ax, (5, 5, 8), (5, 5, 6), 'red') # Confinement Z
        draw_arrow(ax, (5, 5, 2), (5, 5, 4), 'red') # Confinement Z

    elif dimensionality == "Quantum Wire (1D Freedom)":
        # Rod shape
        ax.bar3d(1, 4.5, 4.5, 8, 1, 1, color=object_color, alpha=object_alpha)
        draw_arrow(ax, (5, 5, 5), (9, 5, 5), 'cyan') # Freedom X
        draw_arrow(ax, (5, 8, 5), (5, 6, 5), 'red') # Confinement Y
        draw_arrow(ax, (5, 2, 5), (5, 4, 5), 'red') # Confinement Y
        draw_arrow(ax, (5, 5, 8), (5, 5, 6), 'red') # Confinement Z
        draw_arrow(ax, (5, 5, 2), (5, 5, 4), 'red') # Confinement Z

    elif dimensionality == "Quantum Dot (0D Freedom)":
        # Cube/Dot shape
        ax.bar3d(4, 4, 4, 2, 2, 2, color=object_color, alpha=object_alpha)
        # Confinement in all directions
        draw_arrow(ax, (8, 5, 5), (6, 5, 5), 'red')
        draw_arrow(ax, (2, 5, 5), (4, 5, 5), 'red')
        draw_arrow(ax, (5, 8, 5), (5, 6, 5), 'red')
        draw_arrow(ax, (5, 2, 5), (5, 4, 5), 'red')
        draw_arrow(ax, (5, 5, 8), (5, 5, 6), 'red')
        draw_arrow(ax, (5, 5, 2), (5, 5, 4), 'red')
        
    return fig

def plot_electronic_structure(dim_choice, energy_levels, L_nm, mat_choice):
    fig, ax = plt.subplots(figsize=(8, 5))
    fig.patch.set_facecolor('#0E1117')
    ax.set_facecolor('#0E1117')
    ax.set_ylabel("Energy (eV)", color='white')
    ax.tick_params(colors='white')

    # Special handling for Metal (Gold)
    if "Gold" in mat_choice:
        ax.text(0.5, 0.5, "METALS HAVE CONTINUOUS BANDS\n(No Bandgap)", 
                color='gold', ha='center', va='center', fontsize=14)
        ax.set_xlim(0, 1)
        ax.set_ylim(0, 1)
        ax.set_xticks([])
        ax.set_yticks([])
        ax.set_title("Electronic Structure (Metal)", color='white')
        return fig

    if dim_choice == "Bulk (3D Freedom)":
        E = np.linspace(0.001, 2.0, 300)
        dos = np.sqrt(E)
        ax.plot(dos, E, linewidth=3, color='cyan')
        ax.set_xlabel("Density of States g(E)", color='white')
        ax.set_title("Continuous DOS (Bulk)", color='white')
        ax.fill_between(dos, E, alpha=0.25, color='cyan')
        ax.set_xticks([])

    elif dim_choice == "Quantum Well (2D Freedom)":
        E_levels = [E for q, E in energy_levels] or [0.1]
        plot_limit = E_levels[-1] * 1.2 if E_levels else 1.0
        energies = [0] + E_levels + [plot_limit]
        dos_steps = [0]
        for i in range(1, len(energies)):
            dos_steps.append(i - 1)
        ax.step(dos_steps, energies, where='post', linewidth=3, color='cyan')
        ax.set_ylim(0, plot_limit)
        ax.set_xlim(0, len(E_levels) * 1.1 if E_levels else 1)
        ax.set_xticks([])
        ax.set_title("Staircase DOS (Quantum Well)", color='white')

    elif dim_choice == "Quantum Wire (1D Freedom)":
        E_levels = [E for q, E in energy_levels] or [0.1]
        plot_limit = E_levels[-1] * 1.2 if E_levels else 1.0
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
                ax.plot(dos_shape, E_n + e_axis_rel, linewidth=2, color='cyan')
                ax.hlines(E_n, 0, dos_shape[0], linestyle='--', linewidth=1, color='white')
        ax.set_title("1D DOS (Quantum Wire)", color='white')

    elif dim_choice == "Quantum Dot (0D Freedom)":
        ax.set_xticks([])
        ax.set_xlim(0, 1.4)
        if energy_levels:
            maxE = energy_levels[-1][1]
            for q, E in energy_levels:
                ax.hlines(E, 0, 1, linewidth=3, color='cyan')
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

# ---------------------- HEADER -----------------------
st.title("Quantum Confinement & Nanoscale Electonic Simulator")

# ---------------------- SIDEBAR -----------------------
st.sidebar.header("Simulation Controls")

# Dimensionality
dim_choice = st.sidebar.radio("Dimensionality",
    ["Bulk (3D Freedom)", "Quantum Well (2D Freedom)",
     "Quantum Wire (1D Freedom)", "Quantum Dot (0D Freedom)"], index=3)

# Material
mat_choice = st.sidebar.selectbox("Material", list(MATERIALS.keys()))

# Size Slider
st.sidebar.markdown("---")
# Focusing on 1-25nm where the quantum effects are visible
L_MIN, L_MAX = 1.0, 60.0
L_nm = st.sidebar.slider("Size L (nm)", L_MIN, L_MAX, 4.0, 0.1)

# SA/V Controls
st.sidebar.markdown("---")
st.sidebar.header("Surface-to-Volume Ratio")
shape = st.sidebar.selectbox("Shape", ["Sphere", "Cube", "Rod (cylinder)", "Plate"])
sv_size = st.sidebar.slider("Shape size (nm)", 1.0, 200.0, 50.0)

# Coulomb blockade Controls
st.sidebar.markdown("---")
st.sidebar.header("Coulomb Blockade")
C_total_f = st.sidebar.slider("Total island capacitance (aF)", 0.1, 1000.0, 10.0) * 1e-18
Vg = st.sidebar.slider("Gate potential", -1.0, 1.0, 0.0)
T_CB = st.sidebar.slider("SET Temperature (K)", 0.1, 100.0, 4.0)

# --- CALCULATIONS ---
m_star_val = MATERIALS[mat_choice]["m_star"]
energy_levels, band_gap = calculate_energy_levels(L_nm, m_star_val, dim_choice)

# -----------------------------------------------------------
# 1. Confinement Visual
# -----------------------------------------------------------
st.markdown("---")
st.header("1. Visualizing Confinement")

col1, col2, col3, col4 = st.columns(4)

with col1:
    st.subheader("Bulk")
    fig = create_confinement_plot("Bulk (3D Freedom)", None,
                                  is_selected=(dim_choice == "Bulk (3D Freedom)"), material_name=mat_choice)
    st.pyplot(fig)
    plt.close(fig)

with col2:
    st.subheader("Quantum Well")
    fig = create_confinement_plot("Quantum Well (2D Freedom)", "#3498db",
                                  is_selected=(dim_choice == "Quantum Well (2D Freedom)"), material_name=mat_choice)
    st.pyplot(fig)
    plt.close(fig)

with col3:
    st.subheader("Quantum Wire")
    fig = create_confinement_plot("Quantum Wire (1D Freedom)", "#e74c3c",
                                  is_selected=(dim_choice == "Quantum Wire (1D Freedom)"), material_name=mat_choice)
    st.pyplot(fig)
    plt.close(fig)

with col4:
    st.subheader("Quantum Dot")
    fig = create_confinement_plot("Quantum Dot (0D Freedom)", "#9b59b6",
                                  is_selected=(dim_choice == "Quantum Dot (0D Freedom)"), material_name=mat_choice)
    st.pyplot(fig)
    plt.close(fig)

# -----------------------------------------------------------
# 2. Simulation Results (DOS)
# -----------------------------------------------------------
st.markdown("---")
st.header("2. Simulation Results")

res_col1, res_col2 = st.columns([1, 1.4])

with res_col1:
    st.subheader("Material & Bandgap")
    st.write(f"Material: **{mat_choice.split(' (')[0]}**")
    
    if "Gold" in mat_choice:
         st.warning("Metal Selected: No Band Gap.")
    elif dim_choice != "Bulk (3D Freedom)":
        st.metric("Calculated Band Gap", f"{band_gap:.3f} eV")
        st.info(f"Energy increase due to confinement: {(band_gap - MATERIALS[mat_choice]['Eg0']):.3f} eV")
    else:
        st.info("Bulk: Continuum DOS (Band-like behavior).")

    st.markdown("### Surface-to-Volume")
    SA, V, SV = compute_SA_V(shape, sv_size)
    st.write(f"Shape: **{shape}**")
    st.write(f"S/V ratio ≈ {SV:.3e} m⁻¹")

with res_col2:
    st.subheader("Electronic Structure (Density of States)")
    fig = plot_electronic_structure(dim_choice, energy_levels, L_nm, mat_choice)
    st.pyplot(fig)
    plt.close(fig)

# -----------------------------------------------------------
# 3. Real-World Comparison (Bulk vs Nano)
# -----------------------------------------------------------
st.markdown("---")
st.header("3. Real-World Comparison: Bulk vs. Nano")

# --- COLOR & REAL WORLD LOGIC ---
real_world_bulk_name = "Unknown"
real_world_nano_name = "Unknown"
real_world_bulk_icon = "🪨"
real_world_nano_icon = "✨"

if "Gold" in mat_choice:
    # --- GOLD LOGIC ---
    real_world_bulk_name = "Gold Bar / Jewelry"
    real_world_bulk_icon = "🏆"
    
    if L_nm < 10.0:
        nano_color_hex = "#E0115F" # Ruby Red
        emission_text = "Plasmonic Color: Ruby Red"
        real_world_nano_name = "Ancient Stained Glass (Lycurgus Cup)"
        real_world_nano_icon = "🍷"
    elif L_nm < 30.0:
        nano_color_hex = "#800080" # Purple/Violet
        emission_text = "Plasmonic Color: Violet/Purple"
        real_world_nano_name = "Medical Diagnostic Tests"
        real_world_nano_icon = "🧪"
    elif L_nm < 60.0:
        nano_color_hex = "#0000FF" # Blueish
        emission_text = "Plasmonic Color: Blue Shift"
        real_world_nano_name = "Experimental Sensors"
        real_world_nano_icon = "🔬"
    else:
        nano_color_hex = "#FFD700" # Gold
        emission_text = "Plasmonic Color: Gold (Bulk-like)"
        real_world_nano_name = "Gold Dust / Plating"
        real_world_nano_icon = "✨"
        
else:
    # --- SEMICONDUCTOR LOGIC (CdSe, GaAs, Si) ---
    if "Cadmium" in mat_choice:
        real_world_bulk_name = "Raw Crystal Rock"
        real_world_bulk_icon = "🪨"
        real_world_nano_name = "QLED TV Pixel"
        real_world_nano_icon = ""
    elif "Silicon" in mat_choice:
        real_world_bulk_name = "Computer Wafer / Chip"
        real_world_bulk_icon = "💻"
        real_world_nano_name = "Silicon Photonics / Bio-tag"
        real_world_nano_icon = "🧬"
    elif "Gallium" in mat_choice:
        real_world_bulk_name = "Solar Cell / LED"
        real_world_bulk_icon = "🔋"
        real_world_nano_name = "High-Speed Laser Diode"
        real_world_nano_icon = "📡"

    if band_gap > 0:
        wavelength_nm = (h * 2.998e8) / (band_gap * eV) * 1e9
        rgb_tuple = wavelength_to_rgb(wavelength_nm)
        nano_color_hex = f'#{rgb_tuple[0]:02x}{rgb_tuple[1]:02x}{rgb_tuple[2]:02x}'
        emission_text = f"Emits {nano_color_hex} Light ({wavelength_nm:.1f} nm)"
    else:
        nano_color_hex = "#000000"
        emission_text = "No Emission (IR/Gapless)"
        real_world_nano_name = "Non-Emissive Nanoparticle"

# Get the bulk color
bulk_color_hex = BULK_APPEARANCE.get(mat_choice, "#808080")

# Create two columns for side-by-side comparison
c1, c2 = st.columns(2)

with c1:
    st.subheader("Macroscopic (Bulk)")
    st.write("What a large piece looks like:")
    st.markdown(f"""
        <div style="
            width: 150px; height: 100px; 
            background-color: {bulk_color_hex}; 
            border: 2px solid #FFF;
            border-radius: 5px;
            display: flex; align-items: center; justify-content: center;
            color: {'black' if 'Gold' in mat_choice else 'white'}; 
            font-weight: bold; text-shadow: 1px 1px 2px {'white' if 'Gold' in mat_choice else 'black'};
            margin: auto;">
            BULK
        </div>
    """, unsafe_allow_html=True)
    
    # REAL WORLD ANALOGY CARD
    st.info(f"**Real World Object:**\n\n{real_world_bulk_icon} {real_world_bulk_name}")

with c2:
    st.subheader(f"Nanoscale (L={L_nm} nm)")
    st.write("Nanoparticle Color:")
    
    glow_color = nano_color_hex
    
    st.markdown(f"""
        <div style="
            width: 100px; height: 100px; 
            background-color: {nano_color_hex}; 
            border-radius: 50%; 
            box-shadow: 0 0 20px {glow_color}, 0 0 40px {glow_color};
            margin: auto;">
        </div>
    """, unsafe_allow_html=True)
    
    # REAL WORLD ANALOGY CARD
    st.success(f"**Real World Application:**\n\n{real_world_nano_icon} {real_world_nano_name}\n\n*{emission_text}*")

# Explanation caption
if "Gold" in mat_choice:
    st.caption("""
    **Fun Fact:** The **Lycurgus Cup** (4th Century Roman glass) looks green in daylight but turns **Ruby Red** when light shines through it. This is because the Romans accidentally created Gold Nanoparticles!
    """)
else:
    st.caption(f"""
    **The "Quantum Shift":** **{mat_choice.split(' (')[0]}** changes from a dull {real_world_bulk_name} 
    to a glowing {real_world_nano_name} when confined to {L_nm} nm.
    """)

# -----------------------------------------------------------
# 4. Coulomb Blockade Section
# -----------------------------------------------------------
st.markdown("---")
st.header("4. Coulomb Blockade Demo")

Vbias = np.linspace(-0.1, 0.1, 200)
I_cb, Ec = coulomb_blockade_iv(C_total_f, T_CB, Vg, Vbias)

fig, ax = plt.subplots(figsize=(6, 3))
fig.patch.set_facecolor('#0E1117')
ax.set_facecolor('#0E1117')
ax.plot(Vbias, I_cb, linewidth=2, color='cyan')
ax.set_xlabel("Bias Voltage (V)", color='white')
ax.set_ylabel("Conductance (arb.)", color='white')
ax.tick_params(colors='white')
st.write(f"Charging Energy E₍C₎ ≈ {Ec/eV:.3f} eV")
st.pyplot(fig)
plt.close(fig)

# -----------------------------------------------------------
# 5. Wavefunction Visualizer (The "Guitar String" Analogy)
# -----------------------------------------------------------
st.markdown("---")
st.header("5. Visualizing the Quantum State (Wavefunction)")

col_wave1, col_wave2 = st.columns([1, 2])

with col_wave1:
    st.info("An electron in a box behaves like a standing wave on a guitar string.")
    n_quantum = st.slider("Quantum Number (n)", 1, 5, 1)
    st.write(f"**Mode n={n_quantum}**")
    st.caption("Notice: Higher 'n' means more nodes (humps) and higher energy.")

with col_wave2:
    # Wavefunction calculation: psi = sqrt(2/L) * sin(n * pi * x / L)
    # We plot Probability Density |psi|^2
    x = np.linspace(0, L_nm, 100)
    psi = np.sqrt(2/L_nm) * np.sin(n_quantum * np.pi * x / L_nm)
    prob_density = psi**2

    fig_wave, ax_wave = plt.subplots(figsize=(6, 3))
    fig_wave.patch.set_facecolor('#0E1117')
    ax_wave.set_facecolor('#0E1117')
    
    # Fill the area under the curve
    ax_wave.fill_between(x, prob_density, color='#00CCFF', alpha=0.5)
    ax_wave.plot(x, prob_density, color='white', linewidth=2)
    
    # Draw the "Walls" of the box
    ax_wave.vlines(0, 0, max(prob_density)*1.1, color='red', linewidth=3, linestyle='-')
    ax_wave.vlines(L_nm, 0, max(prob_density)*1.1, color='red', linewidth=3, linestyle='-')
    
    ax_wave.set_title(f"Probability Density |ψ|² for n={n_quantum}", color='white')
    ax_wave.set_xlabel(f"Position x (0 to {L_nm} nm)", color='white')
    ax_wave.set_ylabel("Probability", color='white')
    ax_wave.set_ylim(0, max(prob_density)*1.2)
    ax_wave.tick_params(colors='white')
    
    st.pyplot(fig_wave)
    plt.close(fig_wave)