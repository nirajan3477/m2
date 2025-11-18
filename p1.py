import streamlit as st
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import colorsys
import io

# --- 1. Constants and Physics Definitions ---
h = 6.626e-34      # Planck's constant (J*s)
m_e = 9.11e-31     # Mass of an electron (kg)
eV = 1.602e-19     # Joules per electron-volt (J/eV)
hc_eV_nm = 1240    # h*c in (eV * nm), a useful shortcut

MATERIALS = {
    "Gallium Arsenide (GaAs)": 0.067 * m_e,
    "Cadmium Selenide (CdSe)": 0.13 * m_e,
    "Silicon (Si)": 0.19 * m_e
}

# --- Image Paths ---
BULK_MATERIAL_IMAGES = {
    "Gallium Arsenide (GaAs)": "images/gaas.jpg", 
    "Cadmium Selenide (CdSe)": "images/cdse.jpg",
    "Silicon (Si)": "images/si.jpg" 
}
DEFAULT_BULK_COLOR_HEX = "#A0A0A0" 

# --- 2. Physics Calculation Functions ---
def calculate_energy_levels(L_nm, m_star, dimensionality):
    """
    Calculates energy levels based on the *degrees of freedom*.
    """
    L_m = L_nm * 1e-9  # Convert size from nm to meters
    try:
        constant_factor_J = (h**2) / (8 * m_star * L_m**2)
    except ZeroDivisionError:
        return [], 0
        
    constant_factor_eV = constant_factor_J / eV
    energies = []
    
    if dimensionality == "Quantum Wire (1D Freedom)":
        for n in range(1, 11):
            E = (n**2) * constant_factor_eV
            energies.append(((n,), E))
            
    elif dimensionality == "Quantum Well (2D Freedom)":
        for nx in range(1, 7):
            for ny in range(1, 7):
                E = (nx**2 + ny**2) * constant_factor_eV
                energies.append(((nx, ny), E))
                
    elif dimensionality == "Quantum Dot (0D Freedom)":
        for nx in range(1, 5):
            for ny in range(1, 5):
                for nz in range(1, 5):
                    E = (nx**2 + ny**2 + nz**2) * constant_factor_eV
                    energies.append(((nx, ny, nz), E))
    
    elif dimensionality == "Bulk (3D Freedom)":
        return [], 0 # No discrete levels, energy is a continuum

    energies.sort(key=lambda x: x[1])

    unique_levels = []
    seen_energies = set()
    for q_numbers, E in energies:
        if round(E, 8) not in seen_energies: # Round to avoid float precision issues
            unique_levels.append((q_numbers, E))
            seen_energies.add(round(E, 8))
        if len(unique_levels) >= 6:
            break
            
    band_gap_eV = unique_levels[0][1] if unique_levels else 0
    return unique_levels, band_gap_eV

# --- 3. Visualization Functions ---

def wavelength_to_rgb(wavelength_nm, gamma=0.8):
    """
    Converts a wavelength in nm to an RGB tuple (0-255).
    This approximates the color of *emitted* light.
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

def rgb_to_hex(rgb_tuple):
    """Converts an RGB tuple (0-255) to a hex string."""
    return '#%02x%02x%02x' % rgb_tuple

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
    """
    Selects the correct plot to show:
    - 3D Bulk: Density of States (DOS) curve
    - 2D Well: Staircase DOS plot
    - 1D Wire: 1/sqrt(E) DOS spikes
    - 0D Dot: Discrete energy level diagram (delta functions)
    """
    fig, ax = plt.subplots(figsize=(10, 6))
    ax.set_facecolor('#0E1117')
    fig.set_facecolor('#0E1117')
    ax.set_ylabel("Energy (eV)", fontsize=14, color='white')
    ax.tick_params(axis='y', colors='white')
    ax.spines['bottom'].set_color('white')
    ax.spines['left'].set_color('white')
    ax.tick_params(axis='x', colors='white') # Also set x-axis tick color
    
    # 3D (Bulk) - Plot continuous DOS
    if dim_choice == "Bulk (3D Freedom)":
        E = np.linspace(0.001, 1, 100) # Energy axis
        dos = np.sqrt(E) # g(E) is proportional to sqrt(E)
        ax.plot(dos, E, color='blue', linewidth=3)
        ax.set_xlabel("Density of States g(E)", fontsize=14, color='white')
        ax.set_title("Continuous Density of States (Bulk)", fontsize=16, color='white')
        ax.set_yticks(np.linspace(0, 1, 5))
        ax.set_xticks([])
        ax.fill_between(dos, E, color='blue', alpha=0.3)
        
    # 2D (Quantum Well) - Plot staircase DOS
    elif dim_choice == "Quantum Well (2D Freedom)":
        ax.set_xlabel("Density of States g(E)", fontsize=14, color='white')
        ax.set_title(f"Staircase Density of States (Quantum Well, L={L_nm:.1f}nm)", fontsize=16, color='white')
        
        E_levels = [E for q_num, E in energy_levels]
        if not E_levels:
            E_levels = [0]
            
        plot_limit = E_levels[-1] * 1.2
        
        # Create the steps
        energies = [0] + E_levels + [plot_limit]
        dos_steps = [0] # Start with zero density
        for i in range(1, len(energies)):
            dos_steps.append(i-1) # Each level adds a constant amount to DOS
        
        ax.step(dos_steps, energies, where='post', color='blue', linewidth=3)
        ax.set_ylim(0, plot_limit)
        ax.set_xlim(0, len(E_levels) * 1.1)
        ax.set_xticks([])
        
    # 1D (Wire) - Plot 1/sqrt(E) DOS
    elif dim_choice == "Quantum Wire (1D Freedom)":
        ax.set_xlabel("Density of States g(E)", fontsize=14, color='white')
        ax.set_title(f"1D Density of States (Quantum Wire, L={L_nm:.1f}nm)", fontsize=16, color='white')
        
        E_levels = [E for q_num, E in energy_levels]
        if not E_levels: E_levels = [0.1] # placeholder
        
        plot_limit = E_levels[-1] * 1.2
        ax.set_ylim(0, plot_limit)
        ax.set_xlim(0, 10) # Arbitrary DOS axis
        ax.set_xticks([])
        
        # Calculate the energy difference between first two levels to set a scale
        if len(E_levels) > 1:
            e_scale = (E_levels[1] - E_levels[0]) * 0.8
        else:
            e_scale = plot_limit / 5
        
        # Create the 1/sqrt(E) shape
        e_axis_rel = np.linspace(0.01, e_scale, 100) # Relative energy
        dos_shape = 1 / np.sqrt(e_axis_rel)
        dos_shape = (dos_shape / np.max(dos_shape)) * 8 # Scale it for the plot
        
        # Plot the shape at each energy level
        for E_n in E_levels:
            if E_n < plot_limit:
                ax.plot(dos_shape, E_n + e_axis_rel, color='blue', linewidth=3)
                ax.hlines(E_n, 0, dos_shape[0], color='blue', linestyle='--', linewidth=2)
                ax.text(dos_shape[0] + 0.2, E_n, f'n={energy_levels[E_levels.index(E_n)][0]}', va='center', color='white', fontsize=10)
        
    # 0D (Dot) - Plot discrete levels (delta functions)
    elif dim_choice == "Quantum Dot (0D Freedom)": 
        ax.set_title(f"0D Density of States (Quantum Dot, L={L_nm:.1f}nm)", fontsize=16, color='white')
        ax.set_xticks([])
        ax.set_xlim(0, 1.4)
        ax.set_xlabel("Discrete Energy Levels", fontsize=14, color='white')
        
        if energy_levels:
            max_E = energy_levels[-1][1]
            for q_numbers, E in energy_levels:
                q_str = f"n={q_numbers}"
                # This line represents a delta function at energy E
                ax.hlines(y=E, xmin=0, xmax=1, color='blue', linewidth=3)
                ax.text(1.02, E, f' {q_str} ({E:.3f} eV)', va='center', fontsize=12, color='white')
            ax.set_ylim(0, max_E * 1.2)
        else:
            ax.set_ylim(0, 1)
            ax.text(0.5, 0.5, "No levels found (E=0)", ha='center', color='white')
        ax.grid(axis='y', linestyle='--', alpha=0.7)

    return fig

# --- 4. Streamlit App Layout ---

st.set_page_config(layout="wide")
st.title("🏆 Interactive Quantum Confinement Simulator")
st.write("A project for **Nanoscience and Technology (17B1NPH732)** demonstrating how dimensionality and size impact a nanomaterial's electronic properties.")

# --- Sidebar (User Controls) ---
st.sidebar.header("Simulation Controls")

dim_choice = st.sidebar.radio(
    "Select Dimensionality (Degrees of Freedom)",
    ["Bulk (3D Freedom)", 
     "Quantum Well (2D Freedom)", 
     "Quantum Wire (1D Freedom)", 
     "Quantum Dot (0D Freedom)"],
    index=3  # Default to Quantum Dot
)

mat_choice = st.sidebar.selectbox(
    "Select Material (determines $m^*$)",
    list(MATERIALS.keys())
)

L_MIN = 1.0
L_MAX = 25.0

L_nm = st.sidebar.slider(
    "Nanomaterial Size (L) in nm",
    min_value=L_MIN,
    max_value=L_MAX,
    value=7.0,  # Default size
    step=0.1
)
st.sidebar.markdown("---")
st.sidebar.markdown("""
### **How It Works:**
1.  **Select Material:** Watch the "Bulk" image change.
2.  **Select Dimensionality:** Watch the "Electronic Structure" plot change its fundamental shape (curve, staircase, spikes, or lines).
3.  **Adjust Size & Material:** See how the **color** and **energy levels** change in real-time.
""")

# --- Run Calculations (Done once at the top) ---
m_star_val = MATERIALS[mat_choice]
energy_levels, band_gap = calculate_energy_levels(L_nm, m_star_val, dim_choice)

if band_gap > 0:
    wavelength_nm = hc_eV_nm / band_gap
    emitted_rgb = wavelength_to_rgb(wavelength_nm)
    emitted_hex = rgb_to_hex(emitted_rgb)
    if wavelength_nm < 380:
        color_name = "Ultraviolet (UV)"
    elif wavelength_nm > 780:
        color_name = "Infrared (IR)"
    else:
        color_name = "Visible"
else:
    # Special color for Bulk
    wavelength_nm = float('inf')
    emitted_rgb = (100, 100, 100) # Gray
    emitted_hex = "#646464"
    color_name = "N/A (Continuum)"

# --- Step-by-Step Dynamic Visualization (The "Why") ---
st.markdown("---")
st.header("🔬 Visualizing the Confinement Process (The 'Why')")
st.write(f"This shows the step-by-step reduction in dimensionality, starting from the real bulk material. The nanostructures adapt their **color** ({emitted_hex}) based on your simulation.")

col1, col2, col3, col4 = st.columns(4)
with col1:
    st.subheader("Step 1: Bulk")
    fig = create_confinement_plot("Bulk (3D Freedom)", 
                                  emitted_hex, 
                                  dim_choice == "Bulk (3D Freedom)", 
                                  material_name=mat_choice)
    st.pyplot(fig)
    st.markdown("*(3D Freedom, 0D Confinement)*")

with col2:
    st.subheader("Step 2: Quantum Well")
    fig = create_confinement_plot("Quantum Well (2D Freedom)", 
                                  emitted_hex, 
                                  dim_choice == "Quantum Well (2D Freedom)",
                                  material_name=mat_choice)
    st.pyplot(fig)
    st.markdown("*(2D Freedom, 1D Confinement)*")

with col3:
    st.subheader("Step 3: Quantum Wire")
    fig = create_confinement_plot("Quantum Wire (1D Freedom)", 
                                  emitted_hex, 
                                  dim_choice == "Quantum Wire (1D Freedom)",
                                  material_name=mat_choice)
    st.pyplot(fig)
    st.markdown("*(1D Freedom, 2D Confinement)*")

with col4:
    st.subheader("Step 4: Quantum Dot")
    fig = create_confinement_plot("Quantum Dot (0D Freedom)", 
                                  emitted_hex, 
                                  dim_choice == "Quantum Dot (0D Freedom)",
                                  material_name=mat_choice)
    st.pyplot(fig)
    st.markdown("*(0D Freedom, 3D Confinement)*")

st.markdown("---")

# --- Dynamic Simulation Results (The "Effect") ---
st.header(f"⚡ Simulation Results for: {dim_choice}")

# --- 1. PRE-GENERATE BOTH PLOTS ---
min_radius = 0.1
max_radius = 0.4
radius = min_radius + ((L_nm - L_MIN) / (L_MAX - L_MIN)) * (max_radius - min_radius)

fig_color, ax_color = plt.subplots(figsize=(4, 4))
ax_color.set_facecolor('#0E1117')
fig_color.set_facecolor('#0E1117')

if dim_choice == "Bulk (3D Freedom)":
    circle_color = plt.Circle((0.5, 0.5), 0.4, color=emitted_hex, ec='white')
    ax_color.set_title("Bulk Material Color", color='white')
else:
    circle_color = plt.Circle((0.5, 0.5), radius, color=emitted_hex, ec='white')
    ax_color.set_title(f"Simulated Emitted Color (L={L_nm:.1f}nm)", color='white')
    
ax_color.add_patch(circle_color)
ax_color.set_aspect('equal', adjustable='box')
ax_color.set_xticks([]); ax_color.set_yticks([])
ax_color.set_xlim(0, 1); ax_color.set_ylim(0, 1)

# --- Generate Electronic Structure Plot (fig_dos) ---
fig_dos = plot_electronic_structure(dim_choice, energy_levels, L_nm)


# --- 2. DRAW COLUMNS AND PLOTS ---
res_col1, res_col2 = st.columns([1, 1.5]) 

with res_col1:
    st.subheader("Size-Dependent Emitted Color")
    st.pyplot(fig_color)
    
    st.write(f"Simulating a **{mat_choice.split(' (')[0]}** particle:")
    
    if dim_choice == "Bulk (3D Freedom)":
        st.info("Bulk materials do not have size-dependent quantum effects. Their energy levels form a continuous band.")
    else:
        st.metric(label="Calculated Band Gap (E_g)", value=f"{band_gap:.3f} eV")
        st.metric(label=f"Emission Wavelength (λ)", value=f"{wavelength_nm:.1f} nm ({color_name})")

with res_col2:
    st.subheader("Electronic Structure (CO402.2)")
    st.pyplot(fig_dos)
    
    # --- *** CORRECTED TEXT SECTION *** ---
    if dim_choice == "Bulk (3D Freedom)":
        st.write(r"For a **Bulk** material, the Density of States (DOS) is a **continuous curve** ($g(E) \propto \sqrt{E}$). This means there are available energy states at nearly every energy level, forming a 'band'.")
    elif dim_choice == "Quantum Well (2D Freedom)":
        st.write(r"For a **Quantum Well**, the DOS is a **staircase**. Confinement in one dimension creates discrete 'sub-bands'. The DOS is zero until the first energy level, then jumps.")
    elif dim_choice == "Quantum Wire (1D Freedom)":
        st.write(r"For a **Quantum Wire**, the DOS shows **spikes** ($g(E) \propto 1/\sqrt{E-E_n}$) at the start of each sub-band (at $E_n$, shown as dashed lines).")
    else: # This is now just for Quantum Dot
        st.write(r"For a **Quantum Dot**, the DOS is a series of **delta functions** (infinitely sharp spikes) at the discrete energy levels. This plot shows those discrete levels.")