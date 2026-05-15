import numpy as np
import matplotlib.pyplot as plt

# Parameters
R = 5          # base radius
A = 1.5          # amplitude of sinusoidal ripple
k = 6          # number of waves
N = 5000       # resolution

# Angle array
theta = np.linspace(0, 2*np.pi, N)

# Radial function
r = R + A * np.sin(k * theta)

# Convert to Cartesian
x = r * np.cos(theta)
y = r * np.sin(theta)

# ---- AREA ----
# Polar area formula: A = 1/2 ∫ r(θ)^2 dθ
area = 0.5 * np.trapezoid(r**2, theta)

# ---- PERIMETER ----
# Perimeter = ∫ sqrt( r(θ)^2 + (dr/dθ)^2 ) dθ
# Use numpy trapezoidal integration for robustness.
dr_dtheta = np.gradient(r, theta, edge_order=2)
integrand = np.sqrt(r**2 + dr_dtheta**2)
perimeter = np.trapezoid(integrand, theta)

print(f"Area ≈ {area:.4f}")
print(f"Perimeter ≈ {perimeter:.4f}")

eqR = np.sqrt(area / np.pi)  # Equivalent radius from area
print(f"Equivalent radius from area: {eqR:.4f}")

xec = eqR * np.cos(theta)  # Circle x-coordinates
yec = eqR * np.sin(theta)  # Circle y-coordinates

eqRp = perimeter / (2 * np.pi)  # Equivalent radius from perimeter
print(f"Equivalent radius from perimeter: {eqRp:.4f}") 

xecp = eqRp * np.cos(theta)  # Circle x-coordinates from perimeter
yecp = eqRp * np.sin(theta)  # Circle y-coordinates from perimeter

effArea=perimeter*np.sqrt(area/np.pi)/2
print(f"Effective area from perimeter: {effArea:.4f}")
effRadius=np.sqrt(effArea/np.pi)
print(f"Effective radius from perimeter: {effRadius:.4f}")

# ---- PLOT ----
plt.figure(figsize=(6,6))
plt.plot(x, y, linewidth=2,label='Sinusoidally perturbed circle')
plt.plot(xec, yec, linewidth=2, linestyle='--', label='Equivalent circle from area')
plt.plot(xecp, yecp, linewidth=2, linestyle='-.', label='Equivalent circle from perimeter')
plt.plot(effRadius*np.cos(theta), effRadius*np.sin(theta), linewidth=2, linestyle=':', label='Effective circle from shape tortuosity')
plt.gca().set_aspect('equal', adjustable='box')
plt.axis('off')
#plt.title("Sinusoidally Perturbed Circle")
plt.xlabel("x")
plt.ylabel("y")
plt.grid(False)
plt.tight_layout()
plt.legend(loc='center left', bbox_to_anchor=(1, 0.5))
#plt.show()
plt.savefig("wobblyCircle.png", dpi=300, bbox_inches='tight')

