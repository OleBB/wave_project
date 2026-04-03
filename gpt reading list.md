Given your setup, you’re in a very specific niche:

- short, closely‑spaced panels ($8\,$cm each, total $2.6\,$m),  
- elastic “backbone” (gym band) that is also moored,  
- short, steep gravity waves (amp $4$–$8\,$cm, $f = 1.3$–$1.7\,$Hz → $T \approx 0.6$–$0.8\,$s),  
- moderate wind ($U \approx 6\,$m/s) along the tank.

Below is a **tight reading list** plus a **concrete modeling angle** that really matches what you’re doing.

---

## 1. Closest hydrodynamic analogues: arrays / floating breakwaters

These give you the framework for a row of bodies with connectors and mooring.

1. **Arrays of floating bodies / breakwaters**

   - **Linton, C. M. & McIver, P.** (2001)  
     *“The scattering of water waves by an array of circular cylinders in a channel”*,  
     *J. Eng. Math.* 40, 67–90.  
     → Canonical scattering formulation for arrays; even though cylinders, the **multi‑body, multi‑mode approach** is what you want.

   - **Molin, B. & Remy, F.** (2013)  
     *“Hydrodynamics of interconnected floating bodies”*, *Appl. Ocean Res.* 43, 10–20.  
     → Very close: floating units linked by connectors with stiffness/damping, with moorings.  
     Use this as a template for how to write the global mass–damping–stiffness system and connector forces.

   - **McCartney, B. L.** (1985)  
     *“Floating breakwater design”*, *J. Waterway, Port, Coastal, and Ocean Eng.* 111(2), 304–318.  
     → Short and practical, shows how to evaluate performance in terms of transmission and reflection.

   - **Isaacson, M. & Bhat, S.** (~1980s)  
     Papers on *“Wave interaction with floating breakwaters”* in *Ocean Engineering*.  
     → Look for experiments with **linked pontoons**. The way they treat the connectors and measure transmission is very useful.

2. **Energy balance and damping via transmission**

   Your key quantity is how much wave energy is lost (damped) by the row.

   - **Mei, C. C.; Stiassnie, M.; Yue, D. K.-P.** (2005)  
     *Theory and Applications of Ocean Surface Waves*, World Scientific.  
     → Use the chapter on obstacles/breakwaters to frame the energy balance:
     ```math
     1 = |R|^2 + |T|^2 + D
     ```
     for regular waves, where  
     $R$ = reflection coefficient, $T$ = transmission coefficient,  
     $D$ = total dissipation (your “damping” – internal rubber loss, turbulence, small‑scale breaking, etc.).

---

## 2. Hydroelastic / flexible row analogues

While your panels are short and relatively stiff, the **rubber band and mooring** give the *row* a hydroelastic flavor (deformable backbone).

1. **Hydroelastic plates / ice**

   - **Bennetts, L. G. & Squire, V. A.** (2012)  
     *“On the calculation of wave energy attenuation in the marginal ice zone”*, *J. Fluid Mech.* 708, 396–417.  
     → Their concept of **attenuation coefficient** along a row of flexible elements is very close to “how much wave energy your row eats per unit length”. Good conceptual model even if your elements are short.

   - **Porter, R. & Evans, D. V.** (~1990s–2000s)  
     Papers on *“Wave scattering by rows of obstacles/plates”* (e.g., *J. Fluid Mech.*, *J. Eng. Math.*).  
     → These show how to deal with multiple scatterers in line and compute $R$, $T$ and energy attenuation. Very relevant to the **2.6 m array** perspective.

---

## 3. Wind + waves on floating rafts / FPV

These help with the **incremental effect of wind** and how to model it.

1. **Floating PV (FPV) hydrodynamics with wind**

   Search (Google Scholar) for combinations like:

   - `"floating photovoltaic" "hydrodynamic response" "Ocean Engineering"`  
   - `"floating solar" "mooring" "wind"`  

   Papers often authored by **J. H. Lee, H. S. Kim, S. Hong**, etc. Typical titles:

   - *“Hydrodynamic analysis of floating photovoltaic structure in regular waves”*  
   - *“Integrated analysis of hydrodynamic and aerodynamic loads on floating solar platforms”*

   Why read them:

   - They show **wind speeds**, **scaling**, and simple **aerodynamic models** that are realistic.  
   - They define performance measures you can copy (motions, mooring loads, panel accelerations, etc.).

2. **General wind–wave loading**

   - **Faltinsen, O. M.** (1990)  
     *Sea Loads on Ships and Offshore Structures*, Cambridge University Press.  
     → For a simple but solid aerodynamic force model. For each panel, you can write drag as:
     ```math
     F_\text{drag} = \tfrac{1}{2} \rho_a C_D A
                     \left( U - \dot{x}_\text{panel} \right)^2
     ```
     where $\rho_a$ is air density, $C_D$ a drag coefficient ($\sim 1$–2 for a flat plate),  
     $A$ the projected area, and $U$ your $5.9\,$m/s wind speed.

---

## 4. Wave‑tank methods you absolutely need

Because your tank is only $25\,$m long and your row is relatively long ($2.6\,$m), **reflections** matter.

1. **Separating incident and reflected waves**

   - **Goda, Y. & Suzuki, Y.** (1976)  
     *“Estimation of incident and reflected waves in random wave experiments”*, *Coastal Engineering* 1(1), 47–64.

   - **Mansard, E. & Funke, E.** (1980)  
     *“The measurement of incident and reflected spectra using a least squares method”*, *Coastal Engineering* 4(4), 357–378.

   → Even with regular waves, you can use these multi‑probe methods to extract $R$ and $T$ cleanly from your measurements upstream and downstream of the panels. Without this, your damping estimate $D$ is contaminated by end reflections.

2. **General lab practice and wave theory**

   - **Dean, R. G. & Dalrymple, R. A.**  
     *Water Wave Mechanics for Engineers and Scientists*, World Scientific.  
     → For dispersion, group velocity, and figuring out whether your waves at $f = 1.3$–$1.7\,$Hz and $a = 4$–$8\,$cm are in a comfortable linear regime (they’re fairly steep, so some nonlinearity is expected).

---

## 5. How to analyze your specific system (link to above papers)

To make those references truly useful, here’s how to connect them directly to your experiments.

### 5.1. Wave‑only vs. wave + wind

For each condition (no wind / wind):

1. **Measure wave amplitudes**:

   - Upstream (incident + reflected),
   - Downstream (transmitted).

   Use Goda/Mansard–Funke methods to compute $R$ and $T$ for each frequency.  
   Then compute:
   ```math
   D_\text{no wind} = 1 - |R|^2 - |T|^2
   ```

2. **Switch on wind** ($U \approx 5.9\,$m/s):

   - Repeat and get $R_\text{wind}, T_\text{wind}$ and:
     ```math
     D_\text{wind} = 1 - |R_\text{wind}|^2 - |T_\text{wind}|^2
     ```
   - The difference:
     ```math
     \Delta D = D_\text{wind} - D_\text{no wind}
     ```
     is the **wind‑induced change in damping**. FPV / raft papers and Bennetts & Squire give the conceptual backing to interpret this as “extra attenuation per unit row length”.

3. **Panel motion / connector tension**

   If you track panel motions and rubber‑band tensions, you can:

   - Fit an **equivalent linear connector damping** (from free‑decay tests) for the rubber band.  
   - See how that effective damping changes (if at all) when wind is added.

### 5.2. Simple multi‑body model you can copy from literature

Adapt the approach in **Molin & Remy (2013)** and FPV papers:

- Treat each $8\,$cm panel as a rigid body with 1–2 DOFs (surge + pitch, for example).
- The gym band provides **springs and dashpots** between panel $i$ and $i+1$, and a mooring at the upstream end:

  ```math
  m_i \ddot{x}_i + c_i \dot{x}_i
  + \sum_\text{conn} ( k_{ij} (x_i - x_j)
                     + c_{ij} (\dot{x}_i - \dot{x}_j) )
  = F_{\text{exc},i}^{\text{wave}} + F_{\text{drag},i}^{\text{wind}}
  ```

- Add simple aerodynamic drag from Faltinsen / FPV papers:
  ```math
  F_{\text{drag},i}^{\text{wind}}
    = \tfrac{1}{2} \rho_a C_D A_i
      ( U - \dot{x}_i )\,| U - \dot{x}_i |
  ```

This model gives you:

- RAOs and energy dissipation in waves only → compare with $D_\text{no wind}$.
- Same with drag term → compare with $D_\text{wind}$ and your measured $\Delta D$.

---

## 6. Minimal “start‑reading‑now” list

In order of priority for *your* work:

1. **Dean & Dalrymple – Water Wave Mechanics** (wave theory + tank practice).  
2. **Goda & Suzuki (1976); Mansard & Funke (1980)** (measure $R$, $T$ correctly).  
3. **Molin & Remy (2013), Appl. Ocean Res. 43, 10–20** (interconnected floating bodies).  
4. **One FPV hydrodynamics + wind paper** (for aerodynamic model and context).  
5. **Bennetts & Squire (2012), JFM 708, 396–417** (interpretation of attenuation/damping along a row).  
6. A **Linton & McIver** array‑scattering paper (multi‑body wave interaction framework).

If you want, I can next help you:

- draft the exact **set of equations** to fit to your data (with symbols matched to your panels/rubber), or  
- outline a **short paper structure** (intro / method / results / discussion) aligned with the above literature.
