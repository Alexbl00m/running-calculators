import streamlit as st
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import math
from scipy.optimize import curve_fit

# Set page configuration
st.set_page_config(
    page_title="Critical Power Calculator",
    page_icon="🚴‍♂️",
    layout="wide"
)

# Apply custom CSS
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Montserrat:wght@300;400;500;600;700&display=swap');

html, body, [class*="css"] {
    font-family: 'Montserrat', sans-serif;
}

.main {
    background-color: #FFFFFF;
}

h1, h2, h3, h4, h5, h6 {
    font-family: 'Montserrat', sans-serif;
    font-weight: 600;
    color: #E6754E;
}

.stButton>button {
    background-color: #E6754E;
    color: white;
    font-family: 'Montserrat', sans-serif;
    border: none;
    border-radius: 4px;
    padding: 8px 16px;
}

.stButton>button:hover {
    background-color: #c45d3a;
}

.highlight {
    color: #E6754E;
    font-weight: 600;
}

.result-box {
    background-color: #f8f8f8;
    padding: 20px;
    border-radius: 10px;
    border-left: 5px solid #E6754E;
}

footer {
    font-family: 'Montserrat', sans-serif;
    font-size: 12px;
    color: #888888;
    text-align: center;
    margin-top: 50px;
}
</style>
""", unsafe_allow_html=True)

# Header
st.title("Critical Power Calculator for Cyclists")
st.markdown('<p class="highlight">Science-based cycling performance metrics</p>', unsafe_allow_html=True)

# Sidebar
with st.sidebar:
    st.image("https://via.placeholder.com/150x150.png?text=Your+Logo", width=150)
    st.markdown("## Test Protocols")
    test_method = st.selectbox(
        "Select Testing Method",
        ["3-min All Out Test", "Multiple Time Trial", "Ramp Test", "Time to Exhaustion Test"]
    )
    
    st.markdown("## Rider Information")
    weight = st.number_input("Weight (kg)", min_value=30.0, max_value=150.0, value=70.0, step=0.1)
    
    experience_level = st.radio(
        "Experience Level",
        ["Beginner", "Intermediate", "Advanced", "Elite"]
    )
    
    with st.expander("Advanced Settings"):
        show_formulas = st.checkbox("Show calculation formulas", value=False)
        show_references = st.checkbox("Show scientific references", value=True)

# Helper functions
def calculate_cp_from_3min(max_power, end_power):
    # Based on Vanhatalo et al. (2007)
    cp = end_power
    w_prime = (max_power - end_power) * 180  # 3 minutes in seconds
    return cp, w_prime

def calculate_cp_from_time_trials(powers, durations):
    # 2-parameter critical power model (Monod & Scherrer, 1965)
    def power_model(t, cp, w_prime):
        return cp + (w_prime / t)
    
    popt, _ = curve_fit(power_model, durations, powers)
    cp, w_prime = popt
    return cp, w_prime

def calculate_cp_from_ramp(final_power, time_to_exhaustion, ramp_rate):
    # Based on Poole et al. (2016)
    cp = final_power - (0.5 * ramp_rate * time_to_exhaustion / 60)
    w_prime = 0.5 * ramp_rate * (time_to_exhaustion / 60) ** 2
    return cp, w_prime

def calculate_cp_from_tte(power, time_to_exhaustion):
    # Simplified calculation based on time to exhaustion
    # This is a placeholder - actual implementation would require multiple tests
    if len(power) < 2 or len(time_to_exhaustion) < 2:
        return 0, 0
        
    def power_duration_model(t, cp, w_prime):
        return cp + w_prime / t
    
    popt, _ = curve_fit(power_duration_model, time_to_exhaustion, power)
    cp, w_prime = popt
    return cp, w_prime

def calculate_ftp_from_cp(cp):
    # Typically FTP ≈ 95% of CP (Jeffries et al., 2019)
    return cp * 0.95

def calculate_training_zones(ftp):
    # Traditional 7-zone model
    zones = {
        "Active Recovery": (0, 0.55 * ftp),
        "Endurance": (0.55 * ftp, 0.75 * ftp),
        "Tempo": (0.75 * ftp, 0.90 * ftp),
        "Threshold": (0.90 * ftp, 1.05 * ftp),
        "VO2 Max": (1.05 * ftp, 1.20 * ftp),
        "Anaerobic Capacity": (1.20 * ftp, 1.50 * ftp),
        "Neuromuscular Power": (1.50 * ftp, float('inf'))
    }
    return zones

def w_prime_balance(power_data, cp, w_prime, tau=546):
    """
    Calculate W' balance over a ride
    Based on the W'bal ODE Model by Skiba et al. (2015)
    
    Parameters:
    power_data: array of power data
    cp: critical power
    w_prime: W' (anaerobic work capacity)
    tau: time constant for W' reconstitution
    
    Returns:
    Array of W' balance values
    """
    w_prime_balance = np.ones(len(power_data)) * w_prime
    for i in range(1, len(power_data)):
        expenditure = max(0, (power_data[i-1] - cp)) # W' expenditure
        recovery = min(0, (power_data[i-1] - cp))    # W' recovery
        
        # Apply Skiba's differential equation
        if expenditure > 0:
            w_prime_balance[i] = w_prime_balance[i-1] - expenditure
        else:
            # W' reconstitution
            w_prime_balance[i] = w_prime - (w_prime - w_prime_balance[i-1]) * np.exp(recovery / (tau * cp))
            
        # Ensure W' balance doesn't exceed W'
        w_prime_balance[i] = min(w_prime, max(0, w_prime_balance[i]))
        
    return w_prime_balance

# Main content based on selected method
if test_method == "3-min All Out Test":
    st.markdown("## 3-Minute All-Out Test")
    
    st.markdown("""
    The 3-minute all-out test is a simplified protocol to determine Critical Power (CP) and W′ (W-prime).
    
    ### Protocol:
    1. Perform a thorough warm-up
    2. Start the test with maximal effort from the beginning
    3. Continue with maximum possible effort for the entire 3 minutes
    4. Record your maximum power and average power for the last 30 seconds
    """)
    
    col1, col2 = st.columns(2)
    
    with col1:
        max_power = st.number_input("Maximum Power (Watts)", min_value=100, max_value=2000, value=700)
        end_power = st.number_input("Average Power in Last 30s (Watts)", min_value=100, max_value=2000, value=300)
        
        if st.button("Calculate"):
            cp, w_prime = calculate_cp_from_3min(max_power, end_power)
            ftp = calculate_ftp_from_cp(cp)
            
            with col2:
                st.markdown('<div class="result-box">', unsafe_allow_html=True)
                st.markdown(f"### Results")
                st.markdown(f"**Critical Power (CP):** {cp:.1f} Watts")
                st.markdown(f"**CP/kg:** {cp/weight:.2f} W/kg")
                st.markdown(f"**W′ (W-prime):** {w_prime:.0f} Joules")
                st.markdown(f"**FTP (estimated):** {ftp:.1f} Watts")
                
                # Training zones
                zones = calculate_training_zones(ftp)
                st.markdown("### Training Zones")
                for zone, (lower, upper) in zones.items():
                    st.markdown(f"**{zone}:** {lower:.0f} - {upper:.0f} Watts")
                st.markdown('</div>', unsafe_allow_html=True)

# Add explanation of results
st.markdown("---")
st.markdown("## Interpreting Your Results")

st.markdown("""
### Critical Power (CP)
Critical Power represents the highest intensity you can sustain for a very long time (theoretically 30-60 minutes) without continual fatigue accumulation. It's closely related to your lactate threshold and is a fundamental metric for endurance performance.

### W′ (W-prime)
W′ represents your finite anaerobic work capacity - the amount of work you can perform above your Critical Power before exhaustion. Think of it as your "battery" of high-intensity energy.

### How to use these metrics:
1. **Pacing:** In longer events, stay at or slightly below your CP to avoid depleting W′
2. **Intervals:** Design interval sessions that target CP (to improve it) or W′ (to expand it)
3. **Race strategy:** For time trials, manage your W′ expenditure carefully - save some for hills and finish
4. **Training zones:** Use CP to set precise, physiologically meaningful training zones

### What makes a good CP and W′?
| Level | CP/kg (W/kg) | W′/kg (J/kg) |
|-------|-------------|--------------|
| Recreational | 2.5-3.5 | 10-15 |
| Competitive | 3.5-4.5 | 15-25 |
| Elite | 4.5-5.5 | 20-30 |
| World Class | 5.5+ | 25-35 |

*Note: These values vary based on gender, age, and specialization.*
""")

# Add a footer
st.markdown("---")
st.markdown('<footer>Critical Power Calculator for Cyclists © 2025</footer>', unsafe_allow_html=True)
    
    if show_formulas:
        st.markdown("### Calculation Method")
        st.markdown("""
        - Critical Power (CP) = Average power during the final 30 seconds
        - W′ = (Maximum power - CP) × 180 seconds
        """)
    
    if show_references:
        st.markdown("### Scientific References")
        st.markdown("""
        1. Vanhatalo, A., Doust, J. H., & Burnley, M. (2007). Determination of critical power using a 3-min all-out cycling test. Medicine and Science in Sports and Exercise, 39(3), 548-555.
        
        2. Burnley, M., Doust, J. H., & Vanhatalo, A. (2006). A 3-min all-out test to determine peak oxygen uptake and the maximal steady state. Medicine and Science in Sports and Exercise, 38(11), 1995-2003.
        """)

elif test_method == "Multiple Time Trial":
    st.markdown("## Multiple Time Trial Test")
    
    st.markdown("""
    This method requires performing 3-5 maximal effort time trials of different durations to plot the power-duration relationship.
    
    ### Protocol:
    1. Perform 3-5 time trials of different durations (typically 1-20 minutes)
    2. Record average power for each trial
    3. Rest at least 24 hours between trials
    4. Input the data below to calculate CP and W′
    """)
    
    num_trials = st.number_input("Number of Time Trials", min_value=2, max_value=5, value=3)
    
    cols = st.columns(num_trials)
    durations = []
    powers = []
    
    for i in range(num_trials):
        with cols[i]:
            st.markdown(f"### Trial {i+1}")
            duration = st.number_input(f"Duration (seconds)", min_value=60, max_value=1200, value=120 + i*180, key=f"dur_{i}")
            power = st.number_input(f"Average Power (watts)", min_value=100, max_value=2000, value=400 - i*50, key=f"pow_{i}")
            durations.append(duration)
            powers.append(power)
    
    if st.button("Calculate"):
        if num_trials >= 2:
            cp, w_prime = calculate_cp_from_time_trials(powers, durations)
            ftp = calculate_ftp_from_cp(cp)
            
            col1, col2 = st.columns([3, 2])
            
            with col1:
                st.markdown('<div class="result-box">', unsafe_allow_html=True)
                st.markdown(f"### Results")
                st.markdown(f"**Critical Power (CP):** {cp:.1f} Watts")
                st.markdown(f"**CP/kg:** {cp/weight:.2f} W/kg")
                st.markdown(f"**W′ (W-prime):** {w_prime:.0f} Joules")
                st.markdown(f"**FTP (estimated):** {ftp:.1f} Watts")
                
                # Training zones
                zones = calculate_training_zones(ftp)
                st.markdown("### Training Zones")
                for zone, (lower, upper) in zones.items():
                    st.markdown(f"**{zone}:** {lower:.0f} - {upper:.0f} Watts")
                st.markdown('</div>', unsafe_allow_html=True)
            
            with col2:
                # Plot power-duration curve
                fig, ax = plt.subplots(figsize=(10, 6))
                
                # Plot actual data points
                ax.scatter(durations, powers, color='#E6754E', s=100, label='Time Trials')
                
                # Generate points for the hyperbolic curve
                x_curve = np.linspace(min(durations) * 0.8, max(durations) * 1.2, 100)
                y_curve = [cp + (w_prime / x) for x in x_curve]
                
                # Plot the fitted curve
                ax.plot(x_curve, y_curve, 'b-', label='Power-Duration Curve')
                
                # Add horizontal line for CP
                ax.axhline(y=cp, color='g', linestyle='--', label=f'CP: {cp:.1f}W')
                
                ax.set_xlabel('Duration (seconds)')
                ax.set_ylabel('Power (watts)')
                ax.set_title('Power-Duration Relationship')
                ax.grid(True, alpha=0.3)
                ax.legend()
                
                st.pyplot(fig)
        else:
            st.error("At least 2 time trials are required for calculation")
    
    if show_formulas:
        st.markdown("### Calculation Method")
        st.markdown("""
        The 2-parameter critical power model is used:
        
        P = CP + W′/t
        
        Where:
        - P is the average power for a given duration
        - t is the duration in seconds
        - CP is critical power
        - W′ is the anaerobic work capacity
        
        The parameters are determined through hyperbolic curve fitting.
        """)
    
    if show_references:
        st.markdown("### Scientific References")
        st.markdown("""
        1. Monod, H., & Scherrer, J. (1965). The work capacity of a synergic muscular group. Ergonomics, 8(3), 329-338.
        
        2. Jones, A. M., Vanhatalo, A., Burnley, M., Morton, R. H., & Poole, D. C. (2010). Critical power: Implications for determination of VO2max and exercise tolerance. Medicine and Science in Sports and Exercise, 42(10), 1876-1890.
        
        3. Karsten, B., Jobson, S. A., Hopker, J., Jimenez, A., & Beedie, C. (2014). High agreement between laboratory and field estimates of critical power in cycling. International Journal of Sports Medicine, 35(4), 298-303.
        """)

elif test_method == "Ramp Test":
    st.markdown("## Ramp Test")
    
    st.markdown("""
    The ramp test is an incremental exercise test where power is progressively increased until exhaustion.
    
    ### Protocol:
    1. Start at a low power output
    2. Increase power at a constant rate (e.g., 25W/min)
    3. Continue until exhaustion
    4. Record final power and time to exhaustion
    """)
    
    col1, col2 = st.columns(2)
    
    with col1:
        starting_power = st.number_input("Starting Power (Watts)", min_value=50, max_value=200, value=100)
        ramp_rate = st.number_input("Ramp Rate (Watts/min)", min_value=10, max_value=50, value=25)
        time_to_exhaustion = st.number_input("Time to Exhaustion (seconds)", min_value=300, max_value=1800, value=720)
        
        if st.button("Calculate"):
            final_power = starting_power + (ramp_rate * time_to_exhaustion / 60)
            cp, w_prime = calculate_cp_from_ramp(final_power, time_to_exhaustion, ramp_rate)
            ftp = calculate_ftp_from_cp(cp)
            
            with col2:
                st.markdown('<div class="result-box">', unsafe_allow_html=True)
                st.markdown(f"### Results")
                st.markdown(f"**Final Power:** {final_power:.1f} Watts")
                st.markdown(f"**Critical Power (CP):** {cp:.1f} Watts")
                st.markdown(f"**CP/kg:** {cp/weight:.2f} W/kg")
                st.markdown(f"**W′ (W-prime):** {w_prime:.0f} Joules")
                st.markdown(f"**FTP (estimated):** {ftp:.1f} Watts")
                
                # Training zones
                zones = calculate_training_zones(ftp)
                st.markdown("### Training Zones")
                for zone, (lower, upper) in zones.items():
                    st.markdown(f"**{zone}:** {lower:.0f} - {upper:.0f} Watts")
                st.markdown('</div>', unsafe_allow_html=True)
    
    if show_formulas:
        st.markdown("### Calculation Method")
        st.markdown("""
        For a ramp test with constant increment rate:
        
        - Critical Power (CP) = Pmax - (0.5 × ramp rate × TTE)
        - W′ = 0.5 × ramp rate × TTE²
        
        Where:
        - Pmax is the final power at exhaustion
        - TTE is time to exhaustion in minutes
        - ramp rate is in W/min
        """)
    
    if show_references:
        st.markdown("### Scientific References")
        st.markdown("""
        1. Poole, D. C., Burnley, M., Vanhatalo, A., Rossiter, H. B., & Jones, A. M. (2016). Critical power: An important fatigue threshold in exercise physiology. Medicine and Science in Sports and Exercise, 48(11), 2320-2334.
        
        2. Vanhatalo, A., Jones, A. M., & Burnley, M. (2011). Application of critical power in sport. International Journal of Sports Physiology and Performance, 6(1), 128-136.
        
        3. Morton, R. H. (1994). Critical power test for ramp exercise. European Journal of Applied Physiology and Occupational Physiology, 69(5), 435-438.
        """)

elif test_method == "Time to Exhaustion Test":
    st.markdown("## Time to Exhaustion Test")
    
    st.markdown("""
    This method requires performing multiple constant-power tests until exhaustion at different intensities.
    
    ### Protocol:
    1. Perform tests at different constant power outputs
    2. Record time to exhaustion at each power
    3. Allow sufficient recovery between tests (24-48 hours)
    4. Input the data below to calculate CP and W′
    """)
    
    num_tests = st.number_input("Number of Tests", min_value=2, max_value=5, value=3)
    
    cols = st.columns(num_tests)
    powers = []
    tte_values = []
    
    for i in range(num_tests):
        with cols[i]:
            st.markdown(f"### Test {i+1}")
            power = st.number_input(f"Power (watts)", min_value=100, max_value=2000, value=300 + i*50, key=f"pow_tte_{i}")
            tte = st.number_input(f"Time to Exhaustion (seconds)", min_value=60, max_value=1200, value=600 - i*120, key=f"tte_{i}")
            powers.append(power)
            tte_values.append(tte)
    
    if st.button("Calculate"):
        if num_tests >= 2:
            cp, w_prime = calculate_cp_from_tte(powers, tte_values)
            ftp = calculate_ftp_from_cp(cp)
            
            col1, col2 = st.columns([3, 2])
            
            with col1:
                st.markdown('<div class="result-box">', unsafe_allow_html=True)
                st.markdown(f"### Results")
                st.markdown(f"**Critical Power (CP):** {cp:.1f} Watts")
                st.markdown(f"**CP/kg:** {cp/weight:.2f} W/kg")
                st.markdown(f"**W′ (W-prime):** {w_prime:.0f} Joules")
                st.markdown(f"**FTP (estimated):** {ftp:.1f} Watts")
                
                # Training zones
                zones = calculate_training_zones(ftp)
                st.markdown("### Training Zones")
                for zone, (lower, upper) in zones.items():
                    st.markdown(f"**{zone}:** {lower:.0f} - {upper:.0f} Watts")
                st.markdown('</div>', unsafe_allow_html=True)
            
            with col2:
                # Plot power-duration curve
                fig, ax = plt.subplots(figsize=(10, 6))
                
                # Plot actual data points
                ax.scatter(tte_values, powers, color='#E6754E', s=100, label='TTE Tests')
                
                # Generate points for the hyperbolic curve
                x_curve = np.linspace(min(tte_values) * 0.8, max(tte_values) * 1.2, 100)
                y_curve = [cp + (w_prime / x) for x in x_curve]
                
                # Plot the fitted curve
                ax.plot(x_curve, y_curve, 'b-', label='Power-Duration Curve')
                
                # Add horizontal line for CP
                ax.axhline(y=cp, color='g', linestyle='--', label=f'CP: {cp:.1f}W')
                
                ax.set_xlabel('Time to Exhaustion (seconds)')
                ax.set_ylabel('Power (watts)')
                ax.set_title('Power-Duration Relationship')
                ax.grid(True, alpha=0.3)
                ax.legend()
                
                st.pyplot(fig)
        else:
            st.error("At least 2 tests are required for calculation")
    
    if show_formulas:
        st.markdown("### Calculation Method")
        st.markdown("""
        The linear relationship between power and the inverse of time to exhaustion is used:
        
        P = CP + W′/t
        
        Where:
        - P is the constant power
        - t is the time to exhaustion in seconds
        - CP is critical power
        - W′ is the anaerobic work capacity
        
        The parameters are determined through linear regression when plotting power vs. 1/time.
        """)
    
    if show_references:
        st.markdown("### Scientific References")
        st.markdown("""
        1. Hill, D. W. (1993). The critical power concept. Sports Medicine, 16(4), 237-254.
        
        2. Poole, D. C., Ward, S. A., Gardner, G. W., & Whipp, B. J. (1988). Metabolic and respiratory profile of the upper limit for prolonged exercise in man. Ergonomics, 31(9), 1265-1279.
        
        3. Moritani, T., Nagata, A., deVries, H. A., & Muro, M. (1981). Critical power as a measure of physical work capacity and anaerobic threshold. Ergonomics, 24(5), 339-350.
        """)

# Additional information section
st.markdown("---")
col1, col2 = st.columns([2, 1])

with col1:
    st.markdown("## Understanding Critical Power and W′")
    
    st.markdown("""
    **Critical Power (CP)** represents the highest power output that can be sustained "indefinitely" without fatigue (theoretically 30-60 minutes). It's closely related to the lactate threshold and is a key determinant of endurance performance.
    
    **W′ (W-prime)** represents the finite amount of work that can be performed above critical power. Think of it as your anaerobic work capacity or energy reserves.
    
    Together, these two parameters define your power-duration relationship and can be used to:
    1. Predict performance at different durations
    2. Set appropriate training zones
    3. Optimize pacing strategies
    4. Track changes in fitness over time
    """)

with col2:
    st.markdown("## Recommendations")
    
    beginner_recs = """
    - Focus on building aerobic endurance
    - Start with the 3-min test or ramp test
    - Re-test every 8-12 weeks
    - Target CP/kg: 2.5-3.5 W/kg
    """
    
    intermediate_recs = """
    - Include both threshold and VO2max training
    - Try the multiple time trial approach
    - Re-test every 6-8 weeks
    - Target CP/kg: 3.5-4.2 W/kg
    """
    
    advanced_recs = """
    - Periodize training based on CP and W′
    - Use more precise methods like multiple time trials
    - Monitor changes in both CP and W′
    - Target CP/kg: 4.2-5.0 W/kg
    """
    
    elite_recs = """
    - Detailed analysis of power-duration curve
    - Regular testing with multiple protocols
    - Optimize W′ balance for races
    - Target CP/kg: 5.0+ W/kg
    """
    
    if experience_level == "Beginner":
        st.markdown(beginner_recs)
    elif experience_level == "Intermediate":
        st.markdown(intermediate_recs)
    elif experience_level == "Advanced":
        st.markdown(advanced_recs)
    else:  # Elite
        st.markdown(elite_recs)

# W' Balance Simulator
st.markdown("---")
st.markdown("## W′ Balance Simulator")
st.markdown("""
This simulator helps you understand how W′ (your anaerobic capacity) gets depleted and recharged during a ride based on your estimated Critical Power.
""")

if 'cp' in locals() and 'w_prime' in locals():
    simulate_enabled = True
    default_cp = cp
    default_wprime = w_prime
else:
    simulate_enabled = False
    default_cp = 300
    default_wprime = 20000

col1, col2 = st.columns([1, 1])

with col1:
    if not simulate_enabled:
        st.markdown("#### Enter your values or calculate them using the methods above")
        sim_cp = st.number_input("Critical Power (W)", min_value=100, max_value=500, value=default_cp)
        sim_wprime = st.number_input("W′ (J)", min_value=5000, max_value=50000, value=default_wprime)
    else:
        sim_cp = default_cp
        sim_wprime = default_wprime
        st.markdown(f"#### Using calculated values: CP = {sim_cp:.1f}W, W′ = {sim_wprime:.0f}J")
    
    st.markdown("#### Simulation Parameters")
    duration = st.slider("Ride Duration (minutes)", min_value=5, max_value=60, value=20)
    
    # Create a simple ride profile
    profile_type = st.selectbox("Ride Profile", ["Steady State", "Intervals", "Variable Intensity", "Race Simulation"])
    
    if st.button("Run Simulation"):
        # Generate power data based on profile type
        time_axis = np.linspace(0, duration*60, duration*60)  # second-by-second
        
        if profile_type == "Steady State":
            intensity = st.slider("Intensity (% of CP)", min_value=70, max_value=110, value=95) / 100
            power_data = np.ones_like(time_axis) * sim_cp * intensity
            
        elif profile_type == "Intervals":
            work_intensity = st.slider("Work Interval Intensity (% of CP)", min_value=100, max_value=150, value=120) / 100
            rest_intensity = st.slider("Rest Interval Intensity (% of CP)", min_value=50, max_value=90, value=70) / 100
            interval_length = st.slider("Interval Length (seconds)", min_value=30, max_value=300, value=120)
            rest_length = st.slider("Rest Length (seconds)", min_value=30, max_value=300, value=60)
            
            power_data = np.zeros_like(time_axis)
            for i in range(len(time_axis)):
                cycle_position = i % (interval_length + rest_length)
                if cycle_position < interval_length:
                    power_data[i] = sim_cp * work_intensity
                else:
                    power_data[i] = sim_cp * rest_intensity
        
        elif profile_type == "Variable Intensity":
            # Create a variable ride with random fluctuations
            base_intensity = st.slider("Base Intensity (% of CP)", min_value=70, max_value=100, value=85) / 100
            variability = st.slider("Variability (%)", min_value=5, max_value=30, value=15)
            
            # Generate smooth random variations
            from scipy.ndimage import gaussian_filter1d
            random_variations = np.random.normal(0, variability/100, size=len(time_axis))
            smoothed_variations = gaussian_filter1d(random_variations, sigma=30)  # Smoothing factor
            
            power_data = (base_intensity + smoothed_variations) * sim_cp
            power_data = np.clip(power_data, 0.5*sim_cp, 1.5*sim_cp)  # Limit the range
            
        elif profile_type == "Race Simulation":
            # Simulates a race with a steady start, some attacks, and a finishing sprint
            power_data = np.ones_like(time_axis) * 0.9 * sim_cp  # Base at 90% CP
            
            # Add some attacks (3-4 random attacks)
            num_attacks = np.random.randint(3, 5)
            for _ in range(num_attacks):
                attack_start = np.random.randint(5*60, (duration-3)*60)
                attack_duration = np.random.randint(30, 120)  # 30s to 2min attacks
                attack_intensity = np.random.uniform(1.2, 1.4)  # 120-140% CP
                power_data[attack_start:attack_start+attack_duration] = sim_cp * attack_intensity
                
                # Add final sprint in last 30-60 seconds
                sprint_start = (duration * 60) - np.random.randint(30, 60)
                sprint_duration = np.random.randint(20, 40)
                power_data[sprint_start:sprint_start+sprint_duration] = sim_cp * 1.5  # 150% of CP for sprint
        
        # Calculate W' balance
        w_balance = w_prime_balance(power_data, sim_cp, sim_wprime)
        
        # Plot results
        with col2:
            fig, ax1 = plt.subplots(figsize=(10, 6))
            
            color = '#E6754E'
            ax1.set_xlabel('Time (minutes)')
            ax1.set_ylabel('Power (Watts)', color=color)
            ax1.plot(time_axis/60, power_data, color=color, alpha=0.7)
            ax1.tick_params(axis='y', labelcolor=color)
            ax1.axhline(y=sim_cp, color=color, linestyle='--', alpha=0.7, label=f'CP: {sim_cp:.0f}W')
            
            # Create second y-axis for W' balance
            ax2 = ax1.twinx()
            color = 'blue'
            ax2.set_ylabel('W′ Balance (J)', color=color)
            ax2.plot(time_axis/60, w_balance, color=color)
            ax2.tick_params(axis='y', labelcolor=color)
            ax2.axhline(y=sim_wprime, color=color, linestyle='--', alpha=0.7, label=f'W′: {sim_wprime:.0f}J')
            
            # Add a W' depletion danger zone
            ax2.axhspan(0, 2000, color='red', alpha=0.2, label='Danger Zone')
            
            # Combine legends
            lines1, labels1 = ax1.get_legend_handles_labels()
            lines2, labels2 = ax2.get_legend_handles_labels()
            ax1.legend(lines1 + lines2, labels1 + labels2, loc='upper right')
            
            plt.title('Power Output and W′ Balance Simulation')
            plt.grid(True, alpha=0.3)
            st.pyplot(fig)
            
            # Summary statistics
            st.markdown('<div class="result-box">', unsafe_allow_html=True)
            st.markdown("### Simulation Summary")
            st.markdown(f"**Average Power:** {np.mean(power_data):.1f} Watts ({np.mean(power_data)/sim_cp*100:.1f}% of CP)")
            st.markdown(f"**Normalized Power (estimated):** {(np.mean(power_data**4)**(1/4)):.1f} Watts")
            st.markdown(f"**Time Above CP:** {np.sum(power_data > sim_cp)/60:.1f} minutes")
            st.markdown(f"**Minimum W′ Balance:** {min(w_balance):.0f} Joules")
            st.markdown(f"**W′ Expended:** {sim_wprime - min(w_balance):.0f} Joules ({(sim_wprime - min(w_balance))/sim_wprime*100:.1f}% of total)")
            st.markdown('</div>', unsafe_allow_html=True)
